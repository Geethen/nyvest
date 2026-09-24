"""Combine the TIED ideas — but as an ensemble, not as a stack.

Motivation (user's suggestion, refined by the evidence): swiglu / rmsnorm / sam /
ema / hetero-dropout all TIE the plain MLP. Two ways to combine them:

  1. STACK them into one model (swiglu + rmsnorm + sam + ...). Rejected: the
     per-fold deltas show all five ties are negative-LEANING (12 of 15 fold
     deltas < 0). They are not small positives lost in noise; they are small
     negatives. Stacking stacks the negatives. And the oracle bound (pick the
     best method PER FOLD, with hindsight) is only +0.0011 — below the +0.003
     noise floor. So no combination of these can win as a stack.

  2. ENSEMBLE across them (this file). Different architectures/objectives make
     DIFFERENT errors, and probability-averaging over diverse-but-equal models is
     the one lever that has ever worked here (1-seed 0.7248 -> 5-seed 0.7318).
     Heterogeneous ensembling is strictly more diverse than seed ensembling: the
     members differ in inductive bias, not just initialization.

This is the honest test of "combine the ties": they can't help as a stack, but
their diversity is real and might help as a mixture.

MEMBERS (each contributes N_PER members to the average):
  plain MLP, SwiGLU, RMSNorm  — architectural diversity (all tied)
  + optionally SAM/EMA-trained plain MLPs (VARIANT=all) — objective diversity

VARIANT:
  arch  - plain + swiglu + rmsnorm  (3 archs x N_PER seeds)
  all   - arch + sam-trained + ema-trained plain MLPs (5 kinds x N_PER seeds)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec           # noqa: E402
import data_utils as du           # noqa: E402
import stage3_robust_mlp as s3    # noqa: E402
from exp_llm_ideas import SwiGLUNet, RMSNormNet   # noqa: E402
import exp_generalization as eg   # noqa: E402

VARIANT = os.environ.get("VARIANT", "arch")
N_PER = int(os.environ.get("N_PER", "3"))   # seeds per member kind


def member_specs():
    """(label, build_fn, train_fn) per member kind. train_fn=None -> standard."""
    specs = [
        ("mlp", lambda i, c: s3.MLP(i, c, ec.HIDDEN, ec.DROPOUT), None),
        ("swiglu", lambda i, c: SwiGLUNet(i, c, ec.HIDDEN, ec.DROPOUT), None),
        ("rmsnorm", lambda i, c: RMSNormNet(i, c, ec.HIDDEN, ec.DROPOUT), None),
    ]
    if VARIANT == "all":
        specs += [
            ("sam", lambda i, c: s3.MLP(i, c, ec.HIDDEN, ec.DROPOUT),
             eg.train_one_sam),
            ("ema", lambda i, c: s3.MLP(i, c, ec.HIDDEN, ec.DROPOUT),
             eg.train_one_ema),
        ]
    return specs


def run():
    t0 = time.perf_counter()
    base_pf, base_mean, base_name = ec.baseline_for(ec.RELABEL)
    specs = member_specs()
    name = f"combine_{VARIANT}_n{N_PER}"
    total = len(specs) * N_PER
    print(f"=== {name} ===  kinds={[s[0] for s in specs]}  "
          f"seeds_each={N_PER}  total_members={total}")
    print(f"baseline={base_name} per_fold={base_pf}")
    print("NOTE: baseline uses 5 members; this uses "
          f"{total}. The 15-seed run (ens_15seed) is the size-matched control "
          "that separates 'diversity helped' from 'more members helped'.")

    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats")

    rng = np.random.default_rng(ec.SEED)
    f1s, pcs, per_kind_f1 = [], [], {s[0]: [] for s in specs}
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if ec.CLEAN_CLS12:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        Xtr, ytr, Xte = X[tr_use], y_enc[tr_use], X[te]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)
        n = len(Xtr_s)
        perm = rng.permutation(n)
        nv = int(n * s3.VAL_FRAC)
        vi, ti = perm[:nv], perm[nv:]
        Xtr_t = torch.tensor(Xtr_s[ti], device=ec.DEVICE)
        ytr_t = torch.tensor(ytr[ti], device=ec.DEVICE)
        Xval_t = torch.tensor(Xtr_s[vi], device=ec.DEVICE)
        Xte_t = torch.tensor(Xte_s, device=ec.DEVICE)
        yval = ytr[vi]
        w = s3.class_weights(ytr[ti], n_classes, s3.WEIGHT_MODE)

        P = np.zeros((len(Xte_s), n_classes))
        for label, build, train_fn in specs:
            Pk = np.zeros((len(Xte_s), n_classes))
            for e in range(N_PER):
                tf = train_fn or ec.train_one
                model, vf1 = tf(build, Xtr_t, ytr_t, Xval_t, yval,
                                Xtr_s.shape[1], n_classes, w, ec.SEED + 100 * e)
                p = ec.probs(model, Xte_t, n_classes)
                Pk += p
                P += p
                del model
                torch.cuda.empty_cache()
            # what this kind alone would have scored (diversity diagnostic)
            per_kind_f1[label].append(
                round(du.macro_f1(y_enc[te], (Pk / N_PER).argmax(1), n_classes), 4))
        pred = (P / total).argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  Δ={f1-base_pf[k]:+.4f}  "
              f"({time.perf_counter()-ft:.1f}s)  per-kind="
              f"{ {kk: v[-1] for kk, v in per_kind_f1.items()} }")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    deltas = [f1s[i] - base_pf[i] for i in range(len(f1s))]
    dmean = float(np.mean(deltas))
    all_pos = all(d > 0 for d in deltas)
    verdict = "WIN" if (all_pos and dmean > 0.003) else ("tie" if abs(dmean) <= 0.003 else "LOSS")
    pc_mean = np.mean(pcs, 0)
    print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}  Δ_mean={dmean:+.4f} "
          f"per_fold_Δ={[round(d,4) for d in deltas]}")
    print(f"VERDICT: {verdict}")
    print(f"per-kind solo F1: { {k: round(float(np.mean(v)),4) for k,v in per_kind_f1.items()} }")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")

    rec = {"name": name,
           "notes": f"heterogeneous ensemble of TIED ideas ({'+'.join(s[0] for s in specs)}), "
                    f"{N_PER} seeds each = {total} members; compare vs ens_15seed "
                    "to separate diversity from member count",
           "relabel": ec.RELABEL, "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
           "f1_per_fold": [round(v, 4) for v in f1s], "baseline": base_name,
           "baseline_per_fold": base_pf, "delta_per_fold": [round(d, 4) for d in deltas],
           "delta_mean": round(dmean, 4), "all_folds_positive": all_pos,
           "verdict": verdict,
           "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
           "per_kind_solo_f1": {k: [float(x) for x in v] for k, v in per_kind_f1.items()},
           "n_ensemble": total, "runtime_s": round(time.perf_counter() - t0, 1),
           "config": {"idea": "ensembling", "variant": VARIANT, "n_per": N_PER}}
    (ec.RESULTS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"saved -> {ec.RESULTS_DIR / f'{name}.json'}")


if __name__ == "__main__":
    run()

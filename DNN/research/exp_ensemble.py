"""Batch 4 — better USE of the ensemble, the one lever that has ever worked.

Everything else has tied: architecture (swiglu/rmsnorm/deepnarrow/softmoe),
flat-minima (sam/ema), and — pending — invariance (irm/groupdro). The single
reliable gain in this whole project is the 5-seed probability ensemble
(0.7248 -> 0.7318). So instead of new architectures, interrogate the ENSEMBLE
itself: uniform probability averaging is an arbitrary choice, and prior work
(conformal_compare) showed this model's uncertainty is well calibrated, i.e. it
carries real signal about when a member is trustworthy.

EXP variants:

  logitavg   - Average LOGITS instead of probabilities. Probability averaging is
               a mixture (arithmetic mean); logit averaging is a product-of-
               experts (geometric mean), which sharpens agreement and suppresses
               a single confident-but-wrong member. One-line change, real
               mechanism, never tested here.
  tempscale  - Temperature-scale each member's logits (T fit on the member's own
               val split, leak-free) BEFORE averaging. Prior work found the DNN
               is over-confident (T~0.78); averaging over-confident members
               over-weights whoever is most confidently wrong. Fixes the weight
               distortion that uniform averaging assumes away.
  bigens     - 15-seed ensemble instead of 5. Pure variance reduction along the
               ONE axis known to work. Tests whether the ensemble gain has
               saturated at 5 or still has headroom. If this is the only winner,
               that is itself the finding: buy F1 with compute, not cleverness.
  entropy    - Per-row entropy-weighted averaging: weight each member's
               contribution by (1 - normalized entropy) so confident members
               dominate on rows they understand. Uses the calibrated uncertainty
               as a per-row gate — the "routing" idea that MoE tried and failed
               at, but applied at the OUTPUT where no capacity is added.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec          # noqa: E402
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

EXP = os.environ.get("EXP", "logitavg")
N_ENS = int(os.environ.get("N_ENS", "15" if EXP == "bigens" else "5"))


def raw_logits(model, X_t, n_classes, bs=16384):
    model.eval()
    out = np.zeros((X_t.shape[0], n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], bs):
            out[i:i + bs] = model(X_t[i:i + bs]).cpu().numpy()
    return out


def fit_temperature(logits_val, y_val):
    """1-param temperature fit by NLL on the member's OWN val split (leak-free)."""
    lg = torch.tensor(logits_val, dtype=torch.float32, device=ec.DEVICE)
    y = torch.tensor(y_val, dtype=torch.long, device=ec.DEVICE)
    logT = torch.zeros(1, device=ec.DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp().item())


def run():
    t0 = time.perf_counter()
    base_pf, base_mean, base_name = ec.baseline_for(ec.RELABEL)
    name = {"logitavg": "ens_logit_avg", "tempscale": "ens_temp_scaled",
            "bigens": f"ens_{N_ENS}seed", "entropy": "ens_entropy_weighted"}[EXP]
    print(f"=== {name} ===  n_ens={N_ENS}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats")

    rng = np.random.default_rng(ec.SEED)
    f1s, pcs, temps = [], [], []
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
        yval_np = ytr[vi]
        w = s3.class_weights(ytr[ti], n_classes, s3.WEIGHT_MODE)

        acc = None            # accumulator (probs or logits depending on variant)
        ent_num = None        # entropy-weighted numerator
        ent_den = None
        for e in range(N_ENS):
            model, vf1 = ec.train_one(
                lambda i, c: s3.MLP(i, c, ec.HIDDEN, ec.DROPOUT),
                Xtr_t, ytr_t, Xval_t, yval_np, Xtr_s.shape[1], n_classes,
                w, ec.SEED + 100 * e)
            lg_te = raw_logits(model, Xte_t, n_classes)
            if EXP == "tempscale":
                T = fit_temperature(raw_logits(model, Xval_t, n_classes), yval_np)
                temps.append(round(T, 3))
                lg_te = lg_te / T
            if EXP == "logitavg" or EXP == "tempscale":
                acc = lg_te if acc is None else acc + lg_te
            elif EXP == "bigens":
                p = torch.softmax(torch.tensor(lg_te), 1).numpy()
                acc = p if acc is None else acc + p
            else:  # entropy weighting
                p = torch.softmax(torch.tensor(lg_te), 1).numpy()
                ent = -(p * np.log(p + 1e-12)).sum(1) / np.log(n_classes)  # 0..1
                wt = (1.0 - ent)[:, None]
                ent_num = p * wt if ent_num is None else ent_num + p * wt
                ent_den = wt if ent_den is None else ent_den + wt
            del model
            torch.cuda.empty_cache()

        if EXP == "entropy":
            P = ent_num / np.maximum(ent_den, 1e-9)
        else:
            P = acc / N_ENS
        pred = P.argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  Δ={f1-base_pf[k]:+.4f}  "
              f"({time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    deltas = [f1s[i] - base_pf[i] for i in range(len(f1s))]
    dmean = float(np.mean(deltas))
    all_pos = all(d > 0 for d in deltas)
    verdict = "WIN" if (all_pos and dmean > 0.003) else ("tie" if abs(dmean) <= 0.003 else "LOSS")
    pc_mean = np.mean(pcs, 0)
    print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}  Δ_mean={dmean:+.4f} "
          f"per_fold_Δ={[round(d,4) for d in deltas]}")
    print(f"VERDICT: {verdict}")
    if temps:
        print(f"fitted temperatures: {temps}")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")

    rec = {"name": name, "notes": {
               "logitavg": "average logits (product-of-experts) not probabilities",
               "tempscale": "per-member temperature scaling before averaging",
               "bigens": f"{N_ENS}-seed ensemble (vs 5) — pure variance reduction",
               "entropy": "per-row entropy-weighted member averaging"}[EXP],
           "relabel": ec.RELABEL, "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
           "f1_per_fold": [round(v, 4) for v in f1s], "baseline": base_name,
           "baseline_per_fold": base_pf, "delta_per_fold": [round(d, 4) for d in deltas],
           "delta_mean": round(dmean, 4), "all_folds_positive": all_pos,
           "verdict": verdict,
           "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
           "n_ensemble": N_ENS, "runtime_s": round(time.perf_counter() - t0, 1),
           "config": {"idea": "ensembling", "temps": temps or None}}
    (ec.RESULTS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"saved -> {ec.RESULTS_DIR / f'{name}.json'}")


if __name__ == "__main__":
    run()

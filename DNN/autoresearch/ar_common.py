"""Autoresearch harness — hold everything fixed, vary ONE mechanism, log to wandb.

Successor to `DNN/research/exp_common.py`. Same statistical discipline (paired
per-fold deltas against a FRESH baseline run of this harness, not a stored number
from a different script), plus three things the old harness could not express:

  1. PER-CLASS accounting is first class. The goal here is not just macro-F1 but
     "lift the weak classes (2 rock/sand, 5 grassland, 6 scrub, 7 wetland)
     WITHOUT paying for it out of the strong ones (8 water, 10 built, 3 crop,
     11 sparse-veg, 12 snow, 4 forest)". Every trial reports per-class paired
     deltas and gets a TRADEOFF verdict alongside the macro verdict.

  2. Hooks for mechanisms that are not "a different nn.Module": custom optimizer,
     epoch-aware loss, post-fit surgery on the trained net (classifier
     retraining, tau-normalisation), post-hoc decision rules fitted on a
     leak-free inner split, and full per-fold overrides (specialist cascades,
     self-distillation, transductive adaptation).

  3. A leak-free inner CALIBRATION split. Decision-rule work (per-class priors,
     plug-in macro-F1 maximisation, target-prior EM) must be fitted on data the
     model did not train on AND that is spatially held out, else it tunes to the
     train regions and evaporates on the test fold. `val_mode="group"` carves the
     inner val by whole cell_ids so it is a faithful transfer estimate.

STATISTICS (inherited, non-negotiable):
  fold spread ~0.02 >> effects chased ~0.003. Folds are deterministic
  (GroupKFold on cell_id) so the same fold is the same test set in every run.
  Compare PAIRED per-fold deltas. WIN = all folds positive AND mean delta >
  0.003. Anything else is a tie or a loss, no matter how good the mean looks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

DEVICE = s3.DEVICE
# Seed for the inner val split AND the ensemble members. Overridable so a
# promising result can be re-run on an INDEPENDENT draw: with ~40 trials the best
# one is selected on the same noise it is measured with, and a fresh seed set is
# the only way to tell a real effect from the maximum of many null draws.
SEED = int(os.environ.get("AR_SEED", "0"))
HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("AR_RESULTS_DIR", HERE / "results"))
LOGS_DIR = HERE / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
BASELINE_JSON = RESULTS_DIR / "_baseline.json"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "nyvest-dnn-autoresearch")

# ---------------------------------------------------------------- fixed recipe
# The deployed model: stage-3 robust MLP + stage-8 to12_fix cls12 relabel.
RELABEL = os.environ.get("RELABEL", "to12_fix")
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
HIDDEN = tuple(int(x) for x in os.environ.get("HIDDEN", "256,128").split(","))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "200"))
PATIENCE = int(os.environ.get("PATIENCE", "15"))
BATCH = int(os.environ.get("BATCH", "4096"))
VAL_FRAC = 0.1
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
LABEL_SMOOTH = 0.05

# Weak vs strong classes, by raw class code. Set from the reference per-class F1
# profile (see README / ontology_baseline.json). Trials are judged on lifting
# WEAK without denting STRONG.
WEAK_CLASSES = [2, 5, 6, 7]        # rock/sand .55, grassland .60, scrub .66, wetland .69
STRONG_CLASSES = [3, 4, 8, 10, 11, 12]
HIGH_DROP_TOL = 0.005              # a strong class may not fall more than this


# ------------------------------------------------------------------ trial spec
@dataclass
class Trial:
    """One mechanism under test. Every field not set falls back to the baseline."""
    name: str
    tier: str
    idea: str                                   # what is being changed
    hypothesis: str                             # why it might beat the baseline
    provenance: str = ""                        # paper / origin
    build_fn: Optional[Callable] = None         # (in_dim, n_classes, ctx) -> nn.Module
    opt_fn: Optional[Callable] = None           # (model, ctx) -> torch.optim.Optimizer
    loss_fn: Optional[Callable] = None          # (model, xb, yb, crit, ctx) -> loss
    after_fit: Optional[Callable] = None        # (model, ctx) -> model  (post-fit surgery)
    post_fn: Optional[Callable] = None          # (P_te, P_val, y_val, ctx) -> (P_te', info)
    fold_fn: Optional[Callable] = None          # (ctx) -> (P_te, val_f1, info)  full override
    val_mode: str = "random"                    # random | group  (inner val split)
    extra_gate: Optional[str] = None            # "geo" -> append lon/lat as GATE-ONLY cols
    weight_mode: str = "sqrt"                   # sqrt | inv | none  (class weights in CE)
    label_smooth: float = LABEL_SMOOTH
    n_ensemble: int = N_ENSEMBLE
    transductive: bool = False                  # uses UNLABELLED test features
    is_bound: bool = False                      # uses test LABELS -> not a result
    save_probs: bool = False
    config: dict = field(default_factory=dict)  # extra hyperparameters -> wandb


def default_mlp(in_dim, n_classes, ctx):
    """The exact deployed net: 2 hidden layers, ReLU, dropout 0.3."""
    h1, h2 = HIDDEN
    return nn.Sequential(
        nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(h2, n_classes),
    )


def default_opt(model, ctx):
    return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# ------------------------------------------------------------------- utilities
def set_seed(s):
    s3.set_seed(s)


def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def predict_probs(model, X_t, n_classes, bs=16384):
    """Class probabilities. Modules that already emit LOG-probabilities (the
    hierarchical head, TabM's member average) flag it with `returns_log_probs`
    so we exponentiate instead of double-softmaxing them."""
    model.eval()
    log_p = getattr(model, "returns_log_probs", False)
    out = np.zeros((X_t.shape[0], n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], bs):
            z = logits_of(model, X_t[i:i + bs])
            out[i:i + bs] = (z.exp() if log_p else F.softmax(z, 1)).cpu().numpy()
    return out


def class_weights(y_enc, n_classes, mode):
    return s3.class_weights(y_enc, n_classes, mode) if mode != "none" else None


def val_split(n, groups_tr, rng, mode, frac=VAL_FRAC):
    """Return (val_idx, tr_idx) into the training fold.

    random : the deployed recipe's iid 10% holdout (in-distribution).
    group  : whole cell_ids held out, so inner-val is a spatial-TRANSFER
             estimate — required for anything fitted on val that must survive
             the test fold's unseen regions.
    """
    if mode == "random":
        perm = rng.permutation(n)
        n_val = int(n * frac)
        return perm[:n_val], perm[n_val:]
    cells = np.unique(groups_tr)
    order = rng.permutation(len(cells))
    sizes = pd.Series(groups_tr).value_counts()
    target, taken, chosen = int(n * frac), 0, []
    for ci in order:
        c = cells[ci]
        chosen.append(c)
        taken += int(sizes[c])
        if taken >= target:
            break
    is_val = np.isin(groups_tr, np.array(chosen))
    return np.flatnonzero(is_val), np.flatnonzero(~is_val)


# ------------------------------------------------------------------- training
def train_member(trial, ctx, seed):
    """Train one ensemble member under `trial`'s hooks. Early-stops on inner-val
    macro-F1 exactly like the deployed recipe."""
    set_seed(seed)
    build = trial.build_fn or default_mlp
    model = build(ctx["in_dim"], ctx["n_classes"], ctx).to(DEVICE)
    opt = (trial.opt_fn or default_opt)(model, ctx)
    crit = nn.CrossEntropyLoss(weight=ctx["w"], label_smoothing=trial.label_smooth)
    Xtr_t, ytr_t = ctx["Xtr_t"], ctx["ytr_t"]
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        ctx["epoch"] = epoch
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, BATCH):
            b = order[i:i + BATCH]
            xb, yb = Xtr_t[b], ytr_t[b]
            opt.zero_grad()
            if trial.loss_fn:
                loss = trial.loss_fn(model, xb, yb, crit, ctx)
            else:
                loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = logits_of(model, ctx["Xval_t"]).argmax(1).cpu().numpy()
        f1 = du.macro_f1(ctx["yval_np"], vp, ctx["n_classes"])
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    if trial.after_fit:
        model = trial.after_fit(model, ctx)
    return model, best_f1


def run_fold(trial, ctx):
    """Default per-fold procedure: N-seed probability ensemble (+ optional
    post-hoc decision rule fitted on the inner val)."""
    n_classes = ctx["n_classes"]
    P_te = np.zeros((ctx["Xte_t"].shape[0], n_classes), dtype=np.float64)
    P_val = np.zeros((ctx["Xval_t"].shape[0], n_classes), dtype=np.float64)
    vf1s = []
    for e in range(trial.n_ensemble):
        model, vf1 = train_member(trial, ctx, SEED + 100 * e)
        P_te += predict_probs(model, ctx["Xte_t"], n_classes)
        P_val += predict_probs(model, ctx["Xval_t"], n_classes)
        vf1s.append(vf1)
        ctx["last_model"] = model
        del model
        torch.cuda.empty_cache()
    P_te /= trial.n_ensemble
    P_val /= trial.n_ensemble
    info = {"P_val": P_val.astype(np.float32), "y_val": ctx["yval_np"]}
    if trial.post_fn:
        P_raw = P_te.copy()
        P_te, extra = trial.post_fn(P_te, P_val, ctx["yval_np"], ctx)
        # internal control: the SAME models scored by plain argmax, so the delta
        # attributable to the decision rule alone is perfectly paired.
        info["f1_raw_argmax"] = du.macro_f1(ctx["y_te"], P_raw.argmax(1), n_classes)
        info.update(extra or {})
    return P_te, float(np.mean(vf1s)), info


# ------------------------------------------------------------------ evaluation
def _tradeoff_report(pc_delta, classes):
    """Per-class paired deltas -> the 'did we pay for it?' verdict."""
    idx = {c: i for i, c in enumerate(classes)}
    weak = [pc_delta[idx[c]] for c in WEAK_CLASSES if c in idx]
    strong = [pc_delta[idx[c]] for c in STRONG_CLASSES if c in idx]
    worst_strong_drop = float(min(strong)) if strong else 0.0
    return {
        "weak_gain_mean": float(np.mean(weak)) if weak else 0.0,
        "weak_gain_max": float(np.max(weak)) if weak else 0.0,
        "strong_drop_worst": worst_strong_drop,
        "no_tradeoff": bool(worst_strong_drop >= -HIGH_DROP_TOL),
    }


def load_baseline():
    if not BASELINE_JSON.exists():
        raise SystemExit("no baseline yet — run `python run.py baseline` first")
    return json.loads(BASELINE_JSON.read_text())


def run_trial(trial, wandb_run=None):
    t0 = time.perf_counter()
    is_baseline = trial.name == "baseline"
    base = None if is_baseline else load_baseline()

    print(f"=== {trial.name} [{trial.tier}] ===")
    print(f"idea       : {trial.idea}")
    print(f"hypothesis : {trial.hypothesis}")
    print(f"device={DEVICE} relabel={RELABEL} ens={trial.n_ensemble} "
          f"val_mode={trial.val_mode} weights={trial.weight_mode}")

    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    if RELABEL != "none":
        y_enc, n_ch = du.apply_cls12_relabel(y_enc, classes, RELABEL)
        print(f"relabel={RELABEL}: {n_ch} labels changed")
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs, infos, probs_out, vf1s = [], [], [], {}, []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if CLEAN_CLS12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]

        scaler = StandardScaler().fit(X[tr_use])
        Xtr = scaler.transform(X[tr_use]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)
        # Gate-only columns: appended AFTER the features so a module can slice
        # them off for its router while the experts still see exactly the 67
        # deployed features. Coordinates must never reach the trunk — that would
        # make a position-gated control arm a different MODEL, not a different
        # router, and the comparison it exists for would be meaningless.
        n_gate_extra = 0
        if trial.extra_gate == "geo":
            G = np.stack([lon, lat], 1)
            gsc = StandardScaler().fit(G[tr_use])
            Xtr = np.hstack([Xtr, gsc.transform(G[tr_use])]).astype(np.float32)
            Xte = np.hstack([Xte, gsc.transform(G[te])]).astype(np.float32)
            n_gate_extra = 2
        v_idx, t_idx = val_split(len(Xtr), groups[tr_use], rng, trial.val_mode)
        ctx = {
            "fold": k, "n_classes": n_classes, "classes": classes,
            "in_dim": Xtr.shape[1], "trial": trial, "rng": rng, "scaler": scaler,
            "Xtr_t": torch.tensor(Xtr[t_idx], device=DEVICE),
            "ytr_t": torch.tensor(y_enc[tr_use][t_idx], device=DEVICE),
            "Xval_t": torch.tensor(Xtr[v_idx], device=DEVICE),
            "yval_np": y_enc[tr_use][v_idx],
            "Xte_t": torch.tensor(Xte, device=DEVICE),
            "y_te": y_enc[te],
            "ytr_np": y_enc[tr_use][t_idx],
            "groups_tr": groups[tr_use][t_idx],
            "lidar_cols": [data["feat_cols"].index(c) for c in du.LIDAR_COLS
                           if c in data["feat_cols"]],
            "n_gate_extra": n_gate_extra,
            # raw coordinates, for trials that must PARTITION by geography even
            # though they may not FEED it to the network (regional fine-tuning,
            # locality diagnostics)
            "lonlat_tr": np.stack([lon[tr_use][t_idx], lat[tr_use][t_idx]], 1),
            "lonlat_te": np.stack([lon[te], lat[te]], 1),
        }
        ctx["counts"] = np.bincount(ctx["ytr_np"], minlength=n_classes).astype(np.float64)
        ctx["w"] = class_weights(ctx["ytr_np"], n_classes, trial.weight_mode)

        if trial.fold_fn:
            P, vf1, info = trial.fold_fn(ctx)
        else:
            P, vf1, info = run_fold(trial, ctx)

        pred = P.argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        pc = du.per_class_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        vf1s.append(vf1)
        pcs.append(pc)
        infos.append({kk: vv for kk, vv in info.items() if not isinstance(vv, np.ndarray)})
        if trial.save_probs:
            probs_out[f"fold{k}_P"] = P.astype(np.float32)
            probs_out[f"fold{k}_y"] = y_enc[te]
            if "P_val" in info:
                probs_out[f"fold{k}_Pval"] = info["P_val"]
                probs_out[f"fold{k}_yval"] = info["y_val"]
        msg = f"  fold {k}: F1={f1:.4f}"
        if base:
            msg += f"  Δ={f1 - base['f1_per_fold'][k]:+.4f}"
        msg += f"  (val={vf1:.4f}, {time.perf_counter()-ft:.1f}s)"
        print(msg)
        if wandb_run is not None:
            wandb_run.log({"fold": k, "fold_f1": f1, "fold_val_f1": vf1,
                           **{f"fold_f1_class_{c}": float(v) for c, v in zip(classes, pc)}})
        for key in ("Xtr_t", "ytr_t", "Xval_t", "Xte_t", "last_model"):
            ctx.pop(key, None)
        torch.cuda.empty_cache()

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    rec = {
        "name": trial.name, "tier": trial.tier, "idea": trial.idea,
        "hypothesis": trial.hypothesis, "provenance": trial.provenance,
        "relabel": RELABEL, "transductive": trial.transductive,
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        # Inner-val F1 alongside test F1. The gap between them is the whole story
        # of this model: every study here has found mechanisms that lift val and
        # leave test untouched, and a mechanism that does that has been learned
        # and has failed to cross the fold boundary — a different result from one
        # that simply did not learn.
        "val_f1_per_fold": [round(v, 4) for v in vf1s],
        "val_f1_mean": round(float(np.mean(vf1s)), 4),
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
        "val_mode": trial.val_mode, "weight_mode": trial.weight_mode,
        "n_ensemble": trial.n_ensemble,
        "config": {"hidden": list(HIDDEN), "dropout": DROPOUT, "lr": LR,
                   **trial.config},
        "fold_info": infos,
        "runtime_s": round(time.perf_counter() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # A trial that carves its inner val by whole cells pays the group-val tax
    # (-0.0042) for reasons that have nothing to do with the mechanism under
    # test. Scoring it against the random-val baseline would charge it twice, so
    # record the delta against the matched control as well.
    ctrl_name = "baseline_gval" if trial.val_mode == "group" else None
    ctrl = None
    if ctrl_name and (RESULTS_DIR / f"{ctrl_name}.json").exists():
        ctrl = json.loads((RESULTS_DIR / f"{ctrl_name}.json").read_text())
        cd = [f1s[k] - ctrl["f1_per_fold"][k] for k in range(len(f1s))]
        rec["control"] = f"{ctrl_name} ({ctrl['f1_mean']})"
        rec["delta_vs_control_mean"] = round(float(np.mean(cd)), 4)
        rec["delta_vs_control_per_fold"] = [round(v, 4) for v in cd]

    if base:
        deltas = [f1s[k] - base["f1_per_fold"][k] for k in range(len(f1s))]
        dmean = float(np.mean(deltas))
        all_pos = all(d > 0 for d in deltas)
        # A bound consults the test LABELS to pick among predictions, so it is
        # not achievable and must never enter the leaderboard as a win. It earns
        # its GPU time by saying what the best possible version of a mechanism
        # would score — a null is only interpretable next to its ceiling.
        # A mean above the bar that is NOT positive on every fold is the exact
        # failure mode this harness exists to catch, and it needs its own name.
        # The original rule ("WIN, else tie, else LOSS") sent it to LOSS, which
        # reads as "this made the model worse" when the mean is strongly
        # positive — moe_shared@n_experts=8 scored +0.0040 off a single fold
        # (-0.0003 / +0.0004 / +0.0117) and was labelled a loss. `split` says
        # what actually happened: big mean, one fold carrying it, not a win.
        verdict = ("upper-bound" if trial.is_bound else
                   "WIN" if (all_pos and dmean > 0.003) else
                   "tie" if abs(dmean) <= 0.003 else
                   "split" if dmean > 0.003 else "LOSS")
        pc_delta = np.array([pc_mean[i] - base["f1_per_class"][str(c)]
                             for i, c in enumerate(classes)])
        rec.update({
            "baseline_per_fold": base["f1_per_fold"],
            "baseline_mean": base["f1_mean"],
            "delta_per_fold": [round(d, 4) for d in deltas],
            "delta_mean": round(dmean, 4),
            "all_folds_positive": all_pos, "verdict": verdict,
            "delta_per_class": {str(c): round(float(v), 4)
                                for c, v in zip(classes, pc_delta)},
            **_tradeoff_report(pc_delta, classes),
        })
        print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}  Δ_mean={dmean:+.4f} "
              f"per_fold_Δ={[round(d, 4) for d in deltas]}")
        print(f"VERDICT: {verdict}   no_tradeoff={rec['no_tradeoff']} "
              f"(weak-class mean {rec['weak_gain_mean']:+.4f}, worst strong "
              f"{rec['strong_drop_worst']:+.4f})")
        for c, v, d in zip(classes, pc_mean, pc_delta):
            tag = "weak" if c in WEAK_CLASSES else "    "
            print(f"  class {c:2d} {tag}: {v:.4f}  Δ={d:+.4f}")
    else:
        rec["verdict"] = "baseline"
        print(f"\nBASELINE F1 mean={f1m:.4f} std={f1std:.4f}")
        for c, v in zip(classes, pc_mean):
            print(f"  class {c:2d}: {v:.4f}")

    out = RESULTS_DIR / f"{trial.name}.json"
    out.write_text(json.dumps(rec, indent=2))
    if is_baseline:
        BASELINE_JSON.write_text(json.dumps(rec, indent=2))
    if trial.save_probs:
        np.savez_compressed(RESULTS_DIR / f"{trial.name}_probs.npz", **probs_out)
    print(f"saved -> {out}   ({rec['runtime_s']}s)")
    return rec

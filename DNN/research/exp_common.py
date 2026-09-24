"""Shared harness for the autonomous architecture-research loop.

Every experiment must be comparable to the Stage-3 baseline (0.7318) and the
Stage-8 to12_fix best (0.7341). The ONLY way to guarantee that is to hold the
data, folds, cleaning, class weights, scaler, ensembling and early-stopping
identical, and vary exactly one thing: how a model turns X -> logits.

So an experiment here is just a `build_fn(in_dim, n_classes) -> nn.Module` plus
an optional custom loss. Everything else is this file.

STATISTICS NOTE (why this file reports paired deltas):
  fold std is ~0.009 and the fold-2 vs fold-1 spread is ~0.02, i.e. much LARGER
  than the deltas we are chasing (~0.002). Comparing raw means across runs is
  therefore mostly comparing fold noise. But folds are FIXED (GroupKFold on
  cell_id, deterministic), so the same fold is the same test set in every run.
  We exploit that: report per-fold delta vs the stored baseline per-fold scores
  and their mean/std. A result only counts as real if the paired delta is
  positive on ALL folds and its mean exceeds ~1 seed-noise sigma (~0.003).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Stage-3 baseline per-fold scores (RELABEL=none, lidar, 5-seed ensemble).
BASELINE_PER_FOLD = [0.7378, 0.7392, 0.7184]
BASELINE_MEAN = 0.7318
# Stage-8 to12_fix per-fold (RELABEL=to12_fix) — the overall best.
BASELINE_RELABEL_PER_FOLD = [0.7370, 0.7433, 0.7220]
BASELINE_RELABEL_MEAN = 0.7341

# Shared knobs (defaults = the winning Stage-3 recipe)
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
RELABEL = os.environ.get("RELABEL", "none")   # none | cls12_fix | to12_fix
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
HIDDEN = tuple(int(x) for x in os.environ.get("HIDDEN", "256,128").split(","))
LR = float(os.environ.get("LR", "1e-3"))


def baseline_for(relabel: str):
    if relabel == "none":
        return BASELINE_PER_FOLD, BASELINE_MEAN, "stage3 (0.7318)"
    return BASELINE_RELABEL_PER_FOLD, BASELINE_RELABEL_MEAN, "stage8 to12_fix (0.7341)"


def train_one(build_fn, Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed,
              loss_fn=None):
    """Train one member. `build_fn(in_dim, n_classes) -> nn.Module` emitting logits.
    `loss_fn(model, xb, yb, crit) -> loss` optionally overrides the plain CE step
    (for methods that need aux losses / custom forward)."""
    s3.set_seed(seed)
    model = build_fn(in_dim, n_classes).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=s3.LABEL_SMOOTH)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=s3.WEIGHT_DECAY)
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(s3.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, s3.BATCH):
            b = order[i:i + s3.BATCH]
            xb, yb = Xtr_t[b], ytr_t[b]
            opt.zero_grad()
            loss = loss_fn(model, xb, yb, crit) if loss_fn else crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = _logits(model, Xval_t).argmax(1).cpu().numpy()
        f1 = du.macro_f1(yval_np, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= s3.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def _logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def probs(model, X_t, n_classes, bs=16384):
    model.eval()
    out = np.zeros((X_t.shape[0], n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], bs):
            out[i:i + bs] = F.softmax(_logits(model, X_t[i:i + bs]), 1).cpu().numpy()
    return out


def train_fold(build_fn, Xtr, ytr, Xte, n_classes, rng, loss_fn=None,
               n_ensemble=None):
    n_ensemble = n_ensemble or N_ENSEMBLE
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    in_dim = Xtr.shape[1]
    n = len(Xtr)
    perm = rng.permutation(n)
    n_val = int(n * s3.VAL_FRAC)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr_t = torch.tensor(Xtr[tr_idx], device=DEVICE)
    ytr_t = torch.tensor(ytr[tr_idx], device=DEVICE)
    Xval_t = torch.tensor(Xtr[val_idx], device=DEVICE)
    yval_np = ytr[val_idx]
    Xte_t = torch.tensor(Xte, device=DEVICE)
    w = s3.class_weights(ytr[tr_idx], n_classes, s3.WEIGHT_MODE)
    P = np.zeros((len(Xte), n_classes), dtype=np.float64)
    vf1s = []
    for e in range(n_ensemble):
        model, vf1 = train_one(build_fn, Xtr_t, ytr_t, Xval_t, yval_np, in_dim,
                               n_classes, w, SEED + 100 * e, loss_fn)
        P += probs(model, Xte_t, n_classes)
        vf1s.append(vf1)
        del model
        torch.cuda.empty_cache()
    return P / n_ensemble, float(np.mean(vf1s))


def run_experiment(name, build_fn, loss_fn=None, notes="", extra=None,
                   n_ensemble=None, save_probs=False):
    """Run one architecture over the full 3-fold spatial CV and record results."""
    t0 = time.perf_counter()
    base_pf, base_mean, base_name = baseline_for(RELABEL)
    print(f"=== {name} ===")
    print(f"device={DEVICE} relabel={RELABEL} ensemble={n_ensemble or N_ENSEMBLE} "
          f"hidden={HIDDEN} dropout={DROPOUT} lr={LR}")
    print(f"baseline={base_name} per_fold={base_pf}")

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
    f1s, pcs, probs_out = [], [], {}
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if CLEAN_CLS12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        P, vf1 = train_fold(build_fn, X[tr_use], y_enc[tr_use], X[te], n_classes,
                            rng, loss_fn, n_ensemble)
        pred = P.argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        if save_probs:
            probs_out[f"fold{k}_P"] = P.astype(np.float32)
            probs_out[f"fold{k}_y"] = y_enc[te]
        d = f1 - base_pf[k]
        print(f"  fold {k}: F1={f1:.4f}  Δ={d:+.4f}  (val={vf1:.4f}, "
              f"{time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    deltas = [f1s[k] - base_pf[k] for k in range(len(f1s))]
    dmean = float(np.mean(deltas))
    all_pos = all(d > 0 for d in deltas)
    verdict = ("WIN" if (all_pos and dmean > 0.003) else
               "tie" if abs(dmean) <= 0.003 else "LOSS")
    pc_mean = np.mean(pcs, axis=0)
    print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}   vs {base_name}: "
          f"Δ_mean={dmean:+.4f}  per_fold_Δ={[round(d,4) for d in deltas]}")
    print(f"VERDICT: {verdict}  (needs all-folds-positive AND Δ>0.003)")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")
    dt = time.perf_counter() - t0
    print(f"total {dt:.1f}s")

    rec = {
        "name": name, "notes": notes, "relabel": RELABEL,
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "baseline": base_name, "baseline_per_fold": base_pf,
        "delta_per_fold": [round(d, 4) for d in deltas],
        "delta_mean": round(dmean, 4), "all_folds_positive": all_pos,
        "verdict": verdict,
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
        "n_ensemble": n_ensemble or N_ENSEMBLE, "runtime_s": round(dt, 1),
        "config": {"hidden": list(HIDDEN), "dropout": DROPOUT, "lr": LR, **(extra or {})},
    }
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"saved -> {out}")
    if save_probs:
        np.savez_compressed(RESULTS_DIR / f"{name}_probs.npz", **probs_out)
    return rec

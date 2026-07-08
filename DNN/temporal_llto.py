"""Investigation 1 — temporal robustness via leave-location-AND-time-out (LLTO).

The headline model (F1 0.7341) is validated only on spatially-blocked 3-fold CV
(GroupKFold on cell_id), which mixes all acquisition years 2017-2025 into both
train and test. That answers "does it generalize across space?" but NOT "does it
generalize to an unseen acquisition YEAR?". AlphaEarth embeddings can drift year
to year (sensor, phenology, processing), so a model that silently overfits a
year's radiometry would look fine under spatial CV and fail on next year's data.

LLTO holds out BOTH axes at once so neither leaks:

  for test_year in 2017..2025:
      spatial 3-fold GroupKFold on cell_id (same blocks as the headline protocol)
      for spatial fold k:
          test  = rows in (test_year AND spatial-block k)
          train = rows in (year != test_year  AND  cell_id NOT in block k)
      -> a test row is unseen in time (its year is held out) AND in space
         (its cell_id never appears in train, in any year).

This is strictly harder than either axis alone. We report, per held-out year:
  - macro-F1 (mean over the 3 spatial folds)
  - per-class F1, so we can see WHICH classes are temporally fragile.

The base learner is the headline recipe exactly (stage3 robust MLP, 5-seed
ensemble, sqrt weights, label smoothing, train-only cls12 centroid clean). Test
labels are never touched. We also run a "spatial-only, all years" reference with
the identical learner so the LLTO drop is attributable to the temporal hold-out
alone.

Run: ~/myprojects/recover/.venv/bin/python DNN/temporal_llto.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
N_ENSEMBLE = 5
N_SPATIAL = du.N_FOLDS
OUT_JSON = Path(__file__).resolve().parent / "temporal_llto.json"


def spatial_blocks(groups, n_folds=N_SPATIAL):
    """Assign every row a spatial block 0..n_folds-1 via GroupKFold on cell_id.
    Identical block construction to du.fold_indices, returned as a dense array so
    we can intersect it with a temporal mask."""
    gkf = GroupKFold(n_splits=n_folds)
    blocks = np.empty(len(groups), dtype=int)
    for k, (_, val_idx) in enumerate(gkf.split(np.zeros(len(groups)), None, groups)):
        blocks[val_idx] = k
    return blocks


def fit_predict(Xtr, ytr, Xte, n_classes, rng):
    """Headline recipe: 5-seed StandardScaler+MLP ensemble, probs averaged."""
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    perm = rng.permutation(len(Xtr))
    nv = int(len(Xtr) * s3.VAL_FRAC)
    vi, ti = perm[:nv], perm[nv:]
    Xtr_t = torch.tensor(Xtr[ti], device=DEVICE)
    ytr_t = torch.tensor(ytr[ti], device=DEVICE)
    Xval_t = torch.tensor(Xtr[vi], device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    w = s3.class_weights(ytr[ti], n_classes, s3.WEIGHT_MODE)
    P = np.zeros((len(Xte), n_classes))
    for e in range(N_ENSEMBLE):
        m, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, ytr[vi], Xtr.shape[1],
                            n_classes, w, SEED + 100 * e)
        P += s3.softmax_probs(m, Xte_t, n_classes)
        del m
        torch.cuda.empty_cache()
    return P.argmax(1)


def clean_train(X, y_enc, df, tr_idx, cls12_enc, lon, lat):
    """Train-only cls12 centroid clean, exactly as the headline recipe."""
    if cls12_enc < 0:
        return tr_idx
    keep = du.clean_stale_class_mask(X[tr_idx], y_enc[tr_idx], df.iloc[tr_idx],
                                     cls12_enc, lon[tr_idx], lat[tr_idx])
    return tr_idx[keep]


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  ensemble={N_ENSEMBLE}  spatial_folds={N_SPATIAL}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    years = df["year"].values.astype(int)
    uyears = sorted(np.unique(years).tolist())
    blocks = spatial_blocks(groups)
    rng = np.random.default_rng(SEED)
    print(f"loaded {X.shape[0]:,} rows  years={uyears}  classes={classes}")

    # ---- LLTO: hold out each year AND spatially block ----
    per_year = {}
    for yr in uyears:
        f1s, pcs, ns = [], [], []
        for k in range(N_SPATIAL):
            te = np.flatnonzero((years == yr) & (blocks == k))
            tr = np.flatnonzero((years != yr) & (blocks != k))
            tr = clean_train(X, y_enc, df, tr, cls12_enc, lon, lat)
            pred = fit_predict(X[tr], y_enc[tr], X[te], n_classes, rng)
            f1s.append(du.macro_f1(y_enc[te], pred, n_classes))
            pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
            ns.append(len(te))
        f1m = float(np.mean(f1s))
        pc = np.mean(pcs, axis=0)
        per_year[str(yr)] = {
            "macro_f1": round(f1m, 4),
            "macro_f1_std": round(float(np.std(f1s)), 4),
            "n_test": int(np.sum(ns)),
            "per_class_f1": {str(c): round(float(v), 4) for c, v in zip(classes, pc)},
        }
        print(f"  LLTO year={yr}: macroF1={f1m:.4f} (std {np.std(f1s):.4f})  "
              f"n_test={np.sum(ns):,}  {time.perf_counter()-t0:.0f}s")

    # ---- reference: spatial-only CV, all years mixed (headline protocol) ----
    ref_f1s, ref_pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        tr = clean_train(X, y_enc, df, tr, cls12_enc, lon, lat)
        pred = fit_predict(X[tr], y_enc[tr], X[te], n_classes, rng)
        ref_f1s.append(du.macro_f1(y_enc[te], pred, n_classes))
        ref_pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
    ref_f1 = float(np.mean(ref_f1s))
    ref_pc = np.mean(ref_pcs, axis=0)
    print(f"\n  spatial-only ref (all years): macroF1={ref_f1:.4f}")

    llto_macro = float(np.mean([v["macro_f1"] for v in per_year.values()]))
    print(f"\n=== LLTO temporal robustness ===")
    print(f"  spatial-only (headline)   macroF1 = {ref_f1:.4f}")
    print(f"  LLTO (leave-year+block)   macroF1 = {llto_macro:.4f}  "
          f"(Δ = {llto_macro-ref_f1:+.4f})")
    print(f"  per-year spread: min={min(v['macro_f1'] for v in per_year.values()):.4f} "
          f"max={max(v['macro_f1'] for v in per_year.values()):.4f}")

    OUT_JSON.write_text(json.dumps({
        "investigation": "temporal_llto",
        "scheme": "leave-location-and-time-out (year held out AND cell_id block held out)",
        "n_ensemble": N_ENSEMBLE, "n_spatial_folds": N_SPATIAL,
        "spatial_only_ref": {
            "macro_f1": round(ref_f1, 4),
            "per_class_f1": {str(c): round(float(v), 4) for c, v in zip(classes, ref_pc)},
        },
        "llto_macro_f1_mean": round(llto_macro, 4),
        "llto_vs_spatial_delta": round(llto_macro - ref_f1, 4),
        "per_year": per_year,
    }, indent=2))
    print(f"saved -> {OUT_JSON}  total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""Train + persist the seed model for the active-learning app.

Fits one CatBoost (GPU-trained) on the full stable-allyears set (deduped to one row per
unique location), calibrates a global APS tau via 5-fold cross-conformal, runs a quick
spatial-holdout F1 sanity check, and saves artifacts to models/:
  seed_catboost.cbm, aps_calib.npz, label_encoder.json

This is the FAST live-loop model (CatBoost MultiClass, depth 6, GPU train / CPU predict).
The F1≈0.7074 figure comes from the heavier CatBoost->TabICL CV pipeline; this seed model
is the CatBoost stage-1 alone (~0.69 macro-F1 on spatial holdout), and its APS uncertainty
drives both the map's uncertainty layer and the AL queue. CatBoost was chosen over XGBoost
for ~3× faster GPU training and ~17× faster dense CPU predict (see timber_report.md).

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/active_learning/train_seed_model.py
  (flags: --no-dedup, --cc-cap N, --iterations N, --skip-eval)
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import model_core as mc  # run from scripts/active_learning/, or see sys.path shim below


def _ensure_importable():
    """Allow running from the repo root too."""
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def spatial_holdout_f1(df, y_merged, classes, cc_cap, iterations) -> float:
    """One-shot GroupKFold(3) holdout macro-F1 as a sanity check (not the headline)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score

    groups = df["cell_id"].values if "cell_id" in df.columns else np.arange(len(df))
    gkf = GroupKFold(n_splits=3)
    tr_idx, te_idx = next(gkf.split(df[mc.FEATURE_COLS], y_merged, groups=groups))
    rng = np.random.default_rng(mc.SEED)
    if len(tr_idx) > cc_cap:
        tr_idx = rng.choice(tr_idx, size=cc_cap, replace=False)
    cls_to_col = {c: j for j, c in enumerate(classes)}
    ytr = np.array([cls_to_col[c] for c in y_merged[tr_idx]])
    yte = np.array([cls_to_col[c] for c in y_merged[te_idx]])
    Xtr = df.loc[tr_idx, mc.FEATURE_COLS].values.astype(np.float32)
    Xte = df.loc[te_idx, mc.FEATURE_COLS].values.astype(np.float32)
    cb = mc.make_model(iterations=iterations, n_classes=len(classes))
    cb.fit(Xtr, ytr)
    pred_cols = np.asarray(cb.predict(Xte)).ravel().astype(int)
    return float(f1_score(yte, pred_cols, average="macro",
                          labels=np.arange(len(classes)), zero_division=0))


def main():
    _ensure_importable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dedup", action="store_true",
                    help="train on all rows (default: dedup to latest year per location)")
    ap.add_argument("--cc-cap", type=int, default=mc.CC_CAP)
    ap.add_argument("--iterations", type=int, default=300,
                    help="CatBoost iterations; 300 keeps dense predict fast for ~0 F1 loss")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("Loading stable parquet...")
    df = mc.load_stable()
    print(f"  {len(df):,} rows")
    if not args.no_dedup:
        df = mc.dedup_to_latest_year(df)
        print(f"  deduped to {len(df):,} unique locations")

    y_merged = mc.merge_classes(df[mc.TARGET].values)
    classes = np.array(sorted(np.unique(y_merged)), dtype=int)
    n_classes = len(classes)
    cls_to_col = {c: j for j, c in enumerate(classes)}
    y_col = np.array([cls_to_col[c] for c in y_merged])
    names = mc.load_class_names()
    print(f"classes ({n_classes}): " +
          ", ".join(f"{c}={names.get(str(c), '?')}" for c in classes))

    # subsample for calibration + final fit (speed; matches CV pipeline's cc_cap)
    rng = np.random.default_rng(mc.SEED)
    if len(df) > args.cc_cap:
        idx = rng.choice(len(df), size=args.cc_cap, replace=False)
        print(f"  using {args.cc_cap:,} rows for calibration + fit")
    else:
        idx = np.arange(len(df))
    X = df.loc[idx, mc.FEATURE_COLS].values.astype(np.float32)
    yc = y_col[idx]

    print(f"Cross-conformal APS calibration ({mc.K_INNER}-fold, alpha={mc.ALPHA})...")
    tcc = time.perf_counter()
    pooled = mc.cross_conformal_aps(X, yc, mc.K_INNER, mc.SEED, n_classes)
    tau = mc.conformal_quantile(pooled, mc.ALPHA)
    print(f"  tau={tau:.5f}  ({time.perf_counter()-tcc:.1f}s)")

    print("Fitting final CatBoost...")
    tf = time.perf_counter()
    model = mc.make_model(iterations=args.iterations)
    model.fit(X, yc)
    print(f"  fit {time.perf_counter()-tf:.1f}s")

    # Column order is 0..n_classes-1 here (we encoded yc that way), so the canonical
    # `classes` array IS the column order. Persist it.
    assert np.array_equal(np.asarray(model.classes_).astype(int),
                          np.arange(n_classes)), "unexpected model class order"

    # quick uncertainty sanity: set-size distribution on the fit data
    probs = model.predict_proba(X).astype(np.float64)
    sizes, _ = mc.prediction_set_sizes(probs, tau)
    print(f"  APS set-size on fit rows: mean={sizes.mean():.2f} "
          f"median={int(np.median(sizes))} "
          f"singletons={100*(sizes==1).mean():.1f}% "
          f"empty/≥1 ok (min={sizes.min()})")

    if not args.skip_eval:
        print("Spatial-holdout F1 sanity check (GroupKFold by cell_id)...")
        f1 = spatial_holdout_f1(df, y_merged, classes, args.cc_cap, args.iterations)
        print(f"  macro-F1 (CatBoost stage-1 only) = {f1:.4f} "
              f"(TabICL CV reference ≈0.7074)")

    mc.save_seed_model(model, tau, classes)
    print(f"\nSaved artifacts to {mc.MODELS_DIR}/  "
          f"(total {time.perf_counter()-t0:.1f}s)")
    print(f"  - {mc.MODEL_PATH.name}  - aps_calib.npz  - label_encoder.json")


if __name__ == "__main__":
    main()

"""Train on 2018 ONLY, predict subsequent years — the "label once, deploy forward"
scenario. Compares against the all-years LLTO numbers (temporal_llto.json).

Motivation: the headline model pools 8 training years. In practice you often label
ONE year and apply the model forward. Does a 2018-only model hold up on 2019-2025,
or does temporal drift degrade it relative to the all-years model?

Protocol (spatially-blocked, leak-free — mirrors temporal_llto.py's blocks so the
spatial-generalization penalty is identical and the ONLY difference is training
years):
  blocks = GroupKFold on cell_id (same 3 blocks as LLTO)
  for test_year Y in 2018..2025:
      for block k:
          train = rows in (year == 2018  AND  block != k)
          test  = rows in (year == Y      AND  block == k)
  -> Y==2018 is the in-year (but spatially-held-out) reference; Y>2018 is forward
     deployment onto an unseen year with an unseen location.

Base learner = headline recipe (5-seed MLP ensemble, sqrt weights, label smooth,
train-only cls12 centroid clean). Test labels never touched.

We report per-year macro-F1 + per-class F1, and print the side-by-side delta vs
the all-years LLTO run (loaded from temporal_llto.json if present).

Run: TRAIN_YEAR=2018 ~/myprojects/recover/.venv/bin/python DNN/train2018_forward.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du            # noqa: E402
import stage3_robust_mlp as s3     # noqa: E402
import temporal_llto as tl         # noqa: E402  (spatial_blocks, clean_train, fit_predict)

SEED = 0
N_SPATIAL = du.N_FOLDS
TRAIN_YEAR = int(os.environ.get("TRAIN_YEAR", "2018"))
OUT_JSON = Path(__file__).resolve().parent / f"train{TRAIN_YEAR}_forward.json"
LLTO_JSON = Path(__file__).resolve().parent / "temporal_llto.json"


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={tl.DEVICE}  train_year={TRAIN_YEAR}  spatial_folds={N_SPATIAL}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    years = df["year"].values.astype(int)
    test_years = [y for y in sorted(np.unique(years).tolist()) if y >= TRAIN_YEAR]
    blocks = tl.spatial_blocks(groups)
    rng = np.random.default_rng(SEED)
    print(f"loaded {X.shape[0]:,} rows  train_year={TRAIN_YEAR} "
          f"({int((years==TRAIN_YEAR).sum()):,} rows)  test_years={test_years}")

    # optional: all-years LLTO reference for the delta column
    llto = None
    if LLTO_JSON.exists():
        llto = json.loads(LLTO_JSON.read_text()).get("per_year", {})

    per_year = {}
    for yr in test_years:
        f1s, pcs, ns = [], [], []
        for k in range(N_SPATIAL):
            te = np.flatnonzero((years == yr) & (blocks == k))
            tr = np.flatnonzero((years == TRAIN_YEAR) & (blocks != k))
            tr = tl.clean_train(X, y_enc, df, tr, cls12_enc, lon, lat)
            pred = tl.fit_predict(X[tr], y_enc[tr], X[te], n_classes, rng)
            f1s.append(du.macro_f1(y_enc[te], pred, n_classes))
            pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
            ns.append(len(te))
        f1m = float(np.mean(f1s))
        pc = np.mean(pcs, axis=0)
        allyr = llto.get(str(yr), {}).get("macro_f1") if llto else None
        per_year[str(yr)] = {
            "macro_f1": round(f1m, 4),
            "macro_f1_std": round(float(np.std(f1s)), 4),
            "n_test": int(np.sum(ns)),
            "allyears_llto_f1": allyr,
            "delta_vs_allyears": (round(f1m - allyr, 4) if allyr is not None else None),
            "per_class_f1": {str(c): round(float(v), 4) for c, v in zip(classes, pc)},
        }
        dv = per_year[str(yr)]["delta_vs_allyears"]
        dvs = f"  allyrs={allyr:.4f} Δ={dv:+.4f}" if allyr is not None else ""
        tag = "  <- in-year ref" if yr == TRAIN_YEAR else f"  (+{yr-TRAIN_YEAR}y forward)"
        print(f"  test={yr}: macroF1={f1m:.4f} (std {np.std(f1s):.4f}){dvs}{tag}  "
              f"{time.perf_counter()-t0:.0f}s")

    fwd = [v["macro_f1"] for y, v in per_year.items() if int(y) > TRAIN_YEAR]
    fwd_mean = float(np.mean(fwd)) if fwd else float("nan")
    inyear = per_year[str(TRAIN_YEAR)]["macro_f1"]
    print(f"\n=== Train-{TRAIN_YEAR}-only, deploy forward ===")
    print(f"  in-year (spatially held-out) macroF1 = {inyear:.4f}")
    print(f"  forward-years mean macroF1           = {fwd_mean:.4f}  "
          f"(drop from in-year = {fwd_mean-inyear:+.4f})")
    if llto:
        allyr_fwd = [llto[y]["macro_f1"] for y in per_year if int(y) > TRAIN_YEAR
                     and y in llto]
        if allyr_fwd:
            print(f"  all-years LLTO forward-years mean    = "
                  f"{np.mean(allyr_fwd):.4f}  "
                  f"(train-2018 costs {fwd_mean-np.mean(allyr_fwd):+.4f} vs 8-year)")

    OUT_JSON.write_text(json.dumps({
        "investigation": "train_single_year_forward",
        "train_year": TRAIN_YEAR,
        "scheme": "train on TRAIN_YEAR only (block!=k), test on year Y block==k; "
                  "vs all-years LLTO",
        "n_ensemble": tl.N_ENSEMBLE, "n_spatial_folds": N_SPATIAL,
        "in_year_f1": round(inyear, 4),
        "forward_years_mean_f1": round(fwd_mean, 4),
        "forward_drop_vs_in_year": round(fwd_mean - inyear, 4),
        "per_year": per_year,
    }, indent=2))
    print(f"saved -> {OUT_JSON}  total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()

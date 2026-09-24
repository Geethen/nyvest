"""The three brief-mandated validation checks, runnable standalone:

1. cell_id: recompute cell_id from lon/lat for 500 existing training points
   and assert it matches the parquet's own cell_id (pure local, no GEE).
2. lidar: extract lidar features for 500 existing training points via
   step2_local.sample_lidar and compare to data/lidar_features.parquet
   (join on exact lon/lat).
3. AEF scaling: re-extract AEF bands for 300 existing training points for
   years 2020 and 2024 via GEE and report max |Δ| against the parquet.

Prints a small JSON summary and also returns it (used by report.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import (TRAIN_PARQUET, LIDAR_PARQUET, CONSENSUS_DIR,
                    cell_id_from_lonlat, EXPECTED_BANDS)


def check_cell_id(n=500, seed=42):
    df = pd.read_parquet(TRAIN_PARQUET, columns=["lon", "lat", "cell_id"])
    df = df.drop_duplicates(subset=["lon", "lat"])
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    computed = cell_id_from_lonlat(sample["lon"].to_numpy(), sample["lat"].to_numpy())
    match = (computed == sample["cell_id"].to_numpy())
    return {"n": int(len(sample)), "n_match": int(match.sum()),
            "all_match": bool(match.all())}


def check_lidar(n=500, seed=42):
    from step2_local import sample_lidar
    df = pd.read_parquet(TRAIN_PARQUET, columns=["lon", "lat"])
    df = df.drop_duplicates(subset=["lon", "lat"])
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    fresh = sample_lidar(sample)

    ref = pd.read_parquet(LIDAR_PARQUET)
    merged = fresh.merge(ref, on=["lon", "lat"], how="inner", suffixes=("_fresh", "_ref"))
    cols = ["elevation", "tch", "slope", "aspect_sin", "aspect_cos", "tri"]
    deltas = {}
    for c in cols:
        a = merged[f"{c}_fresh"].to_numpy()
        b = merged[f"{c}_ref"].to_numpy()
        both_valid = ~np.isnan(a) & ~np.isnan(b)
        d = np.abs(a[both_valid] - b[both_valid])
        deltas[c] = float(d.max()) if d.size else None
    return {"n_sampled": int(len(sample)), "n_matched_in_ref": int(len(merged)),
            "max_abs_delta": deltas}


def check_aef_scaling(n=300, years=(2020, 2024), seed=42):
    import gee_extract
    from common import init_gee

    df = pd.read_parquet(TRAIN_PARQUET,
                         columns=["lon", "lat", "year"] + EXPECTED_BANDS)
    sub = df[df["year"].isin(years)].copy()
    # sample n DISTINCT locations, then compare both years for each
    locs = sub[["lon", "lat"]].drop_duplicates().sample(
        n=min(n, sub["lon"].nunique()), random_state=seed).reset_index(drop=True)
    locs["pid"] = np.arange(len(locs))

    init_gee()
    out_path = os.path.join(CONSENSUS_DIR, "_validate_aef_scaling.parquet")
    ckpt_path = out_path + ".checkpoint.json"
    if os.path.exists(out_path):
        os.remove(out_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    gee_extract.run_temporal(locs, out_path, ckpt_path, batch_size=2000,
                             years=years, log=print)
    fresh = pd.read_parquet(out_path)
    fresh = fresh.merge(locs[["pid", "lon", "lat"]], on="pid", how="left")

    ref = sub.merge(locs[["lon", "lat"]], on=["lon", "lat"], how="inner")

    max_delta = {}
    for y in years:
        f_y = fresh[fresh["year"] == y].set_index(["lon", "lat"])
        r_y = ref[ref["year"] == y].set_index(["lon", "lat"])
        common_idx = f_y.index.intersection(r_y.index)
        f_y = f_y.loc[common_idx]
        r_y = r_y.loc[common_idx]
        deltas = (f_y[EXPECTED_BANDS].to_numpy()
                 - r_y[EXPECTED_BANDS].to_numpy())
        max_delta[str(y)] = {"n": int(len(common_idx)),
                             "max_abs_delta": float(np.nanmax(np.abs(deltas)))
                             if len(common_idx) else None}
    return max_delta


def run_all():
    print("== check 1: cell_id ==")
    c1 = check_cell_id()
    print(c1)
    print("== check 2: lidar ==")
    c2 = check_lidar()
    print(c2)
    print("== check 3: AEF scaling ==")
    c3 = check_aef_scaling()
    print(c3)
    result = {"cell_id": c1, "lidar": c2, "aef_scaling": c3}
    with open(os.path.join(CONSENSUS_DIR, "_validation_checks.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    run_all()

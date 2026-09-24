"""Step 2 (main) — orchestrates the full per-point extraction for the
candidate points from step 1:

  data/consensus/candidates.parquet (stratum, cell_id, lon, lat,
    class_2018, class_2024, setsize_2018, setsize_2024)
    -> assign pid
    -> + gk_v1, gk_v2 (local raster sample)
    -> + lidar elevation/tri/tch/... (local, via extract_lidar_features.py)
    -> + wc2020, wc2021, esri_<year>, dw_<year>[,_frac,_n] 2017-2025 (GEE)
    -> data/consensus/points.parquet

  in parallel (same pid space):
    -> aef_long.parquet: pid, year, A00..A63 for 2017-2025 (GEE)

Batches of <=2000 points, checkpointed per (batch, subtask/year) inside
gee_extract.py so a kill + restart resumes without repeating work already
computed. Designed to be run under nohup for the full-size job; --test_mode
runs end-to-end on 200 points first.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from common import CONSENSUS_DIR, init_gee, YEARS
from step2_local import sample_grunnkart, sample_lidar

CANDIDATES_PARQUET = os.path.join(CONSENSUS_DIR, "candidates.parquet")
POINTS_PARQUET = os.path.join(CONSENSUS_DIR, "points.parquet")
AEF_LONG_PARQUET = os.path.join(CONSENSUS_DIR, "aef_long.parquet")

STATIC_TMP = os.path.join(CONSENSUS_DIR, "_points_static.parquet")
STATIC_CKPT = STATIC_TMP + ".checkpoint.json"
AEF_CKPT = AEF_LONG_PARQUET + ".checkpoint.json"

LOCAL_TMP = os.path.join(CONSENSUS_DIR, "_points_local.parquet")


def build_local(test_mode=False, test_n=200, seed=42):
    """Assign pid + sample grunnkart/lidar locally (idempotent, cheap —
    always recomputed, not checkpointed)."""
    df = pd.read_parquet(CANDIDATES_PARQUET)
    if test_mode:
        per_stratum = max(1, test_n // 3)
        parts = [g.sample(n=min(len(g), per_stratum), random_state=seed)
                for _, g in df.groupby("stratum")]
        df = pd.concat(parts, ignore_index=True)
    df["pid"] = range(len(df))
    print(f"  {len(df)} candidate points ({df['stratum'].value_counts().to_dict()})")

    df = sample_grunnkart(df)
    print("  grunnkart v1/v2 sampled locally")
    df = sample_lidar(df)
    print("  lidar sampled locally "
         f"({df['elevation'].notna().sum()}/{len(df)} inside lidar tiles)")
    df.to_parquet(LOCAL_TMP)
    return df


def run(test_mode=False, test_n=200, seed=42, batch_size=2000,
       max_workers_unused=None, skip_local=False, skip_static=False,
       skip_temporal=False):
    import gee_extract
    t0 = time.time()

    if skip_local and os.path.exists(LOCAL_TMP):
        df = pd.read_parquet(LOCAL_TMP)
        print(f"  [skip_local] reused {LOCAL_TMP}: {len(df)} rows")
    else:
        df = build_local(test_mode=test_mode, test_n=test_n, seed=seed)

    init_gee()

    if not skip_static:
        gee_extract.run_static(df, STATIC_TMP, STATIC_CKPT,
                               batch_size=batch_size, years=YEARS, log=print)
    if not skip_temporal:
        gee_extract.run_temporal(df, AEF_LONG_PARQUET, AEF_CKPT,
                                 batch_size=batch_size, years=YEARS, log=print)

    static_df = pd.read_parquet(STATIC_TMP)
    if "geo" in static_df.columns:  # stray all-NaN col GEE sometimes returns
        static_df = static_df.drop(columns=["geo"])
    out = df.merge(static_df, on="pid", how="left")
    out.to_parquet(POINTS_PARQUET)
    print(f"\n[OK] Saved {POINTS_PARQUET}: {len(out)} rows, {out.shape[1]} cols "
         f"in {time.time()-t0:.0f}s")
    if os.path.exists(AEF_LONG_PARQUET):
        aef = pd.read_parquet(AEF_LONG_PARQUET, columns=["pid", "year"])
        print(f"[OK] {AEF_LONG_PARQUET}: {len(aef)} rows "
             f"({aef['pid'].nunique()} pids x up to {aef['year'].nunique()} years)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_mode", action="store_true")
    ap.add_argument("--test_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--skip_local", action="store_true",
                    help="reuse cached _points_local.parquet (resume)")
    ap.add_argument("--skip_static", action="store_true")
    ap.add_argument("--skip_temporal", action="store_true")
    args = ap.parse_args()
    run(test_mode=args.test_mode, test_n=args.test_n, seed=args.seed,
        batch_size=args.batch_size, skip_local=args.skip_local,
        skip_static=args.skip_static, skip_temporal=args.skip_temporal)


if __name__ == "__main__":
    main()

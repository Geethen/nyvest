"""Reference sample: 3,000 existing training points (stratified by class),
product-extracted (grunnkart local + WorldCover/Esri/DW via GEE — no lidar,
no AEF, those already exist for these points in the training parquet).

This measures how often each product agrees with grunnkart on the CLEAN
frame the model was trained on, so agreement rates in the hard-area strata
(flip/uncertain/random) can be interpreted against a baseline.

Output: data/consensus/reference_points.parquet
  pid, lon, lat, class (training label), cell_id, gk_v1, gk_v2,
  wc2020, wc2021, esri_<year>, dw_<year>[,_frac,_n] for 2017-2025
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import TRAIN_PARQUET, CONSENSUS_DIR, init_gee, YEARS
from step2_local import sample_grunnkart

OUT_PARQUET = os.path.join(CONSENSUS_DIR, "reference_points.parquet")
STATIC_CKPT = OUT_PARQUET + ".static_checkpoint.json"
STATIC_TMP = os.path.join(CONSENSUS_DIR, "_reference_static.parquet")


def build_sample(n=3000, seed=42):
    df = pd.read_parquet(TRAIN_PARQUET, columns=["lon", "lat", "class", "cell_id"])
    df = df.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)

    frac = n / len(df)
    parts = []
    for _, g in df.groupby("class"):
        k = max(1, round(len(g) * frac))
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    sample = sample.reset_index(drop=True)
    sample["pid"] = np.arange(len(sample))
    return sample


def run(n=3000, seed=42, batch_size=2000, test_mode=False):
    import gee_extract
    if test_mode:
        n = 200

    sample = build_sample(n=n, seed=seed)
    print(f"  reference sample: {len(sample)} points, "
          f"{sample['class'].nunique()} classes")

    sample = sample_grunnkart(sample)
    print("  grunnkart v1/v2 sampled locally")

    init_gee()
    gee_extract.run_static(sample, STATIC_TMP, STATIC_CKPT,
                           batch_size=batch_size, years=YEARS, log=print)
    static_df = pd.read_parquet(STATIC_TMP)
    if "geo" in static_df.columns:  # stray all-NaN col GEE sometimes returns
        static_df = static_df.drop(columns=["geo"])
    out = sample.merge(static_df, on="pid", how="left")
    out.to_parquet(OUT_PARQUET)
    print(f"[OK] Saved {OUT_PARQUET}: {len(out)} rows, {out.shape[1]} cols")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--test_mode", action="store_true")
    args = ap.parse_args()
    run(n=args.n, seed=args.seed, batch_size=args.batch_size,
        test_mode=args.test_mode)


if __name__ == "__main__":
    main()

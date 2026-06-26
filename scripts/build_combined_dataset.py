"""Build a combined FSCS + Megan dataset at the FSCS 12-class level.

Megan rows with a confirmed FSCS-class equivalent are mapped and appended
to the FSCS stable parquet. All 12 FSCS classes are kept. Features are
aligned to the 64 AlphaEarth bands (Megan's 2 LiDAR columns are dropped).

Approved Megan fallbck → FSCS class mapping
--------------------------------------------
Broadleaved decid. forest  (2)  → 4  forest
Coniferous forests         (5)  → 4
Forest and woodlands       (9)  → 4
Mixed forests             (16)  → 4
Cropland                   (7)  → 3
Grassland                 (10)  → 5
Scrub and heathland       (19)  → 6
Mires, bogs and fens      (15)  → 7  wetland
Lakes and ponds           (13)  → 8  water
Artificial reservoirs      (1)  → 8
Rivers                    (18)  → 8
Canals, ditches & drains   (3)  → 8
Continuous settlement      (6)  → 10
Discontinuous settlement   (8)  → 10
Settlements & other art.  (20)  → 10
Infrastructure            (12)  → 11
Ice sheets / glaciers     (11)  → 12 snow/ice

Dropped (no approved FSCS equivalent):
  4=Coastal beaches/dunes, 14=Marine, 17=Other artificial,
  21=Sparsely vegetated, 22=Urban greenspace

FSCS classes with NO Megan equivalent (kept as FSCS-only):
  1=bare/rock/sand, 2=bare (merged), 9=marine

Output:
  data/grunnkart_nyvest_fscs_alphaearth_combined.parquet
  reports/combined_dataset_summary.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import load_split  # noqa: E402

DATA_DIR    = _REPO / "data"
REPORTS_DIR = _REPO / "reports"
OUT_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth_combined.parquet"
OUT_SUMMARY = REPORTS_DIR / "combined_dataset_summary.csv"
FSCS_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"

FSCS_CLASSES = {
    1: "bare/rock/sand",
    2: "bare (merged 1+2)",
    3: "cropland",
    4: "forest",
    5: "grassland",
    6: "scrub/heathland",
    7: "wetland",
    8: "water",
    9: "marine",
    10: "settlement",
    11: "infrastructure",
    12: "snow/ice",
}

# Approved Megan fallbck → FSCS class mapping
MEGAN_TO_FSCS: dict[int, int] = {
    2:  4,   # Broadleaved deciduous forest
    5:  4,   # Coniferous forests
    9:  4,   # Forest and woodlands
    16: 4,   # Mixed forests
    7:  3,   # Cropland
    10: 5,   # Grassland
    19: 6,   # Scrub and heathland
    15: 7,   # Mires, bogs and fens → wetland
    13: 8,   # Lakes and ponds → water
    1:  8,   # Artificial reservoirs → water
    18: 8,   # Rivers → water
    3:  8,   # Canals, ditches and drains → water
    6:  10,  # Continuous settlement area
    8:  10,  # Discontinuous settlement area
    20: 10,  # Settlements and other artificial areas
    12: 11,  # Infrastructure
    11: 12,  # Ice sheets, glaciers, perennial snowfields → snow/ice
    # Dropped: 4=Coastal, 14=Marine, 17=OtherArtificial, 21=SparselyVeg, 22=UrbanGreenspace
}

# Megan AlphaEarth column names → FSCS A00..A63 (same 64-band embedding, renamed)
MEGAN_AE_COLS = (
    [f"embdd_{i}" for i in range(1, 10)] +  # embdd_1..embdd_9
    [f"embd_{i}"  for i in range(10, 65)]   # embd_10..embd_64
)
FSCS_AE_COLS = [f"A{i:02d}" for i in range(64)]


def load_fscs() -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{FSCS_PARQUET}'").df()
    df[FSCS_AE_COLS] = df[FSCS_AE_COLS].astype(np.float32)
    df = df.dropna(subset=FSCS_AE_COLS + ["class", "cell_id", "lon", "lat"])
    df["source"] = "fscs"
    return df.reset_index(drop=True)


def load_megan_mapped() -> pd.DataFrame:
    """Load all Megan splits, apply FSCS class mapping, align AE features."""
    frames = []
    for split_name in ("train", "val", "test"):
        X_raw, y_raw = load_split(split_name)

        missing = [c for c in MEGAN_AE_COLS if c not in X_raw.columns]
        if missing:
            raise KeyError(f"Missing Megan AE columns in {split_name}: {missing[:3]}")

        # Map fallbck → FSCS class; drop unmapped rows
        fscs_class = y_raw.map(MEGAN_TO_FSCS)
        keep = fscs_class.notna()
        n_dropped = int((~keep).sum())
        X_raw = X_raw.loc[keep].reset_index(drop=True)
        fscs_class = fscs_class[keep].astype(int).reset_index(drop=True)

        # Rename AE columns to FSCS convention and rescale.
        # Megan stores embeddings as integers scaled by ~1000 (e.g. -119 = -0.119).
        # FSCS stores them as floats in [-0.5, 0.5]. Divide Megan by 1000 to align.
        ae = X_raw[MEGAN_AE_COLS].copy()
        ae.columns = FSCS_AE_COLS
        ae = (ae / 1000.0).astype(np.float32)

        df = ae.copy()
        df["class"]   = fscs_class.values
        df["source"]  = "megan"
        df["cell_id"] = pd.array([-1] * len(df), dtype="Int64")
        df["lon"]     = np.nan
        df["lat"]     = np.nan
        print(f"  Megan {split_name}: {len(df):,} rows kept, {n_dropped} dropped")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def run():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FSCS stable parquet...")
    fscs = load_fscs()
    print(f"  FSCS: {len(fscs):,} rows  classes: {sorted(fscs['class'].unique())}")

    print("\nLoading Megan splits and mapping to FSCS classes...")
    megan = load_megan_mapped()
    print(f"  Megan total kept: {len(megan):,} rows  "
          f"classes: {sorted(megan['class'].unique())}")

    # Align to shared columns
    shared_cols = FSCS_AE_COLS + ["class", "source", "cell_id", "lon", "lat"]
    fscs_out  = fscs[shared_cols].copy()
    for col in shared_cols:
        if col not in megan.columns:
            megan[col] = np.nan
    megan_out = megan[shared_cols].copy()

    combined = pd.concat([fscs_out, megan_out], ignore_index=True)
    combined["class"] = combined["class"].astype(int)

    print(f"\nCombined: {len(combined):,} rows  "
          f"(FSCS={len(fscs_out):,}  Megan={len(megan_out):,})")

    # Build summary table
    grp = (combined.groupby(["class", "source"])
                   .size()
                   .unstack(fill_value=0)
                   .reset_index())
    for src in ("fscs", "megan"):
        if src not in grp.columns:
            grp[src] = 0
    grp["total"]       = grp["fscs"] + grp["megan"]
    grp["class_name"]  = grp["class"].map(FSCS_CLASSES)
    grp["pct_megan"]   = (grp["megan"] / grp["total"] * 100).round(1)
    grp["pct_increase"] = (grp["megan"] / grp["fscs"].replace(0, np.nan) * 100).round(1)
    grp = grp.sort_values("class").reset_index(drop=True)

    print("\nClass distribution:")
    print(f"  {'cls':>3}  {'name':<22}  {'fscs':>7}  {'megan':>7}  "
          f"{'total':>7}  {'%megan':>7}  {'%increase':>10}")
    print("  " + "-" * 72)
    for _, row in grp.iterrows():
        print(f"  {int(row['class']):>3}  {row['class_name']:<22}  "
              f"{int(row['fscs']):>7,}  {int(row['megan']):>7,}  "
              f"{int(row['total']):>7,}  {row['pct_megan']:>7.1f}%  "
              f"{row['pct_increase']:>9.1f}%")
    print(f"  {'TOTAL':<26}  {len(fscs_out):>7,}  {len(megan_out):>7,}  "
          f"{len(combined):>7,}")

    grp.to_csv(OUT_SUMMARY, index=False)
    print(f"\nSummary → {OUT_SUMMARY}")

    # Write via duckdb (pyarrow not available in this venv)
    import duckdb as _ddb
    _ddb.sql(f"COPY (SELECT * FROM combined) TO '{OUT_PARQUET}' (FORMAT PARQUET)")
    print(f"Combined parquet → {OUT_PARQUET}  shape={combined.shape}")


if __name__ == "__main__":
    run()

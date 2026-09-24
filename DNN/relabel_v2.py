"""Sample the grunnkart v1 and v2 10m rasters at the FSCS training points.

The training parquet's `class` column came from the GEE asset built off v1.
To isolate the v2 label change we sample BOTH local rasters at the same
(lon, lat) points, so any GEE-vs-local reprojection discrepancy cancels in
the v1->v2 delta.

Writes data/labels_v2.parquet with one row per unique point:
    lon, lat, class_parquet, class_v1, class_v2

Usage
  ~/myprojects/recover/.venv/bin/python DNN/relabel_v2.py
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

_REPO = Path(__file__).resolve().parents[1]
DATA_DIR = _REPO / "data"

STABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"
OUT_PARQUET = DATA_DIR / "labels_v2.parquet"

_GIS = Path(os.environ.get(
    "NYVEST_DATA_DIR", "/data/P-Prosjekter2/154001_nyvest")) / "GIS" / "NIBIO"
RASTER_V1 = _GIS / "Version_2" / "rasterized_10m" / "grunnkart_nyvest_10m.tif"
RASTER_V2 = _GIS / "Version_2" / "rasterized_10m" / "grunnkart_nyvest_10m_v2.tif"

# Authoritative grunnkart codebook (pre-merge). 13 = "other" (settlements &
# artificial areas), excluded from FSCS sampling.
LABELS = {1: "sand", 2: "rock+sand", 3: "crop", 4: "forest", 5: "grassland",
          6: "scrub", 7: "wetland", 8: "water-inland", 9: "water-sea",
          10: "built", 11: "sparse-veg", 12: "snow/ice", 13: "other"}


def sample_raster(path, xs, ys):
    """Nearest-pixel sample of a single-band raster at projected coords."""
    out = np.full(len(xs), -1, dtype=np.int16)
    with rasterio.open(path) as src:
        inv = ~src.transform
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        rows = np.floor(rows).astype(np.int64)
        cols = np.floor(cols).astype(np.int64)
        inside = ((rows >= 0) & (rows < src.height)
                  & (cols >= 0) & (cols < src.width))
        band = src.read(1)
    out[inside] = band[rows[inside], cols[inside]]
    return out


def main():
    pts = duckdb.sql(
        f"SELECT DISTINCT lon, lat, first(class) OVER (PARTITION BY lon, lat) "
        f"AS class_parquet FROM '{STABLE_PARQUET}'").df()
    pts = pts.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    print(f"  points: {len(pts):,}")

    tf = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    xs, ys = tf.transform(pts["lon"].values, pts["lat"].values)

    pts["class_v1"] = sample_raster(RASTER_V1, xs, ys)
    pts["class_v2"] = sample_raster(RASTER_V2, xs, ys)

    # sanity: does the local v1 raster agree with the GEE-sampled parquet label?
    agree = (pts["class_v1"] == pts["class_parquet"]).mean()
    print(f"  local v1 vs parquet label agreement: {agree:.4%}")

    changed = pts["class_v1"] != pts["class_v2"]
    print(f"  v1 -> v2 changed at sample points: {changed.sum():,} "
          f"({changed.mean():.4%})")
    if changed.any():
        xt = (pts[changed].groupby(["class_v1", "class_v2"]).size()
              .reset_index(name="n").sort_values("n", ascending=False))
        xt["from"] = xt["class_v1"].map(LABELS)
        xt["to"] = xt["class_v2"].map(LABELS)
        print(xt[["class_v1", "from", "class_v2", "to", "n"]].to_string(index=False))

    # how many parquet points would leave the 1..12 label space under v2
    leaves = pts["class_v2"].isin([0, 13]).sum()
    print(f"  points whose v2 class is nodata/other (0 or 13): {leaves:,}")

    pts.to_parquet(OUT_PARQUET, index=False)
    print(f"[OK] wrote {OUT_PARQUET}")


if __name__ == "__main__":
    main()

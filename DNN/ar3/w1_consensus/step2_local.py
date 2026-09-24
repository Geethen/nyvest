"""Step 2 (local part) — grunnkart v1/v2 point sample + lidar terrain
features, applied to a set of points (lon, lat in EPSG:4326).

grunnkart: point sample of grunnkart_nyvest_10m[_v2].tif, EPSG:25832 (raster
CRS differs from the 4326 input — reproject before sampling; verified CRS
with rasterio.open(...).crs during setup).

lidar: imports scripts/extraction/extract_lidar_features.py by file path
(read-only import, that script is untouched) and calls its own
tile_index/assign_tiles/sample_tile functions directly, so the extraction is
EXACTLY identical in method, units (mm -> m, MM_TO_M) and tile-priority
resolution — not a reimplementation.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from common import GRUNNKART_V1_TIF, GRUNNKART_V2_TIF, REPO_ROOT

_TF_4326_TO_25832 = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def sample_grunnkart(df, lon_col="lon", lat_col="lat"):
    """Adds gk_v1, gk_v2 (raw class 1..13, 0 = nodata) to df (copy)."""
    df = df.copy()
    x, y = _TF_4326_TO_25832.transform(df[lon_col].to_numpy(),
                                       df[lat_col].to_numpy())
    coords = list(zip(x, y))
    for col, path in (("gk_v1", GRUNNKART_V1_TIF), ("gk_v2", GRUNNKART_V2_TIF)):
        with rasterio.open(path) as ds:
            assert ds.crs.to_string() == "EPSG:25832", f"{path} CRS={ds.crs}"
            vals = [v[0] for v in ds.sample(coords)]
        df[col] = np.array(vals, dtype=np.int64)
    return df


_LIDAR_MOD = None


def _load_lidar_module():
    global _LIDAR_MOD
    if _LIDAR_MOD is not None:
        return _LIDAR_MOD
    path = os.path.join(REPO_ROOT, "scripts", "extraction",
                        "extract_lidar_features.py")
    spec = importlib.util.spec_from_file_location("extract_lidar_features_ro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LIDAR_MOD = mod
    return mod


def sample_lidar(df, lon_col="lon", lat_col="lat"):
    """Adds elevation, tch, slope, aspect_sin, aspect_cos, tri to df (copy).

    Reuses extract_lidar_features.py's own tile_index/assign_tiles/sample_tile
    verbatim (imported by file path, unmodified) for exact fidelity.
    """
    mod = _load_lidar_module()
    df = df.copy()
    pts = pd.DataFrame({"lon": df[lon_col].to_numpy(),
                        "lat": df[lat_col].to_numpy()})
    t = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    pts["x"], pts["y"] = t.transform(pts["lon"].to_numpy(), pts["lat"].to_numpy())

    tiles = mod.tile_index()
    buckets = mod.assign_tiles(pts, tiles)
    frames = []
    for ti, rows in buckets.items():
        frames.append(mod.sample_tile(tiles[ti], rows, pts))
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["lon", "lat", "elevation", "tch", "slope",
                                    "aspect_sin", "aspect_cos", "tri"])
    out = out.drop_duplicates(subset=["lon", "lat"])
    df = df.merge(out, how="left", left_on=[lon_col, lat_col],
                  right_on=["lon", "lat"], suffixes=("", "_lidar"))
    drop_cols = [c for c in ("lon_lidar", "lat_lidar") if c in df.columns]
    if lon_col != "lon" or lat_col != "lat":
        drop_cols += [c for c in ("lon", "lat") if c in df.columns
                     and c not in (lon_col, lat_col)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

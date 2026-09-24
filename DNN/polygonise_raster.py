"""Vectorise a DNN classified raster into smooth per-class polygons (Smoothify).

Pipeline:  classified int16 GeoTIFF  --(rasterio.features.shapes, tiled)-->
per-class raw polygons  --(smoothify, per class)-->  smooth polygons  -> GPKG.

Why tiled: polygonization and Chaikin smoothing hold every vertex in memory, and
smoothing wall-time scales with total vertices (a full-grid smooth of a county is
hours). We cut the raster into square tiles, polygonize each, dissolve+smooth
per-class WITHIN the whole (merged) map, and write one layer. Tiling bounds peak
memory; smoothing runs on all cores (num_cores=0).

Smoothify config is the benchmarked sweet spot (see DNN/smoothify_bench.py):
  merge_field=<class col>   dissolve/smooth WITHIN each class, keep classes distinct
                            (the default merge_collection=True would fuse ALL classes
                             into one geometry — silently destroys the map).
  num_cores=0               all cores (~3x over serial on the 8-core VDI).
  smooth_iterations=3       de-stairsteps 10 m pixels; ~10x vertex inflation. Bump to
                            4 only if edges still read blocky at display scale.
  preserve_area=True        area error ~2e-4 vs ~8e-3 without, for ~same wall time.

Nodata handling: pixels == --nodata-class are dropped (not vectorised).

Run:
  ~/myprojects/recover/.venv/bin/python DNN/polygonise_raster.py \
    --in DNN/data/pdrive_large_tile_classified.tif \
    --out /data/P-Prosjekter2/154001_nyvest/landcover_Geethen/data/polygonise/pdrive_large_tile.gpkg \
    [--nodata-class 0] [--iters 3] [--tile 2048] [--min-pixels 1] [--cores 0]
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.windows import Window
from shapely.geometry import shape as shp_shape

import smoothify

CLASS_COL = "class"


def _write_gpkg(gdf, dest: Path, layer: str):
    """Write a GPKG to `dest`, staging via local disk.

    GPKG is a SQLite database; SQLite cannot hold its file lock / run a
    transaction on a CIFS network mount (the P-drive), so a direct
    to_file(dest) there fails with "Failed to start transaction". Write to a
    local temp file (fast disk, real locking) and copy the finished .gpkg over.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / dest.name
        gdf.to_file(tmp, layer=layer, driver="GPKG")
        os.makedirs(dest.parent, exist_ok=True)
        shutil.copyfile(tmp, dest)


def _tiles(width, height, tx, ty):
    for row in range(0, height, ty):
        h = min(ty, height - row)
        for col in range(0, width, tx):
            w = min(tx, width - col)
            yield Window(col, row, w, h)


def polygonize(path: str, nodata_class: int, tile: int):
    """Classified raster -> per-class polygon GeoDataFrame (nodata dropped).

    Polygonizes tile-by-tile to bound memory. Polygons on a tile seam are split
    there; the per-class dissolve inside smoothify() re-fuses them, so seams do
    not survive into the final map.
    """
    recs = []
    with rasterio.open(path) as src:
        crs = src.crs
        for win in _tiles(src.width, src.height, tile, tile):
            arr = src.read(1, window=win)
            mask = arr != nodata_class
            if not mask.any():
                continue
            wtransform = src.window_transform(win)
            for geom, val in rio_shapes(arr, mask=mask, transform=wtransform):
                recs.append({"geometry": shp_shape(geom), CLASS_COL: int(val)})
    if not recs:
        raise SystemExit("no non-nodata pixels found — nothing to vectorise")
    gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(recs),
                           geometry="geometry", crs=crs)
    return gdf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="output .gpkg path")
    ap.add_argument("--layer", default=None,
                    help="GPKG layer name (default: output file stem)")
    ap.add_argument("--nodata-class", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3,
                    help="smooth_iterations (3 recommended; 4 for smoother edges)")
    ap.add_argument("--cores", type=int, default=0,
                    help="num_cores for smoothify (0=all, 1=serial)")
    ap.add_argument("--tile", type=int, default=2048,
                    help="polygonization tile size in pixels (bounds memory)")
    ap.add_argument("--segment-length", type=float, default=None,
                    help="raster pixel size in map units; None = auto-detect")
    ap.add_argument("--no-preserve-area", action="store_true",
                    help="skip the area-restoration buffer (faster, ~40x more "
                         "area error; not recommended)")
    ap.add_argument("--min-pixels", type=int, default=1,
                    help="drop polygons smaller than this many pixels BEFORE "
                         "smoothing (speckle removal; 1 = keep everything)")
    ap.add_argument("--also-raw", action="store_true",
                    help="also write the un-smoothed polygons to <out stem>_raw.gpkg")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".gpkg":
        raise SystemExit(f"--out must be a .gpkg path, got {out_path.suffix!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer = args.layer or out_path.stem

    t0 = time.perf_counter()
    with rasterio.open(args.inp) as _s:
        px_area = abs(_s.transform.a * _s.transform.e)
    print(f"input: {args.inp}", flush=True)

    gdf = polygonize(args.inp, args.nodata_class, args.tile)
    a_in = float(gdf.geometry.area.sum())
    print(f"polygonize: {len(gdf)} polygons, {len(gdf[CLASS_COL].unique())} classes "
          f"{sorted(gdf[CLASS_COL].unique().tolist())}, area {a_in/1e6:.3f} km^2, "
          f"{time.perf_counter()-t0:.1f}s", flush=True)

    if args.min_pixels > 1:
        thresh = args.min_pixels * px_area
        before = len(gdf)
        gdf = gdf[gdf.geometry.area >= thresh].reset_index(drop=True)
        print(f"speckle filter (< {args.min_pixels}px = {thresh:.0f} m^2): "
              f"{before} -> {len(gdf)} polygons", flush=True)

    if args.also_raw:
        raw_path = out_path.with_name(out_path.stem + "_raw.gpkg")
        _write_gpkg(gdf, raw_path, layer)
        print(f"wrote raw polygons -> {raw_path}", flush=True)

    # Invalid (self-intersecting) polygons are returned unsmoothed by smoothify
    # with a warning — repair first so every geometry actually gets smoothed.
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"repairing {int(invalid.sum())} invalid geometries (make_valid)",
              flush=True)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()

    t_sm = time.perf_counter()
    sm = smoothify.smoothify(
        gdf, segment_length=args.segment_length, num_cores=args.cores,
        merge_field=CLASS_COL, smooth_iterations=args.iters,
        preserve_area=not args.no_preserve_area)
    a_out = float(sm.geometry.area.sum())
    are = abs(a_out - a_in) / a_in if a_in else 0.0
    print(f"smoothify (iters={args.iters}, cores={args.cores}, "
          f"preserve_area={not args.no_preserve_area}): {len(sm)} polygons, "
          f"area_rel_err {are:.2e}, {time.perf_counter()-t_sm:.1f}s", flush=True)

    _write_gpkg(sm, out_path, layer)
    print(f"done -> {out_path} (layer={layer})  "
          f"total {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

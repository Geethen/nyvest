"""Benchmark Smoothify for vectorising DNN classified rasters.

Pipeline under test:  classified int16 raster --(rasterio polygonize)--> per-class
polygons in a GeoDataFrame --(smoothify)--> smooth vectors. Smoothify operates on
VECTORS, so polygonization is a fixed pre-step; we time it once and then sweep the
smoothify hyperparameters that matter (smooth_iterations, num_cores, preserve_area).

For each config we record wall time and quality proxies:
  * n_verts_in / n_verts_out : Chaikin adds vertices; this is the size cost
  * area_rel_err             : |area_out - area_in| / area_in (want ~0 with
                               preserve_area=True; area_tolerance bounds it)
  * n_geoms                  : polygon count (should be invariant)

Run:
  ~/myprojects/recover/.venv/bin/python DNN/smoothify_bench.py \
    --in DNN/data/pdrive_large_tile_classified.tif [--out-dir DNN/data/smoothify_out] \
    [--nodata-class 0] [--iters 2,3,4,5] [--cores 0,1] [--write-gpkg]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape

import smoothify


def _count_verts(gdf: gpd.GeoDataFrame) -> int:
    """Total coordinate count across all geometries (exterior + interiors)."""
    n = 0
    for geom in gdf.geometry.values:
        if geom is None or geom.is_empty:
            continue
        for poly in getattr(geom, "geoms", [geom]):
            n += len(poly.exterior.coords)
            for ring in poly.interiors:
                n += len(ring.coords)
    return n


def polygonize(path: str, nodata_class: int) -> tuple[gpd.GeoDataFrame, float]:
    """Classified raster -> per-class polygon GeoDataFrame (nodata dropped)."""
    t0 = time.perf_counter()
    with rasterio.open(path) as src:
        arr = src.read(1)
        crs = src.crs
        transform = src.transform
        mask = arr != nodata_class
        recs = [
            {"geometry": shp_shape(geom), "class": int(val)}
            for geom, val in rio_shapes(arr, mask=mask, transform=transform)
        ]
    gdf = gpd.GeoDataFrame.from_records(recs)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)
    return gdf, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="if set with --write-gpkg, write the polygonized input and "
                         "each smoothed config to <out-dir>/*.gpkg for QGIS eyeballing")
    ap.add_argument("--nodata-class", type=int, default=0)
    ap.add_argument("--iters", default="2,3,4,5",
                    help="comma list of smooth_iterations to sweep")
    ap.add_argument("--cores", default="0,1",
                    help="comma list of num_cores to sweep (0=all, 1=serial)")
    ap.add_argument("--no-preserve-area", action="store_true",
                    help="also test preserve_area=False (faster, drops the area-"
                         "restoration buffer step)")
    ap.add_argument("--write-gpkg", action="store_true")
    args = ap.parse_args()

    iters = [int(x) for x in args.iters.split(",")]
    cores = [int(x) for x in args.cores.split(",")]
    preserve_opts = [True] + ([False] if args.no_preserve_area else [])

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"input: {args.inp}", flush=True)
    gdf, t_poly = polygonize(args.inp, args.nodata_class)
    v_in = _count_verts(gdf)
    a_in = float(gdf.geometry.area.sum())
    seg = None  # let smoothify auto-detect pixel size
    print(f"polygonize: {len(gdf)} polygons, {v_in:,} vertices, "
          f"area {a_in/1e6:.3f} km^2, {t_poly:.2f}s", flush=True)
    if out_dir and args.write_gpkg:
        gdf.to_file(out_dir / "polygonized.gpkg", driver="GPKG")

    print(f"\n{'iters':>5} {'cores':>5} {'presA':>5} | {'time_s':>8} "
          f"{'v_out':>10} {'v_mult':>6} {'area_rel_err':>12} {'n_geoms':>7}", flush=True)
    print("-" * 74, flush=True)
    rows = []
    for preserve in preserve_opts:
        for nc in cores:
            for it in iters:
                g = gdf.copy()
                t0 = time.perf_counter()
                # merge_field="class": dissolve/smooth WITHIN each class but keep
                # classes distinct. Without it (default merge_collection=True) all
                # polygons dissolve into one geometry — wrong for a multi-class map.
                sm = smoothify.smoothify(
                    g, segment_length=seg, num_cores=nc, merge_field="class",
                    smooth_iterations=it, preserve_area=preserve)
                dt = time.perf_counter() - t0
                v_out = _count_verts(sm)
                a_out = float(sm.geometry.area.sum())
                are = abs(a_out - a_in) / a_in if a_in else 0.0
                print(f"{it:>5} {nc:>5} {str(preserve):>5} | {dt:>8.2f} "
                      f"{v_out:>10,} {v_out/max(v_in,1):>6.2f} {are:>12.2e} "
                      f"{len(sm):>7}", flush=True)
                rows.append(dict(iters=it, cores=nc, preserve_area=preserve,
                                 time_s=dt, v_out=v_out, area_rel_err=are,
                                 n_geoms=len(sm)))
                if out_dir and args.write_gpkg and nc == cores[0] and preserve:
                    sm.to_file(out_dir / f"smooth_it{it}.gpkg", driver="GPKG")

    print(f"\npolygonize time (fixed pre-step): {t_poly:.2f}s | "
          f"input vertices: {v_in:,}", flush=True)


if __name__ == "__main__":
    main()

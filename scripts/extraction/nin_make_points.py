"""Generate FSCS-labelled training points from NiN_v2 polygons.

Reads the field-mapped NiN_v2 nature-type polygons, applies the QC filter and the
naturtype->FSCS crosswalk (nin_crosswalk.py), and emits one point per polygon at
the centroid plus extra interior points for large polygons. Points are kept >=10 m
(one Sentinel-2 pixel) inside the polygon edge to avoid mixed pixels, by sampling
inside a negative-buffered copy of the polygon.

The output point table carries the columns the AlphaEarth extractor and the
downstream loaders need:
  lon, lat (EPSG:4326), class (FSCS code), cell_id (same 25 km grid as the
  stable-allyears parquet), area_m2, tilstand, naturtype  (last two = metadata).

cell_id is computed with the SAME grid definition as
sample_feature_space_stable_allyears.build_grid (25 km cells over the counties
bbox in EPSG:25832, cell_id = j*nx + i), so NiN points share the spatial-blocking
grid used by GroupKFold in the pipeline.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/extraction/nin_make_points.py
  ~/myprojects/recover/.venv/bin/python scripts/extraction/nin_make_points.py \
      --max_interior 8 --out data/nin_points.parquet
"""
from __future__ import annotations

import argparse
import math
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from nin_crosswalk import map_naturtype

SRC = ("/data/P-Prosjekter2/154001_nyvest/GIS/Naturbase/NiN_v2/"
       "NiN_v2_nyvest.shp")

# Grid identical to sample_feature_space_stable_allyears.build_grid
GRID_BBOX_25832 = (234_740.0, 6_435_354.0, 518_617.0, 7_071_896.0)
GRID_CRS = "EPSG:25832"
GRID_SIZE_M = 25_000

EDGE_BUFFER_M = 10.0   # keep points >=1 S2 pixel inside the polygon edge
S2_PIXEL_M = 10.0      # interior-point spacing target


def cell_id_for(x25832: np.ndarray, y25832: np.ndarray) -> np.ndarray:
    """Replicate build_grid's cell_id = j*nx + i for EPSG:25832 coords."""
    xmin, ymin, xmax, ymax = GRID_BBOX_25832
    nx = int(math.ceil((xmax - xmin) / GRID_SIZE_M))
    i = np.floor((x25832 - xmin) / GRID_SIZE_M).astype(int)
    j = np.floor((y25832 - ymin) / GRID_SIZE_M).astype(int)
    return j * nx + i


def interior_points(geom, max_interior: int):
    """Centroid + up to max_interior grid points inside an eroded polygon."""
    eroded = geom.buffer(-EDGE_BUFFER_M)
    if eroded.is_empty or eroded.area == 0:
        # polygon too thin to erode; fall back to a guaranteed-inside point
        return [geom.representative_point()]
    pts = [eroded.representative_point()]
    if max_interior <= 1:
        return pts
    # spacing scales with polygon size so big polygons get more (capped) points
    minx, miny, maxx, maxy = eroded.bounds
    n_target = min(max_interior, 1 + int(eroded.area / (S2_PIXEL_M ** 2 * 400)))
    if n_target <= 1:
        return pts
    step = max(S2_PIXEL_M, math.sqrt(eroded.area / n_target))
    gx = np.arange(minx + step / 2, maxx, step)
    gy = np.arange(miny + step / 2, maxy, step)
    for yy in gy:
        for xx in gx:
            if len(pts) >= max_interior:
                return pts
            p = Point(xx, yy)
            if eroded.contains(p):
                pts.append(p)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_interior", type=int, default=8,
                    help="max points per polygon (centroid + interior)")
    ap.add_argument("--min_area", type=float, default=900.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    out = args.out or os.path.join(base, "data", "nin_points.parquet")

    g = gpd.read_file(
        SRC, columns=["hovedøkosy", "naturtype", "usikkerhet", "tilstand",
                      "area_m2"])
    print(f"read {len(g)} NiN polygons (crs={g.crs})")

    qc = g[(g["usikkerhet"] != "Ja")
           & (g["naturtype"] != "Hule eiker")
           & (g["area_m2"] >= args.min_area)].copy()
    qc["fscs"] = [map_naturtype(nt, eco) for nt, eco
                  in zip(qc["naturtype"], qc["hovedøkosy"])]
    qc = qc.dropna(subset=["fscs"]).copy()
    qc["fscs"] = qc["fscs"].astype(int)
    print(f"after QC + crosswalk: {len(qc)} polygons")

    qc = qc.to_crs(GRID_CRS)  # NiN is 25833; grid is 25832

    records = []
    for _, row in qc.iterrows():
        for p in interior_points(row.geometry, args.max_interior):
            records.append({
                "x25832": p.x, "y25832": p.y,
                "class": row["fscs"], "area_m2": float(row["area_m2"]),
                "tilstand": row["tilstand"], "naturtype": row["naturtype"],
            })
    pts = gpd.GeoDataFrame(
        records,
        geometry=[Point(r["x25832"], r["y25832"]) for r in records],
        crs=GRID_CRS)
    pts["cell_id"] = cell_id_for(
        pts["x25832"].values, pts["y25832"].values)
    wgs = pts.to_crs("EPSG:4326")
    pts["lon"] = wgs.geometry.x
    pts["lat"] = wgs.geometry.y

    df = pts.drop(columns="geometry")[
        ["lon", "lat", "class", "cell_id", "area_m2", "tilstand", "naturtype"]]
    df.to_parquet(out, index=False)
    print(f"\n[OK] wrote {len(df)} points -> {out}")
    print(f"  points per polygon: {len(df) / len(qc):.2f} avg")
    print(f"  distinct cell_id: {df['cell_id'].nunique()}")
    print("\nclass distribution (points):")
    print(df["class"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

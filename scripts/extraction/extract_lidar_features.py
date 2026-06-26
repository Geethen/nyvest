"""Extract lidar terrain features at every unique training point.

Motivation: test whether elevation, canopy height (TCH) and derived terrain
shape (slope, aspect, terrain ruggedness index) improve class separability over
the AlphaEarth embedding alone — in particular the 11 (infrastructure) vs 12
(snow/ice) ceiling, which is an elevation/terrain problem (see
scripts/extraction/extract_dem_features.py and the scheme B handover).

Source rasters: local lidar tiles, 2 bands per tile — DTM (elevation, m) and
chm (canopy height, m) — at 3 m, EPSG:32633. Two directories are merged; on a
tile-code collision the Vestland_Moreromsdal set wins (larger, project-specific):
  .../Nature_types_mapping/Vestland_Moreromsdal_features/lidar/*.tif   (679)
  .../Nature_types_mapping/Features/lidar/*.tif                        (130 extra)

Speed approach: points are sparse relative to the rasters, so we never build a
mosaic. We reproject the unique points to EPSG:32633 once, bucket them into the
tile whose bounds contain them, then open each tile exactly once and read only
the small windows we need. Each point reads a 3x3 DTM window centred on it;
slope/aspect/TRI are computed from that window (Horn 1981 for slope/aspect,
Riley 1999 for TRI). This is edge-safe per tile (the window is clipped to the
tile, and the central 3x3 of a point near a tile edge still comes from the same
source raster) and avoids any global derivative rasters.

Points: union of unique (lon, lat) across the three working parquets, so the
output joins into any downstream dataset by exact (lon, lat) equality — same
contract as dem_features.parquet.

Output: data/lidar_features.parquet
  lon, lat, elevation, tch, slope, aspect_sin, aspect_cos, tri
Checkpointed per tile in data/lidar_features.parquet.checkpoint.json.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/extraction/extract_lidar_features.py
  ~/myprojects/recover/.venv/bin/python scripts/extraction/extract_lidar_features.py --limit-tiles 5   # quick test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _data_root() -> str:
    for c in (os.environ.get("NYVEST_DATA_DIR"),
              "/data/P-Prosjekter2/154001_nyvest",
              "P:/154001_nyvest"):
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("Could not locate the nyvest data root; set NYVEST_DATA_DIR")


DATA_ROOT = _data_root()
# Lower-priority dir first so the higher-priority dir overwrites on collision.
LIDAR_DIRS = [
    os.path.join(DATA_ROOT, "Nature_types_mapping/Features/lidar"),
    os.path.join(DATA_ROOT, "Nature_types_mapping/Vestland_Moreromsdal_features/lidar"),
]
SOURCES = [
    "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet",
    "grunnkart_nyvest_fscs_unstable_alphaearth.parquet",
    "grunnkart_nyvest_fscs_alphaearth.parquet",
]
OUT_PARQUET = os.path.join(DATA_DIR, "lidar_features.parquet")
CHECKPOINT = OUT_PARQUET + ".checkpoint.json"

DTM_BAND, CHM_BAND = 1, 2  # confirmed: descriptions == ('DTM', 'chm')
# Both bands are stored as millimetres (no scale tag set) — a single global
# scale for every tile. Verified against the Copernicus GLO30 DEM at the shared
# training points: /1000 is the best-matching divisor for all 112 sampled tiles
# (DTM /1000 gives 0-2300 m, CHM /1000 gives realistic 0-20 m canopy). An
# earlier per-tile "infer from value range" heuristic mis-scaled low-elevation
# tiles by 10x (read mm as cm) — do not reintroduce it. Nodata is NaN.
MM_TO_M = 0.001


def unique_points() -> pd.DataFrame:
    parts = " UNION ".join(
        f"SELECT lon, lat FROM '{os.path.join(DATA_DIR, s)}'" for s in SOURCES)
    df = duckdb.sql(parts).df()
    return df.sort_values(["lon", "lat"]).reset_index(drop=True)


def tile_index() -> list[dict]:
    """One entry per usable tile: path + bounds, higher-priority dir winning."""
    by_name: dict[str, dict] = {}
    for d in LIDAR_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.tif"))):
            name = os.path.basename(f)
            with rasterio.open(f) as ds:
                b = ds.bounds
            by_name[name] = {"path": f, "left": b.left, "bottom": b.bottom,
                             "right": b.right, "top": b.top}
    return list(by_name.values())


def assign_tiles(pts: pd.DataFrame, tiles: list[dict]) -> dict[int, np.ndarray]:
    """Return {tile_index -> array of point row positions inside that tile}.

    Tiles barely overlap (~3 m seams); first containing tile wins, so each point
    is sampled once. Vectorised mask per tile keeps this fast for ~800 tiles.
    """
    x = pts["x"].to_numpy()
    y = pts["y"].to_numpy()
    unassigned = np.ones(len(pts), dtype=bool)
    buckets: dict[int, np.ndarray] = {}
    for ti, t in enumerate(tiles):
        m = (unassigned
             & (x >= t["left"]) & (x < t["right"])
             & (y >= t["bottom"]) & (y < t["top"]))
        if m.any():
            buckets[ti] = np.flatnonzero(m)
            unassigned &= ~m
    n_out = int(unassigned.sum())
    if n_out:
        print(f"  {n_out:,} points fall outside all lidar tiles (left NaN)")
    return buckets


def _terrain_from_windows(win: np.ndarray, res: float) -> dict[str, np.ndarray]:
    """Vectorised slope/aspect/TRI from a stack of 3x3 DTM windows.

    win: (n, 3, 3) float array, centre pixel = the point. NaNs in the window
    (tile edges / nodata) propagate to NaN outputs for that point.
    """
    # ESRI/Horn 3x3 convention, row 0 = north (top of array):
    #   a b c
    #   d e f
    #   g h i
    z = win
    a, b, c = z[:, 0, 0], z[:, 0, 1], z[:, 0, 2]
    d, _, f = z[:, 1, 0], z[:, 1, 1], z[:, 1, 2]
    g, h, i = z[:, 2, 0], z[:, 2, 1], z[:, 2, 2]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)  # bottom - top (ESRI)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # ESRI aspect: compass degrees, 0 = north, increasing clockwise; the
    # downslope direction. atan2(dzdy, -dzdx) then rotate into compass space.
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = np.mod(450.0 - aspect, 360.0)
    # Flat cells (zero gradient) have undefined aspect; ESRI returns -1. We set
    # aspect_sin/aspect_cos from it anyway (both ~ defined), which is harmless.
    # Riley 1999 TRI: mean abs diff of centre vs its 8 neighbours.
    e = z[:, 1, 1]
    neigh = np.stack([a, b, c, d, f, g, h, i], axis=1)
    diffs = np.abs(neigh - e[:, None])
    valid = ~np.isnan(diffs)
    n_valid = valid.sum(axis=1)
    summed = np.nansum(diffs, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tri = np.where(n_valid > 0, summed / n_valid, np.nan)
    return {"slope": slope, "aspect": aspect, "tri": tri}


def sample_tile(t: dict, rows: np.ndarray, pts: pd.DataFrame) -> pd.DataFrame:
    """Sample one tile for all its points.

    Reads the bounding-box window covering this tile's points (plus a 1 px halo
    for the 3x3 DTM neighbourhood), then gathers per-point 3x3 DTM windows and
    1x1 CHM values with numpy fancy indexing. Reading the window instead of the
    full band is a big win when a tile holds few scattered points; paying the
    read once per tile (not twice per point) is the rest. Tiles run across a
    thread pool (GDAL drops the GIL), taking the full run to ~3 min.
    """
    x = pts["x"].to_numpy()[rows]
    y = pts["y"].to_numpy()[rows]
    out = pd.DataFrame({
        "lon": pts["lon"].to_numpy()[rows],
        "lat": pts["lat"].to_numpy()[rows],
    })
    with rasterio.open(t["path"]) as ds:
        res = ds.res[0]
        H, W = ds.height, ds.width
        nd_dtm = ds.nodatavals[DTM_BAND - 1]
        nd_chm = ds.nodatavals[CHM_BAND - 1]
        # Vectorised world -> (row, col) via the inverse affine; col,row order.
        inv = ~ds.transform
        col_f, row_f = inv * (x, y)
        r = np.floor(row_f).astype(np.int64)
        c = np.floor(col_f).astype(np.int64)
        in_tile = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        if not in_tile.any():
            for col in ("elevation", "tch", "slope", "aspect_sin",
                        "aspect_cos", "tri"):
                out[col] = np.nan
            return out
        # Window bounding the points, with a 1 px halo, clipped to the tile.
        r0 = max(int(r[in_tile].min()) - 1, 0)
        c0 = max(int(c[in_tile].min()) - 1, 0)
        r1 = min(int(r[in_tile].max()) + 2, H)  # exclusive
        c1 = min(int(c[in_tile].max()) + 2, W)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        dtm = ds.read(DTM_BAND, window=win).astype("float64")
        chm = ds.read(CHM_BAND, window=win).astype("float64")
    if nd_dtm is not None:
        dtm = np.where(dtm == nd_dtm, np.nan, dtm)
    if nd_chm is not None:
        chm = np.where(chm == nd_chm, np.nan, chm)
    dtm = dtm * MM_TO_M   # millimetres -> metres (single global scale)
    chm = chm * MM_TO_M

    # Local (row, col) within the window; pad DTM by 1 px of NaN so every 3x3 is
    # in-bounds and edge points correctly pick up NaN past the data extent.
    rl, cl = r - r0, c - c0
    h, w = dtm.shape
    pad = np.full((h + 2, w + 2), np.nan)
    pad[1:-1, 1:-1] = dtm
    dtm_win = np.full((len(rows), 3, 3), np.nan)
    chm_val = np.full(len(rows), np.nan)
    rr, cc = rl[in_tile], cl[in_tile]
    for dy in range(3):
        for dx in range(3):
            dtm_win[in_tile, dy, dx] = pad[rr + dy, cc + dx]
    chm_val[in_tile] = chm[rr, cc]

    out["elevation"] = dtm_win[:, 1, 1]
    out["tch"] = chm_val
    der = _terrain_from_windows(dtm_win, res)
    out["slope"] = der["slope"]
    out["aspect_sin"] = np.sin(np.radians(der["aspect"]))
    out["aspect_cos"] = np.cos(np.radians(der["aspect"]))
    out["tri"] = der["tri"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tiles", type=int, default=0,
                    help="process only the first N populated tiles (testing)")
    ap.add_argument("--workers", type=int, default=8,
                    help="thread pool size for tile reads (GDAL releases the GIL)")
    args = ap.parse_args()

    pts = unique_points()
    t = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    pts["x"], pts["y"] = t.transform(pts["lon"].to_numpy(), pts["lat"].to_numpy())
    print(f"unique points: {len(pts):,}")

    tiles = tile_index()
    print(f"lidar tiles: {len(tiles)}")
    buckets = assign_tiles(pts, tiles)
    populated = sorted(buckets)
    print(f"populated tiles: {len(populated)} "
          f"({sum(len(v) for v in buckets.values()):,} points inside tiles)")
    if args.limit_tiles:
        populated = populated[:args.limit_tiles]

    done: set[int] = set()
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as fh:
            done = set(json.load(fh)["tiles_done"])
        print(f"checkpoint: {len(done)} tiles already done")

    con = duckdb.connect()
    con.sql("CREATE TABLE lidar (lon DOUBLE, lat DOUBLE, elevation DOUBLE, "
            "tch DOUBLE, slope DOUBLE, aspect_sin DOUBLE, aspect_cos DOUBLE, "
            "tri DOUBLE)")
    if os.path.exists(OUT_PARQUET):
        con.sql(f"INSERT INTO lidar SELECT * FROM '{OUT_PARQUET}'")

    todo = [ti for ti in populated if ti not in done]
    t0 = time.time()
    # Reads run in parallel threads (GDAL drops the GIL); duckdb inserts and
    # checkpointing stay on this thread as results arrive.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(sample_tile, tiles[ti], buckets[ti], pts): ti
                for ti in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            ti = futs[fut]
            res = fut.result()
            con.register("_res", res)
            con.sql("INSERT INTO lidar SELECT * FROM _res")
            con.unregister("_res")
            done.add(ti)
            if n % 50 == 0 or n == len(todo):
                con.sql(f"COPY (SELECT DISTINCT * FROM lidar) TO '{OUT_PARQUET}' "
                        f"(FORMAT PARQUET)")
                with open(CHECKPOINT, "w") as fh:
                    json.dump({"tiles_done": sorted(done)}, fh)
                rate = n / (time.time() - t0)
                eta = (len(todo) - n) / max(rate, 1e-9)
                print(f"  {n}/{len(todo)} tiles ({rate:.1f}/s, eta {eta:.0f}s)")

    con.sql(f"COPY (SELECT DISTINCT * FROM lidar) TO '{OUT_PARQUET}' (FORMAT PARQUET)")
    with open(CHECKPOINT, "w") as fh:
        json.dump({"tiles_done": sorted(done)}, fh)
    stats = con.sql(
        "SELECT count(*), count(elevation), count(tch) FROM lidar").fetchone()
    print(f"saved {OUT_PARQUET}: {stats[0]:,} rows "
          f"({stats[1]:,} elevation, {stats[2]:,} tch)")


if __name__ == "__main__":
    main()

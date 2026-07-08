"""Extract AlphaEarth bands (+ all years) at the NiN training points.

Takes the points from nin_make_points.py and samples the annual AlphaEarth
embedding (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) at each, across 2017..2025,
producing a parquet whose schema matches
data/grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet:
  cell_id, class, year, lon, lat, A00..A63   (long form, one row / point / year)

This is the same extraction contract as sample_feature_space_stable_allyears.py
(same collection, .first() per-year reduce, scale 10, sampleRegions), minus the
kMeans coverage step — the NiN points are already chosen, so we just sample bands
at them. Sharded by cell_id with a checkpoint so it resumes after GEE token
expiry, exactly like the stable-allyears extractor.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/extraction/nin_extract_alphaearth.py
  ~/myprojects/recover/.venv/bin/python scripts/extraction/nin_extract_alphaearth.py \
      --max_workers 4 --test_mode
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import duckdb
import ee
import pandas as pd
from tqdm.auto import tqdm

from sample_feature_space_stable_allyears import (
    PROJECT, YEARS, SCALE, EXPECTED_BANDS, init_gee, build_alphaearth_year)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
DEFAULT_IN = os.path.join(DATA, "nin_points.parquet")
DEFAULT_OUT = os.path.join(DATA, "nin_alphaearth.parquet")

TILE_SCALE = 4
SHARD_TIMEOUT_S = 240
FLUSH_EVERY = 10


def points_fc_for_cell(df_cell: pd.DataFrame) -> ee.FeatureCollection:
    """Build a FeatureCollection carrying lon/lat/class/cell_id per point."""
    feats = []
    for _, r in df_cell.iterrows():
        geom = ee.Geometry.Point([float(r["lon"]), float(r["lat"])])
        feats.append(ee.Feature(geom, {
            "lon": float(r["lon"]), "lat": float(r["lat"]),
            "class": int(r["class"]), "cell_id": int(r["cell_id"]),
        }))
    return ee.FeatureCollection(feats)


def sample_cell(df_cell, year_images, years, tile_scale=TILE_SCALE):
    fc = points_fc_for_cell(df_cell)
    frames = []
    for y in years:
        sampled = year_images[y].sampleRegions(
            collection=fc, scale=SCALE, tileScale=tile_scale, geometries=False)
        sampled = sampled.map(lambda f: f.set({"year": y}))
        df_y = ee.data.computeFeatures({
            "expression": sampled, "fileFormat": "PANDAS_DATAFRAME"})
        if df_y is not None and not df_y.empty:
            frames.append(df_y)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--years", default=",".join(str(y) for y in YEARS))
    ap.add_argument("--max_workers", type=int, default=6)
    ap.add_argument("--test_mode", action="store_true")
    args = ap.parse_args()

    years = tuple(int(y) for y in args.years.split(","))
    pts = pd.read_parquet(args.in_path)[["lon", "lat", "class", "cell_id"]]
    cells = sorted(int(c) for c in pts["cell_id"].unique())
    if args.test_mode:
        cells = cells[:2]
        pts = pts[pts["cell_id"].isin(cells)]
        print(f"*** TEST MODE: cells {cells} ({len(pts)} pts) ***")

    checkpoint = args.out + ".checkpoint.json"
    done = set()
    if os.path.exists(args.out) and os.path.exists(checkpoint):
        done = set(json.load(open(checkpoint)).get("done", []))
        print(f"resuming: {len(done)}/{len(cells)} cells done")

    init_gee(PROJECT)
    year_images = {y: build_alphaearth_year(y) for y in years}

    db = duckdb.connect()
    if os.path.exists(args.out):
        db.execute(f"CREATE TABLE data AS SELECT * FROM '{args.out}'")
    lock = Lock()
    schema_checked = [False]

    def flush():
        try:
            n = db.execute("SELECT count(*) FROM data").fetchone()[0]
        except duckdb.CatalogException:
            return 0
        tmp = args.out + ".tmp"
        db.execute(f"COPY data TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        if os.path.exists(args.out):
            os.remove(args.out)
        os.rename(tmp, args.out)
        return n

    def save_cp():
        json.dump({"done": sorted(done)}, open(checkpoint, "w"))

    def work(cid):
        df_cell = pts[pts["cell_id"] == cid]
        df = sample_cell(df_cell, year_images, years)
        if df is None or df.empty:
            return cid, 0
        if not schema_checked[0]:
            missing = [b for b in EXPECTED_BANDS if b not in df.columns]
            if missing:
                raise RuntimeError(f"missing bands {missing[:5]} "
                                   f"(got {len(df.columns)} cols)")
            schema_checked[0] = True
        with lock:
            try:
                db.execute("INSERT INTO data BY NAME SELECT * FROM df")
            except duckdb.CatalogException:
                db.execute("CREATE TABLE data AS SELECT * FROM df")
        return cid, len(df)

    todo = [c for c in cells if c not in done]
    total_rows = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(work, c): c for c in todo}
        since = 0
        with tqdm(total=len(todo), desc="NiN AlphaEarth", ncols=90) as bar:
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    _, n = fut.result()
                    total_rows += n
                    done.add(cid)
                    save_cp()
                    if n:
                        tqdm.write(f"  cell {cid}: {n} rows")
                    since += 1
                    if since >= FLUSH_EVERY:
                        flush()
                        since = 0
                except Exception as e:
                    tqdm.write(f"  [ERROR] cell {cid}: {str(e)[:160]}")
                bar.update(1)

    n = flush()
    db.close()
    print(f"\n[OK] {n:,} rows -> {args.out}")
    if len(done) == len(cells) and os.path.exists(checkpoint):
        os.remove(checkpoint)


if __name__ == "__main__":
    main()

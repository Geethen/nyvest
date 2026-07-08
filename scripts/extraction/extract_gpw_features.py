"""Extract Global Pasture Watch (GPW) features at every unique training point.

Tests whether the GPW 30 m annual layers improve class separability over the
AlphaEarth embedding — in particular the confusable VEGETATION classes (cls5
grassland, cls6 scrub/heath, cls7 wetland, cls3 cropland) that drag macro-F1 in
the scheme B experiment. GPW is purpose-built grassland mapping, so it is a
natural prior for the grassland/cropland boundary.

Five GPW datasets (https://developers.google.com/earth-engine/datasets/publisher/
global-pasture-watch), filtered to 2022 to start (all are annual 2000-2022/2024):
  grassland_c          dominant_class   0=other 1=cultivated 2=nat/semi grassland
  cultiv-grassland_p   probability      0-100  cultivated-grassland probability
  nat-semi-grassland_p probability      0-100  natural/semi-natural prob
  short-veg-height_m   height           0-10 m median veg height (scale 0.1)
  ugpp_m               gc_m2            0-4000 gC/m2/yr gross primary productivity

Output: data/gpw_features.parquet, keyed by exact (lon, lat) like
lidar_features.parquet / dem_features.parquet so the probe joins by coordinate:
  lon, lat, gpw_dominant_class, gpw_cultiv_p, gpw_natsemi_p, gpw_veg_height,
  gpw_ugpp

Points = union of unique (lon, lat) across the three working parquets. Sampled in
chunks via sampleRegions on a single multi-band 2022 image, checkpointed per
chunk so it resumes after GEE token expiry.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/extraction/extract_gpw_features.py
  ~/myprojects/recover/.venv/bin/python scripts/extraction/extract_gpw_features.py --year 2022 --test
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import ee
import pandas as pd
from tqdm.auto import tqdm

from sample_feature_space_stable_allyears import PROJECT, init_gee

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
SOURCES = [
    "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet",
    "grunnkart_nyvest_fscs_unstable_alphaearth.parquet",
    "grunnkart_nyvest_fscs_alphaearth.parquet",
]
OUT = os.path.join(DATA, "gpw_features.parquet")

# (output column, asset id, source band). 30 m, annual.
GPW_LAYERS = [
    ("gpw_dominant_class",
     "projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c", "dominant_class"),
    ("gpw_cultiv_p",
     "projects/global-pasture-watch/assets/ggc-30m/v1/cultiv-grassland_p", "probability"),
    ("gpw_natsemi_p",
     "projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p", "probability"),
    ("gpw_veg_height",
     "projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m", "height"),
    ("gpw_ugpp",
     "projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m", "gc_m2"),
]
OUT_COLS = [c for c, _, _ in GPW_LAYERS]
SCALE = 30
CHUNK = 3000


def unique_points() -> pd.DataFrame:
    parts = " UNION ".join(
        f"SELECT lon, lat FROM '{os.path.join(DATA, s)}'" for s in SOURCES)
    return duckdb.sql(parts).df().sort_values(["lon", "lat"]).reset_index(drop=True)


def build_gpw_image(year: int) -> ee.Image:
    """Single multi-band image for `year`, one band per GPW layer."""
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    bands = []
    for out_col, cid, band in GPW_LAYERS:
        img = (ee.ImageCollection(cid).filterDate(start, end).first()
               .select([band], [out_col]))
        bands.append(img)
    return ee.Image.cat(bands)


def sample_chunk(img, df_chunk) -> pd.DataFrame:
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                        {"lon": float(r.lon), "lat": float(r.lat)})
             for r in df_chunk.itertuples()]
    fc = ee.FeatureCollection(feats)
    sampled = img.sampleRegions(collection=fc, scale=SCALE, tileScale=4,
                                geometries=False)
    return ee.data.computeFeatures({
        "expression": sampled, "fileFormat": "PANDAS_DATAFRAME"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    out = OUT if args.year == 2022 else OUT.replace(".parquet", f"_{args.year}.parquet")
    checkpoint = out + ".checkpoint.json"

    pts = unique_points()
    print(f"unique points: {len(pts):,}")
    chunks = [pts.iloc[i:i + CHUNK] for i in range(0, len(pts), CHUNK)]
    if args.test:
        chunks = chunks[:2]
        print(f"*** TEST: {len(chunks)} chunks ***")

    done = set()
    if os.path.exists(out) and os.path.exists(checkpoint):
        done = set(json.load(open(checkpoint))["done"])
        print(f"resuming: {len(done)}/{len(chunks)} chunks done")

    init_gee(PROJECT)
    img = build_gpw_image(args.year)

    con = duckdb.connect()
    if os.path.exists(out):
        con.sql(f"CREATE TABLE g AS SELECT * FROM '{out}'")

    def flush():
        con.sql(f"COPY (SELECT DISTINCT * FROM g) TO '{out}.tmp' (FORMAT PARQUET)")
        if os.path.exists(out):
            os.remove(out)
        os.rename(out + ".tmp", out)
        json.dump({"done": sorted(done)}, open(checkpoint, "w"))

    def work(idx):
        df = sample_chunk(img, chunks[idx])
        # sampleRegions drops points with no overlap; keep lon/lat + GPW cols
        keep = ["lon", "lat"] + [c for c in OUT_COLS if c in df.columns]
        return idx, df[keep] if len(df) else df

    todo = [i for i in range(len(chunks)) if i not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, i): i for i in todo}
        with tqdm(total=len(todo), desc=f"GPW {args.year}", ncols=90) as bar:
            since = 0
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    _, df = fut.result()
                    if len(df):
                        con.register("_r", df)
                        try:
                            con.sql("INSERT INTO g BY NAME SELECT * FROM _r")
                        except duckdb.CatalogException:
                            con.sql("CREATE TABLE g AS SELECT * FROM _r")
                        con.unregister("_r")
                    done.add(i)
                    since += 1
                    if since >= 20:
                        flush()
                        since = 0
                except Exception as e:
                    tqdm.write(f"  [ERROR] chunk {i}: {str(e)[:160]}")
                bar.update(1)

    flush()
    stats = con.sql(f"SELECT count(*), {', '.join(f'count({c})' for c in OUT_COLS)} "
                    "FROM g").fetchone()
    con.close()
    print(f"\n[OK] {out}: {stats[0]:,} rows")
    for c, n in zip(OUT_COLS, stats[1:]):
        print(f"  {c}: {n:,} non-null")
    if len(done) == len(chunks) and os.path.exists(checkpoint):
        os.remove(checkpoint)


if __name__ == "__main__":
    main()

"""Extract static terrain features (elevation, slope) for every unique point.

Motivation: class 12 (snow/ice) is spectrally inseparable from class 11
(infrastructure) in the annual AlphaEarth embedding — the main accuracy ceiling
of the scheme B pipeline (see common_ground/reports/research/HANDOVER_accuracy.md).
Perennial snow/glaciers sit at high elevation on open slopes; infrastructure is
overwhelmingly low-elevation. A DEM band is the cheapest external discriminator:
it needs only one static sample per unique (lon, lat), not a re-extraction of
the embedding time series.

Source: COPERNICUS/DEM/GLO30 (30 m), sampled at scale=30 with slope from
ee.Terrain.slope on the mosaic.

Points: union of unique (lon, lat) across the three working parquets
(stable allyears, unstable, 2020-only stable), so the features can be joined
into any of the downstream datasets by exact (lon, lat) equality.

Output: data/dem_features.parquet  (lon, lat, elevation, slope)
Checkpointed per chunk in data/dem_features.checkpoint.json — safe to re-run.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/extraction/extract_dem_features.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import ee
import pandas as pd

PROJECT = "ee-gsingh"
CHUNK = 2_000
MAX_WORKERS = 8
MAX_RETRIES = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCES = [
    "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet",
    "grunnkart_nyvest_fscs_unstable_alphaearth.parquet",
    "grunnkart_nyvest_fscs_alphaearth.parquet",
]
OUT_PARQUET = os.path.join(DATA_DIR, "dem_features.parquet")
CHECKPOINT = OUT_PARQUET + ".checkpoint.json"


def unique_points() -> pd.DataFrame:
    parts = " UNION ".join(
        f"SELECT lon, lat FROM '{os.path.join(DATA_DIR, s)}'" for s in SOURCES)
    df = duckdb.sql(parts).df()
    return df.sort_values(["lon", "lat"]).reset_index(drop=True)


def terrain_image() -> ee.Image:
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()
    slope = ee.Terrain.slope(dem.setDefaultProjection("EPSG:4326", None, 30))
    return dem.rename("elevation").addBands(slope.rename("slope"))


def sample_chunk(img: ee.Image, chunk: pd.DataFrame) -> pd.DataFrame:
    feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]),
                        {"i": int(idx)})
             for idx, r in zip(chunk.index, chunk.itertuples())]
    fc = ee.FeatureCollection(feats)
    out = img.sampleRegions(collection=fc, scale=30, geometries=False,
                            tileScale=4).getInfo()
    rows = [{"i": f["properties"]["i"],
             "elevation": f["properties"].get("elevation"),
             "slope": f["properties"].get("slope")}
            for f in out["features"]]
    got = pd.DataFrame(rows).set_index("i")
    res = chunk.copy()
    res["elevation"] = got["elevation"].reindex(chunk.index)
    res["slope"] = got["slope"].reindex(chunk.index)
    return res


def main():
    ee.Initialize(project=PROJECT)
    pts = unique_points()
    print(f"unique points: {len(pts):,}")

    done: set[int] = set()
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            done = set(json.load(f)["chunks_done"])
        print(f"checkpoint: {len(done)} chunks already done")

    chunks = [(ci, pts.iloc[s:s + CHUNK])
              for ci, s in enumerate(range(0, len(pts), CHUNK))
              if ci not in done]
    img = terrain_image()
    con = duckdb.connect()
    con.sql("CREATE TABLE IF NOT EXISTS dem (lon DOUBLE, lat DOUBLE, "
            "elevation DOUBLE, slope DOUBLE)")
    if os.path.exists(OUT_PARQUET):
        con.sql(f"INSERT INTO dem SELECT * FROM '{OUT_PARQUET}'")

    def work(ci, chunk):
        for attempt in range(MAX_RETRIES):
            try:
                return ci, sample_chunk(img, chunk)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(5 * (attempt + 1))
                print(f"  chunk {ci} retry {attempt + 1}: {str(e)[:80]}")

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(work, ci, c): ci for ci, c in chunks}
        for fut in as_completed(futs):
            ci, res = fut.result()
            con.register("_res", res[["lon", "lat", "elevation", "slope"]])
            con.sql("INSERT INTO dem SELECT * FROM _res")
            con.unregister("_res")
            done.add(ci)
            n_done += 1
            if n_done % 5 == 0 or n_done == len(chunks):
                con.sql(f"COPY (SELECT DISTINCT * FROM dem) TO '{OUT_PARQUET}' "
                        f"(FORMAT PARQUET)")
                with open(CHECKPOINT, "w") as f:
                    json.dump({"chunks_done": sorted(done)}, f)
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(chunks)} chunks "
                      f"({rate:.2f}/s, eta {(len(chunks)-n_done)/max(rate,1e-9):.0f}s)")

    con.sql(f"COPY (SELECT DISTINCT * FROM dem) TO '{OUT_PARQUET}' "
            f"(FORMAT PARQUET)")
    with open(CHECKPOINT, "w") as f:
        json.dump({"chunks_done": sorted(done)}, f)
    n = con.sql(f"SELECT count(*), count(elevation) FROM '{OUT_PARQUET}'").fetchone()
    print(f"saved {OUT_PARQUET}: {n[0]:,} rows, {n[1]:,} with elevation")


if __name__ == "__main__":
    main()

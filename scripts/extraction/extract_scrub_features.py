"""Extract literature-backed scrub-discriminating features per unique point.

Context: in the scheme B pipeline, class 6 (scrub/heathland) is the largest
macro-F1 drag — confused with 11 (infrastructure), 4 (forest), 7 (wetland),
5 (grassland). The annual AlphaEarth embedding is a temporal *summary*, so the
features the literature reports as most discriminating for shrubland are exactly
the ones it compresses away: intra-annual phenology, vertical structure, and
seasonal moisture. See HANDOVER_accuracy.md and the literature notes.

Features (all static per (lon, lat); a 2020 reference year, two seasons):
  Sentinel-2 SR (red-edge + SWIR phenology) — summer (Jun-Aug) and
  shoulder (Apr-May + Sep-Oct) median composites, SCL-masked:
    *_summer / *_shoulder for: ndre (B8,B5), cire (B7/B5 - 1), ndvi,
    ndmi (B8,B11), nari (1/B3 - 1/B5 ... anthocyanin), plus raw B5,B6,B7,B11.
    Phenology deltas: d_ndvi, d_ndre, d_ndmi (summer - shoulder).
  Sentinel-1 GRD (structure + moisture) — summer & shoulder:
    vv, vh, vh_vv_ratio (mean), and vv_std/vh_std (temporal variance, captures
    wetland inundation dynamics).
  Canopy height (Lang et al. 2023, ETH 10 m): canopy_height — the clean
  scrub-vs-forest discriminator (top-3 feature in the wetland study).

Reference year 2020 (matches the AlphaEarth sampling year). These are a quick
*screening* set to validate with binary scrub-vs-X probes BEFORE wiring any of
them into the full pipeline (the DEM features were correctly killed this way).

Output: data/scrub_features.parquet  (lon, lat, <features>)
Checkpointed in data/scrub_features.parquet.checkpoint.json — safe to re-run.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/extraction/extract_scrub_features.py
  ~/myprojects/recover/.venv/bin/python scripts/extraction/extract_scrub_features.py --test
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import ee
import pandas as pd

PROJECT = "ee-gsingh"
YEAR = 2020
CHUNK = 1_000          # smaller than DEM: each point pulls ~25 bands
MAX_WORKERS = 8
MAX_RETRIES = 4

SUMMER = (f"{YEAR}-06-01", f"{YEAR}-09-01")
SHOULDER_1 = (f"{YEAR}-04-01", f"{YEAR}-06-01")
SHOULDER_2 = (f"{YEAR}-09-01", f"{YEAR}-11-01")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCES = [
    "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet",
    "grunnkart_nyvest_fscs_unstable_alphaearth.parquet",
    "grunnkart_nyvest_fscs_alphaearth.parquet",
]
OUT_PARQUET = os.path.join(DATA_DIR, "scrub_features.parquet")
CHECKPOINT = OUT_PARQUET + ".checkpoint.json"

S2_COLL = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLL = "COPERNICUS/S1_GRD"
CH_ASSET = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"

# Final feature column order (defines parquet schema)
FEATURES = [
    "canopy_height",
    # summer S2
    "ndvi_summer", "ndre_summer", "cire_summer", "ndmi_summer", "nari_summer",
    "b5_summer", "b6_summer", "b7_summer", "b11_summer",
    # shoulder S2
    "ndvi_shoulder", "ndre_shoulder", "ndmi_shoulder",
    # phenology deltas
    "d_ndvi", "d_ndre", "d_ndmi",
    # S1 SAR
    "vv_summer", "vh_summer", "ratio_summer",
    "vv_shoulder", "vh_shoulder", "ratio_shoulder",
    "vv_std", "vh_std",
]


def init_gee():
    try:
        ee.Initialize(project=PROJECT,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=PROJECT)


def unique_points() -> pd.DataFrame:
    parts = " UNION ".join(
        f"SELECT lon, lat FROM '{os.path.join(DATA_DIR, s)}'" for s in SOURCES)
    return duckdb.sql(parts).df().sort_values(["lon", "lat"]).reset_index(drop=True)


def _mask_s2(img):
    scl = img.select("SCL")
    # drop shadow(3), cloud med/high(8,9), cirrus(10), snow(11)
    keep = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
            .And(scl.neq(10)).And(scl.neq(11)))
    return img.updateMask(keep)


def _s2_indices(comp, suffix):
    ndvi = comp.normalizedDifference(["B8", "B4"]).rename(f"ndvi_{suffix}")
    ndre = comp.normalizedDifference(["B8", "B5"]).rename(f"ndre_{suffix}")
    ndmi = comp.normalizedDifference(["B8", "B11"]).rename(f"ndmi_{suffix}")
    cire = (comp.select("B7").divide(comp.select("B5")).subtract(1)
            .rename(f"cire_{suffix}"))
    nari = (comp.select("B3").pow(-1).subtract(comp.select("B5").pow(-1))
            .divide(comp.select("B3").pow(-1).add(comp.select("B5").pow(-1)))
            .rename(f"nari_{suffix}"))
    raw = comp.select(["B5", "B6", "B7", "B11"],
                      [f"b5_{suffix}", f"b6_{suffix}",
                       f"b7_{suffix}", f"b11_{suffix}"])
    return ndvi.addBands([ndre, ndmi, cire, nari, raw])


def build_feature_image() -> ee.Image:
    s2 = ee.ImageCollection(S2_COLL).filter(
        ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(_mask_s2)
    s2_summer = s2.filterDate(*SUMMER).median()
    s2_shoulder = (s2.filterDate(*SHOULDER_1)
                   .merge(ee.ImageCollection(S2_COLL)
                          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
                          .map(_mask_s2).filterDate(*SHOULDER_2)).median())

    f_summer = _s2_indices(s2_summer, "summer")
    f_shoulder = _s2_indices(s2_shoulder, "shoulder").select(
        ["ndvi_shoulder", "ndre_shoulder", "ndmi_shoulder"])

    d_ndvi = f_summer.select("ndvi_summer").subtract(
        f_shoulder.select("ndvi_shoulder")).rename("d_ndvi")
    d_ndre = f_summer.select("ndre_summer").subtract(
        f_shoulder.select("ndre_shoulder")).rename("d_ndre")
    d_ndmi = f_summer.select("ndmi_summer").subtract(
        f_shoulder.select("ndmi_shoulder")).rename("d_ndmi")

    s1 = (ee.ImageCollection(S1_COLL)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))
    s1_summer = s1.filterDate(*SUMMER)
    s1_shoulder = s1.filterDate(*SHOULDER_1).merge(s1.filterDate(*SHOULDER_2))

    def s1_feats(coll, suffix):
        m = coll.mean()
        vv = m.select("VV").rename(f"vv_{suffix}")
        vh = m.select("VH").rename(f"vh_{suffix}")
        ratio = m.select("VH").subtract(m.select("VV")).rename(f"ratio_{suffix}")
        return vv.addBands([vh, ratio])

    s1_sum = s1_feats(s1_summer, "summer")
    s1_sho = s1_feats(s1_shoulder, "shoulder")
    s1_std = s1.filterDate(*SUMMER).reduce(ee.Reducer.stdDev()).select(
        ["VV_stdDev", "VH_stdDev"], ["vv_std", "vh_std"])

    ch = ee.Image(CH_ASSET).rename("canopy_height").toFloat()

    img = ch.addBands([f_summer, f_shoulder, d_ndvi, d_ndre, d_ndmi,
                       s1_sum, s1_sho, s1_std])
    return img.select(FEATURES)


_NULL = -9999.0


def sample_chunk(img, chunk):
    feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"i": int(idx)})
             for idx, r in zip(chunk.index, chunk.itertuples())]
    fc = ee.FeatureCollection(feats)
    # unmask to a sentinel so a single masked band (e.g. canopy_height over
    # water, or a cloud-gapped season) does not drop the WHOLE point — we want
    # every point back and turn per-band gaps into NaN downstream.
    out = (img.unmask(_NULL).sampleRegions(collection=fc, scale=10,
                                           geometries=False, tileScale=8)
           .getInfo())
    rows = [{"i": f["properties"]["i"],
             **{k: f["properties"].get(k) for k in FEATURES}}
            for f in out["features"]]
    got = pd.DataFrame(rows).set_index("i")
    got = got.replace(_NULL, None)
    res = chunk.copy()
    for k in FEATURES:
        res[k] = got[k].reindex(chunk.index) if k in got else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    init_gee()
    pts = unique_points()
    if args.test:
        pts = pts.iloc[:200]
    print(f"unique points: {len(pts):,}  features: {len(FEATURES)}")

    done = set()
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            done = set(json.load(f)["chunks_done"])
        print(f"checkpoint: {len(done)} chunks done")

    chunks = [(ci, pts.iloc[s:s + CHUNK])
              for ci, s in enumerate(range(0, len(pts), CHUNK))
              if ci not in done]
    img = build_feature_image()

    con = duckdb.connect()
    schema = ", ".join(["lon DOUBLE", "lat DOUBLE"]
                       + [f"{c} DOUBLE" for c in FEATURES])
    con.sql(f"CREATE TABLE feat ({schema})")
    if os.path.exists(OUT_PARQUET):
        con.sql(f"INSERT INTO feat SELECT * FROM '{OUT_PARQUET}'")

    def work(ci, chunk):
        for attempt in range(MAX_RETRIES):
            try:
                return ci, sample_chunk(img, chunk)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(8 * (attempt + 1))
                print(f"  chunk {ci} retry {attempt+1}: {str(e)[:80]}")

    t0 = time.time()
    n_done = 0
    cols = ["lon", "lat"] + FEATURES
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(work, ci, c): ci for ci, c in chunks}
        for fut in as_completed(futs):
            ci, res = fut.result()
            con.register("_res", res[cols])
            con.sql("INSERT INTO feat SELECT * FROM _res")
            con.unregister("_res")
            done.add(ci)
            n_done += 1
            if n_done % 3 == 0 or n_done == len(chunks):
                con.sql(f"COPY (SELECT DISTINCT * FROM feat) TO '{OUT_PARQUET}' "
                        f"(FORMAT PARQUET)")
                with open(CHECKPOINT, "w") as f:
                    json.dump({"chunks_done": sorted(done)}, f)
                rate = n_done / (time.time() - t0)
                eta = (len(chunks) - n_done) / max(rate, 1e-9)
                print(f"  {n_done}/{len(chunks)} chunks "
                      f"({rate:.2f}/s, eta {eta:.0f}s)")

    con.sql(f"COPY (SELECT DISTINCT * FROM feat) TO '{OUT_PARQUET}' "
            f"(FORMAT PARQUET)")
    with open(CHECKPOINT, "w") as f:
        json.dump({"chunks_done": sorted(done)}, f)
    n = con.sql(f"SELECT count(*), count(canopy_height), count(ndre_summer) "
                f"FROM '{OUT_PARQUET}'").fetchone()
    print(f"saved {OUT_PARQUET}: {n[0]:,} rows, "
          f"canopy_height={n[1]:,}, ndre_summer={n[2]:,}")


if __name__ == "__main__":
    main()

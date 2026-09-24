"""Draw a uniform random pixel sample of the 2024 map for area estimation.

The FSCS training points cannot support an area estimate: they are ~100
k-means points per (25 km cell x grunnkart class), CCDC-stable pixels only, so
the sample is stratified on the very label whose area we want, with unknown
within-stratum inclusion probabilities. Area estimators (classical, PPI, cross-
PPI, stratified) all need a PROBABILITY sample. This script builds one.

One streaming pass over the map (row strips, parallel):
  * exact wall-to-wall hard class counts of classified_2024.tif, and the sum of
    the calibrated probabilities uq_2024_pcal.tif per class — the "unlabelled"
    side of PPI with N = every valid pixel;
  * a Poisson sample (p = M / n_valid) of valid pixels, carrying map class,
    9 pcal probabilities, 64 AEF bands (raw int8) and 3 lidar bands.

Then the grunnkart reference class (the same GEE asset the training labels came
from) is sampled at each point. Output:
  data/area_sample_2024.parquet      one row per sampled pixel
  reports/results/area_map_totals_2024.json

Run (server):
    ~/myprojects/recover/.venv/bin/python DNN/area_sample.py --m 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_CACHEMAX", "512")

import rasterio                                   # noqa: E402
from rasterio.windows import Window               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dnn_paths import REPO_DIR, result_path       # noqa: E402

ROOT = Path(os.environ.get("NYVEST_DATA_DIR", "/data/P-Prosjekter2/154001_nyvest"))
LC = ROOT / "landcover_Geethen" / "landcover_2018_2024"
AEF = ROOT / "landcover_Geethen" / "aef_{year}.vrt"
LIDAR = ROOT / "lidar_3band.tif"
GRUNNKART_ASSET = "projects/ee-gsingh/assets/grunnkart_nyvest_10m"
PCAL_SCALE = 60000.0
MAP_CLASSES = [2, 3, 4, 5, 6, 7, 8, 10, 12]      # deployed 9-class ontology
STRIP = 512


def _strip(args):
    year, row0, h, p, seed = args
    rng = np.random.default_rng([seed, row0])
    with rasterio.open(LC / f"classified_{year}.tif") as s:
        cls = s.read(1, window=Window(0, row0, s.width, h))
    valid = cls != 0
    counts = np.bincount(cls[valid].ravel(), minlength=13)[:13]
    with rasterio.open(LC / f"uq_{year}_pcal.tif") as s:
        pc = s.read(window=Window(0, row0, s.width, h))
    psum = np.array([pc[j][valid].astype(np.float64).sum() for j in range(pc.shape[0])]) / PCAL_SCALE
    rr, cc = np.nonzero(valid & (rng.random(cls.shape) < p))
    out = {"row": rr + row0, "col": cc, "map_class": cls[rr, cc].astype(np.int16)}
    for j, c in enumerate(MAP_CLASSES):
        out[f"pcal_{c}"] = pc[j, rr, cc].astype(np.float32) / PCAL_SCALE
    del pc
    if len(rr):
        with rasterio.open(str(AEF).format(year=year)) as s:
            ae = s.read(window=Window(0, row0, s.width, h))
        for b in range(ae.shape[0]):
            out[f"A{b:02d}"] = ae[b, rr, cc]
        del ae
        with rasterio.open(LIDAR) as s:
            lid = s.read(window=Window(0, row0, s.width, h))
        for j, n in enumerate(["elevation", "tri", "tch"]):
            out[n] = lid[j, rr, cc].astype(np.float32)
    return counts, psum, pd.DataFrame(out)


def sample_map(year, m, seed, workers):
    with rasterio.open(LC / f"classified_{year}.tif") as s:
        H, W, transform, crs = s.height, s.width, s.transform, s.crs
    n_valid = 817_880_567                          # change_2018_2024.json valid_px
    p = m / n_valid
    jobs = [(year, r, min(STRIP, H - r), p, seed) for r in range(0, H, STRIP)]
    counts = np.zeros(13, np.int64)
    psum = np.zeros(len(MAP_CLASSES))
    frames = []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, (c, ps, df) in enumerate(ex.map(_strip, jobs)):
            counts += c
            psum += ps
            frames.append(df)
            if i % 10 == 0:
                print(f"  strip {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    xs, ys = rasterio.transform.xy(transform, df["row"].values, df["col"].values)
    df["x"], df["y"] = np.asarray(xs), np.asarray(ys)
    return df, counts, psum, str(crs)


def attach_grunnkart(df, crs, chunk=4000, workers=8):
    import ee
    from pyproj import Transformer
    ee.Initialize(project="ee-gsingh",
                  opt_url="https://earthengine-highvolume.googleapis.com")
    img = ee.Image(GRUNNKART_ASSET).rename("gk").unmask(0)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        df["x"].values, df["y"].values)

    def go(i0):
        idx = range(i0, min(i0 + chunk, len(df)))
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lon[i], lat[i]]), {"i": i})
                                   for i in idx])
        for attempt in range(6):
            try:
                r = ee.data.computeFeatures({
                    "expression": img.sampleRegions(collection=fc, scale=10,
                                                    geometries=False),
                    "fileFormat": "PANDAS_DATAFRAME"})
                return r[["i", "gk"]]
            except Exception as e:                         # rate limits
                time.sleep(10 * 2 ** attempt)
                last = e
        raise last

    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(go, range(0, len(df), chunk)))
    gk = pd.concat(parts).set_index("i")["gk"]
    df["grunnkart"] = gk.reindex(range(len(df))).fillna(0).astype(np.int16).values
    df["lon"], df["lat"] = lon, lat
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--m", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gk-only", action="store_true",
                    help="resume: reuse the map-pass checkpoint, only fetch grunnkart")
    a = ap.parse_args()

    out = REPO_DIR / "data" / f"area_sample_{a.year}.parquet"
    ckpt = out.with_suffix(".mapside.parquet")
    if a.gk_only:
        df = pd.read_parquet(ckpt)
        with rasterio.open(LC / f"classified_{a.year}.tif") as s:
            crs = str(s.crs)
    else:
        df, counts, psum, crs = sample_map(a.year, a.m, a.seed, a.workers)
        n_valid = int(counts.sum())
        totals = {"year": a.year, "n_valid_px": n_valid, "pixel_area_m2": 100.0,
                  "hard_counts": {str(c): int(counts[c]) for c in MAP_CLASSES},
                  "pcal_sums": {str(c): float(v) for c, v in zip(MAP_CLASSES, psum)},
                  "sample_m": len(df), "seed": a.seed}
        result_path(f"area_map_totals_{a.year}.json").write_text(json.dumps(totals, indent=2))
        df.to_parquet(ckpt, index=False)
        print(f"valid px {n_valid:,}; sampled {len(df):,}", flush=True)

    df = attach_grunnkart(df, crs)
    df.to_parquet(out, index=False)
    print(f"-> {out}\n{df['grunnkart'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()

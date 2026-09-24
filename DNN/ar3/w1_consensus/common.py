"""Shared constants + helpers for the W1 consensus-labels pipeline.

Copies GEE conventions EXACTLY from
scripts/extraction/sample_feature_space_stable_allyears.py: project
`ee-gsingh`, high-volume endpoint, AEF collection built per year with
`build_alphaearth_year`, the 25 km grid (`build_grid`, cell_id = j*nx+i,
EPSG:25832 bbox COUNTIES_BBOX_25832), and the 429 retry/backoff pattern.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

# ── paths ────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CONSENSUS_DIR = os.path.join(DATA_DIR, "consensus")
W1_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(W1_DIR, "logs")
os.makedirs(CONSENSUS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TRAIN_PARQUET = os.path.join(
    DATA_DIR, "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet")
LIDAR_PARQUET = os.path.join(DATA_DIR, "lidar_features.parquet")

LANDCOVER_DIR = ("/data/P-Prosjekter2/154001_nyvest/landcover_Geethen/"
                  "landcover_2018_2024")
CLASSIFIED_TIF = {2018: os.path.join(LANDCOVER_DIR, "classified_2018.tif"),
                  2024: os.path.join(LANDCOVER_DIR, "classified_2024.tif")}
SETSIZE_TIF = {2018: os.path.join(LANDCOVER_DIR, "uq_2018_setsize.tif"),
               2024: os.path.join(LANDCOVER_DIR, "uq_2024_setsize.tif")}

GRUNNKART_DIR = ("/data/P-Prosjekter2/154001_nyvest/GIS/NIBIO/Version_2/"
                  "rasterized_10m")
GRUNNKART_V1_TIF = os.path.join(GRUNNKART_DIR, "grunnkart_nyvest_10m.tif")
GRUNNKART_V2_TIF = os.path.join(GRUNNKART_DIR, "grunnkart_nyvest_10m_v2.tif")
GRUNNKART_ASSET = "projects/ee-gsingh/assets/grunnkart_nyvest_10m"

# ── GEE conventions (mirrors sample_feature_space_stable_allyears.py) ──
PROJECT = "ee-gsingh"
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
SCALE = 10
GRID_SIZE_M = 25_000
COUNTIES_BBOX_25832 = (234_740.0, 6_435_354.0, 518_617.0, 7_071_896.0)
COUNTIES_CRS = "EPSG:25832"
YEARS = tuple(range(2017, 2026))  # 2017..2025
EXPECTED_BANDS = [f"A{i:02d}" for i in range(64)]

ESRI_ASSET = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
WORLDCOVER_V100 = "ESA/WorldCover/v100"  # 2020
WORLDCOVER_V200 = "ESA/WorldCover/v200"  # 2021
DYNAMICWORLD = "GOOGLE/DYNAMICWORLD/V1"

# merged-class codes used by the deployed model (see brief context)
MERGED_CLASSES = (2, 3, 4, 5, 6, 7, 8, 10, 12)

# ── 25 km grid: cell_id, pure python (matches build_grid's j*nx+i) ────
_NX = int(math.ceil(
    (COUNTIES_BBOX_25832[2] - COUNTIES_BBOX_25832[0]) / GRID_SIZE_M))
_NY = int(math.ceil(
    (COUNTIES_BBOX_25832[3] - COUNTIES_BBOX_25832[1]) / GRID_SIZE_M))


def grid_dims():
    return _NX, _NY


def cell_id_from_xy_25832(x, y):
    """Vectorised cell_id = j*nx + i, matching build_grid exactly.

    x, y: array-like in EPSG:25832. Points outside the bbox get cell_id -1.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xmin, ymin, xmax, ymax = COUNTIES_BBOX_25832
    i = np.floor((x - xmin) / GRID_SIZE_M).astype(np.int64)
    j = np.floor((y - ymin) / GRID_SIZE_M).astype(np.int64)
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) \
        & (i >= 0) & (i < _NX) & (j >= 0) & (j < _NY)
    cid = j * _NX + i
    cid = np.where(inside, cid, -1)
    return cid


_transformer_4326_to_25832 = None


def cell_id_from_lonlat(lon, lat):
    global _transformer_4326_to_25832
    if _transformer_4326_to_25832 is None:
        from pyproj import Transformer
        _transformer_4326_to_25832 = Transformer.from_crs(
            "EPSG:4326", "EPSG:25832", always_xy=True)
    x, y = _transformer_4326_to_25832.transform(
        np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    return cell_id_from_xy_25832(x, y)


# ── GEE init + retry/backoff (mirrors the reference script) ───────────
def init_gee(project=PROJECT):
    import ee
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
        print(f"[OK] GEE initialised with high-volume endpoint "
              f"(project={project})")
    except Exception as e:
        print(f"  High-volume init failed ({e}); falling back to standard.")
        ee.Initialize(project=project)
        print(f"[OK] GEE initialised (project={project})")


def build_alphaearth_year(year):
    import ee
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(start, end)
        .reduce(ee.Reducer.first())
        .regexpRename("_first$", "")
    )


_RATE_LIMIT_MARKERS = (
    "Too Many Requests", "429", "rate limit", "concurrency limit",
    "user memory limit", "computation timed out",
)
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_BACKOFF_BASE = 15  # seconds; doubles each retry


def is_rate_limit(exc):
    return any(m in str(exc) for m in _RATE_LIMIT_MARKERS)


def compute_features_with_retry(fc, label="", timeout=None):
    """ee.data.computeFeatures with the reference script's 429 backoff."""
    import ee
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return ee.data.computeFeatures({
                "expression": fc,
                "fileFormat": "PANDAS_DATAFRAME",
            })
        except Exception as e:
            if is_rate_limit(e) and attempt < RATE_LIMIT_RETRIES:
                wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                print(f"    [rate-limit] {label} attempt {attempt + 1}/"
                      f"{RATE_LIMIT_RETRIES} — sleeping {wait}s")
                time.sleep(wait)
                continue
            raise

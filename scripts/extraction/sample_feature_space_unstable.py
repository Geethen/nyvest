"""FSCS sampling on *unstable* (CCDC-changed) pixels, across all AlphaEarth years.

Counterpart to `sample_feature_space.py`. Differences:
  1. Inverts the CCDC stability mask — keeps pixels with at least one break
     above changeProb≥0.99 in (CCDC_START, CCDC_END]. Default window matches
     the stable-version: 2017–2020.
  2. Builds a stacked AlphaEarth image with 64 bands per year for every year
     in YEARS (default 2017..2025 → 9 × 64 = 576 bands), so one
     `computeFeatures` call materialises the full time-series for each
     sampled point.
  3. In Python, melts the wide df to long format: one row per (point, year)
     with columns A00..A63 + year + class + cell_id + cluster + lon + lat.

Pipeline per (cell, class) shard:
  embed_stack -> mask(class & UNSTABLE) -> wekaKMeans(k=100) on a sampling
  year's bands -> stratifiedSample 1/cluster, carrying ALL stacked bands ->
  melt to long form -> insert into DuckDB buffer.

The kMeans is trained on a single year's 64-d embeddings (the SAMPLING_YEAR
constant, default 2020) so the coverage logic is comparable with the
stable-version output.

Usage:
  python scripts/extraction/sample_feature_space_unstable.py
  python scripts/extraction/sample_feature_space_unstable.py --test_mode
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout, as_completed
from threading import Lock

import duckdb
import ee
import pandas as pd
from tqdm.auto import tqdm

# ── Constants (mirror sample_feature_space.py so behaviour is comparable) ──
PROJECT = "ee-gsingh"
GRUNNKART_ASSET = "projects/ee-gsingh/assets/grunnkart_nyvest_10m"
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
CCDC_COLLECTION = "GOOGLE/GLOBAL_CCDC/V1"

YEARS = tuple(range(2017, 2026))     # 2017..2025 inclusive (9 years)
SAMPLING_YEAR = 2020                 # train kMeans on this year's 64 bands
SCALE = 10
GRID_SIZE_M = 25_000
K_PER_CLASS = 100
CLASSES = tuple(range(1, 13))        # skip 0 (nodata) and 13 (other)
CCDC_START = 2017                    # window start (exclusive)
CCDC_END = 2020                      # window end (inclusive)
CCDC_PROB = 0.99
SEED = 42

MAX_WORKERS = 10
TILE_SCALE = 4
TILE_SCALES = (4, 8, 16)
SHARD_TIMEOUT_S = 240                # heavier per-shard payload than stable (9× bands)
FLUSH_EVERY_N_SHARDS = 25
KMEANS_NUM_PIXELS = 5000

COUNTIES_BBOX_25832 = (234_740.0, 6_435_354.0, 518_617.0, 7_071_896.0)
COUNTIES_CRS = "EPSG:25832"

EXPECTED_BANDS = [f"A{i:02d}" for i in range(64)]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def init_gee(project=PROJECT):
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
        print(f"[OK] GEE initialised with high-volume endpoint "
              f"(project={project})")
    except Exception as e:
        print(f"  High-volume init failed ({e}); falling back to standard.")
        ee.Initialize(project=project)
        print(f"[OK] GEE initialised (project={project})")


def retry_gee(func, max_retries=3, backoff=2):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff ** (attempt + 1)
            tqdm.write(f"    retry {attempt + 1}/{max_retries} "
                       f"after {wait}s: {str(e)[:120]}")
            time.sleep(wait)


def build_alphaearth_year(year):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(start, end)
        .reduce(ee.Reducer.first())
        .regexpRename("_first$", "")
    )


def build_alphaearth_stack(years):
    """Stack AlphaEarth (64 bands × N years) into one image with renamed bands.

    Band names: A00_2017..A63_2017, A00_2018..A63_2018, ...
    """
    images = []
    for y in years:
        img = build_alphaearth_year(y)
        renamed = img.rename([f"{b}_{y}" for b in EXPECTED_BANDS])
        images.append(renamed)
    out = images[0]
    for img in images[1:]:
        out = out.addBands(img)
    return out


def build_unstable_mask(region, start_year=CCDC_START, end_year=CCDC_END,
                        change_prob=CCDC_PROB):
    """1 where CCDC reports ≥1 break above `change_prob` in (start, end]."""
    ccdc = ee.Image(
        ee.ImageCollection(CCDC_COLLECTION).filterBounds(region).mosaic())
    t_break = ccdc.select("tBreak")
    change_prob_img = ccdc.select("changeProb")
    breaks = (t_break.gt(start_year)
              .And(t_break.lte(end_year))
              .And(change_prob_img.gte(change_prob)))
    n_breaks = breaks.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])
    return n_breaks.gt(0).rename("unstable").unmask(0)


def build_grid(bbox=COUNTIES_BBOX_25832, crs=COUNTIES_CRS,
               cell_size_m=GRID_SIZE_M):
    xmin, ymin, xmax, ymax = bbox
    nx = int(math.ceil((xmax - xmin) / cell_size_m))
    ny = int(math.ceil((ymax - ymin) / cell_size_m))
    proj = ee.Projection(crs)
    cells = []
    for j in range(ny):
        for i in range(nx):
            x0 = xmin + i * cell_size_m
            y0 = ymin + j * cell_size_m
            x1 = min(x0 + cell_size_m, xmax)
            y1 = min(y0 + cell_size_m, ymax)
            cell_id = j * nx + i
            geom = ee.Geometry.Rectangle(
                [x0, y0, x1, y1], proj=proj, evenOdd=True, geodesic=False)
            cells.append((cell_id, geom))
    return cells


def sample_cell_class_points(sampling_year_bands, grunnkart, unstable_mask,
                             cell_geom, cell_id, class_code, k,
                             tile_scale=TILE_SCALE):
    """Stage 1: pick k coverage-sample point geometries for one shard.

    Trains wekaKMeans on the SAMPLING_YEAR bands and returns a small
    FeatureCollection with one point per cluster carrying only cluster id,
    cell_id, class. Payload is tiny (k rows × ~3 cols).
    """
    class_mask = grunnkart.eq(class_code)
    region_mask = class_mask.And(unstable_mask)

    sampling_masked = sampling_year_bands.updateMask(region_mask).clip(cell_geom)
    training = sampling_masked.sample(
        region=cell_geom,
        scale=SCALE,
        numPixels=KMEANS_NUM_PIXELS,
        seed=SEED,
        tileScale=tile_scale,
        dropNulls=True,
    )
    clusterer = ee.Clusterer.wekaKMeans(nClusters=k, seed=SEED).train(training)
    clustered = sampling_masked.cluster(clusterer)

    pts = clustered.stratifiedSample(
        numPoints=1,
        classBand="cluster",
        region=cell_geom,
        scale=SCALE,
        seed=SEED,
        tileScale=tile_scale,
        dropNulls=True,
        geometries=True,
    )
    pts = pts.map(lambda f: f.set({
        "cell_id": cell_id,
        "class": class_code,
    }))
    return pts


def sample_bands_at_points(points_fc, year_image, year, tile_scale=TILE_SCALE):
    """Stage 2: extract `year_image` (64 bands) at the k points.

    Returns a small df: k rows × (64 bands + cluster + cell_id + class
    + lon + lat + year). `sampleRegions` keeps the original geometry +
    properties when `tileScale` is small and the FC is small.
    """
    sampled = year_image.sampleRegions(
        collection=points_fc,
        scale=SCALE,
        tileScale=tile_scale,
        geometries=False,
    )
    sampled = sampled.map(lambda f: f.set({"year": year}))
    return sampled


_EMPTY_SHARD_MARKERS = (
    "No data was found in training input",
    "No valid training data",
)

_solo_executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS * 4,
    thread_name_prefix="fscs-unstable-solo")


def _compute_with_timeout(fc, label):
    """Run ee.data.computeFeatures with the shard-level wall-clock timeout."""
    def _go():
        return ee.data.computeFeatures({
            "expression": fc,
            "fileFormat": "PANDAS_DATAFRAME",
        })
    fut = _solo_executor.submit(_go)
    try:
        return fut.result(timeout=SHARD_TIMEOUT_S)
    except FutTimeout:
        fut.cancel()
        raise TimeoutError(
            f"computeFeatures({label}) exceeded {SHARD_TIMEOUT_S}s")
    except Exception as e:
        if any(m in str(e) for m in _EMPTY_SHARD_MARKERS):
            return None  # empty signal
        raise


def process_shard(cell_id, cell_geom, class_code, year_images,
                  sampling_year_bands, grunnkart, unstable_mask, k,
                  years, db_conn, lock, tile_scale=TILE_SCALE,
                  schema_checked=None):
    """Two-stage extraction to keep per-call payloads small.

    Stage 1: pick k points + cluster ids (small FC, lightweight call).
    Stage 2: per year, sampleRegions to get 64 bands at those points.
    Concatenate per-year dfs into long form and insert.
    """
    pts_fc = sample_cell_class_points(
        sampling_year_bands, grunnkart, unstable_mask,
        cell_geom, cell_id, class_code, k, tile_scale=tile_scale)
    # Attach lon/lat as properties so the per-year df has them.
    pts_fc = pts_fc.map(lambda f: f.set({
        "lon": f.geometry().coordinates().get(0),
        "lat": f.geometry().coordinates().get(1),
    }))

    frames = []
    for y in years:
        sampled = sample_bands_at_points(
            pts_fc, year_images[y], y, tile_scale=tile_scale)
        df_y = _compute_with_timeout(sampled, label=f"y{y}")
        if df_y is None:  # empty
            return 0
        if df_y is None or df_y.empty:
            continue
        frames.append(df_y)

    if not frames:
        return 0

    df_long = pd.concat(frames, ignore_index=True)

    if schema_checked is not None and not schema_checked[0]:
        missing = [b for b in EXPECTED_BANDS if b not in df_long.columns]
        if missing:
            raise RuntimeError(
                f"Long df missing AlphaEarth bands "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''} "
                f"(got {len(df_long.columns)} cols).")
        schema_checked[0] = True

    with lock:
        try:
            db_conn.execute("INSERT INTO data BY NAME SELECT * FROM df_long")
        except duckdb.CatalogException:
            db_conn.execute("CREATE TABLE data AS SELECT * FROM df_long")

    return len(df_long)


def process_shard_with_escalation(cell_id, cell_geom, class_code, year_images,
                                  sampling_year_bands, grunnkart, unstable_mask,
                                  k, years, db_conn, lock, schema_checked,
                                  tile_scales=None, backoff=2):
    if tile_scales is None:
        tile_scales = TILE_SCALES
    last_exc = None
    for attempt, ts in enumerate(tile_scales):
        try:
            return process_shard(
                cell_id, cell_geom, class_code, year_images,
                sampling_year_bands, grunnkart, unstable_mask, k,
                years, db_conn, lock, tile_scale=ts,
                schema_checked=schema_checked)
        except Exception as e:
            last_exc = e
            if attempt == len(tile_scales) - 1:
                raise
            wait = backoff ** (attempt + 1)
            tqdm.write(
                f"    cell {cell_id} class {class_code} retry "
                f"{attempt + 1}/{len(tile_scales) - 1} "
                f"(tileScale {ts}->{tile_scales[attempt + 1]}) "
                f"after {wait}s: {str(e)[:120]}")
            time.sleep(wait)
    raise last_exc


def shard_key(cell_id, class_code):
    return f"{cell_id}:{class_code}"


def run(years, grid_size_m, k_per_class, max_workers, output_path,
        grunnkart_asset, ccdc_start, ccdc_end, ccdc_prob,
        sampling_year=SAMPLING_YEAR, test_mode=False):
    checkpoint_file = output_path + ".checkpoint.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    grunnkart = ee.Image(grunnkart_asset)
    year_images = {y: build_alphaearth_year(y) for y in years}
    sampling_year_bands = year_images[sampling_year]
    region = ee.Geometry.Rectangle(
        list(COUNTIES_BBOX_25832),
        proj=ee.Projection(COUNTIES_CRS), evenOdd=True, geodesic=False)
    unstable_mask = build_unstable_mask(region, ccdc_start, ccdc_end, ccdc_prob)
    print(f"  AlphaEarth years stacked: {list(years)} "
          f"({len(years)} × 64 = {len(years) * 64} bands)")
    print(f"  Sampling year (kMeans training): {sampling_year}")
    print(f"  CCDC UNSTABLE window: ({ccdc_start}, {ccdc_end}] "
          f"@ changeProb≥{ccdc_prob} (inverted mask)")
    print(f"  Grunnkart asset: {grunnkart_asset}")

    cells = build_grid(cell_size_m=grid_size_m)
    print(f"  Grid: {len(cells)} cells of {grid_size_m/1000:.0f} km "
          f"over counties bbox")

    classes = list(CLASSES)
    if test_mode:
        # Pick cells that actually have CCDC-flagged change (avoid the SW
        # stable-forest cluster where the stable-version smoke test ran).
        # Cells ~250 are mid-grid east, more change.
        cells = [c for c in cells if c[0] in (250, 251)]
        classes = [4, 10]   # forest (often disturbed) + built (often changed)
        print(f"  *** TEST MODE: cells {[c[0] for c in cells]} × "
              f"classes {classes} ***")

    shards = [(cid, cgeom, cls) for cid, cgeom in cells for cls in classes]
    total = len(shards)
    expected_pts = total * k_per_class * len(years)
    print(f"  Shards: {total} (cell × class)   "
          f"target ≤ {expected_pts:,} rows (k={k_per_class}/shard "
          f"× {len(years)} years)")

    processed = set()
    prior_failed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            cp = json.load(f)
        processed = set(cp.get("done", []))
        prior_failed = set(cp.get("failed", []))
        print(f"  Resuming: {len(processed)}/{total} done, "
              f"{len(prior_failed)} previously failed (will retry)")

    db_conn = duckdb.connect()
    if os.path.exists(output_path):
        print(f"  Loading existing parquet into buffer "
              f"({os.path.basename(output_path)})...")
        db_conn.execute(f"CREATE TABLE data AS SELECT * FROM '{output_path}'")
        existing = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
        print(f"  [OK] Loaded {existing:,} existing rows")

    lock = Lock()
    failed = set()
    schema_checked = [False]

    def save_checkpoint():
        with lock, open(checkpoint_file, "w") as f:
            json.dump({"done": sorted(processed),
                       "failed": sorted(failed)}, f)

    def flush_to_parquet():
        try:
            n = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
        except duckdb.CatalogException:
            return 0
        if n == 0:
            return 0
        tmp = output_path + ".tmp"
        with lock:
            db_conn.execute(
                f"COPY data TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(tmp, output_path)
        return n

    todo = [s for s in shards if shard_key(s[0], s[2]) not in processed]

    successful = 0
    total_rows = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_key = {}
        for cid, cgeom, cls in todo:
            fut = pool.submit(
                process_shard_with_escalation,
                cid, cgeom, cls, year_images, sampling_year_bands,
                grunnkart, unstable_mask, k_per_class, years,
                db_conn, lock, schema_checked,
            )
            future_to_key[fut] = (cid, cls)

        with tqdm(total=len(todo), desc="FSCS-unstable shards", ncols=100) as pbar:
            since_flush = 0
            for future in as_completed(future_to_key):
                cid, cls = future_to_key[future]
                key = shard_key(cid, cls)
                try:
                    n = future.result()
                    total_rows += (n or 0)
                    successful += 1
                    processed.add(key)
                    failed.discard(key)
                    save_checkpoint()
                    if n:
                        tqdm.write(
                            f"    cell {cid} class {cls}: {n} rows")
                    since_flush += 1
                    if since_flush >= FLUSH_EVERY_N_SHARDS:
                        rows = flush_to_parquet()
                        if rows:
                            tqdm.write(f"    [flush] {rows:,} rows -> "
                                       f"{os.path.basename(output_path)}")
                        since_flush = 0
                except Exception as e:
                    failed.add(key)
                    save_checkpoint()
                    tqdm.write(f"  [ERROR] cell {cid} class {cls}: "
                               f"{str(e)[:200]}")
                pbar.update(1)

    if failed:
        print(f"\n  {len(failed)} shards failed after retries: "
              f"{sorted(failed)[:10]}{'...' if len(failed) > 10 else ''}")
        print(f"  Re-run the same command to retry only the failed shards "
              f"(checkpoint: {os.path.basename(checkpoint_file)})")

    try:
        buf_rows = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
    except Exception:
        buf_rows = 0

    if buf_rows == 0:
        print("  [WARN] No data extracted")
        db_conn.close()
        return

    tmp = output_path + ".tmp"
    db_conn.execute(
        f"COPY data TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    db_conn.close()

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp, output_path)

    file_mb = os.path.getsize(output_path) / 1e6
    print(f"\n[OK] Saved {output_path}")
    print(f"  Rows: {buf_rows:,}   Size: {file_mb:.1f} MB")
    print(f"  Shards: {successful} ok, {len(failed)} failed, "
          f"{len(processed)} total processed")

    if not failed and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def main():
    global SHARD_TIMEOUT_S, TILE_SCALES, KMEANS_NUM_PIXELS, _solo_executor
    default_output = os.path.join(
        DATA_DIR, "grunnkart_nyvest_fscs_unstable_alphaearth.parquet")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--years", default=",".join(str(y) for y in YEARS),
                        help=f"Comma-separated AlphaEarth years to stack "
                             f"(default {YEARS[0]}..{YEARS[-1]})")
    parser.add_argument("--sampling_year", type=int, default=SAMPLING_YEAR,
                        help=f"Year whose bands train the kMeans "
                             f"(default {SAMPLING_YEAR})")
    parser.add_argument("--grunnkart", default=GRUNNKART_ASSET)
    parser.add_argument("--grid_size_m", type=int, default=GRID_SIZE_M)
    parser.add_argument("--k", type=int, default=K_PER_CLASS)
    parser.add_argument("--ccdc_start", type=int, default=CCDC_START)
    parser.add_argument("--ccdc_end", type=int, default=CCDC_END)
    parser.add_argument("--ccdc_prob", type=float, default=CCDC_PROB)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--shard_timeout", type=int, default=SHARD_TIMEOUT_S)
    parser.add_argument("--tile_scales", default=None,
                        help="Comma-separated tileScale ladder "
                             "(default '4,8,16').")
    parser.add_argument("--num_pixels", type=int, default=KMEANS_NUM_PIXELS,
                        help=f"Pixels for kMeans training (default {KMEANS_NUM_PIXELS})")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    SHARD_TIMEOUT_S = args.shard_timeout
    KMEANS_NUM_PIXELS = args.num_pixels
    if args.tile_scales:
        TILE_SCALES = tuple(int(x) for x in args.tile_scales.split(","))
    _solo_executor = ThreadPoolExecutor(
        max_workers=max(MAX_WORKERS, args.max_workers) * 4,
        thread_name_prefix="fscs-unstable-solo")

    years = tuple(int(y) for y in args.years.split(","))
    if args.sampling_year not in years:
        raise SystemExit(
            f"--sampling_year={args.sampling_year} must be in --years={years}")

    init_gee(args.project)
    run(years=years,
        grid_size_m=args.grid_size_m,
        k_per_class=args.k,
        max_workers=args.max_workers,
        output_path=args.output,
        grunnkart_asset=args.grunnkart,
        ccdc_start=args.ccdc_start,
        ccdc_end=args.ccdc_end,
        ccdc_prob=args.ccdc_prob,
        sampling_year=args.sampling_year,
        test_mode=args.test_mode)


if __name__ == "__main__":
    main()

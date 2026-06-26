"""Stratified feature-space coverage sampling of AlphaEarth embeddings.

For each cell of a regular spatial grid covering the three NYVEST counties
(Rogaland + Vestland + Møre og Romsdal), and for each grunnkart class
(1..12, skipping 0/nodata and 13/other), we:

  1. Mask the AlphaEarth (64 bands) image to that class within the cell,
     intersected with a CCDC stability mask (no breaks above changeProb≥0.99
     in the configured window, default 2017–2020).
  2. Train an `ee.Clusterer.wekaKMeans(k=100)` on the masked image so the
     cluster IDs cover the 64-d feature space of that class within that cell.
  3. Sample one pixel per cluster (`stratifiedSample` over the cluster image,
     classBand='cluster', numPoints=1).
  4. Collect AlphaEarth bands A00..A63 + (cell_id, class, cluster_id) and
     write to parquet via DuckDB. Each (cell, class) pair is one shard;
     checkpoint per shard makes the run resumable.

Inputs (Earth Engine assets):
  --grunnkart  default: projects/ee-gsingh/assets/grunnkart_nyvest_10m
  AlphaEarth Satellite Embedding V1 ANNUAL (built-in collection)
  CCDC         GOOGLE/GLOBAL_CCDC/V1

Usage:
  python scripts/extraction/sample_feature_space.py
  python scripts/extraction/sample_feature_space.py --year 2020 --grid_size_m 25000
  python scripts/extraction/sample_feature_space.py --test_mode

Structure mirrors `extract_alphaearth_embeddings.py` from the DegreeofRecovery
repo (init_gee, retry helper, ThreadPoolExecutor shard loop, DuckDB buffer,
JSON checkpoint).
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
from tqdm.auto import tqdm


# ── Constants ───────────────────────────────────────────────────────
PROJECT = "ee-gsingh"
GRUNNKART_ASSET = "projects/ee-gsingh/assets/grunnkart_nyvest_10m"
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
CCDC_COLLECTION = "GOOGLE/GLOBAL_CCDC/V1"

YEAR = 2020
SCALE = 10                           # AlphaEarth native resolution (m)
GRID_SIZE_M = 25_000                 # 25 km × 25 km cells
K_PER_CLASS = 100                    # clusters per (cell, class)
CLASSES = tuple(range(1, 13))        # skip 0 (nodata) and 13 (other)
CCDC_START = 2017                    # stability window start (exclusive)
CCDC_END = 2020                      # stability window end (inclusive)
CCDC_PROB = 0.99                     # changeProb threshold
SEED = 42

MAX_WORKERS = 10                     # kMeans+stratifiedSample is heavier than reduceRegions
TILE_SCALE = 4                       # bump default; clusterer + sampling is memory-hungry
TILE_SCALES = (4, 8, 16)             # escalation ladder for retries
SHARD_TIMEOUT_S = 180                # abandon a shard if computeFeatures hangs past this
FLUSH_EVERY_N_SHARDS = 200           # periodic parquet flush so a hang doesn't lose the buffer
KMEANS_NUM_PIXELS = 5000             # pixels sampled to train kMeans per shard

# Bounding box of Rogaland + Vestland + Møre og Romsdal in EPSG:25832,
# matching the extent of the grunnkart_nyvest_10m raster. Cells whose
# masked area is empty (fjord/sea pixels, missed counties) produce
# empty shards and are skipped quietly.
COUNTIES_BBOX_25832 = (234_740.0, 6_435_354.0, 518_617.0, 7_071_896.0)
COUNTIES_CRS = "EPSG:25832"

EXPECTED_BANDS = [f"A{i:02d}" for i in range(64)]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── GEE helpers ─────────────────────────────────────────────────────
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
    """Run `func()` with exponential-backoff retry on transient GEE errors."""
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


# ── Image stack ─────────────────────────────────────────────────────
def build_alphaearth_image(year):
    """Annual AlphaEarth image with bands A00..A63."""
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(start, end)
        .reduce(ee.Reducer.first())
        .regexpRename("_first$", "")
    )


def build_stable_mask(region, start_year=CCDC_START, end_year=CCDC_END,
                      change_prob=CCDC_PROB):
    """1 where CCDC reports no breaks above `change_prob` in (start, end]."""
    ccdc = ee.Image(
        ee.ImageCollection(CCDC_COLLECTION).filterBounds(region).mosaic())
    t_break = ccdc.select("tBreak")
    change_prob_img = ccdc.select("changeProb")
    breaks = (t_break.gt(start_year)
              .And(t_break.lte(end_year))
              .And(change_prob_img.gte(change_prob)))
    n_breaks = breaks.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])
    return n_breaks.eq(0).rename("stable").unmask(0)


# ── Grid construction ──────────────────────────────────────────────
def build_grid(bbox=COUNTIES_BBOX_25832, crs=COUNTIES_CRS,
               cell_size_m=GRID_SIZE_M):
    """Return a list of (cell_id, ee.Geometry) tiling the bbox."""
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


# ── Sampling ───────────────────────────────────────────────────────
def sample_cell_class(image, grunnkart, stable_mask, cell_geom,
                      cell_id, class_code, k, year, tile_scale=TILE_SCALE):
    """Coverage-sample `k` pixels for one (cell, class) pair.

    Pipeline:
      embed -> mask(class & stable) -> wekaKMeans(k) -> stratifiedSample(1/cluster)
    """
    class_mask = grunnkart.eq(class_code)
    region_mask = class_mask.And(stable_mask)
    masked = image.updateMask(region_mask).clip(cell_geom)

    # Train kMeans on a sample of the masked image inside the cell. Using a
    # `region` here means GEE samples training pixels for us; `k=100` clusters
    # produces ~k coverage points after stratifiedSample.
    training = masked.sample(
        region=cell_geom,
        scale=SCALE,
        numPixels=KMEANS_NUM_PIXELS,
        seed=SEED,
        tileScale=tile_scale,
        dropNulls=True,
    )

    clusterer = ee.Clusterer.wekaKMeans(nClusters=k, seed=SEED).train(training)
    clustered = masked.cluster(clusterer)              # band: 'cluster'

    # Stack cluster id + embedding so the sample carries the band values.
    stacked = clustered.addBands(masked)

    sample = stacked.stratifiedSample(
        numPoints=1,
        classBand="cluster",
        region=cell_geom,
        scale=SCALE,
        seed=SEED,
        tileScale=tile_scale,
        dropNulls=True,
        geometries=True,
    )

    sample = sample.map(lambda f: f.set({
        "cell_id": cell_id,
        "class": class_code,
        "year": year,
    }))
    return sample


_EMPTY_SHARD_MARKERS = (
    "No data was found in training input",
    "No valid training data",
)

# Persistent solo executor for per-shard computeFeatures timeouts. Daemon
# threads + no __exit__ means a stuck GEE call is *abandoned* (it keeps
# burning on its own thread but cannot block the main pool). Workers are
# expendable; we just spawn another. Sized larger than MAX_WORKERS so a
# zombie shard's thread doesn't starve the next live shard's timeout.
_solo_executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS * 4,
    thread_name_prefix="fscs-solo")


def process_shard(cell_id, cell_geom, class_code, year, image, grunnkart,
                  stable_mask, k, db_conn, lock, tile_scale=TILE_SCALE,
                  schema_checked=None):
    fc = sample_cell_class(image, grunnkart, stable_mask, cell_geom,
                           cell_id, class_code, k, year,
                           tile_scale=tile_scale)

    # Attach lon/lat for downstream joins.
    fc = fc.map(lambda f: f.set({
        "lon": f.geometry().coordinates().get(0),
        "lat": f.geometry().coordinates().get(1),
    }))

    def _compute():
        return ee.data.computeFeatures({
            "expression": fc,
            "fileFormat": "PANDAS_DATAFRAME",
        })

    fut = _solo_executor.submit(_compute)
    try:
        df = fut.result(timeout=SHARD_TIMEOUT_S)
    except FutTimeout:
        fut.cancel()   # best-effort; underlying gRPC call may keep running
        raise TimeoutError(
            f"computeFeatures exceeded {SHARD_TIMEOUT_S}s")
    except Exception as e:
        msg = str(e)
        if any(m in msg for m in _EMPTY_SHARD_MARKERS):
            # Cell has no pixels of this class intersected with stable mask.
            return 0
        raise

    if df is None or df.empty:
        return 0

    if schema_checked is not None and not schema_checked[0]:
        missing = [b for b in EXPECTED_BANDS if b not in df.columns]
        if missing:
            raise RuntimeError(
                f"Sampled df missing AlphaEarth bands {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} "
                f"(got {len(df.columns)} cols). Refusing to write a "
                f"partially-banded parquet.")
        schema_checked[0] = True

    with lock:
        try:
            db_conn.execute("INSERT INTO data BY NAME SELECT * FROM df")
        except duckdb.CatalogException:
            db_conn.execute("CREATE TABLE data AS SELECT * FROM df")

    return len(df)


def process_shard_with_escalation(cell_id, cell_geom, class_code, year,
                                  image, grunnkart, stable_mask, k,
                                  db_conn, lock, schema_checked,
                                  tile_scales=None, backoff=2):
    if tile_scales is None:
        tile_scales = TILE_SCALES   # read at call time so CLI override applies
    last_exc = None
    for attempt, ts in enumerate(tile_scales):
        try:
            return process_shard(
                cell_id, cell_geom, class_code, year, image, grunnkart,
                stable_mask, k, db_conn, lock, tile_scale=ts,
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


# ── Runner ─────────────────────────────────────────────────────────
def shard_key(cell_id, class_code):
    return f"{cell_id}:{class_code}"


def run(year, grid_size_m, k_per_class, max_workers, output_path,
        grunnkart_asset, ccdc_start, ccdc_end, ccdc_prob,
        test_mode=False):
    checkpoint_file = output_path + ".checkpoint.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # ── Inputs ──
    grunnkart = ee.Image(grunnkart_asset)
    image = build_alphaearth_image(year)
    region = ee.Geometry.Rectangle(
        list(COUNTIES_BBOX_25832),
        proj=ee.Projection(COUNTIES_CRS), evenOdd=True, geodesic=False)
    stable_mask = build_stable_mask(region, ccdc_start, ccdc_end, ccdc_prob)
    print(f"  AlphaEarth year: {year} (64 bands A00..A63)")
    print(f"  CCDC stability window: ({ccdc_start}, {ccdc_end}] "
          f"@ changeProb≥{ccdc_prob}")
    print(f"  Grunnkart asset: {grunnkart_asset}")

    cells = build_grid(cell_size_m=grid_size_m)
    print(f"  Grid: {len(cells)} cells of {grid_size_m/1000:.0f} km "
          f"over counties bbox")

    classes = list(CLASSES)
    if test_mode:
        # Pick cells near the middle of the bbox — corner cells fall in the
        # North Sea where the grunnkart raster has no data.
        mid = len(cells) // 2
        cells = cells[mid:mid + 2]
        classes = [4, 5]   # forest + grassland: dominant, almost always present
        print(f"  *** TEST MODE: cells {[c[0] for c in cells]} × "
              f"classes {classes} ***")

    shards = [(cid, cgeom, cls) for cid, cgeom in cells for cls in classes]
    total = len(shards)
    expected_pts = total * k_per_class
    print(f"  Shards: {total} (cell × class)   target ≤ {expected_pts:,} pts "
          f"(k={k_per_class}/shard)")

    # ── Checkpoint ──
    processed = set()
    prior_failed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            cp = json.load(f)
        processed = set(cp.get("done", []))
        prior_failed = set(cp.get("failed", []))
        print(f"  Resuming: {len(processed)}/{total} done, "
              f"{len(prior_failed)} previously failed (will retry)")

    # ── DuckDB buffer ──
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
                cid, cgeom, cls, year, image, grunnkart, stable_mask,
                k_per_class, db_conn, lock, schema_checked,
            )
            future_to_key[fut] = (cid, cls)

        with tqdm(total=len(todo), desc="FSCS shards", ncols=100) as pbar:
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

    # ── Export buffer to parquet ──
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


# ── CLI ────────────────────────────────────────────────────────────
def main():
    default_output = os.path.join(
        DATA_DIR, "grunnkart_nyvest_fscs_alphaearth.parquet")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--year", type=int, default=YEAR,
                        help=f"AlphaEarth annual year (default {YEAR})")
    parser.add_argument("--grunnkart", default=GRUNNKART_ASSET,
                        help=f"Grunnkart raster asset id "
                             f"(default {GRUNNKART_ASSET})")
    parser.add_argument("--grid_size_m", type=int, default=GRID_SIZE_M,
                        help=f"Grid cell size in metres "
                             f"(default {GRID_SIZE_M})")
    parser.add_argument("--k", type=int, default=K_PER_CLASS,
                        help=f"kMeans clusters per (cell, class) "
                             f"(default {K_PER_CLASS})")
    parser.add_argument("--ccdc_start", type=int, default=CCDC_START,
                        help=f"CCDC stability window start year, exclusive "
                             f"(default {CCDC_START})")
    parser.add_argument("--ccdc_end", type=int, default=CCDC_END,
                        help=f"CCDC stability window end year, inclusive "
                             f"(default {CCDC_END})")
    parser.add_argument("--ccdc_prob", type=float, default=CCDC_PROB,
                        help=f"CCDC changeProb threshold (default {CCDC_PROB})")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel shard workers (default {MAX_WORKERS})")
    parser.add_argument("--output", default=default_output,
                        help="Output parquet path")
    global SHARD_TIMEOUT_S, TILE_SCALES, KMEANS_NUM_PIXELS, _solo_executor
    parser.add_argument("--shard_timeout", type=int, default=SHARD_TIMEOUT_S,
                        help=f"Per-shard wall-clock timeout in seconds "
                             f"(default {SHARD_TIMEOUT_S})")
    parser.add_argument("--tile_scales", default=None,
                        help="Comma-separated tileScale escalation ladder "
                             "(default '4,8,16'). For retries on heavy "
                             "shards try '16'.")
    parser.add_argument("--num_pixels", type=int, default=KMEANS_NUM_PIXELS,
                        help=f"Pixels sampled to train kMeans per shard "
                             f"(default {KMEANS_NUM_PIXELS}). Drop to 2000 "
                             f"for retry of heavy shards.")
    parser.add_argument("--test_mode", action="store_true",
                        help="Run just 2 cells × 2 classes for a smoke test")
    args = parser.parse_args()

    SHARD_TIMEOUT_S = args.shard_timeout
    KMEANS_NUM_PIXELS = args.num_pixels
    if args.tile_scales:
        TILE_SCALES = tuple(int(x) for x in args.tile_scales.split(","))
    # Resize the solo-executor pool to match the new max_workers (4× buffer
    # so abandoned threads cannot starve live ones).
    _solo_executor = ThreadPoolExecutor(
        max_workers=max(MAX_WORKERS, args.max_workers) * 4,
        thread_name_prefix="fscs-solo")

    init_gee(args.project)
    run(year=args.year,
        grid_size_m=args.grid_size_m,
        k_per_class=args.k,
        max_workers=args.max_workers,
        output_path=args.output,
        grunnkart_asset=args.grunnkart,
        ccdc_start=args.ccdc_start,
        ccdc_end=args.ccdc_end,
        ccdc_prob=args.ccdc_prob,
        test_mode=args.test_mode)


if __name__ == "__main__":
    main()

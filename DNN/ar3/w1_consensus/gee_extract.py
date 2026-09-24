"""GEE batched point extraction: WorldCover, Esri LULC, Dynamic World, AEF.

Batches of <= 2000 points, checkpointed per (batch, sub-task) so a restart
resumes exactly where it left off. Same GEE conventions as
scripts/extraction/sample_feature_space_stable_allyears.py (project
ee-gsingh, high-volume endpoint, retry/backoff on 429s) — including that
script's concurrency pattern: a ThreadPoolExecutor (MAX_WORKERS threads, GEE
calls release the GIL during network I/O) dispatches many (batch, subtask)
computeFeatures calls at once instead of one at a time, since each call is
dominated by GEE round-trip latency (~60-90s) rather than local CPU. A single
lock protects the shared DuckDB buffer + checkpoint file.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import ee
import pandas as pd

from common import (ESRI_ASSET, WORLDCOVER_V100, WORLDCOVER_V200,
                    DYNAMICWORLD, YEARS, SCALE, compute_features_with_retry,
                    build_alphaearth_year, EXPECTED_BANDS)

TILE_SCALE = 4
BATCH_SIZE = 2000
MAX_WORKERS = 10
FLUSH_EVERY = 5


def points_to_fc(df_batch, id_col="pid"):
    feats = []
    for _, r in df_batch.iterrows():
        geom = ee.Geometry.Point([float(r["lon"]), float(r["lat"])])
        feats.append(ee.Feature(geom, {id_col: int(r[id_col])}))
    return ee.FeatureCollection(feats)


# ── per-product images ─────────────────────────────────────────────
def worldcover_image():
    # both are single-image ImageCollections, not bare Image assets
    wc2020 = ee.Image(ee.ImageCollection(WORLDCOVER_V100).first()) \
        .select("Map").rename("wc2020")
    wc2021 = ee.Image(ee.ImageCollection(WORLDCOVER_V200).first()) \
        .select("Map").rename("wc2021")
    return wc2020.addBands(wc2021)


def esri_image_for_year(year):
    ic = (ee.ImageCollection(ESRI_ASSET)
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    return ic.mosaic().select("b1").rename("esri")


def dw_image_for_year(year):
    """June-Sept mode of DW `label`, plus agreement fraction and n obs."""
    dw = (ee.ImageCollection(DYNAMICWORLD)
          .filterDate(f"{year}-06-01", f"{year}-10-01")
          .select("label"))
    n_img = dw.count().rename("dw_n")
    mode_img = dw.reduce(ee.Reducer.mode()).rename("dw")
    eq_mode = dw.map(lambda img: img.eq(mode_img)).sum()
    frac_img = eq_mode.divide(n_img).rename("dw_frac")
    return ee.Image.cat([mode_img, frac_img, n_img])


# ── checkpointed batch driver ───────────────────────────────────────
def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f).get("done", []))
    return set()


def _save_checkpoint(path, done):
    with open(path, "w") as f:
        json.dump({"done": sorted(done)}, f)


def run_static(points_df, out_path, checkpoint_path, id_col="pid",
              batch_size=BATCH_SIZE, tile_scale=TILE_SCALE,
              years=YEARS, log=print, max_workers=MAX_WORKERS):
    """Extract wc2020, wc2021, esri_<year>, dw_<year>[,_frac,_n] for every
    year in `years`. Wide output: one row per pid. Checkpointed per
    (batch_idx, subtask) where subtask in {'wc', 'esri_dw_<year>'}.
    Dispatches all pending subtasks across `max_workers` threads."""
    n = len(points_df)
    n_batches = (n + batch_size - 1) // batch_size
    done = _load_checkpoint(checkpoint_path)
    log(f"  [static] {n} points, {n_batches} batches, "
        f"{len(done)} subtasks already done")

    con = duckdb.connect()
    con.execute("CREATE TABLE IF NOT EXISTS wide (pid BIGINT PRIMARY KEY)")
    if os.path.exists(out_path):
        # `wide` starts as a 1-column (pid) skeleton every process start (this
        # is a fresh in-memory DuckDB connection). Re-add whatever extra
        # columns the on-disk parquet already has (same DOUBLE convention as
        # merge_col below) BEFORE inserting, or the plain column-order INSERT
        # fails with a column-count mismatch on any restart past the first
        # subtask.
        existing_cols = con.execute(
            f"DESCRIBE SELECT * FROM '{out_path}'").fetchdf()["column_name"]
        for c in existing_cols:
            if c == id_col:
                continue
            try:
                con.execute(f'ALTER TABLE wide ADD COLUMN "{c}" DOUBLE')
            except duckdb.CatalogException:
                pass
        con.execute(f"INSERT OR IGNORE INTO wide BY NAME "
                   f"SELECT * FROM '{out_path}'")

    lock = threading.Lock()

    def merge_col(df_new):
        if df_new is None or df_new.empty or id_col not in df_new.columns:
            return
        con.register("_new", df_new[[id_col] + [c for c in df_new.columns
                                                 if c != id_col]])
        for c in df_new.columns:
            if c == id_col:
                continue
            try:
                con.execute(f'ALTER TABLE wide ADD COLUMN "{c}" DOUBLE')
            except duckdb.CatalogException:
                pass
        con.execute(f"""
            INSERT INTO wide ({id_col})
            SELECT {id_col} FROM _new
            ON CONFLICT ({id_col}) DO NOTHING
        """)
        set_clause = ", ".join(f'"{c}" = _new."{c}"' for c in df_new.columns
                               if c != id_col)
        con.execute(f"""
            UPDATE wide SET {set_clause}
            FROM _new WHERE wide.{id_col} = _new.{id_col}
        """)
        con.unregister("_new")

    def flush():
        tmp = out_path + ".tmp"
        con.execute(f"COPY wide TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        os.replace(tmp, out_path)

    tasks = []  # (key, batch_idx, year_or_None)
    for b in range(n_batches):
        wc_key = f"{b}:wc"
        if wc_key not in done:
            tasks.append((wc_key, b, None))
        for y in years:
            key = f"{b}:esri_dw_{y}"
            if key not in done:
                tasks.append((key, b, y))
    total_subtasks = len(done) + len(tasks)
    log(f"  [static] {len(tasks)} subtasks pending, {max_workers} workers")

    def do_task(key, b, y):
        batch = points_df.iloc[b * batch_size:(b + 1) * batch_size]
        fc = points_to_fc(batch, id_col)
        if y is None:
            img = worldcover_image()
        else:
            img = ee.Image.cat([esri_image_for_year(y), dw_image_for_year(y)])
        sampled = img.sampleRegions(collection=fc, scale=SCALE,
                                    tileScale=tile_scale, geometries=False)
        df = compute_features_with_retry(sampled, label=key)
        if y is not None and df is not None and not df.empty:
            rename = {"esri": f"esri_{y}", "dw": f"dw_{y}",
                     "dw_frac": f"dw_{y}_frac", "dw_n": f"dw_{y}_n"}
            df = df.rename(columns=rename)
        return key, df

    since_flush = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(do_task, key, b, y): key for key, b, y in tasks}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                key, df = fut.result()
            except Exception as e:
                log(f"    [ERROR] {key}: {str(e)[:200]} (will retry next run)")
                continue
            with lock:
                merge_col(df)
                done.add(key)
                since_flush += 1
                log(f"    {key}: {0 if df is None else len(df)} rows")
                if since_flush >= FLUSH_EVERY:
                    flush()
                    _save_checkpoint(checkpoint_path, done)
                    log(f"    [flush] {len(done)}/{total_subtasks} "
                        f"subtasks checkpointed")
                    since_flush = 0

    with lock:
        flush()
        _save_checkpoint(checkpoint_path, done)
    n_rows = con.execute("SELECT count(*) FROM wide").fetchone()[0]
    log(f"  [static] done: {n_rows} rows -> {out_path}")
    con.close()


def run_temporal(points_df, out_path, checkpoint_path, id_col="pid",
                 batch_size=BATCH_SIZE, tile_scale=TILE_SCALE,
                 years=YEARS, log=print, max_workers=MAX_WORKERS):
    """Extract AEF A00..A63 per year -> long format pid, year, A00..A63.
    Checkpointed per (batch_idx, year). Dispatches all pending (batch, year)
    subtasks across `max_workers` threads."""
    n = len(points_df)
    n_batches = (n + batch_size - 1) // batch_size
    done = _load_checkpoint(checkpoint_path)
    log(f"  [temporal] {n} points, {n_batches} batches x {len(years)} years, "
        f"{len(done)} (batch,year) already done")

    year_images = {y: build_alphaearth_year(y) for y in years}

    con = duckdb.connect()
    if os.path.exists(out_path):
        con.execute(f"CREATE TABLE data AS SELECT * FROM '{out_path}'")

    lock = threading.Lock()
    schema_checked = [False]

    def flush():
        try:
            n_rows = con.execute("SELECT count(*) FROM data").fetchone()[0]
        except duckdb.CatalogException:
            return 0
        tmp = out_path + ".tmp"
        con.execute(f"COPY data TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        os.replace(tmp, out_path)
        return n_rows

    tasks = []
    for b in range(n_batches):
        for y in years:
            key = f"{b}:{y}"
            if key not in done:
                tasks.append((key, b, y))
    total_subtasks = len(done) + len(tasks)
    log(f"  [temporal] {len(tasks)} (batch,year) subtasks pending, "
        f"{max_workers} workers")

    def do_task(key, b, y):
        batch = points_df.iloc[b * batch_size:(b + 1) * batch_size]
        fc = points_to_fc(batch, id_col)
        sampled = year_images[y].sampleRegions(
            collection=fc, scale=SCALE, tileScale=tile_scale, geometries=False)
        df = compute_features_with_retry(sampled, label=key)
        if df is not None and not df.empty:
            df = df.copy()
            df["year"] = y
        return key, df

    # NOTE: a key is only added to the on-disk checkpoint AFTER the parquet
    # holding its rows has been flushed — checkpointing a key first (before
    # the flush that persists it) would let a crash between the two leave
    # `done` claiming rows that were never written to disk, and a resume
    # would then silently skip re-extracting them.
    since_flush = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(do_task, key, b, y): key for key, b, y in tasks}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                key, df_y = fut.result()
            except Exception as e:
                log(f"    [ERROR] {key}: {str(e)[:200]} (will retry next run)")
                continue
            with lock:
                if df_y is not None and not df_y.empty:
                    if not schema_checked[0]:
                        missing = [c for c in EXPECTED_BANDS if c not in df_y.columns]
                        if missing:
                            raise RuntimeError(
                                f"AEF extraction missing bands {missing[:5]} "
                                f"(got {len(df_y.columns)} cols)")
                        schema_checked[0] = True
                    con.register("_df_y", df_y)
                    try:
                        con.execute("INSERT INTO data BY NAME SELECT * FROM _df_y")
                    except duckdb.CatalogException:
                        con.execute("CREATE TABLE data AS SELECT * FROM _df_y")
                    con.unregister("_df_y")
                done.add(key)
                since_flush += 1
                log(f"    {key}: {0 if df_y is None else len(df_y)} rows")
                if since_flush >= FLUSH_EVERY:
                    n_rows = flush()
                    _save_checkpoint(checkpoint_path, done)
                    log(f"    [flush] total {n_rows} rows checkpointed "
                        f"({len(done)}/{total_subtasks} subtasks done)")
                    since_flush = 0

    with lock:
        n_rows = flush()
        _save_checkpoint(checkpoint_path, done)
    log(f"  [temporal] done: {n_rows} rows -> {out_path}")
    con.close()

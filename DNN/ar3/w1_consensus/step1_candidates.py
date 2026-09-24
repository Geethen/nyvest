"""Step 1 — candidate points from the 2018/2024 local rasters.

Reads small random 512x512 windows of classified_{2018,2024}.tif and
uq_{2018,2024}_setsize.tif (aligned grids, verified: identical transform,
shape, CRS EPSG:32633) and draws pixel centres into three strata:

  flip:      valid in both years, class_2018 != class_2024
  uncertain: valid in both years, class_2018 == class_2024, and
             setsize > 1 in either year (both setsize values must be valid)
  random:    valid in both years, class_2018 == class_2024, NOT uncertain
             (the "stable, unambiguous" control stratum — chosen mutually
             exclusive of the other two so the three strata partition the
             valid-both pixel population cleanly)

Targets: flip 24000, uncertain 12000, random 12000, spread across the
25 km grid cells in ROUNDS (each active cell contributes at most one more
512x512 block per round) so no cell dominates before others get a look in;
a cell with little land area simply contributes what it has.

Dedup:
  - drop any candidate that lands on the same 10 m pixel (classified
    raster's own grid) as an existing training point
  - enforce >=30 m spacing among ALL accepted candidates (any stratum)
    via a 30 m bin occupancy dict

cell_id is computed with the SAME 25 km EPSG:25832 grid as the training
parquet (see common.cell_id_from_lonlat, validated 500/500 against the
parquet in the phase-1 setup).

Output: data/consensus/candidates.parquet
  stratum, cell_id, lon, lat, class_2018, class_2024, setsize_2018, setsize_2024
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from common import (CLASSIFIED_TIF, SETSIZE_TIF, CONSENSUS_DIR, TRAIN_PARQUET,
                    cell_id_from_lonlat)

OUT_PARQUET = os.path.join(CONSENSUS_DIR, "candidates.parquet")
CHECKPOINT = OUT_PARQUET + ".checkpoint.json"

TARGETS = {"flip": 24_000, "uncertain": 12_000, "random": 12_000}
MIN_SPACING_M = 30.0
MAX_PER_BLOCK_PER_STRATUM = 400   # cap the oversample kept from one block
NODATA_CLASS = 0
NODATA_SETSIZE = 255

_tf_32633_to_4326 = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


def _block_center_cell_ids(transform, windows):
    cx = np.array([transform.c + transform.a * (w.col_off + w.width / 2)
                   for _, w in windows])
    cy = np.array([transform.f + transform.e * (w.row_off + w.height / 2)
                   for _, w in windows])
    lon, lat = _tf_32633_to_4326.transform(cx, cy)
    return cell_id_from_lonlat(lon, lat)


def _existing_pixel_set(transform):
    """(col,row) tuples on the classified-raster grid for existing training
    points, for the same-10m-pixel dedup check."""
    df = pd.read_parquet(TRAIN_PARQUET, columns=["lon", "lat"])
    df = df.drop_duplicates()
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    x, y = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    inv = ~transform
    col_f, row_f = inv * (x, y)
    col = np.floor(col_f).astype(np.int64)
    row = np.floor(row_f).astype(np.int64)
    return set(zip(col.tolist(), row.tolist()))


def _read_block(paths, window):
    out = {}
    for key, path in paths.items():
        with rasterio.open(path) as ds:
            out[key] = ds.read(1, window=window)
    return out


def process_block(cell_id, window, existing_px, rng_seed):
    """Read one block, return oversampled candidate rows per stratum
    (spacing/dedup enforced later, globally, in the main thread)."""
    paths = {"c18": CLASSIFIED_TIF[2018], "c24": CLASSIFIED_TIF[2024],
             "s18": SETSIZE_TIF[2018], "s24": SETSIZE_TIF[2024]}
    arrs = _read_block(paths, window)
    c18, c24, s18, s24 = arrs["c18"], arrs["c24"], arrs["s18"], arrs["s24"]

    valid_both = (c18 != NODATA_CLASS) & (c24 != NODATA_CLASS)
    same_class = valid_both & (c18 == c24)
    flip = valid_both & (c18 != c24)
    setsize_valid = (s18 != NODATA_SETSIZE) & (s24 != NODATA_SETSIZE)
    uncertain = same_class & setsize_valid & ((s18 > 1) | (s24 > 1))
    random_mask = same_class & ~uncertain

    rng = np.random.default_rng(rng_seed)
    dy, dx = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    H, W = c18.shape

    rows_out = []
    with rasterio.open(CLASSIFIED_TIF[2018]) as ds:
        transform = ds.transform
    for stratum, mask in (("flip", flip), ("uncertain", uncertain),
                          ("random", random_mask)):
        sub = mask[dy::3, dx::3]
        rr, cc = np.nonzero(sub)
        if rr.size == 0:
            continue
        full_rows = dy + rr * 3
        full_cols = dx + cc * 3
        n = rr.size
        if n > MAX_PER_BLOCK_PER_STRATUM:
            keep = rng.choice(n, size=MAX_PER_BLOCK_PER_STRATUM, replace=False)
            full_rows, full_cols = full_rows[keep], full_cols[keep]
        for r, c in zip(full_rows.tolist(), full_cols.tolist()):
            gcol = window.col_off + c
            grow = window.row_off + r
            if (gcol, grow) in existing_px:
                continue
            x = transform.c + transform.a * (gcol + 0.5)
            y = transform.f + transform.e * (grow + 0.5)
            rows_out.append({
                "stratum": stratum, "cell_id": cell_id,
                "gcol": gcol, "grow": grow, "x": x, "y": y,
                "class_2018": int(c18[r, c]), "class_2024": int(c24[r, c]),
                "setsize_2018": int(s18[r, c]), "setsize_2024": int(s24[r, c]),
            })
    return rows_out


def run(seed=42, max_workers=8, log_every=1):
    t0 = time.time()
    with rasterio.open(CLASSIFIED_TIF[2018]) as ds:
        transform = ds.transform
        all_windows = list(ds.block_windows(1))
    print(f"  {len(all_windows)} blocks total")

    cell_ids = _block_center_cell_ids(transform, all_windows)
    by_cell = {}
    for (ji, w), cid in zip(all_windows, cell_ids):
        if cid < 0:
            continue
        by_cell.setdefault(int(cid), []).append(w)
    active_cells = sorted(by_cell)
    print(f"  {len(active_cells)} active cells")

    rng = random.Random(seed)
    for cid in active_cells:
        rng.shuffle(by_cell[cid])

    print("  Building existing-training-point pixel exclusion set...")
    existing_px = _existing_pixel_set(transform)
    print(f"  {len(existing_px):,} existing-point pixels to exclude")

    accepted = {s: [] for s in TARGETS}
    per_cell_count = {s: {} for s in TARGETS}
    spacing_bins = set()

    # resume from checkpoint if present
    start_round = 0
    if os.path.exists(OUT_PARQUET) and os.path.exists(CHECKPOINT):
        df_prev = pd.read_parquet(OUT_PARQUET)
        with open(CHECKPOINT) as f:
            cp = json.load(f)
        start_round = cp.get("round", 0)
        block_idx = cp.get("block_idx", {})
        # by_cell lists are deterministically shuffled (seeded) and blocks are
        # popped from the END each round, so "remaining" = the first `idx`
        # elements of a freshly-rebuilt shuffle (idx = count left last save).
        for cid_s, idx in block_idx.items():
            by_cell[int(cid_s)] = by_cell[int(cid_s)][:idx]
        for s in TARGETS:
            sub = df_prev[df_prev["stratum"] == s]
            accepted[s] = sub.to_dict("records")
            for cid, cnt in sub["cell_id"].value_counts().items():
                per_cell_count[s][int(cid)] = int(cnt)
        for _, r in df_prev.iterrows():
            bx = round(r["x"] / MIN_SPACING_M)
            by = round(r["y"] / MIN_SPACING_M)
            spacing_bins.add((bx, by))
        print(f"  Resumed from checkpoint: round {start_round}, "
              f"{sum(len(v) for v in accepted.values()):,} rows so far")

    def targets_met():
        return all(len(accepted[s]) >= TARGETS[s] for s in TARGETS)

    round_no = start_round
    seed_ctr = seed * 1000 + start_round
    while not targets_met():
        cells_with_blocks = [c for c in active_cells if by_cell[c]]
        if not cells_with_blocks:
            print("  [WARN] all blocks exhausted before hitting all targets")
            break
        round_no += 1
        t_round = time.time()
        tasks = []
        for cid in cells_with_blocks:
            w = by_cell[cid].pop()
            seed_ctr += 1
            tasks.append((cid, w, seed_ctr))

        n_new = {s: 0 for s in TARGETS}
        # Collect ALL blocks' rows first (keyed by cell), then allocate with
        # a round-robin across cells — NOT first-come-first-served — so the
        # accept order doesn't let whichever threads finish first exhaust a
        # stratum's global target before every cell got a fair share.
        rows_by_cid = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(process_block, cid, w, existing_px, sd): cid
                   for cid, w, sd in tasks}
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    rows_by_cid[cid] = fut.result()
                except Exception as e:
                    print(f"    [ERROR] cell {cid} block read failed: {e}")

        for s in TARGETS:
            if len(accepted[s]) >= TARGETS[s]:
                continue
            pools = {cid: [r for r in rows if r["stratum"] == s]
                     for cid, rows in rows_by_cid.items()}
            pools = {cid: rs for cid, rs in pools.items() if rs}
            order = list(pools)
            rng.shuffle(order)
            oi = 0
            while order and len(accepted[s]) < TARGETS[s]:
                cid = order[oi % len(order)]
                pool_rows = pools.get(cid)
                if not pool_rows:
                    order.remove(cid)
                    if not order:
                        break
                    oi %= len(order)
                    continue
                r = pool_rows.pop(0)
                bx = round(r["x"] / MIN_SPACING_M)
                byy = round(r["y"] / MIN_SPACING_M)
                key = (bx, byy)
                oi += 1
                if key in spacing_bins:
                    continue
                spacing_bins.add(key)
                accepted[s].append(r)
                per_cell_count[s][cid] = per_cell_count[s].get(cid, 0) + 1
                n_new[s] += 1

        elapsed = time.time() - t_round
        totals = {s: len(accepted[s]) for s in TARGETS}
        if round_no % log_every == 0 or targets_met():
            print(f"  round {round_no}: +{n_new} in {elapsed:.1f}s -> "
                  f"totals {totals} (cells_left={len(cells_with_blocks)}) "
                  f"[{time.time()-t0:.0f}s elapsed]")

        # checkpoint
        rows_all = []
        for s in TARGETS:
            rows_all.extend(accepted[s])
        df_ck = pd.DataFrame(rows_all)
        tmp = OUT_PARQUET + ".tmp"
        df_ck.to_parquet(tmp)
        os.replace(tmp, OUT_PARQUET)
        with open(CHECKPOINT, "w") as f:
            json.dump({"round": round_no,
                       "block_idx": {str(c): len(by_cell[c]) for c in active_cells}},
                      f)

    # finalize: build lon/lat + cell_id (recomputed, should match)
    rows_all = []
    for s in TARGETS:
        rows_all.extend(accepted[s])
    df = pd.DataFrame(rows_all)
    lon, lat = _tf_32633_to_4326.transform(df["x"].to_numpy(), df["y"].to_numpy())
    df["lon"] = lon
    df["lat"] = lat
    df["cell_id_check"] = cell_id_from_lonlat(lon, lat)
    mismatches = (df["cell_id_check"] != df["cell_id"]).sum()
    if mismatches:
        print(f"  [WARN] {mismatches} candidates' recomputed cell_id "
              f"differs from block-assigned cell_id (edge-of-block effect); "
              f"using recomputed value as authoritative")
        df["cell_id"] = df["cell_id_check"]
    df = df.drop(columns=["cell_id_check", "gcol", "grow", "x", "y"])
    df = df[["stratum", "cell_id", "lon", "lat", "class_2018", "class_2024",
            "setsize_2018", "setsize_2024"]]

    df.to_parquet(OUT_PARQUET)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\n[OK] Saved {OUT_PARQUET}")
    print(df["stratum"].value_counts())
    print(f"  distinct cells used: {df['cell_id'].nunique()}")
    print(f"  wall time: {time.time()-t0:.0f}s")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--test_mode", action="store_true",
                    help="tiny targets for an end-to-end smoke test")
    args = ap.parse_args()
    if args.test_mode:
        global TARGETS
        TARGETS = {"flip": 100, "uncertain": 50, "random": 50}
    run(seed=args.seed, max_workers=args.max_workers)


if __name__ == "__main__":
    main()

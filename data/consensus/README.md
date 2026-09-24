# data/consensus — W1 consensus-labels dataset

Generated 2026-09-23 15:55 (local server time). Built per `DNN/ar3/briefs/phase1_consensus.md`; code in `DNN/ar3/w1_consensus/`.

## Files

- `candidates.parquet`: 48,000 rows — stratum, cell_id, lon, lat, class_2018, class_2024, setsize_2018, setsize_2024 (step 1 output)
- `points.parquet`: 48,000 rows, 55 columns — candidates + gk_v1/gk_v2 + lidar + wc2020/wc2021 + esri_<year>/dw_<year>[_frac,_n] 2017-2025
- `aef_long.parquet`: 416,265 rows — pid, year, A00..A63 for 2017-2025 (candidate points only)
- `reference_points.parquet`: 3,000 rows — same product extraction on existing training points, stratified by class

## Counts per stratum

| stratum | n |
|---|---|
| flip | 24000 |
| uncertain | 12000 |
| random | 12000 |

## Per-cell histogram (candidates)

| stratum | n_cells | min_per_cell | median_per_cell | max_per_cell |
|---|---|---|---|---|
| flip | 124 | 1 | 212.000 | 428 |
| random | 144 | 4 | 86.000 | 167 |
| uncertain | 124 | 3 | 102.500 | 199 |

## Class-transition counts per stratum (class_2018 -> class_2024, top rows)

| stratum | class_2018 | class_2024 | n |
|---|---|---|---|
| flip | 2 | 6 | 3525 |
| flip | 6 | 7 | 2961 |
| flip | 6 | 2 | 1743 |
| flip | 4 | 5 | 1364 |
| flip | 6 | 4 | 1359 |
| flip | 4 | 6 | 1235 |
| flip | 4 | 7 | 1220 |
| flip | 7 | 4 | 1073 |
| flip | 7 | 6 | 922 |
| flip | 12 | 2 | 919 |
| random | 8 | 8 | 5463 |
| random | 2 | 2 | 2094 |
| random | 4 | 4 | 1997 |
| random | 6 | 6 | 1361 |
| random | 7 | 7 | 416 |
| random | 3 | 3 | 206 |
| random | 12 | 12 | 182 |
| random | 5 | 5 | 169 |
| random | 10 | 10 | 112 |
| uncertain | 6 | 6 | 3155 |
| uncertain | 4 | 4 | 3148 |
| uncertain | 2 | 2 | 2683 |
| uncertain | 7 | 7 | 1396 |
| uncertain | 5 | 5 | 682 |
| uncertain | 3 | 3 | 348 |
| uncertain | 12 | 12 | 251 |
| uncertain | 8 | 8 | 179 |
| uncertain | 10 | 10 | 158 |

## Product agreement with grunnkart (gk_v1)

### By stratum (candidate/hard-area points)

| group | product | year | n_both_defined | n_agree | agreement_rate |
|---|---|---|---|---|---|
| flip | worldcover | 2018 | 23352 | 7841 | 0.336 |
| random | worldcover | 2018 | 11860 | 8841 | 0.745 |
| uncertain | worldcover | 2018 | 11840 | 4798 | 0.405 |
| flip | worldcover | 2024 | 23352 | 7770 | 0.333 |
| random | worldcover | 2024 | 11860 | 8918 | 0.752 |
| uncertain | worldcover | 2024 | 11840 | 4894 | 0.413 |
| flip | esri | 2018 | 23352 | 12778 | 0.547 |
| random | esri | 2018 | 11187 | 8098 | 0.724 |
| uncertain | esri | 2018 | 11840 | 6762 | 0.571 |
| flip | esri | 2024 | 23340 | 12384 | 0.531 |
| random | esri | 2024 | 11185 | 8117 | 0.726 |
| uncertain | esri | 2024 | 11837 | 6560 | 0.554 |
| flip | dw | 2018 | 23352 | 11198 | 0.480 |
| random | dw | 2018 | 11187 | 7501 | 0.671 |
| uncertain | dw | 2018 | 11840 | 5734 | 0.484 |
| flip | dw | 2024 | 23340 | 10091 | 0.432 |
| random | dw | 2024 | 11185 | 7547 | 0.675 |
| uncertain | dw | 2024 | 11837 | 5095 | 0.430 |

### On the reference sample (3,000 existing training points, clean frame)

| product | year | n_both_defined | n_agree | agreement_rate |
|---|---|---|---|---|
| worldcover | 2018 | 2999 | 1420 | 0.473 |
| worldcover | 2024 | 2999 | 1444 | 0.481 |
| esri | 2018 | 2934 | 1586 | 0.541 |
| esri | 2024 | 2932 | 1544 | 0.527 |
| dw | 2018 | 2934 | 1444 | 0.492 |
| dw | 2024 | 2932 | 1391 | 0.474 |

## Survival fraction under `gk_plus2` (vs gk_v1)

| group | year | n_total | n_survive | survive_frac |
|---|---|---|---|---|
| flip | 2018 | 24000 | 10475 | 0.436 |
| random | 2018 | 12000 | 7524 | 0.627 |
| uncertain | 2018 | 12000 | 5501 | 0.458 |
| class=2 | 2018 | 8548 | 315 | 0.037 |
| class=3 | 2018 | 1240 | 163 | 0.131 |
| class=4 | 2018 | 12848 | 10800 | 0.841 |
| class=5 | 2018 | 1172 | 565 | 0.482 |
| class=6 | 2018 | 13910 | 5220 | 0.375 |
| class=7 | 2018 | 997 | 4 | 0.004 |
| class=8 | 2018 | 6993 | 5264 | 0.753 |
| class=10 | 2018 | 430 | 178 | 0.414 |
| class=12 | 2018 | 996 | 991 | 0.995 |
| flip | 2024 | 24000 | 9317 | 0.388 |
| random | 2024 | 12000 | 7588 | 0.632 |
| uncertain | 2024 | 12000 | 4905 | 0.409 |
| class=2 | 2024 | 8548 | 583 | 0.068 |
| class=3 | 2024 | 1240 | 159 | 0.128 |
| class=4 | 2024 | 12848 | 11019 | 0.858 |
| class=5 | 2024 | 1172 | 563 | 0.480 |
| class=6 | 2024 | 13910 | 2961 | 0.213 |
| class=7 | 2024 | 997 | 2 | 0.002 |
| class=8 | 2024 | 6993 | 5413 | 0.774 |
| class=10 | 2024 | 430 | 169 | 0.393 |
| class=12 | 2024 | 996 | 941 | 0.945 |

## Validation checks

1. **cell_id**: 500/500 existing training points' recomputed cell_id matched the parquet's own cell_id (all_match=True).
2. **lidar**: re-extracted 476/500 existing points found in data/lidar_features.parquet; max |Δ| per column: {"elevation": 0.0, "tch": 0.0, "slope": 0.0, "aspect_sin": 0.0, "aspect_cos": 0.0, "tri": 0.0}.
3. **AEF scaling**: max |Δ| vs the training parquet, per year: {"2020": {"n": 300, "max_abs_delta": 0.0}, "2024": {"n": 295, "max_abs_delta": 0.0}}.

## Wall time

- Step 1 (candidate sampling, local rasters only): 8s for all 48,000
  candidates (single round; each of the 312 grid cells with any raster
  block contributed once, matched to its target in that one pass).
- Step 2 local (grunnkart v1/v2 + lidar, 48,000 points): ~90s, folded into
  the first launch attempt's log.
- Step 2 GEE static extraction (WorldCover + Esri + Dynamic World,
  2017-2025) + temporal AEF extraction (2017-2025), 48,000 candidate
  points, run concurrently (10-worker thread pool): 4,075s (~68 min),
  timed by the script itself from just after local extraction to the
  final flush.
- Step 2 reference sample (3,000 points, static products only): completed
  well inside the first ~15 min of the run.
- Validation checks (cell_id / lidar / AEF-scaling, 500+500+300 points):
  under 2 minutes.
- Total wall clock from first launch to all outputs on disk: ~80 minutes
  (14:34-15:54), which INCLUDES an aborted sequential-extraction attempt
  and a mid-run rewrite to add concurrency (see Surprises) — the clean
  68-minute figure above is what a rerun from a warm local-extraction
  cache would take today.

## Surprises / deviations from the brief

- **Sequential extraction was far too slow; rewrote to concurrent
  mid-run.** The first implementation issued one `computeFeatures` call at
  a time. On the 200-point end-to-end test this looked fine (~12s/call),
  but at the real batch size (2,000 points) each call took 60-150s, and
  with 240 static + 216 temporal subtasks that projected to many hours.
  Rewrote `gee_extract.py` to dispatch pending subtasks across a
  10-worker `ThreadPoolExecutor` (matching
  `scripts/extraction/sample_feature_space_stable_allyears.py`'s own
  concurrency pattern, which the sequential-first draft had NOT copied
  despite the brief pointing at that script). This cut per-subtask time to
  ~8s on average — an ~8-10x speedup — bringing the full run down to
  ~68 minutes. No data was lost: the partially-completed sequential run's
  checkpoint/parquet state was reused by the concurrent version.
- **A real checkpoint-resume bug was caught before it corrupted data.**
  `run_static`'s DuckDB buffer started every process run as a 1-column
  (`pid`-only) skeleton table; on a restart with existing multi-column
  data already on disk, `INSERT ... SELECT *` from that parquet into the
  1-column table raised a column-count mismatch. Fixed by re-adding the
  on-disk columns before inserting, and verified with an isolated
  restart test (partial run -> new process -> more years) before
  relaunching the real job.
- **`ESA/WorldCover/v100`/`v200` are single-image ImageCollections, not
  bare Image assets** — `ee.Image(WORLDCOVER_V100)` fails with
  "Image.load: ... is not an Image"; caught by the 200-point test, fixed
  to `ee.Image(ee.ImageCollection(...).first())`.
- **A crosswalk correctness bug was caught by a synthetic unit check, not
  real data.** The first cut of `to_merged()` let ANY product's raw code
  fall back to "identity" if it numerically coincided with one of the 9
  merged-class ids — meaning e.g. a WorldCover code of `4` (not a real
  WorldCover code) would have silently been read as merged class 4
  (forest). Restricted the identity fallback to grunnkart only, whose raw
  code space genuinely IS the merged-class numbering (that's the brief's
  own "others identity" instruction) — WorldCover/Esri/DW codes not in
  their explicit dict now map to `None`.
- **Step-1 candidate spreading needed a round-robin fix.** The first
  accept-order (first-come-first-served across a thread pool) let
  whichever cells' blocks finished first exhaust the global per-stratum
  target, leaving candidates concentrated in only 72 of ~300 candidate
  cells. Switched to round-robin allocation across cells within each
  round; final spread is 124-144 distinct cells per stratum (flip max
  428 vs median 212 per cell — still uneven, since it is genuinely
  bounded by how much flip/uncertain-eligible area each cell has, per the
  brief's own "a cell with little area ... contributes what it has").
- **Esri LULC 2025 coverage is a different (untiled) image than
  2017-2024.** The 2017-2024 images for this AOI carry an `id_no` property
  (`32V_<year>`, one UTM-zone tile); the 2025 image has no `id_no` — it is
  a larger, untiled global mosaic — but does cover the AOI (confirmed by
  direct `reduceRegion` histogram sampling). Recorded as available for all
  9 years; no fallback needed.
- **1/48,000 candidate points (pid 45787) has zero valid AEF rows across
  all 9 years** — every `sampleRegions` call for that point returned
  nothing. Left as a real null (not imputed); negligible at this rate.
- **Two candidate points share the same 30 m spacing bin** under an
  independent re-check (round-trip EPSG:4326<->32633 reprojection can flip
  a boundary point's bin at the sub-cm level even though the enforcement
  at accept-time was correct in the raster's native EPSG:32633 grid) — 2
  of 48,000 pairs, not a real violation of the >=30 m intent.
- **`gk_plus2` survival is very low for wetland and bare in the hard-area
  strata** (class 7: 0.4%/0.2% for 2018/2024; class 2: 3.7%/6.8%) — i.e.
  grunnkart's own label for those classes rarely gets 2 external products
  to agree AND outvote disagreement, in these specific flip/uncertain/
  random pixels. This is reported as a fact about the crosswalk/data, not
  a recommendation.
- Report defaults to `gk_v1` (matching the same raster the training
  parquet's `class` field and the GEE training asset both derive from) as
  "the" grunnkart label for agreement/survival stats; `gk_v2` is also in
  `points.parquet`/`reference_points.parquet` and `report.build()` takes
  `gk_col` as a parameter if the harness wants the other version instead.

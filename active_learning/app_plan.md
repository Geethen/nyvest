# Active-Learning Labeling App — Plan

Interactive app to label Norwegian nature types, seeded by the current best model
(CatBoost→TabICL conformal pipeline, macro-F1 ≈ 0.7074) with conformal uncertainty
layers. Optimized for iteration speed and usability.

## Decisions (confirmed with user)
- **Model in loop:** fast CatBoost live (retrain in seconds); TabICL on-demand for the
  high-quality map (background job, ~85 min/fold on the A40).
- **Map target:** dense raster, **scoped to a SINGLE embedding tile for now** (expand to all
  154 tiles later). Tiles at `…/Nature_types_mapping/Features/embeddings` (~15 GB total,
  bands A00–A63 — the same features the model uses). Points also shown for sample locations.
- **Platform:** local Streamlit web app with a Folium map (works on the A40 via port-forward).
- **Uncertainty signal:** conformal APS prediction-set size (reuses the validated APS code).
- **Inference accelerator:** evaluate `timber-compiler`
  (https://github.com/kossisoroyce/timber) — an AOT compiler that turns the CatBoost model
  into native C99 (~336× single-sample speedup). **Adopt only if it returns full per-class
  probabilities** (the APS layer needs a 10-class probability vector per pixel); otherwise
  fall back to native CatBoost predict_proba (already fast enough for one tile + the live
  loop) and revisit timber for the full-AOI scale-up. Timber's documented interface is a
  batch HTTP server + CatBoost JSON export (`save_model(format='json')`); no in-process
  Python/ctypes API is documented — to be verified empirically in Phase 0b.

## Preconditions / corrections
- **CLAUDE.md is wrong:** rasterio / rioxarray / xarray / dask are NOT in the active
  `recover` venv. Install them (Phase 0) — the dense-raster path needs rasterio at minimum.
- No `models/` dir and no persisted model exist today; the "best model" is a CV script.
  We will add a training-and-persist step (Phase 1).
- Classes are merged 1→2 and 9→8 (12 raw → 10 effective). Class names from
  `reports/conformal_combined_results.json` + `reports/combined_dataset_summary.csv`.

## Architecture
```
active_learning/scripts/
  model_core.py      # shared: load data, merge classes, train CatBoost, APS calibrate,
                     #   predict_proba + prediction-set sizes, persist/load artifacts
  train_seed_model.py# fit CatBoost on all stable rows, cross-conformal APS τ, save to models/
  predict_raster.py  # tile-by-tile dense prediction → class GeoTIFF + uncertainty GeoTIFF
                     #   (CatBoost by default; --stage2 tabicl optional, background)
  app.py             # Streamlit app: map + AL loop
models/
  seed_catboost.cbm, aps_calib.npz, label_encoder.json, class_names.json
active_learning/data/
  labels.parquet     # appended user labels (lon, lat, class, source='user', ts)
  predictions.parquet# per-sample-location: pred, set_size, probs (live CatBoost state)
active_learning/reports/
  predicted_classes.tif, predicted_uncertainty.tif   # dense map (web-mercator COG for Folium)
```

## Phases

### Phase 0 — Env + scaffolding (fast)
- `uv pip install rasterio rioxarray streamlit folium streamlit-folium timber-compiler --python …/recover/.venv/bin/python`
- Create dirs; add a small `class_names.json` (merged-class → label) from the two report files.
- Smoke-test: open one embedding tile with rasterio, confirm 64 bands A00–A63, CRS, res, nodata.

### Phase 0b — Timber spike (decision gate) — DONE: **NO-GO** (2026-06-22)
Result: timber-compiler 0.6.0 cannot compile our 10-class CatBoost — its CatBoost
front-end + tree IR are single-output only (scalar leaf_value; multiclass needs an
n_leaves×n_classes vector-leaf + per-class codegen). Crashes at
`catboost_parser.py:68` on the multiclass `scale_and_bias`. Full writeup:
`active_learning/reports/timber_spike.md`; memory: timber-multiclass-nogo.
→ **Use native CatBoost `predict_proba` everywhere.** It IS fast enough at single-tile
scope. Keep a `--backend {catboost,timber}` seam in Phase 2 for a future revisit.
(Timber does have an in-process `TimberPredictor` API, contrary to its docs — usable
if multiclass support ever lands.)

### Phase 1 — Seed model (`model_core.py` + `train_seed_model.py`)
- Reuse logic from `common_ground/scripts/llto_schemeB_allyears.py`:
  `load_parquet`, `merge_classes`, `cross_conformal_aps`, `aps_score_matrix`,
  `conformal_quantile`, `make_catboost`. **Refactor the reusable bits into `model_core.py`**
  (single source of truth; the CV script keeps working).
- Train one CatBoost on ALL stable rows (deduped to latest year per lon/lat), fit global APS
  τ via 5-fold cross-conformal on a capped subsample. Persist model + τ + encoder + class names.
- This is the LIVE loop model. (TabICL stays as the optional high-quality stage.)

### Phase 2 — Dense raster map (`predict_raster.py`), SINGLE TILE
- `--tile <path>` (default: one representative tile). Read 64 bands → reshape to
  (pixels, 64) → drop nodata → predict_proba → argmax class + APS set size (saved τ).
- Inference backend behind a flag: `--backend {catboost,timber}` (timber only if Phase 0b = GO).
- Write two GeoTIFFs (class, uncertainty=set-size), reproject to EPSG:3857 so Folium overlays
  them as image layers. Block-windowed reads to bound memory; progress logging.
- Designed to loop over a tile list later (`--tiles-glob`) with no code change — just scope.
- `--stage2 tabicl` flag swaps in the TabICL pipeline for a high-quality pass (background,
  guarded by the GPU/`TOTAL_SUP` rules from memory: cuda-only, MemoryMax scope).

### Phase 3 — Streamlit AL app (`app.py`)
- **Map:** Folium with (a) dense class raster overlay + opacity slider, (b) dense uncertainty
  overlay toggle, (c) sample points colored by prediction, sized/filtered by set-size.
- **Active-learning queue:** rank unlabeled sample locations by APS set-size (most ambiguous
  first); "next batch" button surfaces N points; clicking a point shows its embedding-NN context,
  current pred + prediction set, and grunnkart label.
- **Labeling:** dropdown of class names; submit appends to `labels.parquet`.
- **Retrain (live):** button refits CatBoost incorporating user labels (weighted), recomputes
  τ + set-sizes for all sample points, updates the point layer in seconds. Shows F1 on a held-out
  slice + count labeled this session.
- **Regenerate full map:** button kicks off `predict_raster.py` (CatBoost now; TabICL background)
  and reloads the overlays when done.
- Session state persists labels to disk so sessions resume.

### Phase 4 — Validate & document
- Run the seed-model train + a small dense-map test region end-to-end on the A40.
- Verify the AL loop: label a few points, retrain, confirm set-sizes update and map redraws.
- Short README in `active_learning/` (how to launch, port-forward, regenerate map).
- Fix the CLAUDE.md dependency claim (note rasterio et al. are now actually installed).

## Speed/usability guarantees
- Live loop = CatBoost only (seconds), never TabICL.
- Dense raster precomputed to disk; app only overlays images (no per-frame inference).
- APS set-size reused as both the map's uncertainty layer and the AL acquisition function —
  one computation, two uses.

## Risks
- 15 GB of tiles × 64 bands: dense predict is I/O-bound; mitigate with per-tile streaming +
  optional AOI bbox / downsample-preview flag for a quick first map.
- Tile CRS/extent mosaicking; nodata handling at tile edges.
- TabICL background runs must obey the memory guards (see memory: schemeB-allyears-experiment).

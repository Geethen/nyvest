# Brief W1 — consensus labels in error-prone areas

You are building a DATASET, not a model. Work only inside
`DNN/ar3/w1_consensus/` (code) and `data/consensus/` (outputs). Do NOT edit any
other file in the repo, do not git commit, do not delete anything.

Python: `~/myprojects/recover/.venv/bin/python` (has ee, rasterio, geopandas,
duckdb, pyogrio). Wrap any heavy local process in
`systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 ...`.
The machine has 8 CPUs and 62 GB RAM; the P-drive (`/data/P-Prosjekter2/...`) is
CIFS and slow (~200k px/s), so read small windows, never whole rasters.

## Context

`models/dnn_final_moe8_merged.pt` maps 9 classes wall-to-wall for 2018 and 2024:
`/data/P-Prosjekter2/154001_nyvest/landcover_Geethen/landcover_2018_2024/`
`classified_{2018,2024}.tif` (int16, EPSG:32633, 512×512 blocks; check nodata),
`uq_{2018,2024}_setsize.tif` (conformal prediction-set size; 1 = confident).
Class codes (merged): 2 bare (rock+sand+sparse-veg), 3 crop, 4 forest,
5 grassland, 6 scrub, 7 wetland, 8 water, 10 built, 12 snow/ice.

The training points (`data/grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet`)
come only from CCDC-stable pixels, so the pixels where the maps flip between years
are under-sampled. We want new labelled points THERE, labelled by agreement
between the base map (grunnkart) and independent products.

Read `scripts/extraction/sample_feature_space_stable_allyears.py` first. Copy its
GEE conventions exactly: project `ee-gsingh`, high-volume endpoint, AEF
collection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` built per year with
`build_alphaearth_year`, `sampleRegions` at scale 10, the 25 km grid
(`build_grid`: EPSG:25832 bbox `COUNTIES_BBOX_25832`, `cell_id = j*nx + i`), plus
its retry/backoff pattern for 429s.

## Step 1 — candidate points (local rasters)

For each 25 km cell, read a few random 512×512 windows of both classified and
both setsize rasters (aligned grids; verify), and draw pixel centres into these strata:

* `flip`: both years valid, class_2018 != class_2024
* `uncertain`: class_2018 == class_2024 and setsize > 1 in either year
* `random`: any pixel valid in both years (the control stratum)

Targets: flip 24,000, uncertain 12,000, random 12,000. Spread them evenly across
cells (a cell with little area of a stratum contributes what it has). Record
the transition (class_2018, class_2024) and both set sizes. Drop candidates
that fall on the same 10 m pixel as an existing training point. Keep points at
least 30 m apart. Store lon/lat in EPSG:4326 at the pixel centre, and compute
`cell_id` with the SAME grid as the training parquet. Sanity-check it: for 500
existing training points, recompute their cell_id from lon/lat and assert it
matches the parquet's.

## Step 2 — labels and features at the points

Static, one row per point → `data/consensus/points.parquet`:
- `gk_v1`, `gk_v2`: raw grunnkart class (1..13) from
  `/data/P-Prosjekter2/154001_nyvest/GIS/NIBIO/Version_2/rasterized_10m/grunnkart_nyvest_10m.tif`
  and `..._v2.tif` (point sample; check CRS).
- `wc2020`, `wc2021`: ESA WorldCover `ESA/WorldCover/v100`, `ESA/WorldCover/v200`, band `Map`.
- `esri_<year>` for every year available 2017-2025 in the Esri 10 m annual LULC
  time series (GEE community catalog, e.g.
  `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`; verify the
  asset id and class codes from its catalog page or image properties).
- `dw_<year>` for 2017-2025: Dynamic World `GOOGLE/DYNAMICWORLD/V1`, June–September
  images only, per-pixel MODE of `label`, plus `dw_<year>_frac` (share of valid
  observations that equal the mode) and `dw_<year>_n`.
- lidar `elevation`, `tri`, `tch`, extracted EXACTLY as
  `scripts/extraction/extract_lidar_features.py` does (same units and scaling;
  read it). Compare against `data/lidar_features.parquet` on 500 existing points
  to prove the method is identical.
- the stratum fields from step 1.

Temporal → `data/consensus/aef_long.parquet`: `pid, year, A00..A63` for
2017-2025, the same format as the training parquet (float, same scaling).
Prove the scaling matches by re-extracting 300 existing training points for
2020 and 2024 and reporting max |Δ| against the parquet.

Also run the same product extraction (grunnkart, WorldCover, Esri, DW) on a
REFERENCE sample of 3,000 existing training points (stratified by class), saved
to `data/consensus/reference_points.parquet`. It measures how often products
agree with grunnkart on the clean frame. Without that number, agreement in the
hard areas cannot be interpreted.

## Step 3 — crosswalk module (DO NOT pick a final rule)

`DNN/ar3/w1_consensus/crosswalk.py`, importable, with:
- `to_merged(product, codes) -> array of sets or merged codes` mapping each
  product onto the 9 merged classes. Starting crosswalk (verify every code
  against the product docs):
  - grunnkart: 1→2, 9→8, 11→2, 13→None, others identity
  - WorldCover: 10→4, 20→6, 30→5, 40→3, 50→10, 60→2, 70→12, 80→8, 90→7, 100→2, 95→None
  - Dynamic World: 0→8, 1→4, 2→5, 3→7, 4→3, 5→6, 6→10, 7→2, 8→12
  - Esri: water→8, trees→4, flooded veg→7, crops→3, built→10, bare→2,
    snow/ice→12, clouds→None, rangeland→{5,6} (compatible with either)
- `consensus(points_df, year, rule="gk_plus2")` returning (label or -1, n_agree,
  n_disagree). Rule `gk_plus2`: keep the grunnkart label if at least 2
  external products compatible with it for that year (WorldCover nearest
  year, Esri_year, DW_year) AND agreeing > disagreeing. Also implement
  `ext_majority` (majority of external products, ignoring grunnkart) and
  `gk_raw` (grunnkart only), so the harness can run controls.

## Step 4 — report

`data/consensus/README.md`: counts per stratum, per cell histogram, per class;
agreement rate of each product with grunnkart per stratum and on the reference
sample; the fraction of each stratum that survives `gk_plus2` per class; the
three validation checks above (cell_id, lidar, AEF scaling) with numbers; the
wall time; and anything that surprised you. Keep it factual, and include no
recommendations about modelling.

Run GEE extraction in batches of ≤ 2,000 points, checkpoint each batch to disk
so that a restart resumes, and run the long jobs with `nohup` and a log
in `DNN/ar3/w1_consensus/logs/`. Before launching the full run, test end to
end on 200 points.

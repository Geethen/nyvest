# NYVEST Project Overview — `P:\154001_nyvest`

NYVEST is a Norwegian Research Council project producing a wall-to-wall ecosystem/land-cover accounting map for Vestlandet (three western counties: Rogaland, Vestland, Møre og Romsdal). The core pipeline trains a Random Forest classifier on satellite embeddings + LiDAR and predicts land-cover class across the study area at 10 m resolution.

---

## Folder Structure

```
Admin/                      Project administration (budget, invoicing, contracts, team photos)
Feedback/                   Municipality webinar feedback and surveys
GIS/                        All spatial reference data (read-only inputs)
  Boundaries/               Study-area shapefiles (nyvest_fylker, nyvest_kommuner, per-county)
  Naturbase/                Norwegian nature-type databases
    NiN_v2/                 NiN v2 nature types (nationwide + clipped to NYVEST)
    Naturtyper_ku_verdi/    Nature types with ecological condition values
    handbok13/              Handbook-13 nature types
    NiN_verneområder/       Protected areas with NiN mapping
    Fjell_norge_2024/       Mountain boundary layer
  NIBIO/
    Version_2/              NIBIO Grunnkart arealregnskap v2 — the primary land-cover training source
                            (three counties: Rogaland, Vestland, Møre og Romsdal)
  FKB/                      Green-structure base map (gronnstruktur)
  N50/                      Kartverket topographic land-cover polygons
  SSB/                      SSB urban/tettsted boundaries and green-structure
  EEA/                      EUNIS habitat probability rasters (100 m, 1940–2017)
  Horvath_et_al_forestline/ Empirical forest-line raster (Horvath et al. 2019)
  bioklima_longlat/         Vegetation zones & sections (Moen 1998 / Bakkestuen 2008)
  norge_i_bilder/           0.5 m orthophotos downloaded per 2 km tile (Rogaland done)
Nature_types_mapping/       Feature engineering outputs
  Features/
    embeddings/             64-band AlphaEarth satellite embeddings, tiled by SSB 10 km grid (EPSG 25833)
    lidar/                  LiDAR-derived CHM + DTM, same tiling
    tile_ref/               SSB 10 km reference grid shapefiles
  Vestland_Moreromsdal_features/  Same features for northern counties
NiN mapping/                NiN v2 nature-type classification workflow (deep-learning path)
  Data/
    train / validation / Testing_old/   Per-class binary rasters for model training
Landcover_municipalities/   Ecosystem area statistics per municipality/sone-section
  T1-1_training_data/       Field/municipality training polygons
landcover_megan/            Main Random Forest land-cover mapping workflow (Rogaland)
  code/                     All R scripts (see below)
  predicted_tiles/          Per-tile RF class predictions
  tiles/                    Per-tile probability rasters (one band per class)
  polygons/                 Vectorised and smoothed output polygons
  reference_data/           Validation point datasets
  outputs/                  Summary outputs
Pictures/                   Field photos
Presentations/              Project presentations
Reading/                    Literature
```

---

## Scripts

### `landcover_megan/code/` — Main RF land-cover pipeline (R, by Megan)

These scripts run sequentially for Rogaland; the same approach will be applied to other counties.

| Script | Step | What it does |
|---|---|---|
| `prepare_training_data.R` | 1 | Samples 10 000 points from NIBIO Grunnkart polygons; assigns `okosystemtype_2` (level 2) or falls back to level 1; writes lookup tables and training point shapefile |
| `prepare_predictor_stack.R` | 1b | Builds VRTs over tiled embeddings and LiDAR folders; resamples LiDAR to match embeddings; writes `predictors.tif` |
| `extract_predictor_values.R` | 2 | Extracts 64-band embedding + 2-band LiDAR values at training points using `terra::extract`; writes `lc_extract.shp` / train/val/test splits |
| `model_training.R` | 3a | Trains a Random Forest classifier (class labels); generates confusion matrix on val set; saves `rf_model.rds` |
| `randomforest_probabilities.R` | 3b | Trains a second RF for per-class probabilities (uncertainty mapping); tiles the predictor stack into 4 quadrants for memory management |
| `predict.R` | 4 | Predicts land-cover class tile-by-tile using the SSB 10 km grid; saves one raster per tile to `predicted_tiles/` |
| `nyvest_polygonize.R` | 5 | Smooths the classified raster (focal window replaces low-confidence pixels); converts to polygons; dissolves and simplifies |
| `segmentation.R` | — | Experimental: runs image segmentation with `SegmentR` on embedding tiles (not yet in main pipeline) |
| `zonal_stats.R` | 6 | Extracts per-polygon zonal statistics (median, max, sd, % > threshold) from probability raster; one stat block per land-cover class |
| `merge_zonal_stats.R` | 7 | Merges per-class zonal stat shapefiles into a single attributed polygon shapefile |

### `NiN mapping/` — NiN v2 nature-type classification (R, by Ida & Megan)

| Script | What it does |
|---|---|
| `NYVEST NiN mapping - process MI data.R` | Clips the national NiN v2 2025 database to the NYVEST counties; summarises nature-type class counts |
| `NYVEST NiN mapping - process feature data.R` | Prepares per-feature rasters for Rogaland: geology (calcium content), and other environmental layers aligned to Sentinel-2 reference raster |
| `tile_labelled_data.R` | Tiles per-class binary training rasters onto the SSB 10 km grid; reprojects from EPSG 25833 → 32633 to match AlphaEarth embeddings; adds background class 0 |
| `tile_test_val_data.R` | Creates binary (presence/background) rasters for test and validation sets; reprojects and tiles same as above |
| `R script for downloading 2km half meter orto images UTM33.R` | Downloads 0.5 m orthophotos from the `geonorge` WMS in 2 km tiles (parallel); currently processed for Rogaland |

### Root-level scripts

| Script | What it does |
|---|---|
| `empirical_forestline.R` | Uses the Horvath et al. empirical forest-line raster and Kartverket 10 m DEM to classify each DEM tile as above/below forest line (mountain vs. forest); merges tiles |
| `Landcover_municipalities/NYVEST_landcover_statistics.R` | Computes expected ecosystem area per municipality × bioclimatic sone-section using Hovedøkosystemkart; used to prioritise municipalities for field data collection |

### `Nature_types_mapping/prepare_lidar_dataset.py`
Empty stub (Python); LiDAR feature extraction is currently handled in R.

---

## Key Data Sources

| Source | What | Location |
|---|---|---|
| NIBIO Grunnkart v2 | Primary land-cover training polygons (`okosystemtype_1/2`) | `GIS/NIBIO/Version_2/` |
| AlphaEarth embeddings | 64-band satellite embeddings per SSB tile (10 m) | `Nature_types_mapping/Features/embeddings/` |
| LiDAR CHM + DTM | Canopy height + terrain model (10 m) | `Nature_types_mapping/Features/lidar/` |
| NiN v2 (2025) | Norwegian nature-type polygons (Miljødirektoratet) | `GIS/Naturbase/NiN_v2/2025/` |
| Kartverket DEM 10 m | Elevation, used for forest-line analysis | `/data/R/GeoSpatialData/Elevation/Norway_DEM_10m_Kartverket/` |
| Norge i Bilder | 0.5 m orthophotos (WMS download) | `GIS/norge_i_bilder/Rogaland/` |
| Bioklima zones | Vegetation zones × sections for stratified sampling | `GIS/bioklima_longlat/` |

## Coordinate Reference Systems

- Data stored primarily in **EPSG 25833** (UTM zone 33N) and **EPSG 25832** (UTM zone 32N — NIBIO Grunnkart source).  
- AlphaEarth embeddings and the NiN deep-learning pipeline use **EPSG 32633**.  
- Scripts reproject as needed; always check CRS before combining layers.

## Current Status (as of project files)

- Rogaland is the pilot county: embeddings, LiDAR, orthophotos, RF model, and predictions are complete.
- Vestland and Møre og Romsdal features are partially prepared (`Vestland_Moreromsdal_features/`).
- NiN v2 classification (deep-learning path) is in active development.
- The segmentation approach (`SegmentR`) is exploratory and not yet integrated.

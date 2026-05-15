# Grunnkart Rasterization — Preprocessing Methods

**Script:** `scripts/rasterize_grunnkart.py`
**Output:** `GIS/NIBIO/Version_2/rasterized_10m/grunnkart_nyvest_10m.tif`
**Date:** 2026-05-14

---

## 1. Overview

The NIBIO Grunnkart Arealregnskap vector dataset was rasterized to a single 10 m UInt8 GeoTIFF covering the three NYVEST study counties: Rogaland, Vestland, and Møre og Romsdal. A two-source classification was applied — ecosystem and land-cover attributes from the Grunnkart polygons formed the base layer, with built-environment features from FKB Grønnstruktur burned on top as an overwrite pass.

---

## 2. Input Data

### 2.1 NIBIO Grunnkart Arealregnskap (primary source)

Three county-level File Geodatabases, each a sub-directory of `GIS/NIBIO/Version_2/`:

| County | Directory | Inner GDB | Features |
|---|---|---|---|
| Rogaland | `Basisdata_11_Rogaland_25832_GrunnkartArealregnskap` | `11_25832_arealregnskap_gdb.gdb` | 751,887 |
| Møre og Romsdal | `Basisdata_15_More_og_Romsdal_25832` | `15_25832_arealregnskap_gdb.gdb` | 679,974 |
| Vestland | `Basisdata_46_Vestland_25832_GrunnkartArealregnskap` | `46_25832_arealregnskap_gdb.gdb` | 2,055,202 |

All three are published in **EPSG:25832** (UTM zone 32N / ETRS89). Each layer contains one MultiPolygon row per land-cover/ecosystem mapping unit. Only the four columns required for classification are read from disk (`arealdekke`, `grunnforhold`, `okosystemtype_3`, `okosystemtype_1`); all other attributes are discarded at read time to reduce memory usage.

### 2.2 FKB Grønnstruktur (built-environment overlay)

`GIS/FKB/0000_25833_grønnstruktur_gdb.gdb`, layer `grønnstruktur`, published in **EPSG:25833** (UTM zone 33N / ETRS89). This dataset covers built infrastructure including buildings, roads, and grey/hardened surfaces for the full Norwegian west coast. Only polygons with `klasse_navn IN ('FKB_bygg', 'FKB_vei', 'greyArea')` are read; a bounding-box spatial filter to the study-area extent is applied at read time to avoid loading the full national dataset.

### 2.3 County Boundaries

`GIS/Boundaries/nyvest_fylker.shp` — the three county outlines in EPSG:25833 (reprojected to EPSG:25832 internally). Used only to compute the combined grid extent; polygons are not used to clip Grunnkart because each county GDB is already spatially bounded by its county outline.

---

## 3. Output Raster Specification

| Property | Value |
|---|---|
| CRS | EPSG:25832 (UTM zone 32N) |
| Resolution | 10 × 10 m |
| Extent | Union bbox of all 3 counties |
| Grid size | 28,386 × 63,653 px (1,807 Mpx logical) |
| Data type | UInt8 |
| NoData | 0 |
| Active pixels | 918,810,098 (50.9% of grid) |
| File size | 80.3 MB |
| Format | Cloud-optimised GeoTIFF — tiled (512 × 512 px), ZSTD level 3, delta predictor, BIGTIFF, SPARSE\_OK |
| Overviews | 2×, 4×, 8×, 16×, 32× (mode resampling) |

The grid is snapped to the nearest 10 m multiple in each direction (`floor`/`ceil` on the union bbox), ensuring pixel centres align with a common 10 m lattice shared across all counties and all other NYVEST rasters at the same resolution.

GDAL `SPARSE_OK` is used so empty 512 × 512 px blocks (which occur in the substantial gaps between Rogaland and Vestland, and between Vestland and Møre og Romsdal) are never written to disk, keeping the file small despite the large nominal extent.

---

## 4. Classification Scheme

Classification is resolved **in vector space** before rasterization using a sequential priority rule applied to numpy boolean arrays over all polygons simultaneously. Rules are applied in the order listed; a later rule overwrites an earlier one if both match the same polygon (i.e., lower-numbered classes have lower priority and would be overwritten by subsequent matches). In practice the rules are near-exclusive.

| Code | Class | Source attribute(s) |
|---|---|---|
| 0 | nodata / outside study area | — |
| 1 | sand | `arealdekke = "Snaumark_skrinn"` OR `grunnforhold = "Jorddekt"` OR `okosystemtype_3 = "Mineral extraction sites"` |
| 2 | rock | `arealdekke IN ("Snaumark_impediment", "Snaumark_uspesifisert")` |
| 3 | crop | `okosystemtype_1 = "Cropland"` |
| 4 | forest | `okosystemtype_1 = "Forest and woodlands"` |
| 5 | grassland | `okosystemtype_1 = "Grassland"` |
| 6 | scrub | `okosystemtype_1 = "Heathland and shrub"` |
| 7 | wetland | `okosystemtype_1 = "Inland wetlands"` |
| 8 | freshwater | `okosystemtype_1 IN ("Rivers and canals", "Lakes and reservoirs")` |
| 9 | marine | `okosystemtype_1 = "Marine ecosystems"` |
| 10 | built | FKB `klasse_navn IN ("FKB_bygg", "FKB_vei", "greyArea")` — overwrite pass |
| 11 | sparse | `okosystemtype_1 = "Sparsely vegetated ecosystems"` |
| 12 | snow | `arealdekke = "Snoisbre"` |
| 13 | other | Grunnkart polygon matched no rule above |

String matching is case-insensitive and strip-whitespace-normalised (`str.strip().lower()`). All four source attributes are read and matched as-is from the Grunnkart schema; no external lookup table is used.

**Note:** The `okosystemtype_1 = "Heathland and shrub"` value in the Grunnkart data uses "shrub" rather than "scrub" as might be expected from related EEA nomenclature. This is matched exactly as it appears in the database.

**Unclassified polygons (class 13 "other")** in practice cover: `Coastal beaches, dunes and wetlands` (not in the scheme), `Settlements and other artificial areas` polygons that fall outside the FKB Grønnstruktur coverage, and any polygons where all four classification attributes are null. Class 13 accounts for ~0.76% of active pixels across the three counties.

---

## 5. Processing Pipeline

### Step 1 — Grid definition

The output raster transform and dimensions are computed once from the union bounding box of all three county boundaries (reprojected to EPSG:25832). Grid edges are snapped outward to 10 m multiples. The file is created in `w+` (read/write) mode so each subsequent county can read-modify-write into the same file without re-opening.

### Step 2 — Per-county Grunnkart pass (×3, parallelised)

All three counties are processed concurrently using a `ThreadPoolExecutor` with one thread per county. Each thread executes steps 1–3 independently (no shared state); step 4 is serialised in the main thread as each future completes, so subgrid arrays are freed immediately rather than holding all three in memory at once.

1. **Read** — the Grunnkart GDB is read entirely into a GeoPandas GeoDataFrame. Only the four classification columns are loaded via `pyogrio`. No spatial filter is applied because each county GDB already covers only that county's extent. All three reads overlap in wall-clock time (~87 s combined, vs ~150 s serial).

2. **Reclassify** — the four attribute columns are normalised with vectorised pandas string operations (`fillna → astype(str) → str.strip().str.lower()`) and compared element-wise against class rules; the result is a `class_code` uint8 array the same length as the GeoDataFrame. This is an in-memory operation (~0.1 s) that avoids any join or dissolve.

3. **Rasterize to subgrid** — `rasterio.features.rasterize` is called once over all ~750k–2M polygons with `fill=0`, `all_touched=False`. The output is a county-sized uint8 array (e.g. 15,838 × 20,112 px = 319 Mpx for Rogaland, 22,911 × 32,502 px = 745 Mpx for Vestland). All three subgrids are live in memory simultaneously while rasterization overlaps; peak RAM is ~28 GB (including GeoDataFrame geometry overhead), well within the VDI's capacity.

4. **Merge into output** — as each county's future resolves, its subgrid is written into the combined raster block-by-block (512 × 512). For each block, the county array slice is checked for any non-zero pixels; only blocks with data are read-modify-written, preserving any previously written neighbouring county data. Empty blocks across county boundary gaps are never touched (SPARSE\_OK).

5. **Free memory** — each subgrid array is deleted immediately after its merge completes.

### Step 3 — FKB built overwrite pass

FKB Grønnstruktur is read with:
- A bounding-box spatial filter derived from the county boundaries (reprojected to EPSG:25833, the native CRS of FKB)
- An OGR SQL `WHERE klasse_navn IN ('FKB_bygg', 'FKB_vei', 'greyArea')` filter applied at the GDAL driver level

This returns 3,359,153 built polygons (~160 s to read from the national GDB). Null/empty geometries are dropped **before** reprojection to avoid transforming features that will be discarded. Remaining geometries are reprojected from EPSG:25833 → EPSG:25832.

Because the built polygons are spatially clustered in urban and road corridors, a **block-by-block sindex burn** is used rather than an in-memory subgrid: an R-tree spatial index is built over all 3.36M polygons, and for each 512 × 512 block within the data extent the index is queried for intersecting polygons. Blocks with no intersections (the majority, covering forests, marine, etc.) are skipped entirely. Only touched blocks are rasterized and written as overwrite (existing pixel values are replaced wherever `burnt != 0`).

This approach was preferred over the subgrid approach used for Grunnkart because the combined built-layer subgrid would be ~1.8 GB (full 3-county extent), whereas the sindex approach uses only a 512 × 512 working buffer.

### Step 4 — Overviews

Internal GDAL overviews at factors 2, 4, 8, 16, 32 are built while the file is still open, using **mode** resampling (appropriate for categorical data — takes the most frequent class within each 2×2 window).

---

## 6. Class Distribution (final output)

Across all three counties combined:

| Class | Name | Pixels | % of active |
|---|---|---|---|
| 1 | sand | 522,291 | 0.06% |
| 2 | rock | 914,277 | 0.10% |
| 3 | crop | 16,609,081 | 1.81% |
| 4 | forest | 162,290,631 | 17.66% |
| 5 | grassland | 9,798,295 | 1.07% |
| 6 | scrub | 157,350,143 | 17.13% |
| 7 | wetland | 13,649,144 | 1.49% |
| 8 | freshwater | 33,349,552 | 3.63% |
| 9 | marine | 340,226,487 | 37.03% |
| 10 | built | 8,756,572 | 0.95% |
| 11 | sparse | 154,519,308 | 16.82% |
| 12 | snow | 13,805,190 | 1.50% |
| 13 | other | 7,019,127 | 0.76% |
| **total active** | | **918,810,098** | **50.9% of grid** |

The dominance of marine (37%) reflects the extensive fjord and coastal waters within the three study counties. Forest, scrub, and sparse vegetation together account for ~52% of land surface. Snow (1.5%) is notable given the coastal range and mountain plateau terrain.

---

## 7. Key Design Decisions

**Vector-first reclassification.** Reclassifying in vector space before rasterization is faster and more memory-efficient than rasterizing all attributes to separate bands and then applying a per-pixel lookup. It also means the classification logic is in pure Python/numpy and easily auditable.

**Single combined raster with SPARSE\_OK.** A single file is simpler for downstream use than a VRT over three separate county tiles. GDAL's sparse block support means the physical file size (80.3 MB) reflects only active pixels — approximately the same size as three separate county files would be.

**Two burn strategies.** Grunnkart polygons form large, contiguous land-cover units that cover most of each county. Rasterizing the full county at once is fastest for dense, space-filling polygons. FKB built polygons are topologically dense in cities and roads but spatially sparse across the study area; the sindex block-by-block approach avoids allocating a full 1.8 Gpx working array for the built pass.

**No per-polygon boundary clipping.** Grunnkart is published per-county and already bounded by the county outline, so explicit intersection of 750k–2M polygons against the county boundary (a slow and topology-error-prone operation) is unnecessary. FKB built polygons are bounded at read time via the bbox filter.

**UInt8 dtype.** 13 classes + nodata fits comfortably in a single byte. At 10 m this keeps even a naively uncompressed file at ~1.8 GB; with ZSTD level 3 + the delta predictor the spatial autocorrelation of land-cover classes compresses the output to 80.3 MB.

**Parallel county rasterization.** The three county read→reclassify→rasterize pipelines are independent and run concurrently via `ThreadPoolExecutor`. `pyogrio` and `rasterio.features.rasterize` both release the GIL during their hot loops, giving real thread-level concurrency. The merge into the shared output file remains serial. Total runtime reduced from ~30 min to ~23 min.

---

## 8. Reproducing the Output

```bash
# Full run (~23 min on the Linux VDI)
~/myprojects/recover/.venv/bin/python scripts/rasterize_grunnkart.py

# Options
#   --data-root <path>   Override auto-detected NYVEST data root
#   --out <path>         Override output path
#   --resolution 10.0    Pixel size in map units (default 10)
#   --only rogaland      Process a subset of counties
#   --skip-built         Skip the FKB Grønnstruktur overwrite pass
#   --built-only         Re-run only the built pass on an existing output file
```

# Data sourcing + inference — current optimal approach

This is the practical "how do I get AlphaEarth embeddings and run the DNN over
them" reference. For model/training details see `DNN/README.md`. For per-source
benchmark details see `DNN/README.md`'s "Inference performance & wall-to-wall
county scaling" and "Inference data sourcing" sections — this file is the
condensed decision + script-path guide.

## End-to-end workflow (source.coop → county-clipped classified map + UQ)

This is the full pipeline as actually run end-to-end for the 3-county AOI
(2026-07-03), in order. Each step's script lives in `DNN/`.

```bash
PY=~/myprojects/recover/.venv/bin/python
BASE=/data/P-Prosjekter2/154001_nyvest/landcover_Geethen/data   # or any working dir
AOI=/data/P-Prosjekter2/154001_nyvest/GIS/Boundaries/nyvest_fylker.shp  # 3 counties

# 0. one-time: s5cmd (not in the venv; install via pixi or grab a release)
pixi global install s5cmd

# 1. one-time: download the AEF spatial index (~78 MB)
curl -o aef_index.parquet https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet

# 2. compute AOI bounds from the ACTUAL boundary vector, not from memory/an old
#    run. This step is the one that silently wrecked an earlier run of this
#    pipeline (see "AOI-bounds pitfall" below) — never hand-type bounds.
$PY -c "
import geopandas as gpd
b = gpd.read_file('$AOI').to_crs('EPSG:32633').total_bounds
print(f'{b[0]:.0f} {b[1]:.0f} {b[2]:.0f} {b[3]:.0f}')
"
# -> e.g. -99551 6466105 212826 7092052 for the 3-county nyvest AOI

# 3. find which raw AEF tiles cover those bounds/year (46 tiles for the 3-county AOI)
$PY DNN/fetch_aef_sourcecoop.py --index aef_index.parquet \
    --bounds -99551 6466105 212826 7092052 --bounds-crs EPSG:32633 --year 2024

# 4. bulk-download the whole tiles with s5cmd (0 errors, ~166 MB/s local disk;
#    slower to a CIFS-mounted P-drive, still fine). Path-style env vars are
#    mandatory (source.coop's proxy is path-style only).
export AWS_S3_FORCE_PATH_STYLE=true AWS_REGION=us-east-1 AWS_EC2_METADATA_DISABLED=true
s5cmd --no-sign-request --endpoint-url https://data.source.coop \
    cp -c 48 's3://tge-labs/aef/v1/annual/2024/32N/<tile>.tiff' "$BASE/aef_2024/32N/"
# repeat per tile/zone, or generate an s5cmd run-file (one `cp` line per tile from
# step 3's output) and `s5cmd run file.txt` — safer than a glob, which would
# pull every tile in the zone, not just the ones covering your AOI.

# 5. preprocess: dequant int8->float32, reproject 31N/32N->32633, flip north-up,
#    snap to a shared 10 m lattice. Idempotent (skips existing outputs), so
#    re-running after adding more raw tiles only processes the new ones.
$PY DNN/prep_aef_tiles.py --glob "$BASE/aef_2024/**/*.tiff" \
    --out-dir "$BASE/aef_2024_32633/" --workers 4

# 6. mosaic into one VRT
$PY DNN/build_vrt.py --glob "$BASE/aef_2024_32633/*.tif" --out "$BASE/aef_2024.vrt"

# 7. sanity-check the VRT actually covers the AOI BEFORE spending time on lidar/
#    inference. TWO checks — the first catches wrong bounds (extent), the second
#    catches the zone-overlap nodata bug (per-pixel coverage). Both silently
#    truncated earlier runs; always run both.
$PY -c "
import geopandas as gpd, numpy as np, rasterio
from shapely.geometry import box
from rasterio.features import geometry_mask
import rasterio.transform as RT
ae = rasterio.open('$BASE/aef_2024.vrt')
gdf = gpd.read_file('$AOI').to_crs(ae.crs)
aoi = gdf.union_all()
# 7a. extent: does the tile RECTANGLE cover the AOI? (catches wrong bounds)
missing = aoi.difference(box(*ae.bounds))
print(f'AOI area outside the AEF rectangle: {100*missing.area/aoi.area:.2f}%  (want ~0)')
# 7b. per-pixel: is the AE data actually PRESENT inside the AOI? (catches the
#     zone-overlap nodata-overwrite bug — a correct build_vrt.py gives ~88%+;
#     the buggy SimpleSource version gave 65%, the missing 23% being real data
#     clobbered by all-nodata adjacent-zone tiles along the UTM seam)
shp=(2000,1200)
b1 = ae.read(1, out_shape=shp)
valid = np.isfinite(b1) & (b1 != 0)
dt = ae.transform * rasterio.Affine.scale(ae.width/shp[1], ae.height/shp[0])
inside = geometry_mask(list(gdf.geometry.values), out_shape=shp, transform=dt, invert=True, all_touched=True)
print(f'AE-valid within AOI: {100*(valid&inside).sum()/inside.sum():.1f}%  '
      f'(want ~88%+; large fjords are genuine gaps)')
"
# 7a must print ~0.00% — if not, the bounds in step 2/3 were wrong; fix + re-download
# 7b should be ~88%+ — if it is ~65%, build_vrt.py painted nodata over data (needs
#    the ComplexSource/NODATA fix); rebuild the VRT before continuing

# 8. build the lidar raster (elevation, tri, tch) on the AE grid, parallel
$PY DNN/build_lidar_raster.py --like "$BASE/aef_2024.vrt" \
    --out "$BASE/lidar_3band.tif" --workers 8

# 9. build a lidar coverage mask for investigation/QA (1=lidar, 0=median-filled,
#    255=outside AOI) — load alongside classified_2024.tif to see which areas
#    are lidar-informed vs. median-filled
$PY DNN/build_lidar_coverage_mask.py --lidar "$BASE/lidar_3band.tif" \
    --out "$BASE/lidar_coverage.tif" --aoi "$AOI"

# 10. run inference, clipped to the AOI, with UQ. --uq-out is a STEM: it writes
#     three typed rasters (uq_2024_pcal.tif uint16 proba, uq_2024_setsize.tif
#     uint8, uq_2024_inset.tif uint8) — ~3x smaller than the old 21-band float32
#     stack (~10 GB vs ~31 GB) and each opens on its own. See "UQ output" below.
$PY DNN/predict_raster.py --in "$BASE/aef_2024.vrt" \
    --out "$BASE/classified_2024.tif" --uq-out "$BASE/uq_2024.tif" \
    --lidar-raster "$BASE/lidar_3band.tif" --aoi "$AOI" \
    --readers 6 --mask-allzero
#     ^ NO --scale: prep_aef_tiles.py output is already unit-norm float32.

# 11. write a manifest documenting every output (class legend, bands, encodings,
#     nodata) so the end user doesn't reverse-engineer the GeoTIFF tags.
$PY DNN/write_manifest.py --dir "$BASE" --out "$BASE/MANIFEST"
```

**Measured for the 3-county AOI (2026-07-03/04):** 46 tiles, ~104 GB raw
download (2 batches, ~2-3 min actual s5cmd transfer each), prep ~15 min (4
workers), lidar raster ~6.5 min (8 workers, 814 blocks), inference **~69 min**
for **817M classified px** (**I/O-bound on the P-drive CIFS mount, not
GPU-bound** — see "Inference throughput" below). Final grid: 44,612 × 75,014
(3.35B px), **88.4% AE-valid within the AOI** (the remaining ~11% is genuine
water — one large contiguous fjord/lake region plus scattered speckle). Lidar
covers ~56% of classified pixels (rest median-filled); see "Lidar coverage"
below.

> **Zone-overlap nodata bug (fixed 2026-07-04):** an earlier run of this exact
> pipeline classified only 601M px (65% AE-valid within the AOI) because
> `build_vrt.py` used plain `<SimpleSource>` mosaic entries. AEF ships one tile
> per UTM zone and each tile's footprint spills into the neighbouring zone's
> area as all-nodata; with no nodata masking, GDAL painted a later all-nodata
> tile OVER an earlier data-bearing tile along every 31N/32N seam, silently
> deleting ~23% of the real embeddings (large grey holes in the classified map,
> which look like source gaps but are not — AEF is a wall-to-wall product).
> `build_vrt.py` now emits `<ComplexSource>` + `<NODATA>` (+ band `<NoDataValue>`)
> when tiles have a nodata value, so valid data always wins an overlap; the
> P-drive path (nodata=None, all-zero gaps + `--mask-allzero`) is unchanged.
> Step 7b above is the check that catches this.

## AOI-bounds pitfall (read this before typing coordinates)

**Always derive AOI bounds from the actual boundary vector, never from memory,
an old run, or a hand-typed guess.** An earlier run of this pipeline used bounds
copied from an old note that only covered 39% of the true 3-county polygon (the
whole northern ~61%, 56,000 km², was silently never downloaded, never
classified — `classified_2024.tif` had large "missing" regions that looked like
a masking bug but were actually a wrong-extent bug). The AEF tile fetch, the
VRT, the lidar raster, and inference all inherit whatever rectangle you hand
them with no warning if it's wrong — nothing in the chain checks "does this
actually cover my AOI" except step 7 above, which is why that check exists and
should never be skipped.

The county/AOI boundary used here: `GIS/Boundaries/nyvest_fylker.shp`
(EPSG:25833, 3 features = Rogaland + Vestland + Møre og Romsdal, 92,045 km²
total). Individual county files also exist (`rogaland.shp`, `vestland.shp`,
`more_romsdal.shp`) if you need to predict/inspect one county at a time.

## Restricting predictions to an AOI (`predict_raster.py --aoi`)

AEF tiles are square/rectangular and the raw download's bounding rectangle
almost always overflows the actual study area — for the 3-county AOI, the tile
rectangle was **108,696 km² while the counties are only 92,045 km²** (67% of
the rectangle was ocean/neighbouring land even with CORRECT bounds, since
square tiles can't hug an irregular coastline). `predict_raster.py` has an
`--aoi <vector>` flag (any OGR format: shp/gpkg/geojson) that rasterizes the
polygon(s) per-block and skips pixels outside — output is `nodata` there, same
as an AE-nodata pixel. Reprojects the AOI to the raster's CRS automatically and
fails fast if the AOI doesn't intersect the input at all (near-certain
CRS/extent mistake).

Verified (2026-07-03): on a boundary-straddling test crop, `--aoi` produced
**zero classified pixels outside the polygon** and **100% class agreement**
with an unmasked run on the pixels inside — masking changes extent only, not
predictions. Re-verified on the full 3.35B-px 3-county run: 0 leaked pixels
across the whole grid.

You may see a `NotGeoreferencedWarning` from `rasterio.features._rasterize`
during a multi-reader run — this is a **benign thread-safety artifact in
Python's `warnings` module** (GDAL's C-level warning printing racing across the
`--readers` threads' warning-filter state), not a real georeferencing problem.
Reproduced deliberately: every individual thread's own recorded warnings are
empty and its mask output is bit-identical to a single-threaded reference call
on the same window. Confirmed harmless by the "zero classified px outside AOI"
check above, run on real output. Safe to ignore.

## Lidar raster derivation (elevation + TRI + tch)

`predict_raster.py --lidar-raster` needs a 3-band `elevation, tri, tch` raster
(band order matters — must match `LIDAR_COLS` in `predict_raster.py`/
`data_utils.py`) on the **exact same grid** as the AE input — same transform,
CRS, size; no internal resampling. `DNN/build_lidar_raster.py` builds this from
the raw P-drive lidar tiles:

- Raw tiles are 2-band (DTM, chm), float64, 3 m, EPSG:32633, millimetre-scaled
  (÷1000 → m; verified vs Copernicus GLO30). Two source dirs are merged, the
  larger/project-specific one winning a filename collision:
  `Nature_types_mapping/Features/lidar/` (130 tiles) +
  `Nature_types_mapping/Vestland_Moreromsdal_features/lidar/` (679 tiles) = 809
  deduped tiles.
- TRI (Riley 1999: mean abs diff of a pixel vs its 8 neighbours) is computed at
  the lidar's **native 3 m resolution** (matches training), then all 3 bands are
  reprojected/resampled onto the target 10 m grid with `Resampling.average`.
- **Parallelized**: the target grid is split into blocks computed across a
  `ProcessPoolExecutor` (`--workers`, default = cpu count), with the parent
  process serializing GDAL writes. TRI is computed in float32 (bandwidth-bound;
  halves the memory traffic vs float64, error ~1e-4 m — negligible against
  metre-scale elevations). This is a ~13× speedup over an earlier
  single-threaded float64 version (profiled: TRI was 73% of per-block time,
  memory-bandwidth-bound over 8 neighbour passes) — verified bit-for-bit
  equivalent aside from float32 rounding (0 NaN-pattern mismatches, max abs
  diff 2.4e-4 m over 207M compared pixels). A 44612×75014 (3.35B px) grid takes
  **~6.5 min on 8 cores** (814 blocks); scale roughly linearly with pixel count
  and inversely with `--workers`.
- Only blocks that overlap the lidar footprint do any work; the rest are left
  NaN and median-filled by `predict_raster.py` at inference (median comes from
  the training set, saved in the model checkpoint — safe fallback, not a
  failure mode).

## Lidar coverage: what fraction of the AOI actually has real lidar?

**Answer (3-county AOI, verified 2026-07-03): 56.1%** of classified pixels get
real lidar; the other 44% are median-filled. Build this number yourself with
`DNN/build_lidar_coverage_mask.py --lidar lidar_3band.tif --out
lidar_coverage.tif --aoi nyvest_fylker.shp` — it reads band 1 (elevation) of the
**already-built** `lidar_3band.tif` (not a re-derivation from raw tiles), so it
always reflects what `predict_raster.py` will actually use, and writes a
single-band uint8 raster (`1`=lidar, `0`=no-lidar/median-filled,
`255`=outside-AOI) you can load in QGIS next to `classified_2024.tif` to see
exactly where coverage is thin.

**Do not estimate coverage from a decimated raster read** (e.g. `.read(1,
out_shape=(small, small))` for a quick check) — this was tried and gave **54-60%
too, but for the wrong reason and against the wrong (pre-bounds-fix) grid**, and
separately, decimating a raster with sparse/patchy valid data by ~10-25× can
bias the *apparent* coverage in either direction because each output cell
mixes many source sub-pixels. Always measure coverage at full resolution
(or verify a decimated estimate against a full-res spot-check) before reporting
a number. The three numbers that appeared during development of this pipeline —
81.6%, 54.1%, 56.1% — were not three independent measurements of the same
quantity: the first two were computed against an incomplete AEF grid (see
"AOI-bounds pitfall"), and only 56.1%, computed after the bounds fix, is the
real figure for the full 3-county AOI.

## Inference throughput: I/O-bound on the P-drive, not GPU-bound

`predict_raster.py`'s pipeline (reader threads → GPU → writer) is designed to
be decompression-bound on local disk (DATA_INFERENCE's original benchmark:
~1.4 M px/s with `--readers 6` on local SSD). **Against the CIFS-mounted
P-drive (`//netapp3-cifs.nina.no`), throughput measured ~0.2 M px/s** for the
3-county run (601M px in 3282s) — 7x slower. Confirmed I/O-bound, not a code
regression: `nvidia-smi` sampled repeatedly during the run showed the GPU at
**0% utilization** the entire time, and a single 2048×2048×64-band read off the
P-drive VRT benchmarked only 130-183 MB/s (vs. local-disk speeds the original
benchmark assumed). Running a second CIFS-heavy job concurrently (e.g. a
coverage-verification script re-reading raw lidar tiles) measurably worsens
this further — avoid launching competing P-drive-reading jobs during a live
inference run.

If this becomes a bottleneck worth fixing (current: accepted as-is), the
options in order of effort: (1) clip the AE VRT + lidar raster to the AOI
bounding box before inference (no code change, cuts I/O volume by however much
of the rectangle is outside the AOI — for this run that would have been ~67%
less to read); (2) stage `aef_2024_32633/*.tif` + `lidar_3band.tif` to local
`/home` disk before predicting, predict from there (avoids CIFS during the hot
loop, costs ~85 GB local disk); (3) profile whether more `--readers` helps
against CIFS specifically (untested — the original readers-scaling benchmark
was against local disk, so it may not transfer).

## When the P-drive doesn't have the AlphaEarth tiles you need

**Local/mosaicked AlphaEarth tiles already on the P-drive → `predict_raster.py`
directly, read-infer-write, no separate download step.** This is the fast path
whenever the AOI is already covered by the existing P-drive tiles.

```bash
PY=~/myprojects/recover/.venv/bin/python

# single tile or a VRT mosaic over many tiles
$PY DNN/predict_raster.py \
    --in /data/P-Prosjekter2/154001_nyvest/Nature_types_mapping/Features/embeddings/embeddings_<id>.tif \
    --out classified.tif \
    --uq-out uq.tif \
    --lidar-raster lidar_10m.tif \
    --scale 1000 --readers 6 --block 2048
```

- **`--scale 1000` is required for P-drive tiles** — they're packed ~1000×
  larger than the unit-norm embeddings the model trained on (verified: L2 norm
  ~996 in the raster vs 1.0 in training data). Omitting it silently produces
  garbage predictions. Not needed for GEE-direct or geedim output (already
  unit-norm float64); source.coop/`prep_aef_tiles.py` output needs its own
  dequantization (see below), not `--scale`.
- **154 P-drive tiles exist but don't cover the full 3-county AOI** (only
  ~15%, mostly Rogaland) — see "End-to-end workflow" above for pulling the rest
  from source.coop.
- **Mosaic many tiles first** with `DNN/build_vrt.py` so inference runs once
  instead of per-tile:
  ```bash
  $PY DNN/build_vrt.py \
      --glob '/data/P-Prosjekter2/154001_nyvest/Nature_types_mapping/Features/embeddings/*.tif' \
      --out embeddings.vrt
  $PY DNN/predict_raster.py --in embeddings.vrt --out classified.tif \
      --scale 1000 --readers 6 --mask-allzero
  ```
  **`--mask-allzero` is required for VRT mosaics** — `build_vrt.py` writes no
  `NoDataValue`, so the gaps between tiles read as all-band-zero and, without
  this flag, get confidently classified (a real AE embedding is unit-norm, so
  all-zero is never valid data). Harmless on a single fully-covered tile.
- **`--readers 6`** on the 8-core VDI gives ~2.8× over 1 reader on local disk
  (measured 1.4 M px/s vs 0.5 M px/s) — decompression-bound there, not
  GPU-bound. Against a network mount this benchmark does not hold (see
  "Inference throughput" above).
- **Skip `--uq-out`** unless you need calibrated proba / conformal sets — even
  after the size fix it's the bulk of the output (three typed rasters, ~10 GB at
  county scale, vs. ~60 MB for the int16 class map). See "UQ output" below for
  the layout.
- **`--aoi <vector>`**: restrict predictions to a study-area polygon, e.g. when
  the input rectangle overflows the true AOI (see above).
- Lidar (`--lidar-raster`): see "Lidar raster derivation" above. Omitting it
  entirely falls back to training-set medians, which is safe but less accurate.

## Pulling more AlphaEarth data from source.coop

**The winner is: bulk-download whole tiles with `s5cmd`, preprocess with
`prep_aef_tiles.py`, build a local VRT, then predict against local files.** Do
NOT stream per-window through GDAL/`vsicurl` against source.coop — that specific
access pattern trips the proxy's rate limiter (see "Why not GDAL streaming"
below). See "End-to-end workflow" above for the full command sequence,
including the critical AOI-bounds-from-vector step.

**Disk watch:** a single raw tile is up to 3.4 GB; the 3-county AOI needed 46
tiles (~104 GB compressed). `prep_aef_tiles.py` output is comparable in size.
On a box with limited free space, download → prep → predict → **delete the raw
tile** in a per-tile loop caps peak disk. `prep_aef_tiles.py` is idempotent
(skips existing outputs), so it's safe to re-run after adding more raw tiles —
only new ones get processed.

Key facts about the source.coop tiles (all handled by `DNN/prep_aef_tiles.py`;
raw-tile constants live in `DNN/fetch_aef_sourcecoop.py`):
- Data is **int8-quantized** — dequantize with `((v/127.5)**2)*sign(v)`
  (−128 = nodata → NaN). Do NOT skip this: raw int8 values fed to the model
  directly are garbage. `prep_aef_tiles.py` applies this via a 256-entry LUT
  indexed by `raw+128` (1.37× faster than the elementwise formula, and it
  matters at 4.3e9 elements/tile; verified bit-exact vs the reference formula).
  `predict_raster.py` still has no AEF path — dequant happens in the prep step,
  not inference, so `predict_raster.py --in` must be a *prepped* tile/VRT.
- Source tiles are in native **UTM zones 31N/32N** (a 3-county Norwegian AOI
  spans both), not the project's working CRS (32633), and are **bottom-up**
  (row 0 = south). `prep_aef_tiles.py` reprojects + flips with an in-process
  `rasterio.vrt.WarpedVRT` (nearest resampling, so dequant-after-warp is exact)
  and **snaps each tile's origin to a shared 10 m lattice** — without the snap,
  `calculate_default_transform` picks each tile's origin independently and the
  tiles don't share a pixel grid, which `build_vrt.py` requires. Do NOT use the
  shipped static per-tile `.vrt` (its warp machinery is slow — 41s for a
  512×512×64 window vs 2–4s for the raw `.tiff`).
- `s5cmd` requesting large contiguous byte-range parts (`-c 48`) gets **0
  HTTP 500s**; download to `/home` or the P-drive, not `/tmp` (a VDI `/tmp` may
  be only a few GB, smaller than one raw tile). Prefer downloading the *exact*
  tile keys from `fetch_aef_sourcecoop.py`'s output (an s5cmd run-file, one `cp`
  per tile) over a zone-wide glob (`2024/32N/*.tiff`), which pulls every tile in
  that UTM zone globally, not just the ones covering your AOI.
- `source.coop's` proxy is **path-style only**: `AWS_S3_FORCE_PATH_STYLE=true`
  + `AWS_REGION=us-east-1` + `AWS_EC2_METADATA_DISABLED=true` are mandatory —
  without them s5cmd's SDK uses virtual-host addressing
  (`tge-labs.data.source.coop`, doesn't resolve) and returns bogus 404s.

**Why not GDAL streaming:** source.coop is a Cloudflare Worker proxy in front
of cloud storage. GDAL/`vsicurl` reading a COG window issues many small,
scattered per-band-block range requests (64 bands × N blocks across the
BAND-interleaved file), and the proxy **rate-limits that pattern with HTTP
500s** — measured 50–100 500s reading a *single* 2000×2000×64 window, every
run, un-tunable (varying multiplex / threads / band-batch / sub-window
granularity did not help). `s5cmd` requesting large contiguous byte-range parts
gets **0 500s** at concurrency up to 64, so the trigger is GDAL's request
*pattern*, not raw concurrency. If you must retry GDAL against the proxy,
`GDAL_HTTP_MAX_RETRY=5` + `GDAL_HTTP_RETRY_DELAY=1` are mandatory (the baseline
env in `fetch_aef_sourcecoop.py` has no retry config and crashes on the first
500 with a misleading `ZSTDDecode: Unknown frame descriptor`).

**Full-AOI throughput (measured 2024; per-window streaming figures are
throttle-dependent and NOT reproducible — treat as best-case-only):**

| option | throughput | notes |
|---|---:|---|
| **s5cmd bulk whole-tile GET** | **~166 MB/s @ c=48–64, 0 errors** | winner; robust, reproducible |
| source.coop per-window GDAL streaming | ~184k px/s *best day*, retry-bound on a throttled day | unreliable — proxy throttles the pattern |
| geedim (`max_requests=100`) | 98.3k px/s | not in-repo, `pip install geedim`; serves analysis-ready float64 (no dequant) |
| GEE `computePixels` hand-rolled | ~12k px/s | superseded, see below |

## Future work: predicting the same AOI repeatedly

The download → `prep_aef_tiles.py` → VRT → predict path above is optimal for a
**one-off** wall-to-wall pass. If you will re-read the same AOI many times
(retraining loops, multiple years, tiling/blocking experiments), the prepped
per-tile GeoTIFFs are not the ideal long-lived layout — they're
ZSTD/deflate-compressed, band-interleaved-ish, and each re-read re-decodes the
whole footprint. The higher-leverage move (per the Earthmover cloud-native
array posts) is a **one-time rewrite into a local Zarr / Icechunk store** you
own: dequant + reproject baked in, pixel-interleaved ~3–15 MB chunks, on `/home`
or the P-drive. Then `rioxarray`+`dask` (or a plain `xarray`/`xbatcher` →
PyTorch dataloader) reads it with async chunk prefetch and no proxy in the loop.
Only worth it if you re-read the AOI — for a single pass, converting to Zarr
just to read once and discard is pure overhead. `s5cmd` + `prep_aef_tiles.py`'s
dequant/reproject front-end is reusable as the ingest stage of that pipeline.
(`zarr`/`icechunk`/`s3fs`/`obstore` are NOT yet in the venv — `dask`/`xarray`/
`rioxarray`/`fsspec` are.) The CIFS I/O bottleneck (see "Inference throughput")
would also be a strong argument for staging data locally in this scenario.

## Running inference once you have the data

```bash
PY=~/myprojects/recover/.venv/bin/python

# raster tiles / VRT / COG URL (local, source.coop, GCS, S3 all supported via rasterio)
$PY DNN/predict_raster.py --in <ae_tile_or_vrt_or_url> --out classified.tif \
    [--uq-out uq.tif] [--lidar-raster lidar_3band.tif] [--aoi study_area.shp] \
    [--scale 1000]   # only for P-drive-style pre-scaled tiles
    [--readers 6] [--block 2048] [--mask-allzero]  # --mask-allzero for VRT mosaics

# tabular / parquet points (columns A00..A63 [+ elevation,tri,tch])
$PY DNN/predict.py --in points.parquet --out preds.parquet
```

`predict.py` always writes `pred_class` + UQ columns (`pcal_{c}` calibrated
proba, `set_size`, `inset_{c}` conformal-set membership) using
`models/dnn_final.pt` + `models/dnn_final_calib.npz`. See `DNN/README.md` for
the calibration-method comparison (Venn-Abers won over temperature scaling) and
the LAC+Mondrian conformal-set recipe.

## UQ output (`--uq-out` = a stem, three typed rasters)

`predict_raster.py --uq-out uq.tif` writes THREE files, not one, because the UQ
components have different value domains and packing them all as float32 wasted
~3x the disk (a 3-county run was ~31 GB as one 21-band float32 stack; the split
is ~10 GB). Each opens independently and is self-describing (band descriptions +
nodata set):

| file | dtype | bands | contents | decode |
|---|---|---|---|---|
| `uq_pcal.tif` | uint16 | C (10) | per-class calibrated probability, scaled ×60000 | `proba = value / 60000` (∈[0,1]; bands sum ~1) |
| `uq_setsize.tif` | uint8 | 1 | conformal set size 0..C (1 = confident) | value as-is |
| `uq_inset.tif` | uint8 | C (10) | per-class 0/1 conformal-set membership | value as-is |

Nodata: `65535` (pcal), `255` (setsize/inset). The uint16 proba is lossless to
~1e-5 (max round-trip error 8e-6, well under the calibration error itself); the
argmax over `uq_pcal` matches the `classified` map's raw ensemble prediction on
~97% of pixels (the rest are pixels where Venn-Abers recalibration nudges the
winning class — expected, not a bug). Band descriptions carry the class code
(`pcal_c12`, `inset_c11`, …); codes 1 and 9 never appear (training merges 1→2,
9→8). `DNN/write_manifest.py` writes a `MANIFEST.json`/`.md` documenting all of
this (class legend, per-file bands/dtype/nodata/scale) next to the outputs.

## Alternative approaches tested (and why they lost)

| script / approach | what it does | verdict |
|---|---|---|
| GDAL/`vsicurl` per-window streaming from source.coop (all variants: single-big-read, 4-sub-window streaming, band-batched, `GDAL_NUM_THREADS=1`, multiplex on/off) | read COG windows directly off the proxy via `rasterio`/`WarpedVRT` | **lost — proxy throttles the scattered-range pattern with 50–100 HTTP 500s per window, un-tunable.** Superseded by s5cmd whole-tile download |
| `s5cmd cat`/`cp` whole-tile GET | large contiguous byte-range parts (`-c 48 -p 4`) instead of GDAL's per-band-block scatter | **winner** — 0 HTTP 500s across many runs at concurrency 8–64, ~166 MB/s. Requires the path-style env vars (proxy is path-style only) |
| geedim (`ee.Image.gd` accessor) | `prepareForExport` + `toGeoTIFF(max_requests=N)`; thread-pools GEE's `computePixels` internally | easiest to wire up (no UTM-zone reprojection or int8 dequantization needed), but ~1.7–1.9× slower than source.coop streaming; good for a quick one-off test pull, not production |
| GEE `ee.data.computePixels`, hand-rolled tiling | direct synchronous pixel API calls, manually split into ≤290×290px/64-band requests to stay under the 48 MB/request cap | ~12k px/s, ~6 hr for a full-AOI single-threaded pull — superseded by geedim's internal thread-pooling of the same API |
| `Export.image.toCloudStorage` / `toDrive` | GEE's async server-side-tiled batch export, no 48 MB cap | not benchmarked (queues as a background job, minutes–hours latency) — unsuitable for interactive iteration |
| GCS direct access (`/vsigs/alphaearth_foundations`) | read the AEF bucket directly instead of via source.coop's mirror | blocked — bucket is requester-pays, needs a real billing-linked GCP project |
| static per-tile `.vrt` (source.coop's shipped VRT) | corrects the raw tiff's bottom-up row order and reprojects via a pre-built `VRTWarpedDataset` XML file | works but slow — 41s for a 512×512×64 window vs 2–4s reading the raw `.tiff` directly. Use `rasterio.vrt.WarpedVRT` in-process instead |
| `DNN/build_vrt.py` | mosaics many aligned single-grid tiles into one virtual raster (plain XML `VRTDataset`, no gdalbuildvrt CLI needed); emits `<ComplexSource>`+`<NODATA>` when tiles carry a nodata value so overlapping tiles don't clobber each other, plain `<SimpleSource>` when they don't | current best practice for consolidating many tiles into one `predict_raster.py --in` target. The nodata masking is REQUIRED for AEF tiles (adjacent UTM zones overlap with all-nodata margins — see "Zone-overlap nodata bug"); harmless for non-overlapping P-drive tiles |
| `DNN/prep_aef_tiles.py` | dequant int8→float32 (LUT, nodata→NaN) + reproject 31N/32N→32633 + flip north-up + snap to a shared 10 m lattice, windowed (~0.5 GB/worker) | **the required preprocessing** for raw source.coop AEF tiles before `build_vrt.py`/`predict_raster.py`. Output is unit-norm (no `--scale`) |
| single-threaded float64 `build_lidar_raster.py` | original lidar-raster derivation, one process, TRI in float64 | **superseded** — ~13× slower than the current parallel/float32 version for identical output (0 NaN-pattern mismatches, max abs diff 2.4e-4 m). Kept only as a historical note; the current script IS the fast version |
| estimating lidar coverage from a decimated raster read | quick `.read(1, out_shape=(small,small))` + `isfinite().mean()` | **unreliable** — biased by resampling of sparse/patchy data at ~10-25× decimation. Use `build_lidar_coverage_mask.py` at full resolution instead |

## Verification done

- P-drive tile vs. source.coop tile, same 3km×3km crop, both dequantized/scaled
  correctly and reprojected to a shared pixel grid: raw embeddings show 0.96
  mean per-band correlation, both correctly unit-norm (L2 ≈ 1.0), 92% pixel-
  level class-map agreement. Residual differences concentrate in known
  spectrally-confusable vegetation classes and are consistent with the two
  sources being different temporal composites (P-drive multi-year mosaic vs.
  source.coop single-year 2024 snapshot), not a scaling/transform bug.
- `prep_aef_tiles.py`: dequant LUT verified **bit-exact** vs the
  `fetch_aef_sourcecoop.py` reference formula on all 254 valid int8 values;
  synthetic bottom-up UTM32N tile round-trips to L2≈1.0 unit-norm, flips
  correctly, and nodata→NaN maps 1:1 to `predict_raster.py`'s nodata-class.
  Windowed peak ~0.5 GB/worker vs ~50 GB for a naive whole-tile load.
- `build_lidar_raster.py` (parallel/float32 rewrite): verified against the
  original single-threaded float64 version over the full 3-county grid —
  207M+ compared pixels, 0 NaN-pattern mismatches, max abs diff 2.4e-4 m
  (float32 rounding only). ~13× faster (164s vs ~36 min on an earlier,
  smaller grid; ~6.5 min for the full 3.35B-px 3-county grid on 8 cores).
- `predict_raster.py --aoi`: verified on a boundary-straddling crop (zero
  leakage outside the polygon, 100% class agreement inside vs. unmasked) and
  re-verified on the full 3-county production run (0 classified px outside
  the AOI across 3.35B pixels).
- Full AOI-bounds-from-vector → download → prep → VRT → lidar → coverage mask
  → AOI-clipped inference(+UQ) chain run clean end to end for the 3-county AOI
  (46 tiles, 3.35B px grid, **817M classified px**, split typed UQ (uint16 proba + uint8 setsize/inset), ~56% lidar
  coverage, 88.4% AE-valid within AOI, 0 classified px outside AOI). Two failure
  modes found and fixed in practice, each with an explicit checkpoint (step 7a/7b
  above): (1) wrong AOI bounds silently truncating the extent, (2) `build_vrt.py`
  painting all-nodata adjacent-zone tiles over real data along the UTM seam
  (the ComplexSource/NODATA fix). Both looked like "missing data" in the map and
  neither raised an error — hence the two coverage checks are mandatory.

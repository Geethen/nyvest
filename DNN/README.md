# DNN replacement for the CatBoost + TabICL pipeline

A GPU DNN classifier for the nyvest land-cover task, built to replace the
two-stage CC-APS CatBoost → TabICL model. Evaluated on the **exact same
protocol** as `common_ground/scripts/llto_schemeB_allyears.py` so macro-F1 is
directly comparable:

- stable-allyears parquet (663,740 rows, 12 classes)
- class merge 1→2 and 9→8 (⇒ 10 effective classes)
- 3-fold spatial CV via `GroupKFold` on `cell_id`
- macro-F1 over all classes (`zero_division=0`)
- 64 AlphaEarth bands + 3 lidar features (elevation, tri, tch)
- test-fold labels never cleaned/relabelled (leak-free)

**Reference to beat: TabICL+lidar macro-F1 = 0.7139.**

## Results (3-fold spatial CV macro-F1)

| Stage | Model | F1 mean | std | vs 0.7139 | wall time |
|------|-------|--------:|----:|----------:|----------:|
| 1 | plain 2-layer MLP, class-weighted CE | 0.7112 | 0.009 | −0.0027 | ~60 s |
| 2 | residual + BatchNorm MLP, SWA, ensemble | 0.7106 | 0.007 | −0.0033 | ~610 s |
| 3 | robust MLP (mixup + input noise + 5-seed ensemble + cls12 clean) | 0.7283 | 0.009 | +0.0144 | ~430 s |
| 3 (final) | MLP + 5-seed ensemble + cls12 clean (mixup/noise off) | 0.7318 | 0.010 | +0.0179 | ~270 s |
| **8 (best)** | **+ targeted cls12-only relabel (to12_fix)** | **0.7341** | 0.009 | **+0.0202** | ~370 s |

**Key finding 1 — capacity is not the lever.** Stage 2's residual net drove
val-F1 to 0.965 while test-F1 stayed at 0.71: depth/BatchNorm memorizes the train
fold and does not transfer across spatial (cell_id) folds. The bottleneck is
*spatial generalization*, not expressiveness.

**Key finding 2 — ensembling is the lever; augmentation is not.** The Stage-3
ablation (drop one lever at a time from the 0.7283 config):

| variant | F1 | Δ vs 0.7283 |
|---|---:|---:|
| no mixup | 0.7289 | +0.0006 |
| no input-noise | 0.7297 | +0.0014 |
| no cls12 clean | 0.7279 | −0.0004 |
| single seed (no ensemble) | 0.7248 | −0.0035 |
| **no mixup + no noise** | **0.7318** | **+0.0035** |

mixup and input-noise slightly *hurt* here — dropping both is best. The reliable
gain is the **5-seed probability ensemble** (single-seed loses 0.0035). cls12
centroid cleaning is macro-F1-neutral but kept for label correctness and
leak-safety. The final/default config is therefore the simplest one: a 2-layer
class-weighted MLP, 5-seed ensembled, with train-only cls12 cleaning.

## Stage 4 — conformal (CC-APS) pseudo-labelling from the unstable parquet

Self-training: per fold, train the DNN ensemble on stable-fit, calibrate an APS
gate on a held-out stable-cal slice (α=0.05, same as the reference), pseudo-label
the unstable rows whose APS set is a **singleton**, then retrain on stable +
pseudo. Leak-free (test fold never touched).

**Result: F1 = 0.7320 ± 0.009 — a tie with 0.7318 (+0.0002, within fold noise).**
The gate works (keeps ~31% of unstable rows at ~89% pseudo-label accuracy), but
adding ~5k pseudo rows to an already-664k-row training set is a rounding error.
This is the opposite of TabICL, where support is capped at 32k so each added row
counts. The unstable set is also small (16.8k rows) and has **no class 12**, so
it can't help the hardest class. **Verdict: no-op for the DNN; not adopted.**

## Architecture sweep (Stage 5) and MoE (Stage 6) — all lose to the plain MLP

Same robust recipe (5-seed ensemble, sqrt weights, label smoothing, cls12
clean), only the architecture changes:

| Architecture | F1 mean | vs MLP best (0.7318) |
|---|---:|---:|
| **plain MLP** | **0.7318** | — |
| GLU / GEGLU gated MLP | 0.7288 | −0.0030 |
| MoE, 4 soft experts (stage6) | 0.7294 | −0.0024 |
| residual GLU | 0.7206 | −0.0112 |
| SNN (SELU + AlphaDropout) | 0.7075 | −0.0243 |

More expressive nets overfit the spatial folds (residual-GLU val-F1 0.88,
test 0.72) — same lesson as Stage 2. The MoE did **not** specialize its way to
better grassland/sparse-veg; problem classes were flat-to-worse. Capacity is not
the lever.

## Spatial MoE (Stage 7) — experts specialized by GEOGRAPHY also lose

`stage7_spatial_moe.py`: experts route by location, not class. Two modes —
`soft` (gating MLP on lon/lat → softmax) and `hard` (KMeans regions on train
coords, leak-free, one expert per region). Same robust recipe otherwise.

| Config | F1 mean | std | vs MLP best (0.7318) | gate behavior |
|---|---:|---:|---:|---|
| **plain MLP** | **0.7318** | .010 | — | — |
| class-MoE E=4 | 0.7294 | .008 | −0.0024 | — |
| soft spatial E=4 | 0.7280 | .011 | −0.0038 | 3-4 experts shared |
| soft spatial E=8 | 0.7221 | .008 | −0.0097 | ~5 shared |
| hard KMeans E=4 | 0.7166 | .012 | −0.0152 | ~1 region/fold |
| hard KMeans E=8 | 0.7092 | .005 | −0.0226 | ~1 region/fold (worst) |

**All lose; soft > hard; more experts = worse.** The gate-usage diagnostic
explains hard's collapse: each spatial-CV test fold (GroupKFold on cell_id) falls
almost entirely inside ONE KMeans region, so the gate routes ~84-95% of the fold
to a single expert (e.g. E=8 fold0→expert0 @0.839, fold1→expert1 @0.952). The
other experts never see the test region and are dead weight, leaving one
under-trained expert to predict the whole fold. **Hard spatial experts are
redundant with, and antagonistic to, the geographic CV.** Soft routing blends and
degrades gracefully but still can't beat a single model that already sees all
regions. Confirms again: the bottleneck is feature-space class overlap, not a
routing/capacity problem — neither class-experts nor region-experts help.

## Feature attention (Stage 9) — also loses, but confirms the key features

With 67 covariates, attention is tempting. Two forms tested:
FT-Transformer feature self-attention (Stage 5, partial run trended <0.7318) and
lightweight feature GATING (`stage9_feat_attn.py`: squeeze-excite + competitive
softmax channel attention in front of the best MLP).

| config | macro F1 | std | vs plain MLP (0.7318) |
|---|---:|---:|---:|
| plain MLP | 0.7318 | .010 | — |
| SE feature-attention | 0.7283 | .010 | −0.0035 |
| softmax feature-attention | 0.7281 | .009 | −0.0037 |

**Both lose** — feature attention genuinely doesn't help (not just an FT-is-heavy
artifact). The 64 AlphaEarth bands are ALREADY a transformer-derived embedding, so
re-attending across features is redundant; the plain MLP's first layer already
learns feature importance implicitly, and explicit gating just adds overfit
variance on the spatial folds.

**But the learned gate weights are informative and corroborate `lidar-features`:**
top-weighted features are **tri, tch, elevation** (the 3 lidar terrain features) +
bands A10/A14/A11/A37/A48; lowest are A55/A56/A43/A26/A04. Attention independently
rediscovered that lidar terrain is the most valuable signal — matching the
CatBoost-importance finding by a completely different method. Acting on it
(reweighting) doesn't help; the MLP already uses those features.

## Class-wise learning curves — problem classes are confusion-bound, not data-starved

`learning_curves.py` / `lc_point.py` (parallel) / `lc_aggregate.py` train the
best recipe on 5%→100% of fold-0 train and plot per-class test F1 vs train size
(`learning_curve.png`). Reading the slope over the last data doubling:

- **Macro F1 saturates by ~70% (310k rows): 0.7326 → 0.7309 at 100%** — a tiny
  dip. More data of the same distribution does **not** help.
- **Grassland (5) and sparse-veg (2) — the two worst classes — are dead flat**
  (5: 0.578→0.585; 2: 0.481→0.495, even dips at 100%). They are
  **confusion-bound**: the ceiling is spectral separability, not label volume.
- open-upland (6), mire (7), forests (3/4), water/bare/built all saturated early.
- **Snow/ice (12)** climbs steeply to ~40% (177k) then peaks/declines — its issue
  is rarity + stale-label noise, fixed by centroid cleaning, not bulk data.

**Implication:** gains must come from better FEATURES that separate the
vegetation classes (lidar already proved this lever) or targeted relabeling of
noisy classes — not more data, and not a bigger/fancier network. This is also
why every alternative architecture (GLU/SNN/FT/MoE) failed: you cannot model past
genuine feature-space overlap.

## Targeted class-12 relabel — NEW BEST (Stage 8)

cls12 (snow/ice) is the ONE noise-limited class, and its noise is static
mislabelling: 29/146 locations are suspect in *every* year 2017-2025 (no temporal
trend) — permanently mislabelled rock/debris polygons, not deglaciation. Blanket
relabel banks cls12 but wrecks grassland/mire (−0.0065 macro). So
`stage8_cls12_relabel.py` applies leak-free perfold corrected labels to cls12-
touching rows ONLY:

| config | macro F1 | cls12 | grassland/5 | mire/7 |
|---|---:|---:|---:|---:|
| best (centroid clean) | 0.7318 | 0.702 | 0.599 | 0.691 |
| cls12_fix (recode bad ice) | 0.7332 | 0.724 | 0.598 | 0.690 |
| **to12_fix (also recover→12)** | **0.7341** | **0.726** | 0.599 | 0.691 |

**to12_fix is the new best: +0.0023 macro, +0.024 cls12, grassland/mire untouched**
— exactly the surgical win the diagnostics predicted. Caveat: cls12 is rare and
fold-2 (207 test rows) is volatile (cls12 there *dropped* to 0.61 while folds 0/1
rose to 0.77-0.79), so the cls12 gain is real-but-noisy; macro std unchanged
(0.009). Headroom beyond this needs external ground truth (Sentinel-2 NDSI
persistence / NVE glacier inventory), parked.

## Per-class label-noise diagnosis (leak-free)

Four tests separate *label noise* (fixable by relabeling) from *feature overlap*
(irreducible) per class. All leak-free, reusing the per-fold cleanlab artifact
(`clean_labels_perfold.npz`, test labels never touched). Scripts:
`noise_diagnostics.py` (A/B/C), `noise_clean_gain.py` (D).

- **A. Transition matrix** — est. label-noise rate per class (1 − P(true=c|obs=c)).
- **B. Confident-disagreement** — % of a class's rows the DNN confidently contradicts.
- **C. Noise-corrected ceiling** — how much low F1 is just noisy scoring.
- **D. Clean-vs-noisy F1 gain** (decisive) — retrain on leak-free corrected labels;
  per-class F1 Δ. Rise ⇒ noise-limited; flat/drop ⇒ confusion-bound.

| class | F1 | noise% (A) | confDis% (B) | relabel Δ (D) | verdict |
|---|---:|---:|---:|---:|---|
| sparse-veg/2 | 0.55 | 19% | 11% | −0.00 | confusion-bound |
| **grassland/5** | 0.60 | 20% | 9% | **−0.022** | **confusion-bound** (relabel HURTS) |
| open-upland/6 | 0.66 | 19% | 10% | +0.00 | confusion-bound |
| mire/7 | 0.69 | 19% | 4% | **−0.032** | confusion-bound (relabel HURTS) |
| **snow-ice/12** | 0.70 | 13% | 15% | **+0.037** | **NOISE-LIMITED** → cls11 |
| forest/3,4 | 0.74-0.79 | 14% | 4% | −0.01 to −0.02 | confusion-bound |
| bare/10, water/8 | 0.85-0.95 | 2-7% | 1-3% | ~0 | clean & easy |

**Verdict: label noise is a real, fixable problem for ONE class — snow/ice (12)**,
whose labels are systematically stale (glacier/snow → bare-rock/infrastructure,
both noise and model arrows point to class 11). Centroid cleaning already
exploits this. For the worst classes (grassland 5, sparse-veg 2, mire 7), noise
is NOT the issue — cleanlab relabeling makes them *worse* (it recodes genuine
grassland as forest because they overlap), confirming the flat learning curves:
they are confusion-bound. **Bulk relabeling hurts the DNN macro-F1 (−0.0065)** —
note this differs from TabICL where perfold_fix gave +0.007, because the DNN
trains on the full 664k rows with class weighting + ensembling and is already
more noise-robust than the 32k-support TabICL.

## Files

- `data_utils.py` — shared loader, spatial-fold splitter, cls12 centroid
  cleaning, macro-F1. Mirrors the reference pipeline exactly.
- `stage1_mlp.py` — plain MLP baseline.
- `stage2_resmlp.py` — residual+BN MLP (capacity probe; no gain).
- `stage3_robust_mlp.py` — **the model** (winning config). Knobs are env vars.
- `ablation_one.py` / `run_ablation.sh` — Stage-3 lever ablation.
- `conformal_utils.py` + `stage4_pseudo.py` — CC-APS pseudo-labelling (no-op).
- `stage5_arch.py` — architecture sweep (glu/residual_glu/snn/ft via ARCH env).
- `stage6_moe.py` — soft Mixture-of-Experts (added by subagent).
- `learning_curves.py` (serial) / `lc_point.py` + `lc_aggregate.py` (parallel,
  one process per (frac,seed) point) — class-wise learning curves +
  `learning_curve.png`.

## Run

```bash
PY=~/myprojects/recover/.venv/bin/python
# canonical winning config (all defaults)
systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
  $PY DNN/stage3_robust_mlp.py
```

GPU memory: trivial (<1 GB) — the whole frame is 664k×67 float32 and the net is
tiny. No need for the chunking/`TOTAL_SUP` ceilings that bound TabICL.

## Refactored library — fast iteration + deployable inference

The `stageN` scripts above are the *experiment record*; each grew its own copy of
the scaler → val-split → ensemble → predict loop. `dnn_core.py` consolidates that
into one fast, reusable core so a new idea is ~30 lines and a chosen recipe can be
baked into a single portable artifact for production inference on AlphaEarth
rasters.

### `dnn_core.py` — the shared core
Same winning maths (class-weighted CE, label smoothing, early-stopped 2-layer MLP,
5-seed probability ensemble, StandardScaler), verified to reproduce the reference
fold-0 F1 (0.737). Speed/ergonomics over the old per-stage loops:

- **frame cache** (`load_cached`): memoises the duckdb parquet load to `.cache/`,
  self-invalidating on parquet mtime — repeat runs go **2.1 s → 0.2 s**.
- **whole dataset on-GPU**, no per-epoch H2D copies.
- **on-GPU per-epoch val macro-F1** (`gpu_macro_f1`, bit-matches sklearn) — kills
  the `.cpu()`+sklearn call every epoch.
- **`Config`** dataclass — every hyperparameter, all env-overridable (same env
  names as stage3), so sweeps stay one-liners.
- **`Ensemble`** — a fitted object you can `predict_proba`/`predict`/`save`/`load`;
  save is bit-exact on reload. This is the deployable model.
- **`cv_evaluate(data, cfg, relabel_fn=...)`** — the reference 3-fold spatial CV in
  one call; `relabel_fn` injects per-fold train-only label surgery (e.g. stage8).

**Note on AMP / `torch.compile`:** both are wired (`DNN_AMP=1`, `DNN_COMPILE=1`)
but **default OFF** — measured *slower* on this tiny 256→128 MLP (fp16 cast +
GradScaler overhead beats the tensor-core gain; single-fold 20.6 s → 34.3 s with
AMP, and slightly lower F1). The workload is Python-minibatch-loop-bound on a tiny
net, so the real levers are the cache + on-GPU eval, not lower precision. Keep the
switches for any larger architecture probed later.

New experiment sketch:
```python
import dnn_core as C
data = C.load_cached("lidar")
res  = C.cv_evaluate(data, C.Config(hidden=(512, 256)))   # your idea here
```

### `train_final.py` — bake the recipe into one artifact
Trains the winning recipe (5-seed ensemble + `to12_fix` cls12 relabel, the only
label fix that helped) on the **entire** stable frame — no held-out fold, so it
uses the full-data cleanlab artifact `clean_labels_full.npz` (no test to leak
into). Writes `models/dnn_final.pt` (weights + scaler stats + class decode map +
feature order + lidar medians) and a `.meta.json`.
```bash
systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
  $PY DNN/train_final.py            # RELABEL=to12_fix EXTRA=lidar by default
```

### `fit_calibration.py` — uncertainty quantification (calibrated proba + conformal sets)
Companion to `train_final.py`. `train_final.py`'s own training probabilities are
optimistic (trained on everything, no held-out fold) so they can't be used to
calibrate uncertainty. This script instead runs the reference leak-free 3-fold
spatial CV to produce out-of-fold (OOF) probabilities over the whole stable
dataset, then fits two things:

- **LAC + Mondrian (class-conditional) conformal prediction sets**, alpha=0.1
  (90% target coverage) — the method Investigation 3 (`conformal_compare.py`)
  found best: tightest sets among valid methods, no empty sets, per-class
  coverage spread 0.29→0.035 vs plain split (see the Conformal UQ section
  above).
- **Point calibration**, head-to-head between temperature scaling (Investigation
  3's method) and Venn-Abers (one-vs-rest binary calibrators — the `venn-abers`
  package is binary-only, so multiclass here means one calibrator per class,
  fit on `[1-p_c, p_c]` vs `y==c`, renormalized at inference). Both are scored
  by ECE on a held-out half of the OOF set; the lower-ECE method is saved and
  used for the `pcal_*` output columns/bands. Both numbers are recorded in
  `dnn_final_calib.meta.json` regardless of which wins.

Writes `models/dnn_final_calib.npz` (+ `.meta.json`), consumed by `predict.py`
and `predict_raster.py`'s `dnn_core.Calibration`.
```bash
systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
  $PY DNN/fit_calibration.py        # ALPHA=0.1 EXTRA=lidar RELABEL=to12_fix by default
```

**Result (OOF macro-F1 0.7321, close to the 0.7341 CV reference): Venn-Abers
wins decisively.**

| method | ECE before | ECE after | fit time | apply time (331,870 rows) |
|---|---:|---:|---:|---:|
| temperature scaling (T=0.78) | 0.0694 | 0.0394 | 18s | 12.8s |
| **Venn-Abers OvR** | 0.0694 | **0.0042** | 485s | **1.6s** |

Venn-Abers is ~9x better calibrated (ECE 0.0042 vs 0.0394) AND ~8x faster to
*apply* (the isotonic-regression fit is a one-time ~8 min cost at calibration
time; inference is a vectorized `searchsorted`, cheaper than temperature
scaling's softmax). It won on both axes, so `dnn_final_calib.npz` uses
Venn-Abers for the `pcal_*` columns/bands. LAC+Mondrian conformal sets: 90.0%
coverage (target), avg set size 1.55, singleton rate 57%, per-class coverage
tight to 0.9000–0.9020 — matches the Investigation-3 reference behavior.

### `predict_raster.py` — inference on AlphaEarth GeoTIFF stacks
Block-windowed (bounded memory) inference on 64-band AE tiles (band order
A00..A63, matching the GEE/source.coop export contract). Writes a single-band
classified GeoTIFF of **raw class codes** (e.g. 12 = snow/ice) + optional
`--uq-out` multiband GeoTIFF: per-class calibrated probability + prediction-set
size + per-class 0/1 conformal-set-inclusion, `2C+1` float32 bands with
self-describing band names (`pcal_c{v}`, `set_size`, `inset_c{v}`). Nodata
pixels → `--nodata-class` (default 0). If the model uses lidar, pass
`--lidar-raster` (bands elevation,tri,tch on the same grid) or omit it to
median-fill from the saved training medians.
```bash
$PY DNN/predict_raster.py --in ae_tile.tif --out classified.tif \
    [--uq-out uq.tif] [--lidar-raster lidar.tif] [--scale 10000]
```
`--scale` divides the AE bands if a source packs them as scaled ints; omit for the
float embedding used in training.

### `predict.py` — inference on tabular AE points (parquet)
Same artifacts, for point/parquet inputs shaped like the training data (columns
A00..A63 [+ lidar]). Missing lidar columns are median-filled. Always writes
`pred_class` + the UQ columns (`pcal_{c}`, `set_size`, `inset_{c}`).
```bash
$PY DNN/predict.py --in points.parquet --out preds.parquet
```

## Inference performance & wall-to-wall county scaling

Target use case: classify 3 Norwegian counties wall-to-wall at 10 m ≈ **1.3
billion pixels**. Measured facts (A40, shared) that shape the design:

| stage | throughput | 1.3B px |
|---|---:|---:|
| **model compute only** (5-seed ensemble, on-GPU) | ~6 M px/s | **~4 min** |
| fused batched-bmm ensemble vs python loop | +10% | (compute isn't the wall) |
| `torch.compile` on the forward | fragile / OOM-prone, ~10% | not worth it |
| **reading + decompressing** a 64-band deflate tile (1 handle) | **~0.7 M px/s** | **~30 min** |

**The model is not the bottleneck — raster I/O (decompression) is.** A 64-band
float32 tile is huge (a 3000×3000 tile = 695 MB compressed), and decompression, not
the MLP, dominates. So `predict_raster.py` is a **producer/consumer pipeline** that
overlaps read → GPU → write and, crucially, gives each reader thread its **own GDAL
handle** so decompression parallelizes across cores (a shared handle serializes
reads and kills the gain):

| readers | end-to-end throughput |
|---:|---:|
| 1 | 0.5 M px/s |
| 3 | 1.2 M px/s |
| 6 | 1.4 M px/s (8-core box; read-ceiling) |

Results are **bit-identical across reader counts** (deterministic). On the 8-core
VDI, `--readers 6` gives ~2.8×; the ceiling is the machine's decompression
bandwidth, not the GPU (which idles most of the run).

### Recommendations for the county-scale run
- **Tile the counties** (e.g. a manifest of ~5–10k px tiles) and run one process
  per tile batch across machines/cores — the pipeline scales with **read cores**,
  so more cores/nodes beats a faster GPU. A single 8-core box does 1.3B px in
  ~15–25 min of wall time; more read cores shorten it proportionally until the GPU
  (~4 min) becomes the floor.
- **Source-agnostic**: `--in` accepts any rasterio path — local GeoTIFF, VRT, or a
  COG URL (`/vsicurl/…`, GCS, S3). For source.coop AE COGs, block reads become
  HTTP range requests; more `--readers` then overlaps network latency too.
- **Skip `--uq-out`** unless you need it — the class map is one int16 band; the
  UQ stack is `2C+1` float32 bands (21 for the 10-class model, 20×+ the write
  volume + memory).
- **AMP / torch.compile stay off** — measured slower / OOM-prone on this tiny net.
- If read bandwidth ever stops being the wall, distilling the 5-seed ensemble into
  one net drops compute 5× (but compute is already ~4 min, so it's rarely worth
  the F1 risk).

```bash
# fast wall-to-wall tile (local or COG URL), 6 read threads:
$PY DNN/predict_raster.py --in ae_tile.tif --out classified.tif --readers 6 --block 2048
```

## Inference data sourcing — the P-drive embeddings don't cover the AOI

`embeddings.vrt` on the P-drive is framed to the full 3-county AOI (15002×18002
px) but only 154 of the tiles that would fill it actually exist — most of the
mosaic is empty. `DNN/fetch_aef_sourcecoop.py` investigates 3 ways to get
AlphaEarth embeddings for the missing area (tested on a 2024 subset):

| option | cost/auth | throughput (measured, apples-to-apples on a 20×20km/2000×2000px/64-band 2024 window) | full AOI (270M px) | verdict |
|---|---|---:|---:|---|
| **source.coop, 1 handle, streaming reads** | free, no auth | ~184k px/s (1 open, 4 sequential sub-window reads) | **~24 min** | **winner, and scales further with parallel readers (untested)** |
| **source.coop, 1 handle, single big read** | free, no auth | 124k–152k px/s | ~30–36 min | good, simpler code |
| **geedim `img.gd.toGeoTIFF()`** | free (existing `ee-gsingh` project) | 98.3k px/s (`max_requests=100`); 57.6k px/s at default `max_requests=32` | 46 min / 78 min | easiest to wire up, no UTM/dequant handling |
| source.coop, reopen handle per window | free, no auth | 85k px/s (4× reopen cost dominates) | ~53 min | **anti-pattern — don't reopen per block** |
| GEE `ee.data.computePixels` (raw, hand-tiled) | free | ~12k px/s (48 MB/request cap forces ~290×290 px tiles) | ~6 hr single-threaded | superseded by geedim |
| GCS `alphaearth_foundations` | requester-pays, needs billing + creds | not measured — anonymous access denied | — | blocked on credentials |

**Correction to an earlier pass in this investigation:** an initial source.coop
estimate of "~2 min for the full AOI" was extrapolated from small (512×512px)
windows and was wrong — retested at the same 2000×2000px scale geedim was
benchmarked at, single-connection source.coop is **~30 min**, not ~2 min.
Still faster than geedim but only ~1.3–1.9×, not ~20×. Small-window
extrapolation was misleading; always benchmark at the scale you'll actually
run at.

**Streaming (1 persistent handle, many reads) vs. reopening per request
matters a lot, and directly favors source.coop's architecture:** opening a
fresh `rasterio.open()` + `WarpedVRT` handle costs real fixed time (TIFF
header/IFD fetch + warp-transformer setup). Reopening once per window to
cover 4 sub-tiles of the test area took 46.7s total; keeping ONE handle open
and reading all 4 sub-windows through it took 21.6s (2.2× faster). This is
exactly the pattern `predict_raster.py`'s reader threads already use (one
GDAL handle per thread, reused across all its assigned blocks) — so wiring
source.coop in via that existing architecture, rather than a naive
open-per-tile downloader, should land close to the streaming number above.
**Not yet tested: multiple parallel reader threads each streaming from their
own handle** (`predict_raster.py` already supports `--readers N` for local/
COG sources) — this is the highest-value next experiment, since the existing
6-reader local-tile benchmark showed ~2.8× over 1 reader before hitting the
machine's decompression ceiling, and source.coop's bottleneck (network RTT +
zstd decompression) may parallelize similarly.

**geedim** (`ee.Image.gd` accessor, v2.0 API) wraps the `computePixels`
tiling problem: `img.gd.prepareForExport(crs=, scale=, region=)` then
`prepped.gd.toGeoTIFF(path, max_requests=N)` auto-splits the region into
GEE-sized tiles and downloads them with a thread pool, writing directly into
one GeoTIFF. Tested against the same 2024 AlphaEarth mosaic
(`.reduce(ee.Reducer.first())`, matching `sample_feature_space_stable_allyears
.build_alphaearth_year`): `max_requests=32` (default) gave 57.6k px/s;
`max_requests=100` gave 98.3k px/s (1.7×) with no quota errors — GEE's
per-project concurrent-request quota didn't bite at that level on this small
test, but a sustained ~30-46 min run at that concurrency was NOT tested and
could throttle differently. Output is **already reprojected to 32633 and
analysis-ready float64** (unit-norm verified, L2=1.0000516) — no
dequantization or WarpedVRT reprojection step needed, unlike source.coop. Also
uses the collection's per-pixel `.reduce(first())` mosaic directly, so tile
seams are handled by GEE server-side (source.coop tiles are the *raw*,
non-mosaicked per-UTM-zone EE images — cross-zone seam behavior vs. the
collection's mosaic was not verified either way in this pass).

**Key findings:**
- source.coop (`data.source.coop/tge-labs/aef`) mirrors the exact GCS AEF
  dataset over plain HTTPS, no auth, `accept-ranges: bytes` (vsicurl-friendly).
  An index (`aef_index.parquet`, 302k rows, ~78 MB) maps year + WGS84 bbox +
  UTM zone → tile URL — filter it instead of scanning directories.
- **GDAL's default vsicurl config is ~5× too slow on this dataset**: the COGs
  are band-interleaved (64 separate per-band blocks), and without HTTP/2
  multiplexing GDAL fetches them serially (41 s for a 1024×1024×64 block).
  Setting `GDAL_HTTP_MULTIPLEX=YES` + `GDAL_HTTP_VERSION=2` drops that to 8 s
  (~5×) and pushes small-window (512×512) throughput to ~140 MB/s. This is
  the single biggest lever and is easy to miss.
- Data is **int8-quantized** (dequantize: `((v/127.5)**2)*sign(v)`, −128 =
  nodata — see `dequantize()`) and the raw `.tiff` is **bottom-up**
  (row 0 = south). The dataset ships a per-tile `.vrt` that corrects this,
  but it hardcodes `/vsis3/...` (needs `AWS_NO_SIGN_REQUEST=YES`, works fine)
  and its `VRTWarpedDataset` warp is itself slow (41 s for a 512×512×64
  window vs 2–4 s reading the raw `.tiff` directly) — **the warp machinery is
  the cost, not the transport.** Fastest correct path: read the raw `.tiff`
  via `/vsicurl`, flip the row axis and dequantize in numpy.
- The AOI straddles **UTM zones 31N and 32N** in the source data (only 20
  tiles total cover the 3 counties for one year), neither matching the
  project's working CRS (32633). Reprojecting via a static warped-VRT file
  is slow for the same reason above; `rasterio.vrt.WarpedVRT` used
  in-process at read time is fast (0.09 s/window once GDAL's block cache is
  warm) and correctly handles the flip + reprojection in one pass — this is
  what `open_tile_window()` does. Verified end-to-end (source.coop tile →
  `WarpedVRT` reproject → dequantize → `dnn_core.Ensemble` inference) on a
  live 1024×1024 window: dequantized L2 norm 1.000 ± 0.01 (correct unit-norm
  embeddings), predictions ran in <1 s on GPU.
- GEE's synchronous `ee.data.computePixels` is capped at exactly 50,331,648
  bytes/request (48 MiB) and serves **float64**, not int8, so a request maxes
  out around 290×290 px for 64 bands. **geedim solves this properly** — see
  above — by tiling + thread-pooling the same `computePixels` calls
  internally, reaching 98.3k px/s at `max_requests=100` vs ~12k px/s hand-
  rolled. `Export.image.toCloudStorage`/`toDrive` (async batch, server-side
  tiled, no 48 MB cap) was not benchmarked — it avoids the per-request cap
  but queues as a background job (minutes–hours latency), unsuitable for
  interactive iteration; worth using for a one-time bulk backfill export
  instead of point-in-time streaming, if source.coop's 20-tile-per-year
  footprint is ever unavailable or wrong.
- GCS direct access (`/vsigs/...`) failed with `AWS`-equivalent anonymous
  access — the bucket is requester-pays and needs a real billing-linked GCP
  project (`GS_NO_SIGN_REQUEST` does not bypass this, unlike source.coop's
  public S3 mirror). Not pursued further since source.coop serves identical
  data for free.

**Recommendation:** for a **one-off test pull**, geedim is the path of least
resistance — `pip install geedim`, reuses the existing `ee-gsingh` auth, one
function call gives an analysis-ready reprojected GeoTIFF, ~46 min for the
full AOI at `max_requests=100`. For **`predict_raster.py`-style streaming
inference** (read-infer-write per block, no full download), source.coop with
persistent per-thread handles is faster (~24–36 min full-AOI equivalent,
single-threaded, untested with multiple parallel readers) and
infrastructure-free, at the cost of handling the UTM-zone reprojection
(`WarpedVRT`) and int8 dequantization yourself — wire `open_tile_window`-style
per-tile reads into `predict_raster.py`'s reader threads (one handle per
thread, reused across all its blocks — do NOT reopen per block, see above)
rather than pre-downloading/mosaicking 13 GB of raw UTM31N/32N tiles to disk
first. **Next step:** benchmark `--readers N > 1` for source.coop the same
way the local-tile case was benchmarked (1/3/6 readers, measuring aggregate
px/s) to see if it parallelizes as well as decompression-bound local reads
did, or whether network/GEE-side limits cap it lower.

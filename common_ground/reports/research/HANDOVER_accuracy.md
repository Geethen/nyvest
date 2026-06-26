# Scheme B Accuracy — Handover Summary

**Goal:** improve macro-F1 on the scheme B allyears pipeline. Reference to beat: **0.6975**
(CC-APS α=0.05, 2020-only stable, TabICL stage-2).

## UPDATE 2026-06-12 — new best 0.7074 (TOTAL_SUP=32k); two levers ruled out

- **New best: `perfold_all_fix` + `TOTAL_SUP=32000` → F1 0.7074** (cls12 0.7042,
  result JSON `_perfold_fix_sup32k`). Support-size curve: 25k → 0.7036,
  32k → 0.7074 (+0.004 on every fold), 36k → 0.7076 (saturated), 40k → CUDA OOM
  on the A40-24Q. `TOTAL_SUP` is now an env var. Use 32k.
- **nc-gate (noise-corrected APS) is a NO-OP**: wired in as `GATE_MODE=nc`
  (leak-free per-fold impostor estimate), result `_perfold_fix_ncgate` = 0.7036,
  exact tie with baseline. The gate only ever had ~9 cls12 candidates to act on.
  Don't iterate further.
- **DEM features are a dead end**: GLO30 elevation+slope extracted for all 89,286
  unique points (`data/dem_features.parquet`, join via `EXTRA_FEATURES=dem`).
  Binary 11-vs-12 probe: embed AUC 0.9752 → embed+dem 0.9762 (nothing). The
  embedding already encodes elevation; class 11 itself sits at median 1011 m.
- **Where F1 is actually lost** (confusion matrix, fixed test): 11→6 17.8%,
  3→5 19.4%, 7→6 19.3%, 6→11 11.1%, 5→3 15.6%. Big natural-class confusions
  dominate; cls12 errors are negligible in count. Next ideas should target
  these (e.g. seasonal/phenology features — but probe cheaply with the binary
  CatBoost pattern before burning a pipeline run).
- **MEMORY SAFETY (mandatory)**: a CPU-fallback TabICL run OOMs 64 GB host RAM
  and kills the whole VDI session. The script now fails fast if device != cuda
  with TOTAL_SUP > 8k, frees fold-local arrays per fold, and drops half-built
  stage-2 models on fold failure (a CUDA OOM previously leaked ~20 GB host).
  Always launch via:
  `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 CLEAN_MODE=... TOTAL_SUP=32000 OUT_SUFFIX=... ~/myprojects/recover/.venv/bin/python common_ground/scripts/llto_schemeB_allyears.py`
  Expected RSS: ~1.4 G during CC, ~23 G during the TabICL stage (32k support).

## Environment
- Repo `/home/geethen.singh/myprojects/nyvest`; venv `~/myprojects/recover/.venv/bin/python` (shared with `recover`).
- GPU A40-24Q works via **torch 2.5.1+cu121** (downgraded from 2.12+cu130, which the CUDA-12.2 driver couldn't run). TabICL + TabPFN both run on GPU.
- Pipeline: `common_ground/scripts/llto_schemeB_allyears.py`. 3-fold GroupKFold(cell_id), TabICL stage-2, n=25k support. Env switches: `CLEAN_MODE`, `STAGE2_MODEL` (tabicl|tabpfn), `OUT_SUFFIX`. Results: `common_ground/reports/research/schemeB_allyears_results*.json`.

## Current best (all leak-free, fixed ~221k test set)
| config | mean F1 | class-12 |
|---|---|---|
| **perfold_all_fix** (best mean) | **0.7036** | 0.6867 |
| **centroid** (best class-12) | 0.6970 | **0.7061** |
| no-force baseline | 0.6964 | 0.6654 |
- `perfold_all_fix` = leak-free cleanlab RELABEL of train rows to suggested class (per-fold, train-only).
- `centroid` = remove class-12 rows nearer another class centroid than their own (`clean_stale_class`).
- **They do NOT combine** (combo = 0.7012 / cls12 0.6595, worse than both).

## Key findings (what's been ruled out)
1. **Allyears ≈ 2020-only** at equal 25k support (0.696 vs 0.6975). Extra years don't help overall.
2. **Class 12 (snow/ice) is the main drag** and is spectrally inseparable from class 11 (infrastructure) in the annual AlphaEarth embedding (centroid-distance ratio 1.09; 95% of stray cls12 rows fall nearest 11). Labels are partly stale (glaciers retreated 2017-25; T estimates ~17% of cls12 labels wrong, 92% of errors → class 11). **No label cleaning or sampling breaks this ceiling** — needs a real 11/12 discriminator.
3. **Cleaning is a confusable-class denoiser:** perfold_fix lifts scrub +0.024, wetland +0.018, snow/ice +0.021; bare −0.006 (collateral). Gains concentrated in spectrally-overlapping classes.
4. **cleanlab uses dataset-wide stats → train/test leak is real** (~0.006 mean, 0.028 cls12). Always flag per-fold train-only. cleanlab REMOVE is mostly leak (perfold_rm ≈ no-force); RELABEL survives.
5. **TabPFN v3 (tabpfn 8.0.6, TabPFNV3) ties TabICL v2** on accuracy (0.7002 vs 0.7036, within noise) but is ~5× slower at predict. **Keep TabICL.**
6. **Test partition:** keep noisy labels (leak-free lower bound). Model reproduces the noise, so noisy ≈ true F1. A trustworthy corrected number needs external relabel.

## Highest-value next steps (label cleaning is exhausted)
The remaining upside is in **features/labels**, not models or sampling:
1. **Seasonal/elevation feature for 11↔12** (HIGHEST CEILING): summer-only embedding, NDSI snow-persistence band, or DEM. The annual embedding fundamentally can't separate snow-covered infrastructure from perennial snow. **Needs GEE extraction (currently parked).**
2. **External relabel of class 12** from NVE 2018-19 glacier inventory or Sentinel-2 NDSI persistence — gold-standard fix + a trustworthy test metric. Needs extraction.
3. Cheap/quick: noise-corrected APS in the gate (diagnostic showed cls12 τ 0.97→0.85; would tighten snow/ice pseudo-labels) — conformal-native alternative to data cleaning, not yet wired into the GPU pipeline.

## Paths (all relative to repo root /home/geethen.singh/myprojects/nyvest)

### Scripts
- `common_ground/scripts/llto_schemeB_allyears.py` — MAIN pipeline (CC-APS gate + stage-2). Env: CLEAN_MODE, STAGE2_MODEL, OUT_SUFFIX.
- `common_ground/scripts/clean_labels_artifacts.py` — builds `clean_labels_full.npz` (global OOS pred_probs, issue masks, suggested labels, noise matrix T). ~50 min.
- `common_ground/scripts/clean_labels_perfold.py` — builds `clean_labels_perfold.npz` (leak-free per-fold remove/relabel + per-fold test transition matrix). ~3 hr.
- `common_ground/scripts/clean_labels_compare.py` — characterise centroid vs cleanlab flag counts.
- `common_ground/scripts/noise_corrected_aps.py` — conformal noise-correction diagnostic (reuses clean_labels_full.npz).
- `common_ground/scripts/noise_corrected_test_metric.py` — post-hoc test-noise correction (unreliable; see test-label-noise.md).
- Reference (2020-only) pipeline: `common_ground/scripts/llto_schemeB_alpha_sweep.py`.

### Data (parquet)
- `data/grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet` — 663,740 rows, 2017-25, 12 classes, 64 bands A00-A63 (+ cell_id, lon, lat, year). MAIN stable source.
- `data/grunnkart_nyvest_fscs_unstable_alphaearth.parquet` — unstable rows (CC-APS pseudo-label pool).
- `data/grunnkart_nyvest_fscs_alphaearth.parquet` — 2020-only stable (the 0.6975 reference source).
- Class legend (FSCS_CLASSES in `scripts/build_combined_dataset.py`): 2=bare,3=cropland,4=forest,5=grassland,6=scrub,7=wetland,8=water,10=settlement,**11=infrastructure,12=snow/ice** (1→2, 9→8 merged).

### Precomputed artifacts (reuse — expensive to rebuild)
- `common_ground/reports/research/clean_labels_full.npz` — global masks + noise matrix T.
- `common_ground/reports/research/clean_labels_perfold.npz` — per-fold leak-free masks.

### Result JSONs (`common_ground/reports/research/schemeB_allyears_results_*.json`)
- `_noforce_canonical` (0.6964) · `_clean_centroid` (0.6970, best cls12) · `_perfold_all_fix` (0.7036, BEST mean) · `_perfold_all_rm` · `_cleanlab_all_fix` (0.7084 but minor leak) · `_combo_centroid_perfold_fix` (0.7012) · `_tabpfn_perfold_fix` (0.7002).
- Reference: `schemeB_alpha_sweep_results.json` → results.cc_aps_a05 (0.6975).

## Memory pointers
See `~/.claude/projects/-home-geethen-singh-myprojects-nyvest/memory/`: `schemeB-class12.md`,
`test-label-noise.md`, `tabpfn-v3.md`, `tabicl-cpu-oom.md`, `schemeB-allyears-experiment.md`.

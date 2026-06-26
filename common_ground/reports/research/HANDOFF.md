# Scheme B Research — Handoff

**As of 2026-05-21.** Read this before continuing.

## Current best model

**`tabicl_25k_n16_kvon`** — TabICL zero-shot, `n_estimators=16`, `kv_cache=True`,
25 000-row random support drawn from `stable + Stage-1 pseudo-labels on unstable`.

- **Macro-F1 = 0.6881** (3-fold spatial LLTO, σ = 0.018) on Scheme B = 10 classes.
- 12-class CatBoost cg_llto reference = 0.6197.

## Scheme B (the label space we're using)

Merge `1 → 2` (sand+rock → "bare") and `9 → 8` (freshwater+marine → "water").
Final classes: **[2, 3, 4, 5, 6, 7, 8, 10, 11, 12]**.
Both mergers are ecologically defensible (same `Snaumark` and same `water` umbrellas).
Reason for 10 classes: TabICL fine-tune cap is `max_classes = 10`.

## Pipeline (LLTO Common Ground)

1. **Spatial folds**: KMeans(k=3, seed=0) on `(lon, lat)` from the stable parquet.
   Unstable rows are assigned by `km.predict`. Same physical location → same fold.
2. **Stage 1**: CatBoost 500 iter (CPU) on 80 % of the fold's stable_train,
   stratified split (random_state=0). Predicts pseudo-labels for `unstable_train`.
3. **Stage 2 (this is the best model)**: TabICL on
   `support = random 25 000 rows from (stable_train + pseudo-labelled unstable_train)`,
   then predict on `stable_test`.
4. **Score** against grunnkart `class` (merged) at `stable_test` only.

## Files

| Path | What |
|---|---|
| `common_ground/scripts/llto_research_schemeB_round2.py` | Round 2 (ensemble/support sweep/FT) |
| `common_ground/scripts/llto_schemeB_diagnostics.py` | Confusion + Stage-1 quality + per-class learning curves |
| `common_ground/scripts/llto_research_scheme.py` | Generic scheme runner (3 model variants per scheme) |
| `common_ground/reports/research/schemeB_round2_report.html` | Round-2 results & KV cache analysis |
| `common_ground/reports/research/schemeB_diagnostics_report.html` | Root-cause analysis, must-read |
| `common_ground/reports/research/schemeB_diagnostics.json` | Raw diagnostic data (CM, pseudo conf, learning curves) |
| `common_ground/reports/research/schemeB_round3_results.json` | n_est × support-size sweep results |
| `data/grunnkart_nyvest_fscs_alphaearth.parquet` | Stable, year=2020, 79 119 rows |
| `data/grunnkart_nyvest_fscs_unstable_alphaearth.parquet` | Unstable, 2017–2025, 16 844 rows |

Features: 64 AlphaEarth embedding bands `A00..A63`.

## Environment gotchas — read these or you will lose hours

| Issue | Fix |
|---|---|
| **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** | DO NOT SET. Causes `CUDA driver error: operation not supported` on torch 2.5.1+cu124 with driver 535.274.02. |
| **TabICL fine-tune `amp=True`** (default) | Set `amp=False`. Same driver error otherwise. |
| **CatBoost-GPU + TabICL-CUDA in one process** | Exhausts the A40's 24 GB. Use `task_type="CPU"` for CatBoost when TabICL also runs. CPU CatBoost takes ~60–125 s per fold at 500/1000 iter. |
| **TabICL `kv_cache=True` requires ≤ `max_classes = 10`** | Scheme B sits exactly at this boundary → safe. Don't try with 11+ classes. |
| **TabICL fine-tune ≤ 10 classes only** | The loss path `logits[..., :n_classes]` doesn't route through the hierarchical many-class strategy. Scheme B fine-tune works; 12-class does not. |
| **OOM during prediction at large support** | `n_estimators=32 + 20k support` OOMs on the A40. Clear cache between calls (`torch.cuda.empty_cache()`). Cap fine-tune at `max_data_size=3000`, `n_estimators_finetune=1`. |
| **venv** | `~/myprojects/recover/.venv/bin/python` (Python 3.12, shared with the `recover` project). |

## Weak classes (F1 < 0.65 under current best)

| Class | F1 | n_train/fold | Stage-1 pseudo conf | Dominant confusion | Verdict |
|---|---|---|---|---|---|
| 5 grassland | **0.40** | 3 159 | 0.55 | → 3 crop (42 % of test) | feature-bound + partly data-bound |
| 2 bare (1+2) | **0.53** | 635 | 0.46 | → 6 scrub (21 %) | **DATA-BOUND** |
| 7 wetland | **0.62** | 4 359 | 0.52 | → 6 scrub (17 %) | data-helpful + feature gap |

Per-class learning curves are in `schemeB_diagnostics.json` under `learning_curve`.
Curve slopes (Δ F1 from 5k → 25k support):
- DATA-BOUND (curve still rising at 25k): class **2** (+0.068)
- data-helpful (still gaining, slowing): class **7** (+0.040), class **12** (+0.037), class **5** (+0.035)
- FEATURE-BOUND (plateaued): classes 4 (+0.008), 11 (+0.007), 8 (+0.002)

## Suggested improvements — prioritised

These have NOT been tested yet. Order is by expected value × ease.

### Tier 1 — quick wins (<15 min each)

1. **Biased support draw for class 5** (and possibly class 2).
   Currently random subsampling of the 25k support gives ~5.6 % class 5 / ~0.15 % class 2.
   Force class 5 to ~16 % (4 000 rows) and class 2 to ~3 % (750 rows) via stratified-without-replacement
   draw — not oversampling. Tested at 5k support in earlier rounds and it hurt macro-F1 by ~0.01;
   the hypothesis is that at 25k the macro damage is much smaller while minority recall holds.
   Single script: extend `llto_research_schemeB_round2.py` with a custom `subsample` that
   caps per-class draws.

2. **Confidence-gated pseudo-labels for weak classes.**
   Currently all Stage-1 pseudo-labels go into Stage 2 regardless of confidence. Filter:
   for pseudo-labels that came out as classes 2, 5, 7 in Stage 1, keep only those with
   top-1 prob ≥ 0.6. Pseudo conf data is in `schemeB_diagnostics.json` → `pseudo_quality`.

3. **Iterate Stage 1 → Stage 2.** Use the current best (`tabicl_25k_n16_kvon`) as Stage 1
   instead of CatBoost-500, then re-train Stage 2 (also TabICL) on the cleaner pseudo-labels.
   Expected: lower noise in pseudo-labels for classes 5 and 7, which currently have 33–62 %
   low-conf pseudo-labels.

### Tier 2 — feature engineering (medium effort, big upside)

4. **Multi-temporal phenology features for class 5.**
   The unstable parquet has 9 years of AlphaEarth per cell (2017–2025). For each `cell_id`,
   compute per-band mean/std/range across years and join to the stable rows. Cell overlap
   is only ~99 of 238 stable cells — fall back to zeros where missing. Targets the 5↔3 crop
   confusion (crops have sharp annual phenology, grasslands flatter).
   Extraction is already documented at
   `scripts/extraction/sample_feature_space_unstable.py`.

5. **Sentinel-1 SAR features for class 7.**
   Wetlands have distinct VV/VH backscatter (volume + surface scattering). Not currently in
   the feature set. Would target the 7↔6 (scrub) and 7↔4 (forest) confusions at riparian
   edges. Requires a new GEE extraction.

6. **Hydrology features for class 7.** Distance-to-water, slope, drainage density —
   derivable from existing DEM + hydrography layers. Cheaper than SAR.

### Tier 3 — model / routing changes

7. **Binary grassland-vs-crop expert.** Train a focused CatBoost classifier on classes 3 and 5
   only, with whatever new features help most (phenology). At inference: if TabICL predicts
   3 or 5, defer to the binary expert. Expected: class-5 F1 up by 0.05–0.10.

8. **Two-model routing for weak classes.** CatBoost for {3, 4, 6, 8, 10, 11, 12}; TabICL for
   {2, 5, 7}. Use predictions from whichever model is best at each class. Test set predictions
   from the diagnostic run are saved per fold in the JSON.

### Tier 4 — collect more data (longest lead time)

9. **More sand / gravel / exposed-rock labels** for class 2 — clearly data-bound (curve still
   rising at 25k support).
10. **Snow class** (12) is also data-bound — n=159 in the full 12-class set, only 53/fold.
    The current 0.72 F1 is a tabular-foundation-model gift; more data would lift it further
    but priority is lower since it's already > 0.65.

## What NOT to spend time on

- **Larger TabICL support beyond 25k.** Tested; curve has flattened.
- **`n_estimators=32`.** OOMs on the A40 at any support size we'd want.
- **Ensemble of TabICL + CatBoost.** Tested; CatBoost's weak fold drags the average down.
- **Fine-tuning for the macro-F1 win.** Tested; FT gives σ reduction (0.014 vs 0.027) but mean F1 is below zero-shot. Useful only if production worst-fold stability matters more than mean.
- **Stratified support with FT.** Hurts FT consistently (overfits to over-represented classes).
- **Softmax-temperature changes.** Tested at 0.5; no-op.

## Reference numbers to beat

| Model | F1 (Scheme B, 10 cls) | σ | Per-class weak |
|---|---|---|---|
| **tabicl_25k_n16_kvon** (current best) | **0.6881** | 0.018 | 2: 0.530, 5: 0.392, 7: 0.622 |
| tabicl_20k_n8_kvoff | 0.6871 | 0.014 | 2: 0.531, 5: 0.399, 7: 0.624 |
| catboost_high_iter (CPU 1000 iter) | 0.6582 | 0.027 | 2: 0.499, 5: 0.344, 7: 0.583 |
| catboost_baseline (CPU 500 iter) | 0.6502 | 0.032 | 2: 0.486, 5: 0.334, 7: 0.581 |
| tabicl_ft_30ep | 0.6559 | **0.004** | (lowest σ, mean below zero-shot) |

## Test protocol

- Always 3-fold KMeans spatial LLTO with seed=0 for direct comparability.
- Always score against grunnkart `class` (merged via Scheme B map) on `stable_test`.
- Always report macro-F1 mean ± std across folds.
- Persist results as JSON with the structure used by other scripts in this directory
  so they can be combined into HTML reports.

## Sanity-check the environment before doing anything

```bash
~/myprojects/recover/.venv/bin/python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
m = torch.nn.Linear(10,10).to('cuda')
print('to-cuda OK')
from tabicl import TabICLClassifier, FinetunedTabICLClassifier
print('tabicl FT OK')
"
nvidia-smi --query-gpu=memory.free --format=csv
```

If any of the above errors, do not start an experiment — fix the env first.
The most common cause is `PYTORCH_CUDA_ALLOC_CONF` leaked into the shell.

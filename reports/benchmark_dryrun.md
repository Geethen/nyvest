# Tabular benchmark — full run

**Date:** 2026-04-22
**Host:** `t2lipvdiml07` (VDI, Linux) · 8 CPUs · 62.8 GB RAM · NVIDIA A40-24Q
**Venv:** `~/myprojects/recover/.venv` (Python 3.12) · torch 2.5.1+cu124

## Config

| | |
|---|---|
| Data | `/data/P-Prosjekter2/154001_nyvest/landcover_megan/{train,val}_values.shp` |
| Train rows | 69 552 (full, after NA drop) |
| Val rows | 19 999 (full, after NA drop) |
| Features | 66 (64 AlphaEarth embeddings + 2 LiDAR) |
| Target | `fallbck` — 22 classes |
| `n_estimators` | 500 (tree models) |
| `n_jobs` | 4 |
| Device | `cuda` (XGBoost, CatBoost) |
| Foundation models | capped at 5 000 train / 5 000 val (module constants) |
| TabPFN | wrapped in `ManyClassClassifier(alphabet_size=10)` (22 > 10) |
| W&B | disabled |

Command:

```bash
~/myprojects/recover/.venv/bin/python scripts/benchmark_tabular.py \
  --wandb-mode disabled --n-estimators 500
```

## Results

Sorted by macro-F1 (val). Raw numbers in [benchmark_results.csv](benchmark_results.csv).

| Model    | n_train | F1 (macro) | Bal. acc. | Train (s) | Val (s) | Peak RSS (MB) |
|----------|--------:|-----------:|----------:|----------:|--------:|--------------:|
| xgboost  | 69 552  | **0.697**  | **0.698** |     21.26 |    0.30 |         2 885 |
| lightgbm | 69 552  | 0.696      | 0.695     |     57.16 |    7.46 |         2 911 |
| rf       | 69 552  | 0.669      | 0.672     |     65.73 |    0.85 |         3 652 |
| catboost | 69 552  | 0.665      | 0.667     |      6.50 |    0.05 |         3 012 |
| tabicl   | 5 000   | 0.649      | 0.654     |      0.95 |   28.51 |         3 448 |
| tabpfn   | 5 000   | 0.645      | 0.651     |      0.04 |  614.55 |         1 566 |
| linear   | 69 552  | 0.626      | 0.630     |     20.86 |    0.02 |           633 |
| dummy    | 69 552  | 0.004      | 0.046     |      0.00 |    0.00 |           603 |

## Observations

- **XGBoost wins** at full scale (F1 0.697). LightGBM essentially ties it (0.696) but trains 2.7× slower and predicts 25× slower. **XGBoost on GPU is the current Pareto point**: best accuracy, fast train, fast predict, moderate memory.
- **CatBoost on GPU trains 10× faster than LightGBM** (6.5 s vs 57 s) but gives up 3 F1 points — likely under-fit; could close the gap with tuning.
- **Foundation models lose at scale.** In the 2 000-row dry run, tabicl led; at 70 k rows, both tabicl and tabpfn drop below the tree ensembles because they're still capped at 5 000 support rows (`FOUNDATION_MAX_TRAIN`). TabPFN's val-predict took ~10 min (614 s) on 5 000 rows — not practical at inference time for tile-wide prediction.
- **Linear baseline is non-trivial** (F1 0.626) — half the gap between dummy (0.004) and XGBoost (0.697) is captured by a linear model on the 66 features. Embeddings + LiDAR are well-conditioned.
- **ManyClassClassifier works**: 22-class TabPFN ran end-to-end via the output-coding wrapper. `alphabet_size=10` must be passed explicitly — left unset it raises `alphabet_size must be specified when base estimator has no limit`.
- **GPU utilisation**: XGBoost + CatBoost used the A40; LightGBM used CPU (GPU build not installed); tabpfn/tabicl used torch CUDA for their transformer backbones.
- **RF memory spike**: RF `fit_mem_delta_mb` = 3 006 MB — highest of the tree ensembles. Cap `max_depth=20` is already in place; consider lower for full production pipeline if memory is tight.

## Next steps

1. **Tune the leaders** (XGBoost, LightGBM) on `val` before picking a production model. Current configs are defaults + `n_estimators=500`.
2. **Re-enable W&B** (`--wandb-mode online`) for tracked sweeps.
3. **Test set**: once tuning is done, score on the untouched [test_values.shp](../../../data/P-Prosjekter2/154001_nyvest/landcover_megan/test_values.shp) (10 936 rows).
4. **Foundation models at larger support**: raise `FOUNDATION_MAX_TRAIN` to 10 000–20 000 and re-benchmark if inference-time budget allows — may close the accuracy gap at some VRAM/latency cost.
5. **Drop `linear` + `dummy`** from routine runs; they are sanity baselines and the gap is well-established.

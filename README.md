# nyvest

Geospatial land-cover modelling on the nyvest FSCS feature space. The repo is now trimmed around the runs that clear 0.7 macro-F1 under spatially blocked CV.

See [CLAUDE.md](CLAUDE.md) for environment and data-root conventions.

## Current Best Run

- Workflow: CC-APS pseudo-labelling + TabICL stage 2 on stable all-years AlphaEarth support.
- Feature set: 64 AlphaEarth bands plus 3 local lidar features: `elevation`, `tri`, `tch`.
- Result: macro-F1 `0.7139 +/- 0.0110`, balanced accuracy `0.7107`.
- Report: [common_ground/reports/research/schemeB_allyears_best_report.html](common_ground/reports/research/schemeB_allyears_best_report.html).
- Canonical JSON: [common_ground/reports/research/schemeB_allyears_results_perfold_fix_sup32k_lidar.json](common_ground/reports/research/schemeB_allyears_results_perfold_fix_sup32k_lidar.json).

The retained baseline JSONs are the high-F1 comparison points around the same family: AlphaEarth-only `sup32k`, saturated `sup36k`, `perfold_all_fix`, `ncgate`, and the TabPFN per-fold check.

## Retained Scripts

- [common_ground/scripts/llto_schemeB_allyears.py](common_ground/scripts/llto_schemeB_allyears.py) - canonical >0.7 evaluation workflow.
- [common_ground/scripts/clean_labels_perfold.py](common_ground/scripts/clean_labels_perfold.py) - train-fold clean-label prerequisite for leak-free cleaning modes.
- [scripts/extraction/sample_feature_space_stable_allyears.py](scripts/extraction/sample_feature_space_stable_allyears.py) - builds the stable all-years AlphaEarth support parquet.
- [scripts/extraction/sample_feature_space_unstable.py](scripts/extraction/sample_feature_space_unstable.py) - companion unstable all-years extraction used by the workflow.
- [scripts/extraction/extract_lidar_features.py](scripts/extraction/extract_lidar_features.py) - extracts the retained 3 m lidar features.
- [scripts/feature_probe_lidar.py](scripts/feature_probe_lidar.py) - documents/probes why `elevation`, `tri`, and `tch` are kept.

## Active Learning

Active-learning app files are kept separately under [active_learning](active_learning). That folder contains the Streamlit/Folium labeling app, seed-model training utilities, prediction/raster helpers, model metadata, and the timber benchmark notes from the app spike. Runtime labels, predictions, CatBoost model binaries, APS calibration arrays, and regenerated map rasters are ignored by git.

## Reproduce The Best Run

```bash
TOTAL_SUP=32000 \
CLEAN_MODE=perfold_all_fix \
OUT_SUFFIX=_perfold_fix_sup32k_lidar \
EXTRA_FEATURES=lidar \
~/myprojects/recover/.venv/bin/python common_ground/scripts/llto_schemeB_allyears.py
```

Input data are intentionally not tracked in git. On the Linux VDI they live under `/data/P-Prosjekter2/154001_nyvest`; project-local generated parquets under `data/` are ignored by git.

To open the retained report:

```bash
xdg-open common_ground/reports/research/schemeB_allyears_best_report.html
```
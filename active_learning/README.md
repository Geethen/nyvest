# Active Learning

Self-contained active-learning app and supporting utilities for interactive Norwegian nature-type labeling.

## Contents

- [scripts/app.py](scripts/app.py) - Streamlit + Folium labeling app.
- [scripts/train_seed_model.py](scripts/train_seed_model.py) - trains the fast CatBoost seed model and APS calibration.
- [scripts/build_predictions.py](scripts/build_predictions.py) - scores unique sample locations for the queue.
- [scripts/predict_raster.py](scripts/predict_raster.py) - generates a dense single-tile class and uncertainty raster.
- [models](models) - tracked class metadata; generated `seed_catboost.cbm` and `aps_calib.npz` are ignored.
- [reports](reports) - app spike notes, timber benchmark notes, and small retained preview artifacts.

## Run

```bash
~/myprojects/recover/.venv/bin/python -m streamlit run active_learning/scripts/app.py \
  --server.address 0.0.0.0 --server.port 8501
```

## Generate Seed Artifacts

```bash
~/myprojects/recover/.venv/bin/python active_learning/scripts/train_seed_model.py
~/myprojects/recover/.venv/bin/python active_learning/scripts/build_predictions.py
```

The app writes runtime labels and predictions to `active_learning/data/`, and dense map rasters to `active_learning/reports/`. Those generated outputs are ignored by git.
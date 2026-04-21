# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a shared virtual environment at `geo/` (Python 3.13, managed by `uv`). Always use it explicitly:

```bash
# Run a script
geo/Scripts/python script.py

# Install a package
uv pip install <package> --python geo/Scripts/python.exe

# Launch Jupyter
geo/Scripts/jupyter lab
```

The `geo` venv contains the full geospatial + ML stack: geopandas, rasterio, rioxarray, xarray, dask, torch, torchgeo, scikit-learn, lightgbm, xgboost, earthengine-api, geemap, and more.

## Project structure

```
data/        # raw and processed spatial/tabular data
notebooks/   # exploratory Jupyter notebooks
scripts/     # standalone processing scripts
models/      # saved model artefacts
plots/       # generated figures
reports/     # outputs and summaries
geo/         # shared virtual environment (not a Python package)
```

This is a geospatial analysis and modelling project. Work flows from `data/` → `notebooks/` or `scripts/` → `models/` + `plots/` + `reports/`.

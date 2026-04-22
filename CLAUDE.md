# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The venv differs by platform — always invoke the interpreter explicitly rather than relying on `PATH`:

| Platform | Venv root | Python | Jupyter |
|---|---|---|---|
| Local (Windows) | `geo/` (Python 3.13, in this repo) | `geo/Scripts/python.exe` | `geo/Scripts/jupyter` |
| Server / VDI (Linux) | `~/myprojects/recover/.venv` (Python 3.12, shared with the `recover` project) | `~/myprojects/recover/.venv/bin/python` | `~/myprojects/recover/.venv/bin/jupyter` |

```bash
# Install a package into the right venv
uv pip install <package> --python ~/myprojects/recover/.venv/bin/python   # server
uv pip install <package> --python geo/Scripts/python.exe                  # local
```

Dependencies are recorded in this repo's [pyproject.toml](pyproject.toml). On the server they are installed into the shared `recover` venv (`recover`'s own `pyproject.toml` is not modified). Stack includes: geopandas, pyogrio, rasterio, rioxarray, xarray, dask, torch, torchgeo, scikit-learn, lightgbm, xgboost, catboost, tabpfn, tabicl, wandb, earthengine-api, geemap.

### Platform differences

- **Data root**: `P:/154001_nyvest` on Windows local; `/data/P-Prosjekter2/154001_nyvest` on the Linux VDI. Scripts auto-detect both; override with `NYVEST_DATA_DIR` pointing at the project root.
- **GPU**: server has an NVIDIA A40 (24 GB). Scripts that support GPU take `--device {auto,cpu,cuda}` (default `auto`). `benchmark_tabular.py` uses it for XGBoost (`device="cuda"`) and CatBoost (`task_type="GPU"`); LightGBM GPU requires a custom build and is not wired up.

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

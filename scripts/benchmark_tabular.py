"""Benchmark tabular classifiers on NYVEST land-cover training data.

Inputs (from landcover_megan R pipeline):
    train_values.shp / val_values.shp / test_values.shp
    - 64 AlphaEarth embedding bands (embdd_1..embd_64)
    - 2 LiDAR bands (lidar_1, lidar_2)
    - target: fallbck (ecosystem class code)

Logs per-model train time, val predict time, macro F1, and balanced accuracy
to Weights & Biases, and prints a summary table.

Usage:
    geo/Scripts/python scripts/benchmark_tabular.py
    geo/Scripts/python scripts/benchmark_tabular.py --models rf,xgboost
    geo/Scripts/python scripts/benchmark_tabular.py --wandb-mode offline
"""

from __future__ import annotations

import argparse
import gc
import os
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("P:/154001_nyvest/landcover_megan")
TARGET = "fallbck"
DROP_COLS = ("split", "geometry")
WANDB_PROJECT = "nyvest-tabular-benchmark"

# Foundation models (TabPFN, TabICL) are trained in-context; cap the support
# set to keep inference tractable and stay near their pretraining regime.
# Inference memory is O(n_train x n_val) so keep both modest and chunk predict.
FOUNDATION_MAX_TRAIN = 5_000
FOUNDATION_MAX_VAL = 5_000
FOUNDATION_PREDICT_CHUNK = 500


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a split, dropping geometry at read-time and casting to float32.

    Using pyogrio with ignore_geometry avoids materialising ~75k geometries
    we never use; float32 halves the feature matrix footprint.
    """
    path = DATA_DIR / f"{name}_values.shp"
    # pyogrio's read_dataframe returns a plain pandas DF when geometry is ignored
    try:
        import pyogrio
        df = pyogrio.read_dataframe(path, read_geometry=False)
    except Exception:
        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        del gdf

    df = df.dropna(subset=[TARGET]).dropna()
    y = df[TARGET].astype(int)
    drop = [TARGET] + [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    float_cols = X.select_dtypes(include="float").columns
    X[float_cols] = X[float_cols].astype(np.float32)
    del df
    gc.collect()
    return X, y


def subsample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = 0):
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _peak_rss_mb(proc):
    # On Windows peak_wset is peak working set since process start.
    mi = proc.memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 1024**2


def _predict_chunked(model, X_val, chunk_size):
    if chunk_size is None or chunk_size >= len(X_val):
        return model.predict(X_val)
    parts = []
    for start in range(0, len(X_val), chunk_size):
        X_chunk = X_val.iloc[start:start + chunk_size] if hasattr(X_val, "iloc") else X_val[start:start + chunk_size]
        parts.append(model.predict(X_chunk))
        gc.collect()
    return np.concatenate(parts)


def fit_predict(model, X_train, y_train, X_val, predict_chunk_size=None):
    proc = psutil.Process()
    rss_start = proc.memory_info().rss / 1024**2

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    rss_after_fit = proc.memory_info().rss / 1024**2

    t0 = time.perf_counter()
    y_pred = _predict_chunked(model, X_val, predict_chunk_size)
    val_time = time.perf_counter() - t0
    rss_after_pred = proc.memory_info().rss / 1024**2
    peak_rss = _peak_rss_mb(proc)

    mem_stats = {
        "rss_start_mb": rss_start,
        "rss_after_fit_mb": rss_after_fit,
        "rss_after_pred_mb": rss_after_pred,
        "peak_rss_mb": peak_rss,
    }
    return y_pred, train_time, val_time, mem_stats


def build_model(name: str, n_classes: int, seed: int = 0, n_estimators: int = 500, n_jobs: int = 4):
    if name == "dummy":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="most_frequent", random_state=seed)
    if name == "linear":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=seed,
            max_depth=20,  # cap tree depth to limit memory
        )
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_estimators,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            eval_metric="mlogloss",
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=n_estimators,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
        )
    if name == "tabpfn":
        import json
        from tabpfn import TabPFNClassifier
        # TabPFN stores an install state file; user_id != null means authed.
        state_path = Path(os.environ.get("LOCALAPPDATA", "")) / "priorlabs" / ".tabpfn" / "state.json"
        authed = False
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                authed = bool(state.get("user_id"))
            except Exception:
                pass
        # TabPFN reads TABPFN_TOKEN (see tabpfn/browser_auth.py) and cached files
        # at ~/.cache/tabpfn/auth_token or ~/.tabpfn/token.
        token_files = [
            Path.home() / ".cache" / "tabpfn" / "auth_token",
            Path.home() / ".tabpfn" / "token",
        ]
        has_token_file = any(p.exists() and p.read_text().strip() for p in token_files)
        has_env = bool(os.environ.get("TABPFN_TOKEN"))
        # On Windows, setx writes to HKCU\Environment but isn't visible to the
        # current shell's child processes until a new login. Fall back to the
        # registry so the user doesn't need to restart terminals.
        if not has_env and os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                    try:
                        val, _ = winreg.QueryValueEx(k, "TABPFN_TOKEN")
                        if val:
                            os.environ["TABPFN_TOKEN"] = val
                            has_env = True
                    except FileNotFoundError:
                        pass
            except Exception:
                pass
        if not (authed or has_env or has_token_file):
            raise RuntimeError(
                "TabPFN auth not found. Set TABPFN_TOKEN env var, or run interactively once:\n"
                "  geo/Scripts/python -c \"from tabpfn import TabPFNClassifier; import numpy as np; "
                "TabPFNClassifier().fit(np.random.randn(10,4), np.random.randint(0,2,10))\"\n"
                "Paste your API key from https://ux.priorlabs.ai/account when prompted."
            )
        return TabPFNClassifier(
            random_state=seed,
            ignore_pretraining_limits=True,
        )
    if name == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def run_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_classes: int,
    seed: int,
    wandb_mode: str,
    n_estimators: int,
    n_jobs: int,
) -> dict:
    import wandb

    is_foundation = name in {"tabpfn", "tabicl"}
    if is_foundation:
        Xt, yt = subsample(X_train, y_train, FOUNDATION_MAX_TRAIN, seed)
        Xv, yv = subsample(X_val, y_val, FOUNDATION_MAX_VAL, seed)
        predict_chunk_size = FOUNDATION_PREDICT_CHUNK
    else:
        Xt, yt = X_train, y_train
        Xv, yv = X_val, y_val
        predict_chunk_size = None

    mem = psutil.virtual_memory()
    mem_available_gb = mem.available / 1024**3
    print(
        f"  [mem] available: {mem_available_gb:.1f} GB "
        f"({mem.percent}% used)  train: {len(Xt)}  val: {len(Xv)}"
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=name,
        group="initial-benchmark",
        mode=wandb_mode,
        reinit=True,
        config={
            "model": name,
            "n_train": len(Xt),
            "n_val": len(Xv),
            "n_features": X_train.shape[1],
            "n_classes": n_classes,
            "seed": seed,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
        },
    )

    model = None
    try:
        model = build_model(
            name, n_classes=n_classes, seed=seed,
            n_estimators=n_estimators, n_jobs=n_jobs,
        )
        y_pred, train_time, val_time, mem_stats = fit_predict(
            model, Xt, yt, Xv, predict_chunk_size=predict_chunk_size
        )
        f1 = f1_score(yv, y_pred, average="macro")
        bal_acc = balanced_accuracy_score(yv, y_pred)
        wandb.log(
            {
                "train_time_s": train_time,
                "val_time_s": val_time,
                "f1_macro": f1,
                "balanced_accuracy": bal_acc,
                **mem_stats,
            }
        )
        delta_fit = mem_stats["rss_after_fit_mb"] - mem_stats["rss_start_mb"]
        result = {
            "model": name,
            "n_train_used": len(Xt),
            "n_val_used": len(Xv),
            "train_time_s": round(train_time, 2),
            "val_time_s": round(val_time, 2),
            "f1_macro": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "peak_mem_mb": round(mem_stats["peak_rss_mb"], 0),
            "fit_mem_delta_mb": round(delta_fit, 0),
        }
    except Exception as e:
        wandb.log({"error": str(e)})
        result = {"model": name, "error": str(e)}
    finally:
        run.finish()
        if model is not None:
            del model
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="dummy,linear,rf,xgboost,lightgbm,catboost,tabpfn,tabicl",
        help="Comma-separated subset of: dummy,linear,rf,xgboost,lightgbm,catboost,tabpfn,tabicl",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Trees/iterations for RF, XGBoost, CatBoost (ignored by foundation models)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel workers for RF/XGBoost/CatBoost (default: half of CPU cores)",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Cap training rows (applies to all models). Foundation models are still capped further.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="Cap validation rows (applies to all models).",
    )
    args = parser.parse_args()

    print(f"Loading data from {DATA_DIR} ...")
    X_train, y_train_raw = load_split("train")
    X_val, y_val_raw = load_split("val")

    if args.max_train_rows is not None:
        X_train, y_train_raw = subsample(X_train, y_train_raw, args.max_train_rows, args.seed)
    if args.max_val_rows is not None:
        X_val, y_val_raw = subsample(X_val, y_val_raw, args.max_val_rows, args.seed)

    le = LabelEncoder().fit(
        pd.concat([y_train_raw, y_val_raw], ignore_index=True)
    )
    y_train = pd.Series(le.transform(y_train_raw), name=TARGET)
    y_val = pd.Series(le.transform(y_val_raw), name=TARGET)
    n_classes = len(le.classes_)

    mem_total_gb = psutil.virtual_memory().total / 1024**3
    print(
        f"  train: {X_train.shape}  val: {X_val.shape}  "
        f"features: {X_train.shape[1]}  classes: {n_classes}"
    )
    print(f"  system: {os.cpu_count()} CPUs, {mem_total_gb:.1f} GB RAM, using n_jobs={args.n_jobs}")
    print(f"  class distribution (train): {pd.Series(y_train).value_counts().to_dict()}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []
    out = Path(__file__).parent.parent / "reports" / "benchmark_results.csv"
    out.parent.mkdir(exist_ok=True)

    for name in models:
        print(f"\n=== {name} ===")
        res = run_model(
            name, X_train, y_train, X_val, y_val,
            n_classes=n_classes, seed=args.seed, wandb_mode=args.wandb_mode,
            n_estimators=args.n_estimators, n_jobs=args.n_jobs,
        )
        results.append(res)
        summary = pd.DataFrame(results)
        print("\n--- Results so far ---")
        print(summary.to_string(index=False))
        summary.to_csv(out, index=False)

    print(f"\nFinal summary saved to: {out}")


if __name__ == "__main__":
    main()

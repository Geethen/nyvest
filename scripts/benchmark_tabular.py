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
import math
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

TARGET = "fallbck"
# Known project roots across environments. P:/ is the Windows local mount;
# /data/P-Prosjekter2 is the Linux VDI mount of the same share.
_DATA_ROOT_CANDIDATES = (
    Path("/data/P-Prosjekter2/154001_nyvest"),
    Path("P:/154001_nyvest"),
)


def get_data_dir() -> Path:
    """Resolve the landcover_megan data directory across local/server envs.

    Override with NYVEST_DATA_DIR (points to the project root containing
    landcover_megan/).
    """
    env = os.environ.get("NYVEST_DATA_DIR")
    if env:
        root = Path(env)
    else:
        root = next((p for p in _DATA_ROOT_CANDIDATES if p.exists()), None)
        if root is None:
            raise FileNotFoundError(
                "Could not locate NYVEST project root. Tried: "
                + ", ".join(str(p) for p in _DATA_ROOT_CANDIDATES)
                + ". Set NYVEST_DATA_DIR to the project root."
            )
    data_dir = root / "landcover_megan"
    if not data_dir.exists():
        raise FileNotFoundError(f"landcover_megan not found under {root}")
    return data_dir


DATA_DIR = get_data_dir()
DROP_COLS = ("split", "geometry")
WANDB_PROJECT = "nyvest-tabular-benchmark"

# Foundation models (TabPFN, TabICL) are trained in-context; cap the support
# set to keep inference tractable and stay near their pretraining regime.
# Inference memory is O(n_train x n_val). Keep validation modest and use larger
# predict batches; tiny repeated predict calls can be disproportionately slow.
FOUNDATION_MAX_TRAIN = 5_000
FOUNDATION_MAX_VAL = 5_000
FOUNDATION_PREDICT_CHUNK = 1_000


def _resolve_tabpfn_model_version(version_name: str):
    """Resolve tabpfn ModelVersion enum member by name across releases."""
    try:
        from tabpfn.constants import ModelVersion
    except Exception:
        try:
            # Backward compatibility for older tabpfn import paths.
            from tabpfn.model import ModelVersion
        except Exception as e:
            raise RuntimeError(
                "Could not import tabpfn ModelVersion. Upgrade tabpfn to a "
                "release that exposes create_default_for_version()."
            ) from e

    if hasattr(ModelVersion, version_name):
        return getattr(ModelVersion, version_name)

    available = [k for k in dir(ModelVersion) if k.isupper()]
    raise RuntimeError(
        f"Requested TabPFN version '{version_name}' is not available in the "
        f"installed tabpfn package. Available: {available}"
    )


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


def _predict_chunked(model, X_val, chunk_size, show_progress=False, progress_label="predict"):
    if chunk_size is None or chunk_size >= len(X_val):
        if show_progress:
            print(f"  [progress] {progress_label}: single-batch predict on {len(X_val)} rows", flush=True)
        return model.predict(X_val)

    parts = []
    total = len(X_val)
    n_chunks = (total + chunk_size - 1) // chunk_size
    t0 = time.perf_counter()
    for chunk_idx, start in enumerate(range(0, total, chunk_size), start=1):
        X_chunk = X_val.iloc[start:start + chunk_size] if hasattr(X_val, "iloc") else X_val[start:start + chunk_size]
        parts.append(model.predict(X_chunk))
        if show_progress:
            done = min(start + chunk_size, total)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("inf")
            eta_str = f"{eta:.1f}s" if math.isfinite(eta) else "n/a"
            print(
                f"  [progress] {progress_label}: chunk {chunk_idx}/{n_chunks} "
                f"rows {done}/{total} elapsed {elapsed:.1f}s eta {eta_str}",
                flush=True,
            )
        gc.collect()
    return np.concatenate(parts)


def fit_predict(
    model,
    X_train,
    y_train,
    X_val,
    predict_chunk_size=None,
    show_progress=False,
    progress_label="model",
):
    proc = psutil.Process()
    rss_start = proc.memory_info().rss / 1024**2

    t0 = time.perf_counter()
    if show_progress:
        print(f"  [progress] {progress_label}: fit started", flush=True)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    if show_progress:
        print(f"  [progress] {progress_label}: fit finished in {train_time:.2f}s", flush=True)
    rss_after_fit = proc.memory_info().rss / 1024**2

    t0 = time.perf_counter()
    y_pred = _predict_chunked(
        model,
        X_val,
        predict_chunk_size,
        show_progress=show_progress,
        progress_label=f"{progress_label} predict",
    )
    val_time = time.perf_counter() - t0
    if show_progress:
        print(f"  [progress] {progress_label}: predict finished in {val_time:.2f}s", flush=True)
    rss_after_pred = proc.memory_info().rss / 1024**2
    peak_rss = _peak_rss_mb(proc)

    mem_stats = {
        "rss_start_mb": rss_start,
        "rss_after_fit_mb": rss_after_fit,
        "rss_after_pred_mb": rss_after_pred,
        "peak_rss_mb": peak_rss,
    }
    return y_pred, train_time, val_time, mem_stats


def resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' if a CUDA device is visible, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_model(
    name: str,
    n_classes: int,
    seed: int = 0,
    n_estimators: int = 500,
    n_jobs: int = 4,
    device: str = "cpu",
):
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
            device=device,
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
            task_type="GPU" if device == "cuda" else "CPU",
        )
    if name in {"tabpfn", "tabpfn_v26", "tabpfn_v3"}:
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
        if name in {"tabpfn_v26", "tabpfn_v3"}:
            # Prefer explicit V3 when present. If unavailable but V2_6 exists,
            # use that as the next-best explicit modern default.
            if name == "tabpfn_v3":
                try:
                    version = _resolve_tabpfn_model_version("V3")
                except RuntimeError:
                    version = _resolve_tabpfn_model_version("V2_6")
            else:
                version = _resolve_tabpfn_model_version("V2_6")
            base = TabPFNClassifier.create_default_for_version(
                version,
                random_state=seed,
                ignore_pretraining_limits=True,
            )
        else:
            base = TabPFNClassifier(
                random_state=seed,
                ignore_pretraining_limits=True,
            )
        # TabPFN natively supports up to 10 classes; beyond that, wrap with the
        # ManyClassClassifier output-coding extension.
        # https://docs.priorlabs.ai/extensions/many-class
        if n_classes > 10:
            from tabpfn_extensions.many_class import ManyClassClassifier
            # alphabet_size = TabPFN's native class limit; current API requires
            # passing it explicitly when the base estimator doesn't expose a limit.
            return ManyClassClassifier(base, alphabet_size=10, random_state=seed)
        return base
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
    device: str,
    foundation_max_train: int,
    foundation_max_val: int,
    show_progress: bool,
) -> dict:
    import wandb

    is_foundation = name in {"tabpfn", "tabpfn_v26", "tabpfn_v3", "tabicl"}
    if is_foundation:
        Xt, yt = subsample(X_train, y_train, foundation_max_train, seed)
        Xv, yv = subsample(X_val, y_val, foundation_max_val, seed)
        predict_chunk_size = FOUNDATION_PREDICT_CHUNK
        model_progress = show_progress
    else:
        Xt, yt = X_train, y_train
        Xv, yv = X_val, y_val
        predict_chunk_size = None
        model_progress = False

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
            "device": device,
        },
    )

    model = None
    try:
        model = build_model(
            name, n_classes=n_classes, seed=seed,
            n_estimators=n_estimators, n_jobs=n_jobs, device=device,
        )
        y_pred, train_time, val_time, mem_stats = fit_predict(
            model,
            Xt,
            yt,
            Xv,
            predict_chunk_size=predict_chunk_size,
            show_progress=model_progress,
            progress_label=name,
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
        default="dummy,linear,rf,xgboost,lightgbm,catboost,tabpfn,tabpfn_v26,tabpfn_v3,tabicl",
        help="Comma-separated subset of: dummy,linear,rf,xgboost,lightgbm,catboost,tabpfn,tabpfn_v26,tabpfn_v3,tabicl",
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
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for GPU-capable models (xgboost, catboost). "
             "'auto' picks cuda if torch sees a GPU, else cpu.",
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
    parser.add_argument(
        "--foundation-max-train",
        type=int,
        default=FOUNDATION_MAX_TRAIN,
        help="Cap train rows for foundation models (tabpfn/tabpfn_v26/tabpfn_v3/tabicl).",
    )
    parser.add_argument(
        "--foundation-max-val",
        type=int,
        default=FOUNDATION_MAX_VAL,
        help="Cap validation rows for foundation models (tabpfn/tabpfn_v26/tabpfn_v3/tabicl).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress logs for long foundation-model fit/predict stages (default: enabled).",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

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
    print(f"  system: {os.cpu_count()} CPUs, {mem_total_gb:.1f} GB RAM, using n_jobs={args.n_jobs}, device={device}")
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
            n_estimators=args.n_estimators, n_jobs=args.n_jobs, device=device,
            foundation_max_train=args.foundation_max_train,
            foundation_max_val=args.foundation_max_val,
            show_progress=args.progress,
        )
        results.append(res)
        summary = pd.DataFrame(results)
        print("\n--- Results so far ---")
        print(summary.to_string(index=False))
        summary.to_csv(out, index=False)

    print(f"\nFinal summary saved to: {out}")


if __name__ == "__main__":
    main()

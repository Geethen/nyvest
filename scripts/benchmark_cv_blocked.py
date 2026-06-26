"""3-fold spatially-blocked CV on the FSCS AlphaEarth parquet.

Folds are formed by GroupKFold on `cell_id` so all points within a 25 km grid
cell stay together — preventing the over-optimistic estimates that come from
having near-neighbour pixels split across train/val.

Models, build_model, fit_predict, resolve_device are reused from
benchmark_tabular.py so this script is the CV harness, nothing more.

Inputs:
  data/grunnkart_nyvest_fscs_alphaearth.parquet
    columns: A00..A63 (features), class (label), cell_id (group), lon, lat, ...

Outputs:
  reports/benchmark_cv_blocked_per_fold.csv  one row per (model, fold)
  reports/benchmark_cv_blocked_summary.csv   one row per model (mean/std)

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/benchmark_cv_blocked.py
  ~/myprojects/recover/.venv/bin/python scripts/benchmark_cv_blocked.py \
      --models linear,rf,xgboost --wandb-mode offline
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import model factory + fit helper from the existing benchmark script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from benchmark_tabular import (  # noqa: E402
    FOUNDATION_MAX_TRAIN,
    FOUNDATION_MAX_VAL,
    FOUNDATION_PREDICT_CHUNK,
    build_model,
    fit_predict,
    resolve_device,
    subsample,
)

PARQUET = _HERE.parent / "data" / "grunnkart_nyvest_fscs_alphaearth.parquet"
REPORTS = _HERE.parent / "reports"
WANDB_PROJECT = "nyvest-tabular-benchmark"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"
GROUP = "cell_id"

DEFAULT_MODELS = ("dummy,linear,rf,xgboost,lightgbm,catboost,"
                  "tabpfn,tabpfn_v26,tabpfn_v3,tabicl")
FOUNDATION = {"tabpfn", "tabpfn_v26", "tabpfn_v3", "tabicl"}


def load_fscs(parquet=PARQUET):
    import duckdb
    df = duckdb.sql(f"SELECT * FROM '{parquet}'").df()
    # Float32 to halve memory; matches the existing benchmark.
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    print(f"  loaded {len(df):,} rows × {len(df.columns)} cols "
          f"from {Path(parquet).name}")
    print(f"  classes: {sorted(df[TARGET].unique().tolist())}")
    print(f"  unique cells: {df[GROUP].nunique()}")
    return df


def run_fold(name, X_tr, y_tr, X_va, y_va, n_classes, seed, n_estimators,
             n_jobs, device, fold_idx, wandb_mode, show_progress,
             foundation_max_train=FOUNDATION_MAX_TRAIN,
             foundation_max_val=FOUNDATION_MAX_VAL):
    import wandb

    is_foundation = name in FOUNDATION
    if is_foundation:
        # foundation_max_train=0 means "no cap"
        if foundation_max_train and foundation_max_train > 0:
            Xt, yt = subsample(X_tr, y_tr, foundation_max_train, seed)
        else:
            Xt, yt = X_tr, y_tr
        if foundation_max_val and foundation_max_val > 0:
            Xv, yv = subsample(X_va, y_va, foundation_max_val, seed)
        else:
            Xv, yv = X_va, y_va
        chunk = FOUNDATION_PREDICT_CHUNK
    else:
        Xt, yt = X_tr, y_tr
        Xv, yv = X_va, y_va
        chunk = None

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{name}-fold{fold_idx}",
        group="cv-blocked-3fold",
        mode=wandb_mode,
        reinit=True,
        config={
            "model": name, "fold": fold_idx,
            "n_train": len(Xt), "n_val": len(Xv),
            "n_features": len(FEATURE_COLS), "n_classes": n_classes,
            "seed": seed, "n_estimators": n_estimators,
            "n_jobs": n_jobs, "device": device,
        },
    )

    model = None
    try:
        model = build_model(
            name, n_classes=n_classes, seed=seed,
            n_estimators=n_estimators, n_jobs=n_jobs, device=device,
        )
        y_pred, train_time, val_time, mem_stats = fit_predict(
            model, Xt, yt, Xv,
            predict_chunk_size=chunk,
            show_progress=show_progress and is_foundation,
            progress_label=f"{name}-fold{fold_idx}",
        )
        f1 = f1_score(yv, y_pred, average="macro")
        bal_acc = balanced_accuracy_score(yv, y_pred)
        wandb.log({
            "train_time_s": train_time, "val_time_s": val_time,
            "f1_macro": f1, "balanced_accuracy": bal_acc, **mem_stats,
        })
        res = {
            "model": name, "fold": fold_idx,
            "n_train_used": len(Xt), "n_val_used": len(Xv),
            "train_time_s": round(train_time, 2),
            "val_time_s": round(val_time, 2),
            "f1_macro": round(float(f1), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "peak_mem_mb": round(mem_stats["peak_rss_mb"], 0),
        }
    except Exception as e:
        wandb.log({"error": str(e)})
        res = {"model": name, "fold": fold_idx, "error": str(e)}
    finally:
        run.finish()
        if model is not None:
            del model
        gc.collect()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(PARQUET),
                        help="Path to input parquet (must have A00..A63, class, "
                             "cell_id columns).")
    parser.add_argument("--reports", default=str(REPORTS),
                        help="Directory to write CSV outputs.")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--foundation-max-train", type=int,
                        default=FOUNDATION_MAX_TRAIN,
                        help="Cap train rows for foundation models "
                             "(tabpfn*/tabicl). Use 0 to disable cap.")
    parser.add_argument("--foundation-max-val", type=int,
                        default=FOUNDATION_MAX_VAL,
                        help="Cap val rows for foundation models. "
                             "Use 0 to disable cap.")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing per-fold CSV instead of "
                             "overwriting (useful for adding new models).")
    parser.add_argument("--summary-suffix", default="",
                        help="Append this suffix to the summary CSV name, "
                             "e.g. '_fm_nocap' -> benchmark_cv_blocked_"
                             "summary_fm_nocap.csv.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    parquet = Path(args.parquet)
    reports = Path(args.reports)
    print(f"Loading {parquet} ...")
    df = load_fscs(parquet)
    df = df.dropna(subset=FEATURE_COLS + [TARGET, GROUP]).reset_index(drop=True)
    print(f"  after dropna: {len(df):,} rows")

    le = LabelEncoder().fit(df[TARGET].astype(int))
    y_all = pd.Series(le.transform(df[TARGET].astype(int)), name=TARGET)
    X_all = df[FEATURE_COLS].reset_index(drop=True)
    groups = df[GROUP].reset_index(drop=True)
    n_classes = len(le.classes_)
    print(f"  classes (encoded): {n_classes}  groups: {groups.nunique()}")

    cv = GroupKFold(n_splits=args.n_splits)
    folds = list(cv.split(X_all, y_all, groups=groups))
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i}: train={len(tr):,}  val={len(va):,}  "
              f"train_cells={groups.iloc[tr].nunique()}  "
              f"val_cells={groups.iloc[va].nunique()}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    reports.mkdir(exist_ok=True)
    suffix = args.summary_suffix
    per_fold_csv = reports / f"benchmark_cv_blocked_per_fold{suffix}.csv"
    summary_csv = reports / f"benchmark_cv_blocked_summary{suffix}.csv"

    rows = []
    if args.append and per_fold_csv.exists():
        rows = pd.read_csv(per_fold_csv).to_dict(orient="records")
        print(f"  appending to {per_fold_csv.name} ({len(rows)} prior rows)")
    for name in models:
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        for fold_idx, (tr, va) in enumerate(folds):
            X_tr = X_all.iloc[tr].reset_index(drop=True)
            y_tr = y_all.iloc[tr].reset_index(drop=True)
            X_va = X_all.iloc[va].reset_index(drop=True)
            y_va = y_all.iloc[va].reset_index(drop=True)
            res = run_fold(
                name, X_tr, y_tr, X_va, y_va, n_classes,
                seed=args.seed, n_estimators=args.n_estimators,
                n_jobs=args.n_jobs, device=device,
                fold_idx=fold_idx, wandb_mode=args.wandb_mode,
                show_progress=args.progress,
                foundation_max_train=args.foundation_max_train,
                foundation_max_val=args.foundation_max_val,
            )
            rows.append(res)
            pd.DataFrame(rows).to_csv(per_fold_csv, index=False)
            print(f"  fold {fold_idx}: "
                  f"f1={res.get('f1_macro')}  "
                  f"bal_acc={res.get('balanced_accuracy')}  "
                  f"err={res.get('error', '')[:80]}")
        print(f"  [{name}] total {time.perf_counter()-t0:.1f}s")

    # Aggregate
    df_pf = pd.DataFrame(rows)
    if "f1_macro" in df_pf.columns:
        agg = (df_pf.dropna(subset=["f1_macro"])
                   .groupby("model")
                   .agg(f1_mean=("f1_macro", "mean"),
                        f1_std=("f1_macro", "std"),
                        bal_acc_mean=("balanced_accuracy", "mean"),
                        bal_acc_std=("balanced_accuracy", "std"),
                        train_time_mean=("train_time_s", "mean"),
                        peak_mem_mean=("peak_mem_mb", "mean"),
                        n_folds=("f1_macro", "count"))
                   .sort_values("f1_mean", ascending=False)
                   .reset_index())
        agg.to_csv(summary_csv, index=False)
        print(f"\n--- Summary (ranked by mean macro-F1) ---")
        print(agg.to_string(index=False))
        print(f"\nSaved: {per_fold_csv}")
        print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()

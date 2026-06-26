"""Learning curves for models from the blocked-CV benchmark.

Reads `reports/benchmark_cv_blocked_summary.csv` for default model selection
(top-k by macro-F1), then refits each model on log-spaced training-size cuts
within the same 3-fold GroupKFold split on cell_id. The training subsample
at each size is stratified by class so rare classes don't disappear from
very small training sets.

Per-class macro-F1 (one column per encoded class label) is recorded
alongside the aggregate macro-F1 / balanced accuracy.

Outputs:
  reports/learning_curves.csv               (model, fold, n_train, f1_macro,
                                             bal_acc, train_time_s, f1_class_*)
  plots/learning_curves.png                 (mean ± std curve per model)
  plots/learning_curves_per_class.png       (small-multiples: panel per class,
                                             models overlaid)

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/learning_curves.py
  ~/myprojects/recover/.venv/bin/python scripts/learning_curves.py \
      --models linear,xgboost,catboost,tabicl \
      --sizes-tabicl 1000,5000,full
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
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
PLOTS = _HERE.parent / "plots"
SUMMARY_CSV = REPORTS / "benchmark_cv_blocked_summary.csv"
LC_CSV = REPORTS / "learning_curves.csv"
LC_PNG = PLOTS / "learning_curves.png"
LC_PER_CLASS_PNG = PLOTS / "learning_curves_per_class.png"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"
GROUP = "cell_id"
FOUNDATION = {"tabpfn", "tabpfn_v26", "tabpfn_v3", "tabicl"}


def stratified_cap(X, y, n, seed):
    n = int(n)
    if n >= len(X):
        return X, y
    vc = y.value_counts()
    if (vc < 2).any():
        return subsample(X, y, n, seed)
    X_keep, _, y_keep, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed)
    return X_keep.reset_index(drop=True), y_keep.reset_index(drop=True)


def load_fscs():
    import duckdb
    df = duckdb.sql(f"SELECT * FROM '{PARQUET}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df


def top_models_from_summary(k=3, override=None):
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found — run benchmark_cv_blocked.py first.")
    s = pd.read_csv(SUMMARY_CSV)
    s = s[s["model"] != "dummy"]
    return s.sort_values("f1_mean", ascending=False).head(k)["model"].tolist()


def log_sizes(n_max, n_sizes=5, n_min=1000):
    n_min = min(n_min, max(100, n_max // 50))
    sizes = np.logspace(np.log10(n_min), np.log10(n_max), n_sizes)
    sizes = sorted({int(round(s)) for s in sizes})
    if sizes[-1] != n_max:
        sizes[-1] = n_max
    return sizes


def parse_size_spec(spec: str, n_max: int) -> list[int]:
    """Parse a comma-separated list of sizes. 'full' or 'max' → n_max."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("full", "max"):
            out.append(n_max)
        else:
            out.append(min(int(tok), n_max))
    return sorted(set(out))


def fit_and_score(name, X_tr, y_tr, X_va, y_va, n_classes, class_labels,
                  seed, n_estimators, n_jobs, device):
    is_foundation = name in FOUNDATION
    if is_foundation:
        # Foundation cap only applies if the requested train slice exceeds it
        # AND we want to apply it (TabICL uncapped paths pass through here too;
        # we trust the outer loop's n_train choice). Keep val capped as before.
        if len(X_tr) > FOUNDATION_MAX_TRAIN:
            Xt, yt = subsample(X_tr, y_tr, FOUNDATION_MAX_TRAIN, seed) \
                if FOUNDATION_MAX_TRAIN > 0 else (X_tr, y_tr)
        else:
            Xt, yt = X_tr, y_tr
        if FOUNDATION_MAX_VAL > 0 and len(X_va) > FOUNDATION_MAX_VAL:
            Xv, yv = subsample(X_va, y_va, FOUNDATION_MAX_VAL, seed)
        else:
            Xv, yv = X_va, y_va
        chunk = FOUNDATION_PREDICT_CHUNK
    else:
        Xt, yt = X_tr, y_tr
        Xv, yv = X_va, y_va
        chunk = None

    model = build_model(
        name, n_classes=n_classes, seed=seed,
        n_estimators=n_estimators, n_jobs=n_jobs, device=device,
    )
    y_pred, train_time, val_time, _ = fit_predict(
        model, Xt, yt, Xv,
        predict_chunk_size=chunk,
        show_progress=False,
    )
    f1_macro = f1_score(yv, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(yv, y_pred)
    per_class = f1_score(yv, y_pred, labels=class_labels,
                         average=None, zero_division=0)
    result = dict(
        n_train_used=len(Xt),
        n_val_used=len(Xv),
        train_time_s=round(train_time, 2),
        val_time_s=round(val_time, 2),
        f1_macro=round(float(f1_macro), 4),
        balanced_accuracy=round(float(bal_acc), 4),
    )
    for cls, f1c in zip(class_labels, per_class):
        result[f"f1_class_{int(cls)}"] = round(float(f1c), 4)
    del model
    gc.collect()
    return result


def plot_aggregate(df, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = (df.groupby(["model", "n_train"])
             .agg(f1_mean=("f1_macro", "mean"),
                  f1_std=("f1_macro", "std"))
             .reset_index())
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in agg.groupby("model"):
        g = g.sort_values("n_train")
        ax.errorbar(g["n_train"], g["f1_mean"], yerr=g["f1_std"],
                    marker="o", capsize=3, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("training rows (log)")
    ax.set_ylabel("macro-F1 (3-fold blocked CV)")
    ax.set_title("Learning curves — GroupKFold on cell_id")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_per_class(df, class_labels, class_names, out_png):
    """Small-multiples: one panel per class, all models overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_cols = [f"f1_class_{int(c)}" for c in class_labels]
    models = sorted(df["model"].dropna().unique().tolist())
    n_cls = len(class_cols)
    ncols = 5
    nrows = int(np.ceil(n_cls / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    for idx, col in enumerate(class_cols):
        ax = axes[idx]
        for m in models:
            g = (df[df["model"] == m]
                 .groupby("n_train")[col]
                 .agg(["mean", "std"])
                 .reset_index()
                 .sort_values("n_train"))
            if g.empty:
                continue
            ax.errorbar(g["n_train"], g["mean"], yerr=g["std"].fillna(0),
                        marker="o", markersize=3, capsize=2,
                        linewidth=1, color=color_map[m], label=m)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which="both", alpha=0.3)
        label = class_names[idx] if class_names else col.replace("f1_class_", "class ")
        ax.set_title(label, fontsize=9)

    for j in range(n_cls, len(axes)):
        axes[j].axis("off")

    for ax in axes[::ncols]:
        ax.set_ylabel("F1")
    for ax in axes[-ncols:]:
        ax.set_xlabel("n_train (log)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Per-class learning curves (3-fold blocked CV)", y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list (overrides auto-top-k)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-sizes", type=int, default=5,
                        help="Default cuts for models without an explicit "
                             "--sizes-<model> override.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    # Per-model size overrides; values are comma-separated ints or 'full'.
    for m in ("xgboost", "catboost", "linear", "rf", "lightgbm",
              "tabpfn", "tabpfn_v3", "tabpfn_v26", "tabicl"):
        parser.add_argument(f"--sizes-{m}", default=None,
                            help=f"Override training-size cuts for {m} "
                                 "(comma list, 'full' for fold max)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    models = top_models_from_summary(args.top_k, override=args.models)
    print(f"Models for learning curves: {models}")

    df_all = load_fscs()
    df_all = df_all.dropna(subset=FEATURE_COLS + [TARGET, GROUP]).reset_index(drop=True)
    le = LabelEncoder().fit(df_all[TARGET].astype(int))
    y_all = pd.Series(le.transform(df_all[TARGET].astype(int)), name=TARGET)
    X_all = df_all[FEATURE_COLS].reset_index(drop=True)
    groups = df_all[GROUP].reset_index(drop=True)
    n_classes = len(le.classes_)
    class_labels = list(range(n_classes))
    class_names = [str(int(c)) for c in le.classes_]
    print(f"  classes (raw → encoded): "
          f"{dict(zip([int(c) for c in le.classes_], class_labels))}")

    cv = GroupKFold(n_splits=args.n_splits)
    folds = list(cv.split(X_all, y_all, groups=groups))
    fold_train_sizes = [len(tr) for tr, _ in folds]
    print(f"  fold train sizes: {fold_train_sizes}  classes: {n_classes}")

    def sizes_for(name, fold_n_max):
        override = getattr(args, f"sizes_{name}".replace("-", "_"), None)
        if override:
            return parse_size_spec(override, fold_n_max)
        return log_sizes(fold_n_max, n_sizes=args.n_sizes)

    REPORTS.mkdir(exist_ok=True)
    rows = []
    for name in models:
        print(f"\n=== {name} ===")
        for fold_idx, (tr, va) in enumerate(folds):
            X_tr_full = X_all.iloc[tr].reset_index(drop=True)
            y_tr_full = y_all.iloc[tr].reset_index(drop=True)
            X_va = X_all.iloc[va].reset_index(drop=True)
            y_va = y_all.iloc[va].reset_index(drop=True)
            sizes = sizes_for(name, len(X_tr_full))
            print(f"  fold {fold_idx} sizes: {sizes}")
            for n_train in sizes:
                t0 = time.perf_counter()
                X_tr, y_tr = stratified_cap(
                    X_tr_full, y_tr_full, n_train, args.seed + fold_idx)
                try:
                    res = fit_and_score(
                        name, X_tr, y_tr, X_va, y_va, n_classes,
                        class_labels=class_labels,
                        seed=args.seed, n_estimators=args.n_estimators,
                        n_jobs=args.n_jobs, device=device,
                    )
                    res.update({"model": name, "fold": fold_idx,
                                "n_train": n_train,
                                "wall_s": round(time.perf_counter() - t0, 1)})
                except Exception as e:
                    res = {"model": name, "fold": fold_idx,
                           "n_train": n_train,
                           "error": str(e)[:200],
                           "wall_s": round(time.perf_counter() - t0, 1)}
                rows.append(res)
                pd.DataFrame(rows).to_csv(LC_CSV, index=False)
                print(f"  fold {fold_idx} n_train={n_train:>6,} "
                      f"f1={res.get('f1_macro')} "
                      f"bal_acc={res.get('balanced_accuracy')} "
                      f"wall={res.get('wall_s')}s "
                      f"err={res.get('error','')[:60]}")

    df = pd.DataFrame(rows)
    if "f1_macro" in df.columns:
        valid = df.dropna(subset=["f1_macro"])
        plot_aggregate(valid, LC_PNG)
        plot_per_class(valid, class_labels, class_names, LC_PER_CLASS_PNG)
        print(f"\nSaved: {LC_CSV}")
        print(f"Saved: {LC_PNG}")
        print(f"Saved: {LC_PER_CLASS_PNG}")


if __name__ == "__main__":
    main()

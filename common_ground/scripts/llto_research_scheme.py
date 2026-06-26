"""Compare a class-merging scheme: CatBoost (CPU) + TabICL zero-shot (GPU).

Usage:
  python llto_research_scheme.py --scheme A   # 11 classes: 1+2 -> bare
  python llto_research_scheme.py --scheme B   # 10 classes: 1+2, 8+9
  python llto_research_scheme.py --scheme C   # 9 classes: 1+2, 8+9, 11+12

For each scheme runs three configurations per fold (3 LLTO folds):
  catboost_baseline   CatBoost 500 iter, CPU, full stable+pseudo
  catboost_high_iter  CatBoost 1 000 iter, CPU
  tabicl_zero_10k     TabICL zero-shot, n_est=8, random 10 000-row support

Writes:
  common_ground/reports/research/scheme_<X>_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import resolve_device  # noqa: E402

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

STABLE_PARQUET   = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"

# Merge maps. Source class -> target class. Unlisted classes untouched.
SCHEMES = {
    "A": {1: 2},           # 11 classes: sand + rock -> bare (we keep id=2)
    "B": {1: 2, 9: 8},     # 10 classes: + freshwater + marine -> water (id=8)
    "C": {1: 2, 9: 8, 12: 11},  # 9 classes: + snow -> sparse
}
SCHEME_NAMES = {
    "A": "1+2 -> bare (11 classes)",
    "B": "1+2 -> bare,  8+9 -> water (10 classes)",
    "C": "1+2 -> bare,  8+9 -> water,  11+12 -> sparse/snow (9 classes)",
}

N_FOLDS    = 3
SEED       = 0
SUPPORT_10K = 10_000


def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id",
                                            "lon", "lat"]).reset_index(drop=True)


def merge_classes(y: np.ndarray, merge_map: dict) -> np.ndarray:
    y = y.copy().astype(int)
    for src, tgt in merge_map.items():
        y[y == src] = tgt
    return y


def make_catboost_cpu(iterations):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU",
        loss_function="MultiClass",
    )


def make_tabicl_zero(n_estimators=8):
    from tabicl import TabICLClassifier
    return TabICLClassifier(
        n_estimators=n_estimators, random_state=SEED,
        verbose=False, device="cuda",
    )


def random_subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def fit_and_eval(model, X_tr, y_tr, X_te, y_te, n_classes):
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = np.asarray(model.predict(X_te)).ravel().astype(int)
    t_pred = time.perf_counter() - t0
    f1m = f1_score(y_te, y_pred, average="macro",
                   labels=np.arange(n_classes), zero_division=0)
    bal = balanced_accuracy_score(y_te, y_pred)
    f1_pc = f1_score(y_te, y_pred, labels=np.arange(n_classes),
                     average=None, zero_division=0)
    return {"f1_macro": round(float(f1m), 4),
            "bal_acc":  round(float(bal), 4),
            "fit_s":    round(t_fit, 2),
            "pred_s":   round(t_pred, 2),
            "f1_per_class": [round(float(v), 4) for v in f1_pc]}


def run(scheme: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if scheme not in SCHEMES:
        raise SystemExit(f"Unknown scheme {scheme!r}; choose from {list(SCHEMES)}")
    merge_map = SCHEMES[scheme]
    print(f"Scheme {scheme}: {SCHEME_NAMES[scheme]}")
    print(f"Merge map: {merge_map}")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    y_stable_m   = merge_classes(stable[TARGET].values, merge_map)
    y_unstable_m = merge_classes(unstable[TARGET].values, merge_map)
    merged_classes = sorted(set(y_stable_m.tolist()) | set(y_unstable_m.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    print(f"merged classes: {decoded}  n={n_classes}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    results: dict[str, list] = {}
    exp_meta: dict[str, str] = {}
    t_global = time.perf_counter()

    for k in range(N_FOLDS):
        s_test  = (stable_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_s_tr = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr = le.transform(y_stable_m[s_train])
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(y_stable_m[s_test])
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)

        # Stage 1: CatBoost CPU 500 iter on 80% of stable_train
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost_cpu(500)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        probs_u  = stage1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_u = probs_u.argmax(axis=1)

        X_aug = np.concatenate([X_s_tr, X_u_tr], axis=0)
        y_aug = np.concatenate([y_s_tr, pseudo_u], axis=0)

        rng = np.random.default_rng(SEED + k)
        Xs_10k, ys_10k = random_subsample(X_aug, y_aug, SUPPORT_10K, rng)

        print(f"\n=== fold {k}  n_aug={len(y_aug):,}  n_test={len(y_s_te):,} ===")

        experiments = [
            ("catboost_baseline",
             "CatBoost 500 iter, CPU, full stable+pseudo (Stage 2)",
             lambda: make_catboost_cpu(500), X_aug, y_aug),
            ("catboost_high_iter",
             "CatBoost 1 000 iter, CPU, full stable+pseudo",
             lambda: make_catboost_cpu(1000), X_aug, y_aug),
            ("tabicl_zero_10k",
             "TabICL zero-shot, n_est=8, random 10 000-row support",
             lambda: make_tabicl_zero(8), Xs_10k, ys_10k),
        ]

        for name, desc, model_fn, X_tr, y_tr in experiments:
            if name not in results:
                results[name] = []
                exp_meta[name] = desc
            try:
                m = fit_and_eval(model_fn(), X_tr, y_tr, X_s_te, y_s_te, n_classes)
            except Exception as e:
                err = str(e)[:200]
                print(f"  [{name:<22}] FAILED: {type(e).__name__}: {err}")
                m = {"f1_macro": float("nan"), "bal_acc": float("nan"),
                     "fit_s": 0.0, "pred_s": 0.0,
                     "f1_per_class": [float("nan")] * n_classes,
                     "error": err}
            m["fold"] = k
            m["n_train"] = int(len(y_tr))
            results[name].append(m)
            print(f"  [{name:<22}] F1={m['f1_macro']:.4f}  "
                  f"bal={m['bal_acc']:.4f}  fit={m['fit_s']:.1f}s  pred={m['pred_s']:.1f}s")

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    summary = {}
    for name, folds in results.items():
        f1s  = [f["f1_macro"] for f in folds if not np.isnan(f["f1_macro"])]
        bals = [f["bal_acc"]  for f in folds if not np.isnan(f["bal_acc"])]
        valid_pcs = [f["f1_per_class"] for f in folds
                     if not any(np.isnan(v) for v in f["f1_per_class"])]
        pc_mean = (np.array(valid_pcs).mean(axis=0).tolist()
                   if valid_pcs else [float("nan")] * n_classes)
        summary[name] = {
            "description": exp_meta[name],
            "f1_mean":  round(float(np.mean(f1s)),  4) if f1s else float("nan"),
            "f1_std":   round(float(np.std(f1s)),   4) if f1s else float("nan"),
            "bal_mean": round(float(np.mean(bals)), 4) if bals else float("nan"),
            "bal_std":  round(float(np.std(bals)),  4) if bals else float("nan"),
            "f1_per_class": {str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)},
            "per_fold": folds,
        }

    out_json = OUT_DIR / f"scheme_{scheme}_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": scheme,
                   "scheme_name": SCHEME_NAMES[scheme],
                   "merge_map": {str(k): v for k, v in merge_map.items()},
                   "merged_classes": decoded,
                   "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1),
                   "n_folds": N_FOLDS,
                   "results": summary}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    print(f"\n{'Experiment':<22}  {'F1':>8}  {'std':>6}  {'bal':>6}")
    print("-" * 50)
    for name, s in sorted(summary.items(),
                          key=lambda x: -(x[1]["f1_mean"] if not np.isnan(x[1]["f1_mean"]) else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<22}  {'NaN':>8}  {'NaN':>6}  {'NaN':>6}  [FAIL]")
        else:
            print(f"{name:<22}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  {s['bal_mean']:>6.4f}")

    return out_json


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", required=True, choices=list(SCHEMES))
    args = p.parse_args()
    run(args.scheme)

"""TabICL experiments for LLTO Common Ground (nyvest).

Translates the round-1/round-2 CatBoost ideas to TabICL where possible,
and adds TabICL-specific levers (ensemble passes, support set construction,
softmax temperature).

CatBoost knob -> TabICL adaptation
  iterations / depth / l2_reg       -> not applicable (frozen transformer)
  class_weight / sample_weight      -> not supported by TabICL API
                                       -> approximated via stratified support set
  pseudo-label hard filters         -> directly applicable (changes training set)

Experiments
  tabicl_baseline       n_est=8,  random 5 000-row support
  tabicl_n16            n_est=16, random 5 000-row support
  tabicl_n32            n_est=32, random 5 000-row support
  tabicl_strat_5k       n_est=8,  stratified 5 000-row support (~417/class)
  tabicl_strat_n16      n_est=16, stratified 5 000-row support
  tabicl_temp_05        n_est=8,  softmax_temperature=0.5 (sharper)
  tabicl_pseudo_top80   n_est=8,  random 5 000 from top-80%-confidence pseudo-set
  tabicl_support_10k    n_est=8,  random 10 000-row support

Stage 1 is CatBoost 500 iter (same as round 1/2) so only Stage 2 varies.

Output
  common_ground/reports/research/tabicl_results.json
  common_ground/reports/research/tabicl_report.html
"""

from __future__ import annotations

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

BASELINE_F1  = 0.6197
BASELINE_PC  = {1: 0.0382, 2: 0.5479, 3: 0.7338, 4: 0.6913,
                5: 0.3635, 6: 0.6257, 7: 0.5839, 8: 0.7908,
                9: 0.9065, 10: 0.7561, 11: 0.7304, 12: 0.6679}

N_FOLDS    = 3
SEED       = 0
SUPPORT_5K = 5_000
SUPPORT_10K = 10_000


def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id",
                                            "lon", "lat"]).reset_index(drop=True)


def make_catboost_stage1(device):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="GPU" if device == "cuda" else "CPU",
        loss_function="MultiClass",
    )


def make_tabicl(n_estimators=8, softmax_temperature=0.9):
    from tabicl import TabICLClassifier
    return TabICLClassifier(
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        random_state=SEED,
        verbose=False,
    )


def random_subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def stratified_subsample(X, y, n_per_class_target, n_classes, rng):
    """Take min(n_per_class_target, available) from each class.
    Returns the concatenated stratified support set (no replacement, no oversampling).
    """
    parts_X, parts_y = [], []
    for c in range(n_classes):
        mask = y == c
        avail = int(mask.sum())
        if avail == 0:
            continue
        take = min(n_per_class_target, avail)
        cls_idx = np.flatnonzero(mask)
        chosen = rng.choice(cls_idx, size=take, replace=False)
        parts_X.append(X[chosen])
        parts_y.append(y[chosen])
    X_out = np.concatenate(parts_X, axis=0)
    y_out = np.concatenate(parts_y, axis=0)
    # shuffle so TabICL doesn't see a class-ordered context
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def fit_and_eval(model, X_tr, y_tr, X_te, y_te, n_classes):
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


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  seed={SEED}")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    classes = sorted(set(stable[TARGET].astype(int).unique()) |
                     set(unstable[TARGET].astype(int).unique()))
    le = LabelEncoder().fit(np.array(classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    n_per_class_target = SUPPORT_5K // n_classes  # 416 per class
    print(f"classes: {decoded}  n={n_classes}  per-class target={n_per_class_target}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    rng_global = np.random.default_rng(SEED)
    results: dict[str, list] = {}
    exp_meta: dict[str, str] = {}
    t_global = time.perf_counter()

    for k in range(N_FOLDS):
        s_test  = (stable_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_s_tr = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr = le.transform(stable.loc[s_train, TARGET].astype(int).values)
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(stable.loc[s_test,  TARGET].astype(int).values)
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)

        # Stage 1: CatBoost (cheap, consistent with round 1/2)
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost_stage1(device)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        probs_u  = stage1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_u = probs_u.argmax(axis=1)
        conf_u   = probs_u.max(axis=1)

        # Full augmented (stable + all pseudo) — same as cg_llto baseline
        X_aug_full = np.concatenate([X_s_tr, X_u_tr], axis=0)
        y_aug_full = np.concatenate([y_s_tr, pseudo_u], axis=0)
        # Filtered augmented (stable + pseudo above 80th-pct conf)
        thresh = float(np.percentile(conf_u, 80))
        mask80 = conf_u >= thresh
        X_aug_top80 = np.concatenate([X_s_tr, X_u_tr[mask80]], axis=0)
        y_aug_top80 = np.concatenate([y_s_tr, pseudo_u[mask80]], axis=0)

        print(f"\n=== fold {k}  n_test={s_test.sum():,}  "
              f"aug_full={len(y_aug_full):,}  aug_top80={len(y_aug_top80):,}  "
              f"thresh_top80={thresh:.4f} ===")

        # Build support sets per experiment
        rng = np.random.default_rng(SEED + k)
        # random 5k from full
        Xs_rand5k, ys_rand5k = random_subsample(X_aug_full, y_aug_full, SUPPORT_5K, rng)
        # stratified 5k from full
        Xs_strat5k, ys_strat5k = stratified_subsample(
            X_aug_full, y_aug_full, n_per_class_target, n_classes, rng)
        # random 5k from top80 pseudo-filtered set
        Xs_top80_5k, ys_top80_5k = random_subsample(X_aug_top80, y_aug_top80, SUPPORT_5K, rng)
        # random 10k from full
        Xs_rand10k, ys_rand10k = random_subsample(X_aug_full, y_aug_full, SUPPORT_10K, rng)

        print(f"   rand_5k:    {len(ys_rand5k):,} rows, class dist={np.bincount(ys_rand5k, minlength=n_classes).tolist()}")
        print(f"   strat_5k:   {len(ys_strat5k):,} rows, class dist={np.bincount(ys_strat5k, minlength=n_classes).tolist()}")
        print(f"   top80_5k:   {len(ys_top80_5k):,} rows, class dist={np.bincount(ys_top80_5k, minlength=n_classes).tolist()}")
        print(f"   rand_10k:   {len(ys_rand10k):,} rows")

        experiments = [
            ("tabicl_baseline",
             "n_estimators=8, random 5 000-row support (matches existing benchmark)",
             make_tabicl(n_estimators=8), Xs_rand5k, ys_rand5k),

            ("tabicl_n16",
             "n_estimators=16, random 5 000-row support (≈ CatBoost high_iter analogue)",
             make_tabicl(n_estimators=16), Xs_rand5k, ys_rand5k),

            ("tabicl_n32",
             "n_estimators=32, random 5 000-row support (more ensemble passes)",
             make_tabicl(n_estimators=32), Xs_rand5k, ys_rand5k),

            ("tabicl_strat_5k",
             f"n_estimators=8, stratified 5 000-row support (~{n_per_class_target}/class)",
             make_tabicl(n_estimators=8), Xs_strat5k, ys_strat5k),

            ("tabicl_strat_n16",
             f"n_estimators=16 + stratified 5 000-row support (~{n_per_class_target}/class)",
             make_tabicl(n_estimators=16), Xs_strat5k, ys_strat5k),

            ("tabicl_temp_05",
             "n_estimators=8, softmax_temperature=0.5 (sharper class probabilities)",
             make_tabicl(n_estimators=8, softmax_temperature=0.5), Xs_rand5k, ys_rand5k),

            ("tabicl_pseudo_top80",
             "n_estimators=8, random 5 000 from top-80%-confidence pseudo-set",
             make_tabicl(n_estimators=8), Xs_top80_5k, ys_top80_5k),

            ("tabicl_support_10k",
             "n_estimators=8, random 10 000-row support (more context)",
             make_tabicl(n_estimators=8), Xs_rand10k, ys_rand10k),
        ]

        for name, desc, model, Xs, ys in experiments:
            if name not in results:
                results[name] = []
                exp_meta[name] = desc
            try:
                metrics = fit_and_eval(model, Xs, ys, X_s_te, y_s_te, n_classes)
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")
                metrics = {"f1_macro": float("nan"), "bal_acc": float("nan"),
                           "fit_s": 0.0, "pred_s": 0.0,
                           "f1_per_class": [float("nan")] * n_classes,
                           "error": str(e)}
            metrics["fold"] = k
            metrics["n_support"] = int(len(ys))
            results[name].append(metrics)
            print(f"  [{name:<24}] F1={metrics['f1_macro']:.4f}  "
                  f"bal={metrics['bal_acc']:.4f}  "
                  f"fit={metrics['fit_s']:.1f}s  pred={metrics['pred_s']:.1f}s  "
                  f"n_sup={metrics['n_support']:,}")

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
            "delta_vs_baseline": round(float(np.mean(f1s)) - BASELINE_F1, 4) if f1s else float("nan"),
            "f1_per_class": {str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)},
            "per_fold": folds,
        }

    out_json = OUT_DIR / "tabicl_results.json"
    with open(out_json, "w") as f:
        json.dump({"elapsed_s": round(elapsed, 1),
                   "n_folds": N_FOLDS,
                   "device": device,
                   "classes": decoded,
                   "weak_classes": [1, 2, 5, 6, 7],
                   "baseline_f1": BASELINE_F1,
                   "baseline_per_class": BASELINE_PC,
                   "results": summary}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    print(f"\n{'Experiment':<26}  {'F1 mean':>8}  {'±std':>6}  {'Δ baseline':>10}")
    print("-" * 56)
    for name, s in sorted(summary.items(),
                          key=lambda x: -(x[1]["f1_mean"] if not np.isnan(x[1]["f1_mean"]) else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<26}  {'NaN':>8}  {'NaN':>6}  {'NaN':>10}  [FAIL]")
            continue
        flag = " ★" if s["delta_vs_baseline"] > 0.003 else ""
        print(f"{name:<26}  {s['f1_mean']:>8.4f}  "
              f"{s['f1_std']:>6.4f}  {s['delta_vs_baseline']:>+10.4f}{flag}")

    return out_json


if __name__ == "__main__":
    run()

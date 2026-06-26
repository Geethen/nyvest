"""Round-2 auto-research for LLTO Common Ground (nyvest).

Builds on round-1 findings:
  - high_iter (1 000) and class_weight (CatBoost) were the keepers.
  - Oversampling is explicitly excluded.

Round-2 experiments
  baseline                 -- CatBoost 500 iter, no weighting
  high_iter                -- CatBoost 1 000 iter
  class_weight             -- CatBoost 500 iter + full inverse-freq weights
  high_iter_1500           -- CatBoost 1 500 iter (does benefit plateau?)
  high_iter_cw_full        -- 1 000 iter + full inverse-freq weights
  high_iter_cw_targeted    -- 1 000 iter + 3x weight on weak classes {1,2,5,6,7}
  pseudo_top80             -- drop pseudo-labels with top-1 prob below 80th pct
  pseudo_class5_strict     -- class-5 pseudo-labels need top-1 prob >= 0.70
  catboost_depth8          -- depth=8 at 500 iter
  catboost_l2_10           -- l2_leaf_reg=10 at 500 iter (stronger regularisation)
  xgboost_gpu              -- XGBoost on GPU, 500 trees

Output
  common_ground/reports/research/round2_results.json
  common_ground/reports/research/round2_report.html
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

WEAK_CLASSES = [1, 2, 5, 6, 7]   # below 0.65 in baseline

N_FOLDS = 3
SEED    = 0
N_JOBS  = 4


def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id",
                                            "lon", "lat"]).reset_index(drop=True)


def cw_inverse_freq(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


def cw_targeted(decoded: list, weak_decoded: list, boost: float) -> np.ndarray:
    """Weights array indexed by *encoded* class; 1.0 for non-weak, `boost` for weak."""
    w = np.ones(len(decoded), dtype=np.float64)
    weak_set = set(weak_decoded)
    for i, cls in enumerate(decoded):
        if int(cls) in weak_set:
            w[i] = boost
    return w


def make_catboost(iterations, device, class_weights=None, **kw):
    from catboost import CatBoostClassifier
    kwargs = dict(
        iterations=iterations, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=N_JOBS,
        task_type="GPU" if device == "cuda" else "CPU",
        loss_function="MultiClass",
    )
    if class_weights is not None:
        kwargs["class_weights"] = class_weights.tolist()
    kwargs.update(kw)
    return CatBoostClassifier(**kwargs)


def make_xgb(device, n_classes):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=n_classes,
        random_state=SEED, n_jobs=N_JOBS,
        tree_method="hist", device="cuda" if device == "cuda" else "cpu",
        verbosity=0,
    )


def fit_and_eval(model, X_tr, y_tr, X_te, y_te, n_classes,
                 sample_weight=None):
    t0 = time.perf_counter()
    if sample_weight is not None:
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    y_pred = np.asarray(model.predict(X_te)).ravel().astype(int)
    f1m = f1_score(y_te, y_pred, average="macro",
                   labels=np.arange(n_classes), zero_division=0)
    bal = balanced_accuracy_score(y_te, y_pred)
    f1_pc = f1_score(y_te, y_pred, labels=np.arange(n_classes),
                     average=None, zero_division=0)
    return {"f1_macro": round(float(f1m), 4),
            "bal_acc": round(float(bal), 4),
            "fit_s": round(fit_s, 2),
            "f1_per_class": [round(float(v), 4) for v in f1_pc]}


def build_aug(X_s_tr, y_s_tr, X_u_tr, pseudo_u, mask):
    """Return X_aug, y_aug using stable + masked pseudo unstable."""
    X_u_kept = X_u_tr[mask]
    y_u_kept = pseudo_u[mask]
    X_aug = np.concatenate([X_s_tr, X_u_kept], axis=0)
    y_aug = np.concatenate([y_s_tr, y_u_kept], axis=0)
    return X_aug, y_aug


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
    print(f"classes: {decoded}  n={n_classes}")
    weak_encoded = [decoded.index(c) for c in WEAK_CLASSES]
    print(f"weak (below 0.65 baseline): {WEAK_CLASSES} -> encoded {weak_encoded}")

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
        y_s_tr = le.transform(stable.loc[s_train, TARGET].astype(int).values)
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(stable.loc[s_test,  TARGET].astype(int).values)
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)

        # Stage 1 (always plain CatBoost 500)
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost(500, device)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        probs_u  = stage1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_u = probs_u.argmax(axis=1)
        conf_u   = probs_u.max(axis=1)

        # filters
        mask_all      = np.ones(len(pseudo_u), dtype=bool)
        thresh_top80  = np.percentile(conf_u, 80)
        mask_top80    = conf_u >= thresh_top80
        # class-5-strict: keep all non-class-5; class-5 only when conf >= 0.70
        c5_enc        = decoded.index(5)
        mask_c5strict = ~((pseudo_u == c5_enc) & (conf_u < 0.70))

        # class weights
        cw_full = cw_inverse_freq(y_s_tr, n_classes)
        cw_targ = cw_targeted(decoded, WEAK_CLASSES, boost=3.0)

        # ── build experiment list per fold (so weights see y_s_tr) ──
        experiments = [
            ("baseline", "CatBoost 500 iter, no weighting (sanity re-run)",
             make_catboost(500, device), mask_all, None),
            ("high_iter", "CatBoost 1 000 iter, no weighting",
             make_catboost(1000, device), mask_all, None),
            ("class_weight", "CatBoost 500 iter + full inverse-freq class weights",
             make_catboost(500, device, class_weights=cw_full), mask_all, None),
            ("high_iter_1500", "CatBoost 1 500 iter, no weighting",
             make_catboost(1500, device), mask_all, None),
            ("high_iter_cw_full", "CatBoost 1 000 iter + full inverse-freq weights",
             make_catboost(1000, device, class_weights=cw_full), mask_all, None),
            ("high_iter_cw_targeted", f"CatBoost 1 000 iter + 3x weight on classes {WEAK_CLASSES}",
             make_catboost(1000, device, class_weights=cw_targ), mask_all, None),
            ("pseudo_top80", "CatBoost 500 iter; drop pseudo-labels below 80th-pct confidence",
             make_catboost(500, device), mask_top80, None),
            ("pseudo_class5_strict", "CatBoost 500 iter; class-5 pseudo-labels need top-1 prob >= 0.70",
             make_catboost(500, device), mask_c5strict, None),
            ("catboost_depth8", "CatBoost 500 iter, depth=8 (vs default 6)",
             make_catboost(500, device, depth=8), mask_all, None),
            ("catboost_l2_10", "CatBoost 500 iter, l2_leaf_reg=10 (stronger regularisation)",
             make_catboost(500, device, l2_leaf_reg=10), mask_all, None),
            ("xgboost_gpu", "XGBoost 500 trees, depth=6, GPU",
             make_xgb(device, n_classes), mask_all, None),
        ]

        print(f"\n=== fold {k}  n_train_stable={s_train.sum():,}  "
              f"n_test={s_test.sum():,}  n_unstable={u_train.sum():,} ===")
        print(f"   pseudo_top80 thresh={thresh_top80:.4f} → kept {mask_top80.sum()}/{len(mask_top80)}")
        print(f"   pseudo_class5_strict → kept {mask_c5strict.sum()}/{len(mask_c5strict)}")

        for name, desc, model, mask, sw_fn in experiments:
            if name not in results:
                results[name] = []
                exp_meta[name] = desc
            X_aug, y_aug = build_aug(X_s_tr, y_s_tr, X_u_tr, pseudo_u, mask)
            sw = None if sw_fn is None else sw_fn(y_aug)
            try:
                metrics = fit_and_eval(model, X_aug, y_aug,
                                       X_s_te, y_s_te, n_classes,
                                       sample_weight=sw)
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")
                metrics = {"f1_macro": float("nan"), "bal_acc": float("nan"),
                           "fit_s": 0.0, "f1_per_class": [float("nan")] * n_classes,
                           "error": str(e)}
            metrics["fold"] = k
            metrics["n_train"] = int(len(y_aug))
            results[name].append(metrics)
            print(f"  [{name:<24}] F1={metrics['f1_macro']:.4f}  "
                  f"bal={metrics['bal_acc']:.4f}  fit={metrics['fit_s']:.1f}s  "
                  f"n={metrics['n_train']:,}")

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

    out_json = OUT_DIR / "round2_results.json"
    with open(out_json, "w") as f:
        json.dump({"elapsed_s": round(elapsed, 1),
                   "n_folds": N_FOLDS,
                   "device": device,
                   "classes": decoded,
                   "weak_classes": WEAK_CLASSES,
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

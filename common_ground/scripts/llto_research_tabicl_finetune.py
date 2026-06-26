"""TabICL fine-tuning experiments for LLTO Common Ground (nyvest).

Background
  TabICL v2.1.1 added FinetunedTabICLClassifier (in-context fine-tuning of the
  frozen transformer's backbone). The pretrained model supports up to
  `max_classes = 10` natively. Our grunnkart task has 12 classes and the
  multi-class fine-tuning code path triggers a CUDA assert
  (`logits[..., :n_classes]` does not route through the hierarchical
  many-class strategy that zero-shot prediction uses).

  Workaround: merge the two smallest classes into their dominant confusion
  targets so we drop to 10 classes:
    class 1  (n=352) -> class 6  (dominant: 123 of 352 misclassified to 6)
    class 12 (n=159) -> class 11 (dominant: 56 of 159 misclassified to 11)
  Remaining classes are unchanged.

Caveat
  All fine-tune metrics here are 10-class macro-F1; they are NOT comparable
  to the 12-class baseline F1 = 0.6197. To make the comparison fair this
  script ALSO scores CatBoost on the same merged 10-class label space and
  reports both. The 10-class CatBoost number is the head-to-head reference.

  Also: `amp=False` is required at fit time — the default `amp=True` triggers
  "CUDA driver error: operation not supported" on this driver/torch combo
  (driver 535.274.02 / torch 2.5.1+cu124).

Experiments (all on the merged 10-class problem)
  catboost_10cls_baseline      CatBoost 500 iter on merged labels (reference)
  catboost_10cls_high_iter     CatBoost 1 000 iter (carryover of round-2 winner)
  tabicl_10cls_zero            TabICL zero-shot, n_estimators=8, random 5 000 support
  tabicl_10cls_zero_strat      TabICL zero-shot, n_estimators=8, stratified 5 000 support
  tabicl_10cls_ft_30ep         TabICL fine-tune, 30 epochs, lr=1e-5, max_data=5 000
  tabicl_10cls_ft_60ep_lowlr   TabICL fine-tune, 60 epochs, lr=5e-6 (slower, longer)
  tabicl_10cls_ft_strat        TabICL fine-tune on stratified 5 000 support set

Output
  common_ground/reports/research/tabicl_ft_results.json
  appended to the round-3 HTML report
"""

from __future__ import annotations

import json
import os
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
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — it triggers
# "CUDA driver error: operation not supported" on torch 2.5.1+cu124 with this
# system's NVIDIA driver (535.274.02). The OOM risk is managed instead by
# capping support sets and n_estimators_inference.

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import resolve_device  # noqa: E402

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

STABLE_PARQUET   = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"

# 10-class merge map: source -> target. Anything not listed stays the same.
CLASS_MERGE = {1: 6, 12: 11}

N_FOLDS    = 3
SEED       = 0
SUPPORT_5K = 5_000


def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id",
                                            "lon", "lat"]).reset_index(drop=True)


def merge_classes(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype(int)
    for src, tgt in CLASS_MERGE.items():
        y[y == src] = tgt
    return y


def make_catboost(iterations, device):
    # Force CPU: CatBoost-GPU runs in the same process as TabICL-CUDA and
    # the two together exhaust the 24 GB A40 in this shared/multi-user env.
    # CatBoost CPU takes ~60s vs ~4s but is reliable; only used for the
    # 10-class reference numbers (2 experiments × 3 folds = 6 fits).
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU",
        loss_function="MultiClass",
    )


def random_subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def stratified_subsample(X, y, n_per_class_target, n_classes, rng):
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
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def fit_and_eval(model_fn, X_tr, y_tr, X_te, y_te, n_classes,
                 needs_val: bool = False, val_frac: float = 0.20):
    # Aggressively clear CUDA cache before each TabICL run; the previous
    # estimator's KV/activation cache otherwise sits there.
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    t0 = time.perf_counter()
    if needs_val:
        X_t, X_v, y_t, y_v = train_test_split(
            X_tr, y_tr, test_size=val_frac, stratify=y_tr, random_state=SEED)
        model = model_fn()
        model.fit(X_t, y_t, X_val=X_v, y_val=y_v)
    else:
        model = model_fn()
        model.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = np.asarray(model.predict(X_te)).ravel().astype(int)
    t_pred = time.perf_counter() - t0
    # Capture extras before freeing the model
    extras = {}
    if hasattr(model, "best_epoch_"):
        extras["best_epoch"] = int(getattr(model, "best_epoch_", -1) or -1)
    # Free GPU after predict (don't `del model` because Python then complains
    # about referencing a deleted local further down even if guarded).
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    f1m = f1_score(y_te, y_pred, average="macro",
                   labels=np.arange(n_classes), zero_division=0)
    bal = balanced_accuracy_score(y_te, y_pred)
    f1_pc = f1_score(y_te, y_pred, labels=np.arange(n_classes),
                     average=None, zero_division=0)
    return ({"f1_macro": round(float(f1m), 4),
             "bal_acc":  round(float(bal), 4),
             "fit_s":    round(t_fit, 2),
             "pred_s":   round(t_pred, 2),
             "f1_per_class": [round(float(v), 4) for v in f1_pc],
             **extras})


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  seed={SEED}")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    # ---- Merge to ≤10 classes ----
    y_stable_merged   = merge_classes(stable[TARGET].values)
    y_unstable_merged = merge_classes(unstable[TARGET].values)
    merged_classes = sorted(set(y_stable_merged.tolist()) | set(y_unstable_merged.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    print(f"merged classes: {decoded}  n={n_classes}  (originally 12; merged 1→6, 12→11)")
    n_per_class_target = SUPPORT_5K // n_classes
    print(f"per-class target for stratified sampling: {n_per_class_target}")

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
        y_s_tr = le.transform(y_stable_merged[s_train])
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(y_stable_merged[s_test])
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)

        # Stage 1 (CatBoost 500 iter on 80% of merged stable_train)
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost(500, device)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        probs_u  = stage1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_u = probs_u.argmax(axis=1)

        X_aug_full = np.concatenate([X_s_tr, X_u_tr], axis=0)
        y_aug_full = np.concatenate([y_s_tr, pseudo_u], axis=0)

        rng = np.random.default_rng(SEED + k)
        Xs_rand5k, ys_rand5k = random_subsample(X_aug_full, y_aug_full, SUPPORT_5K, rng)
        Xs_strat5k, ys_strat5k = stratified_subsample(
            X_aug_full, y_aug_full, n_per_class_target, n_classes, rng)
        # Smaller support set for fine-tune to stay within GPU memory
        SUPPORT_FT = 3000
        Xs_rand_ft, ys_rand_ft = random_subsample(X_aug_full, y_aug_full, SUPPORT_FT, rng)
        ft_per_class_target = SUPPORT_FT // n_classes
        Xs_strat_ft, ys_strat_ft = stratified_subsample(
            X_aug_full, y_aug_full, ft_per_class_target, n_classes, rng)

        print(f"\n=== fold {k}  n_train_stable={s_train.sum():,}  "
              f"n_test={s_test.sum():,}  n_unstable={u_train.sum():,} ===")
        print(f"   rand_5k class dist:  {np.bincount(ys_rand5k, minlength=n_classes).tolist()}")
        print(f"   strat_5k class dist: {np.bincount(ys_strat5k, minlength=n_classes).tolist()}")

        # ── experiments ─────────────────────────────────────────
        # CatBoost on merged 10-class problem — head-to-head reference
        def cb_baseline():
            return make_catboost(500, device)
        def cb_high_iter():
            return make_catboost(1000, device)

        def tabicl_zero(n_est=4):
            # n_est=4 (not 8) keeps prediction within the GPU memory budget
            # alongside the ~5 000-row support and ~25 000-row test set.
            from tabicl import TabICLClassifier
            return TabICLClassifier(n_estimators=n_est, random_state=SEED, verbose=False, device="cuda")

        def tabicl_ft(epochs=30, lr=1e-5, n_est_inf=4, max_data=3000):
            # Memory-tight defaults: n_est_finetune=1 (vs 2), max_data=3000
            # (vs 5000), n_est_inf=4. Fine-tune holds optimizer state +
            # activations for backward; this fits in ~20 GB once the zero-shot
            # model is no longer cached.
            from tabicl import FinetunedTabICLClassifier
            return FinetunedTabICLClassifier(
                epochs=epochs, learning_rate=lr, weight_decay=0.01,
                early_stopping=True, patience=6, min_delta=1e-4,
                eval_metric="log_loss", random_state=SEED, verbose=False,
                n_estimators_finetune=1, n_estimators_validation=1,
                n_estimators_inference=n_est_inf,
                max_data_size=max_data, device="cuda", amp=False,
            )

        experiments = [
            # Catboost reference (full augmented set)
            ("catboost_10cls_baseline",
             "CatBoost 500 iter, full stable + pseudo (10-class merged)",
             cb_baseline, X_aug_full, y_aug_full, False),
            ("catboost_10cls_high_iter",
             "CatBoost 1 000 iter, full stable + pseudo (10-class merged)",
             cb_high_iter, X_aug_full, y_aug_full, False),

            # TabICL zero-shot (n_est=4 for GPU memory budget)
            ("tabicl_10cls_zero",
             "TabICL zero-shot, n_est=4, random 5 000 support",
             tabicl_zero, Xs_rand5k, ys_rand5k, False),
            ("tabicl_10cls_zero_strat",
             f"TabICL zero-shot, n_est=4, stratified 5 000 support (~{n_per_class_target}/class)",
             tabicl_zero, Xs_strat5k, ys_strat5k, False),

            # TabICL fine-tune (smaller support set + n_est_finetune=1 for GPU budget)
            ("tabicl_10cls_ft_30ep",
             f"TabICL fine-tune, 30 epochs, lr=1e-5, {SUPPORT_FT}-row random support",
             lambda: tabicl_ft(epochs=30, lr=1e-5, max_data=SUPPORT_FT),
             Xs_rand_ft, ys_rand_ft, True),
            ("tabicl_10cls_ft_60ep_lowlr",
             f"TabICL fine-tune, 60 epochs, lr=5e-6 (slower decay), {SUPPORT_FT}-row support",
             lambda: tabicl_ft(epochs=60, lr=5e-6, max_data=SUPPORT_FT),
             Xs_rand_ft, ys_rand_ft, True),
            ("tabicl_10cls_ft_strat",
             f"TabICL fine-tune, 30 epochs, lr=1e-5, stratified {SUPPORT_FT}-row support",
             lambda: tabicl_ft(epochs=30, lr=1e-5, max_data=SUPPORT_FT),
             Xs_strat_ft, ys_strat_ft, True),
        ]

        for name, desc, model_fn, X_tr, y_tr, needs_val in experiments:
            if name not in results:
                results[name] = []
                exp_meta[name] = desc
            try:
                metrics = fit_and_eval(model_fn, X_tr, y_tr, X_s_te, y_s_te,
                                       n_classes, needs_val=needs_val)
            except Exception as e:
                err = str(e)[:150]
                print(f"  [{name:<28}] FAILED: {type(e).__name__}: {err}")
                metrics = {"f1_macro": float("nan"), "bal_acc": float("nan"),
                           "fit_s": 0.0, "pred_s": 0.0,
                           "f1_per_class": [float("nan")] * n_classes,
                           "error": err}
            metrics["fold"] = k
            metrics["n_support"] = int(len(y_tr))
            results[name].append(metrics)
            be = f"  best_ep={metrics.get('best_epoch','-')}" if "best_epoch" in metrics else ""
            print(f"  [{name:<28}] F1={metrics['f1_macro']:.4f}  "
                  f"bal={metrics['bal_acc']:.4f}  "
                  f"fit={metrics['fit_s']:.1f}s  pred={metrics['pred_s']:.1f}s{be}")

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

    out_json = OUT_DIR / "tabicl_ft_results.json"
    with open(out_json, "w") as f:
        json.dump({"elapsed_s": round(elapsed, 1),
                   "n_folds": N_FOLDS,
                   "device": device,
                   "merged_classes": decoded,
                   "class_merge_map": {str(k): v for k, v in CLASS_MERGE.items()},
                   "results": summary}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # head-to-head: catboost_10cls_baseline is the reference
    if "catboost_10cls_baseline" in summary and not np.isnan(summary["catboost_10cls_baseline"]["f1_mean"]):
        ref = summary["catboost_10cls_baseline"]["f1_mean"]
    else:
        ref = None
    print(f"\n{'Experiment':<28}  {'F1 mean':>8}  {'±std':>6}  {'Δ ref10':>10}")
    print("-" * 60)
    for name, s in sorted(summary.items(),
                          key=lambda x: -(x[1]["f1_mean"] if not np.isnan(x[1]["f1_mean"]) else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<28}  {'NaN':>8}  {'NaN':>6}  {'NaN':>10}  [FAIL]")
            continue
        delta = s["f1_mean"] - ref if ref is not None else float("nan")
        flag = " ★" if (delta is not None and not np.isnan(delta) and delta > 0.003) else ""
        ds = f"{delta:+.4f}" if not np.isnan(delta) else "—"
        print(f"{name:<28}  {s['f1_mean']:>8.4f}  "
              f"{s['f1_std']:>6.4f}  {ds:>10}{flag}")

    return out_json


if __name__ == "__main__":
    run()

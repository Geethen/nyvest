"""Scheme B round-2: ensemble, support-size sweep, and fine-tuning.

Scheme B: merge 1→2 (bare), 9→8 (water) → 10 classes.
Classes: [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

Experiments (all 3-fold LLTO, same CatBoost Stage-1 pseudo-labels):

  ── Ensemble ──────────────────────────────────────────────────
  ensemble_avg          Average probs from CB_high_iter + TabICL_10k
  ensemble_geomean      Geometric mean of probs (sharper than avg)

  ── Support-size sweep ────────────────────────────────────────
  tabicl_zero_5k        n_est=8, 5 000-row random support  (reference)
  tabicl_zero_10k       n_est=8, 10 000-row random support (best from round 1)
  tabicl_zero_15k       n_est=8, 15 000-row random support
  tabicl_zero_20k       n_est=8, 20 000-row random support (if memory allows)

  ── Fine-tune (Scheme B = exactly 10 classes) ─────────────────
  tabicl_ft_30ep        Fine-tune 30 ep, lr=1e-5, random 3 000 support
  tabicl_ft_strat       Fine-tune 30 ep, lr=1e-5, stratified 3 000 support

Notes:
  - CatBoost runs on CPU to avoid GPU contention with TabICL.
  - amp=False required on this driver/torch stack for fine-tuning.
  - Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
  - CUDA cache is cleared between TabICL calls to avoid OOM.
  - ensemble builds on top of saved CB and TabICL probabilities so no
    extra model fits are needed.

Output:
  common_ground/reports/research/schemeB_round2_results.json
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

MERGE_MAP = {1: 2, 9: 8}   # Scheme B

# Reference numbers from prior runs (for the leaderboard Δ column)
REF_CB_BASE  = 0.6502
REF_CB_HI    = 0.6582
REF_TABICL10 = 0.6728

N_FOLDS = 3
SEED    = 0


def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id",
                                            "lon", "lat"]).reset_index(drop=True)


def merge_classes(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype(int)
    for src, tgt in MERGE_MAP.items():
        y[y == src] = tgt
    return y


def make_catboost_cpu(iterations):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU", loss_function="MultiClass",
    )


def clear_cuda():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def random_subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def stratified_subsample(X, y, n_per_class, n_classes, rng):
    parts_X, parts_y = [], []
    for c in range(n_classes):
        mask = y == c
        avail = int(mask.sum())
        if avail == 0:
            continue
        take = min(n_per_class, avail)
        chosen = rng.choice(np.flatnonzero(mask), size=take, replace=False)
        parts_X.append(X[chosen])
        parts_y.append(y[chosen])
    X_out = np.concatenate(parts_X)
    y_out = np.concatenate(parts_y)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def score(y_true, y_pred, n_classes):
    f1m = f1_score(y_true, y_pred, average="macro",
                   labels=np.arange(n_classes), zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1_pc = f1_score(y_true, y_pred, labels=np.arange(n_classes),
                     average=None, zero_division=0)
    return float(f1m), float(bal), [float(v) for v in f1_pc]


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  scheme=B")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    y_stable_m   = merge_classes(stable[TARGET].values)
    y_unstable_m = merge_classes(unstable[TARGET].values)
    merged_classes = sorted(set(y_stable_m.tolist()) | set(y_unstable_m.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    print(f"classes: {decoded}  n={n_classes}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    # results[name] = list of per-fold dicts
    results: dict[str, list] = {}
    descs: dict[str, str] = {}
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

        # Stage 1: CatBoost CPU 500 on 80% of stable_train
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost_cpu(500)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        pseudo_u = stage1.predict_proba(X_u_tr).astype(np.float32).argmax(axis=1)

        X_aug = np.concatenate([X_s_tr, X_u_tr])
        y_aug = np.concatenate([y_s_tr, pseudo_u])

        rng = np.random.default_rng(SEED + k)
        Xs = {n: random_subsample(X_aug, y_aug, n, rng)
              for n in (5_000, 10_000, 15_000, 20_000)}
        n_per_cls_3k = 3000 // n_classes
        Xs_strat3k = stratified_subsample(X_aug, y_aug, n_per_cls_3k, n_classes, rng)
        Xs_rand3k  = random_subsample(X_aug, y_aug, 3000, rng)

        print(f"\n=== fold {k}  n_aug={len(y_aug):,}  n_test={len(y_s_te):,} ===")

        # ── 1. CatBoost models (needed for ensemble) ──────────────
        clear_cuda()
        cb_hi = make_catboost_cpu(1000)
        t0 = time.perf_counter()
        cb_hi.fit(X_aug, y_aug)
        t_cb = time.perf_counter() - t0
        probs_cb = cb_hi.predict_proba(X_s_te).astype(np.float64)
        pred_cb  = probs_cb.argmax(axis=1)
        f1_cb, bal_cb, pc_cb = score(y_s_te, pred_cb, n_classes)
        print(f"  [catboost_high_iter    ] F1={f1_cb:.4f}  bal={bal_cb:.4f}  fit={t_cb:.1f}s")
        _record(results, descs, "catboost_high_iter",
                "CatBoost 1 000 iter, CPU (Scheme B reference)",
                k, f1_cb, bal_cb, pc_cb, t_cb, 0.0, len(y_aug))

        # ── 2. TabICL support-size sweep ──────────────────────────
        from tabicl import TabICLClassifier
        probs_tabicl_10k = None   # save for ensemble

        for sup_n in (5_000, 10_000, 15_000, 20_000):
            name = f"tabicl_zero_{sup_n//1000}k"
            desc = f"TabICL zero-shot, n_est=8, {sup_n:,}-row random support"
            X_sup, y_sup = Xs[sup_n]
            clear_cuda()
            try:
                t0 = time.perf_counter()
                clf = TabICLClassifier(n_estimators=8, random_state=SEED,
                                       verbose=False, device="cuda")
                clf.fit(X_sup, y_sup)
                t_fit = time.perf_counter() - t0
                t0 = time.perf_counter()
                probs_tz = clf.predict_proba(X_s_te).astype(np.float64)
                t_pred = time.perf_counter() - t0
                pred_tz = probs_tz.argmax(axis=1)
                f1_tz, bal_tz, pc_tz = score(y_s_te, pred_tz, n_classes)
                print(f"  [{name:<22}] F1={f1_tz:.4f}  bal={bal_tz:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s  n_sup={len(y_sup):,}")
                if sup_n == 10_000:
                    probs_tabicl_10k = probs_tz   # save for ensemble
                _record(results, descs, name, desc, k, f1_tz, bal_tz, pc_tz,
                        t_fit, t_pred, len(y_sup))
            except Exception as e:
                print(f"  [{name:<22}] FAILED: {type(e).__name__}: {str(e)[:120]}")
                _record_fail(results, descs, name, desc, k, n_classes, len(y_sup))
            clear_cuda()

        # ── 3. Ensemble (CB_hi + TabICL_10k) ──────────────────────
        if probs_tabicl_10k is not None:
            # arithmetic mean
            probs_ens = (probs_cb + probs_tabicl_10k) / 2.0
            pred_ens  = probs_ens.argmax(axis=1)
            f1_e, bal_e, pc_e = score(y_s_te, pred_ens, n_classes)
            print(f"  [ensemble_avg          ] F1={f1_e:.4f}  bal={bal_e:.4f}")
            _record(results, descs, "ensemble_avg",
                    "Probability average: CB_high_iter + TabICL_zero_10k",
                    k, f1_e, bal_e, pc_e, 0.0, 0.0, 0)

            # geometric mean (numerically stable via log)
            eps = 1e-9
            log_p = (np.log(probs_cb + eps) + np.log(probs_tabicl_10k + eps)) / 2.0
            log_p -= log_p.max(axis=1, keepdims=True)
            probs_geo = np.exp(log_p)
            probs_geo /= probs_geo.sum(axis=1, keepdims=True)
            pred_geo  = probs_geo.argmax(axis=1)
            f1_g, bal_g, pc_g = score(y_s_te, pred_geo, n_classes)
            print(f"  [ensemble_geomean      ] F1={f1_g:.4f}  bal={bal_g:.4f}")
            _record(results, descs, "ensemble_geomean",
                    "Geometric mean of probs: CB_high_iter + TabICL_zero_10k",
                    k, f1_g, bal_g, pc_g, 0.0, 0.0, 0)
        else:
            print("  [ensemble_*            ] skipped — TabICL_10k failed")

        # ── 4. Fine-tuning (Scheme B = 10 classes, fine-tune eligible) ──
        from tabicl import FinetunedTabICLClassifier

        def make_ft(epochs=30, lr=1e-5):
            return FinetunedTabICLClassifier(
                epochs=epochs, learning_rate=lr, weight_decay=0.01,
                early_stopping=True, patience=6, min_delta=1e-4,
                eval_metric="log_loss", random_state=SEED, verbose=False,
                n_estimators_finetune=1, n_estimators_validation=1,
                n_estimators_inference=4,
                max_data_size=3000, device="cuda", amp=False,
            )

        for name, desc, X_sup, y_sup in [
            ("tabicl_ft_30ep",
             "TabICL fine-tune 30 ep, lr=1e-5, random 3 000-row support",
             *Xs_rand3k),
            ("tabicl_ft_strat",
             f"TabICL fine-tune 30 ep, lr=1e-5, stratified 3 000-row support (~{n_per_cls_3k}/cls)",
             *Xs_strat3k),
        ]:
            clear_cuda()
            try:
                X_t, X_v, y_t, y_v = train_test_split(
                    X_sup, y_sup, test_size=0.20, stratify=y_sup, random_state=SEED)
                t0 = time.perf_counter()
                ft = make_ft()
                ft.fit(X_t, y_t, X_val=X_v, y_val=y_v)
                t_fit = time.perf_counter() - t0
                t0 = time.perf_counter()
                pred_ft = np.asarray(ft.predict(X_s_te)).ravel().astype(int)
                t_pred = time.perf_counter() - t0
                best_ep = getattr(ft, "best_epoch_", "?")
                f1_ft, bal_ft, pc_ft = score(y_s_te, pred_ft, n_classes)
                print(f"  [{name:<22}] F1={f1_ft:.4f}  bal={bal_ft:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s  best_ep={best_ep}")
                _record(results, descs, name, desc, k, f1_ft, bal_ft, pc_ft,
                        t_fit, t_pred, len(X_sup), best_epoch=best_ep)
            except Exception as e:
                print(f"  [{name:<22}] FAILED: {type(e).__name__}: {str(e)[:120]}")
                _record_fail(results, descs, name, desc, k, n_classes, len(X_sup))
            clear_cuda()

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── summarise ────────────────────────────────────────────────
    summary = {}
    for name, folds in results.items():
        f1s   = [f["f1_macro"] for f in folds if not np.isnan(f["f1_macro"])]
        bals  = [f["bal_acc"]  for f in folds if not np.isnan(f["bal_acc"])]
        valid_pcs = [f["f1_per_class"] for f in folds
                     if not any(np.isnan(v) for v in f["f1_per_class"])]
        pc_mean = np.array(valid_pcs).mean(axis=0).tolist() if valid_pcs else [float("nan")] * n_classes
        f1_mean = float(np.mean(f1s)) if f1s else float("nan")
        summary[name] = {
            "description": descs[name],
            "f1_mean":  round(f1_mean, 4),
            "f1_std":   round(float(np.std(f1s)),  4) if f1s else float("nan"),
            "bal_mean": round(float(np.mean(bals)), 4) if bals else float("nan"),
            "bal_std":  round(float(np.std(bals)),  4) if bals else float("nan"),
            "f1_per_class": {str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)},
            "per_fold": folds,
        }

    out_json = OUT_DIR / "schemeB_round2_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": "B",
                   "merged_classes": decoded,
                   "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1),
                   "n_folds": N_FOLDS,
                   "references": {"catboost_baseline": REF_CB_BASE,
                                  "catboost_high_iter": REF_CB_HI,
                                  "tabicl_zero_10k": REF_TABICL10},
                   "results": summary}, f, indent=2)
    print(f"\nSaved → {out_json}")

    # leaderboard
    refs = {"tabicl_zero_10k": REF_TABICL10}
    print(f"\n{'Experiment':<24}  {'F1':>8}  {'std':>6}  {'Δ TabICL-10k':>13}")
    print("-" * 57)
    for name, s in sorted(summary.items(),
                          key=lambda x: -(x[1]["f1_mean"] if not np.isnan(x[1]["f1_mean"]) else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<24}  {'NaN':>8}  {'NaN':>6}  [FAIL]")
        else:
            d = s["f1_mean"] - REF_TABICL10
            flag = " ★" if d > 0.003 else ""
            print(f"{name:<24}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  {d:>+13.4f}{flag}")


def _record(results, descs, name, desc, fold, f1, bal, pc, fit_s, pred_s, n_sup, **extras):
    if name not in results:
        results[name] = []
        descs[name] = desc
    results[name].append({"fold": fold, "f1_macro": round(f1, 4),
                           "bal_acc": round(bal, 4), "fit_s": round(fit_s, 2),
                           "pred_s": round(pred_s, 2), "n_support": n_sup,
                           "f1_per_class": [round(v, 4) for v in pc], **extras})


def _record_fail(results, descs, name, desc, fold, n_classes, n_sup):
    _record(results, descs, name, desc, fold,
            float("nan"), float("nan"), [float("nan")] * n_classes,
            0.0, 0.0, n_sup)


if __name__ == "__main__":
    run()

"""Scheme B diagnostics for the current best model (tabicl_25k_n16_kvon).

Diagnoses why classes 2 (bare), 5 (grassland), 7 (wetland) score < 0.65, and
identifies which classes would benefit from more training data.

Tests run (3-fold LLTO, Scheme B = 10 classes):
  1. Confusion matrices per fold + aggregated — where do misclassifications go?
  2. Class support distribution (train / test) per fold.
  3. Stage-1 pseudo-label quality on unstable rows:
       - mean top-1 confidence per pseudo-class
       - per-pseudo-class entropy
       - what % of "uncertain" pseudo-labels (top-1 < 0.5) belong to which class
  4. Per-class learning curves for the best TabICL config:
       - support sizes: {1k, 2k, 5k, 10k, 15k, 20k, 25k}
       - n_estimators=16 throughout (for speed; KV cache on)
       - track F1 per class as support grows
       - classes whose curve is still rising at 25k are data-bound
       - classes that plateaued early are feature-bound

Output:
  common_ground/reports/research/schemeB_diagnostics.json
  + an HTML report assembled afterwards
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
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score)
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
MERGE_MAP = {1: 2, 9: 8}

N_FOLDS = 3
SEED    = 0
SUPPORT_SIZES_LC = [1000, 2000, 5000, 10000, 15000, 20000, 25000]
N_EST_LC = 16


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


def per_class_entropy(probs_for_class):
    """Mean entropy of the probability distributions for rows assigned to a given class."""
    if len(probs_for_class) == 0:
        return float("nan")
    eps = 1e-12
    ent = -np.sum(probs_for_class * np.log(probs_for_class + eps), axis=1)
    return float(np.mean(ent))


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

    # Per-fold storage
    per_fold = []
    learning_curve = {sup: [] for sup in SUPPORT_SIZES_LC}   # sup -> list of per-fold dict
    t_global = time.perf_counter()

    from catboost import CatBoostClassifier
    from tabicl import TabICLClassifier

    for k in range(N_FOLDS):
        print(f"\n========== Fold {k} ==========")
        s_test  = (stable_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_s_tr = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr = le.transform(y_stable_m[s_train])
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(y_stable_m[s_test])
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)

        # ── 1. Class support distribution ────────────────────────────
        train_dist = np.bincount(y_s_tr, minlength=n_classes).tolist()
        test_dist  = np.bincount(y_s_te, minlength=n_classes).tolist()

        # ── 2. Stage-1 + pseudo-label quality ────────────────────────
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        s1 = CatBoostClassifier(iterations=500, random_seed=SEED, verbose=False,
                                allow_writing_files=False, thread_count=4,
                                task_type="CPU", loss_function="MultiClass")
        s1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        probs_u = s1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_u = probs_u.argmax(axis=1)
        conf_u   = probs_u.max(axis=1)

        # per-pseudo-class diagnostics
        pseudo_dist = np.bincount(pseudo_u, minlength=n_classes).tolist()
        pseudo_conf_mean = []
        pseudo_entropy = []
        pseudo_low_conf_pct = []   # % of pseudo-labels for this class with top-1 < 0.5
        for c in range(n_classes):
            mask = (pseudo_u == c)
            if mask.sum() == 0:
                pseudo_conf_mean.append(float("nan"))
                pseudo_entropy.append(float("nan"))
                pseudo_low_conf_pct.append(float("nan"))
                continue
            pseudo_conf_mean.append(float(conf_u[mask].mean()))
            pseudo_entropy.append(per_class_entropy(probs_u[mask]))
            pseudo_low_conf_pct.append(float((conf_u[mask] < 0.5).mean()))

        # ── 3. Run best TabICL config and compute confusion ──────────
        X_aug = np.concatenate([X_s_tr, X_u_tr])
        y_aug = np.concatenate([y_s_tr, pseudo_u])
        rng = np.random.default_rng(SEED + k)

        # Best config = 25k support, n_est=16, kv_cache=True
        Xs, ys = random_subsample(X_aug, y_aug, 25000, rng)
        clear_cuda()
        clf = TabICLClassifier(n_estimators=16, random_state=SEED,
                               verbose=False, device="cuda", kv_cache=True)
        t0 = time.perf_counter()
        clf.fit(Xs, ys)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = np.asarray(clf.predict(X_s_te)).ravel().astype(int)
        t_pred = time.perf_counter() - t0
        del clf
        clear_cuda()

        cm = confusion_matrix(y_s_te, y_pred, labels=np.arange(n_classes))
        f1_pc = f1_score(y_s_te, y_pred, labels=np.arange(n_classes),
                         average=None, zero_division=0)
        f1m = f1_score(y_s_te, y_pred, average="macro",
                       labels=np.arange(n_classes), zero_division=0)
        bal = balanced_accuracy_score(y_s_te, y_pred)

        print(f"  best-config TabICL: F1={f1m:.4f}  bal={bal:.4f}  "
              f"fit={t_fit:.1f}s  pred={t_pred:.1f}s")
        print(f"  Stage-1 pseudo-conf mean (per class encoded):")
        for c in range(n_classes):
            cls_label = decoded[c]
            print(f"    cls {cls_label} ({pseudo_dist[c]:>5} pseudo):  "
                  f"conf_mean={pseudo_conf_mean[c]:.3f}  "
                  f"entropy={pseudo_entropy[c]:.3f}  "
                  f"low_conf_pct={pseudo_low_conf_pct[c]*100:.1f}%")

        per_fold.append({
            "fold": k,
            "n_test": int(s_test.sum()),
            "n_train": int(s_train.sum()),
            "n_unstable": int(u_train.sum()),
            "train_dist": train_dist,
            "test_dist": test_dist,
            "pseudo_dist": pseudo_dist,
            "pseudo_conf_mean": pseudo_conf_mean,
            "pseudo_entropy": pseudo_entropy,
            "pseudo_low_conf_pct": pseudo_low_conf_pct,
            "cm": cm.tolist(),
            "f1_per_class": [round(float(v), 4) for v in f1_pc],
            "f1_macro": round(float(f1m), 4),
            "bal_acc": round(float(bal), 4),
        })

        # ── 4. Per-class learning curve ──────────────────────────────
        print(f"  Learning curve (n_est={N_EST_LC}):")
        for sup in SUPPORT_SIZES_LC:
            if sup > len(X_aug):
                print(f"    sup={sup}: SKIPPED (only {len(X_aug)} aug rows)")
                continue
            Xs_lc, ys_lc = random_subsample(X_aug, y_aug, sup, rng)
            clear_cuda()
            try:
                t0 = time.perf_counter()
                clf = TabICLClassifier(n_estimators=N_EST_LC, random_state=SEED,
                                       verbose=False, device="cuda", kv_cache=True)
                clf.fit(Xs_lc, ys_lc)
                y_pred_lc = np.asarray(clf.predict(X_s_te)).ravel().astype(int)
                t_lc = time.perf_counter() - t0
                del clf
                clear_cuda()
                f1_pc_lc = f1_score(y_s_te, y_pred_lc, labels=np.arange(n_classes),
                                    average=None, zero_division=0)
                f1m_lc = float(f1_score(y_s_te, y_pred_lc, average="macro",
                                         labels=np.arange(n_classes), zero_division=0))
                print(f"    sup={sup:>5}: F1_macro={f1m_lc:.4f}  t={t_lc:.1f}s")
                learning_curve[sup].append({
                    "fold": k,
                    "f1_macro": round(f1m_lc, 4),
                    "f1_per_class": [round(float(v), 4) for v in f1_pc_lc],
                    "n_actual": int(len(ys_lc)),
                    "time_s": round(t_lc, 2),
                })
            except Exception as e:
                print(f"    sup={sup}: FAILED: {type(e).__name__}: {str(e)[:120]}")
                clear_cuda()

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── aggregate ─────────────────────────────────────────────────
    cm_total = np.array([f["cm"] for f in per_fold]).sum(axis=0)
    f1_pc_mean = np.array([f["f1_per_class"] for f in per_fold]).mean(axis=0).tolist()

    # learning curve summary
    lc_summary = {}
    for sup in SUPPORT_SIZES_LC:
        runs = learning_curve[sup]
        if not runs:
            continue
        f1m_mean = float(np.mean([r["f1_macro"] for r in runs]))
        pc_mat = np.array([r["f1_per_class"] for r in runs])
        pc_mean = pc_mat.mean(axis=0).tolist()
        lc_summary[sup] = {
            "f1_macro": round(f1m_mean, 4),
            "f1_per_class": {str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)},
            "n_folds": len(runs),
            "time_s_mean": round(float(np.mean([r["time_s"] for r in runs])), 2),
        }

    # aggregated pseudo-quality (mean of fold values)
    agg_pseudo_conf = []
    agg_pseudo_ent  = []
    agg_pseudo_low  = []
    for c in range(n_classes):
        vals_c = [f["pseudo_conf_mean"][c] for f in per_fold]
        vals_e = [f["pseudo_entropy"][c]   for f in per_fold]
        vals_l = [f["pseudo_low_conf_pct"][c] for f in per_fold]
        agg_pseudo_conf.append(round(float(np.nanmean(vals_c)), 4))
        agg_pseudo_ent.append(round(float(np.nanmean(vals_e)), 4))
        agg_pseudo_low.append(round(float(np.nanmean(vals_l)), 4))

    train_mean = np.array([f["train_dist"] for f in per_fold]).mean(axis=0).tolist()
    test_mean  = np.array([f["test_dist"]  for f in per_fold]).mean(axis=0).tolist()
    pseudo_mean = np.array([f["pseudo_dist"] for f in per_fold]).mean(axis=0).tolist()

    out = {
        "n_classes": n_classes,
        "decoded_classes": decoded,
        "elapsed_s": round(elapsed, 1),
        "best_config": "tabicl_25k_n16_kvon",
        "f1_per_class_mean": {str(decoded[i]): round(v, 4) for i, v in enumerate(f1_pc_mean)},
        "cm_total": cm_total.tolist(),
        "support": {
            "train_mean_per_fold": {str(decoded[i]): round(v, 1) for i, v in enumerate(train_mean)},
            "test_mean_per_fold":  {str(decoded[i]): round(v, 1) for i, v in enumerate(test_mean)},
            "pseudo_mean_per_fold":{str(decoded[i]): round(v, 1) for i, v in enumerate(pseudo_mean)},
        },
        "pseudo_quality": {
            "conf_mean": {str(decoded[i]): agg_pseudo_conf[i] for i in range(n_classes)},
            "entropy":   {str(decoded[i]): agg_pseudo_ent[i]  for i in range(n_classes)},
            "low_conf_pct": {str(decoded[i]): agg_pseudo_low[i] for i in range(n_classes)},
        },
        "learning_curve": lc_summary,
        "per_fold": per_fold,
    }
    out_json = OUT_DIR / "schemeB_diagnostics.json"
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"\nSaved → {out_json}")

    # Print learning-curve summary per weak class
    print("\n=== Learning curve per class (weak classes highlighted) ===")
    print(f"{'sup':>6}  ", end="")
    for c in decoded:
        marker = "*" if str(c) in ("2","5","7") else " "
        print(f"{marker}cls{c:>2}", end="  ")
    print()
    for sup in SUPPORT_SIZES_LC:
        if sup not in lc_summary:
            continue
        print(f"{sup:>6}  ", end="")
        for c in decoded:
            v = lc_summary[sup]["f1_per_class"][str(c)]
            print(f" {v:>5.3f}", end="  ")
        print()


if __name__ == "__main__":
    run()

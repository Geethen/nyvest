"""Characterize label-noise cleaning strategies on the stable-allyears parquet.

Compares three approaches for flagging mislabelled rows (focus: class 12 snow/ice,
but cleanlab scans all classes):
  1. centroid heuristic  — row nearer another class centroid than its own (current)
  2. cleanlab (all)      — confident-learning find_label_issues over all classes
  3. cleanlab (cls12)    — cleanlab issues restricted to class 12

pred_probs for cleanlab are OUT-OF-SAMPLE (5-fold cross-val CatBoost), as required.
Writes a per-class flag summary and the row-index sets so the retrain can reuse them.

Output: common_ground/reports/research/clean_labels_compare.json
        common_ground/reports/research/clean_labels_masks.npz  (keep masks)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO / "data"
OUT_DIR = _REPO / "common_ground" / "reports" / "research"
STABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"
MERGE_MAP = {1: 2, 9: 8}
SEED = 0
K = 5
CLS = 12


def merge_classes(y):
    y = y.copy().astype(int)
    for s, t in MERGE_MAP.items():
        y[y == s] = t
    return y


def make_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU", loss_function="MultiClass")


def oos_pred_probs(X, y_enc, n_classes):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    P = np.zeros((len(y_enc), n_classes), dtype=np.float64)
    for i, (tr, va) in enumerate(skf.split(X, y_enc)):
        cb = make_catboost()
        cb.fit(X[tr], y_enc[tr])
        # align columns to global class order (CatBoost preserves sorted classes)
        proba = cb.predict_proba(X[va]).astype(np.float64)
        cls_order = cb.classes_.astype(int)
        P[np.ix_(va, cls_order)] = proba
        print(f"  oos fold {i} done")
    return P


def centroid_suspect(X, y_merged, target):
    cents = {int(c): X[y_merged == c].mean(0) for c in np.unique(y_merged)}
    tgt = np.flatnonzero(y_merged == target)
    Xt = X[tgt]
    d_own = np.linalg.norm(Xt - cents[target], axis=1)
    others = [c for c in cents if c != target]
    OC = np.stack([cents[c] for c in others])
    d_other = np.linalg.norm(Xt[:, None, :] - OC[None, :, :], axis=2).min(1)
    mask = np.zeros(len(X), dtype=bool)
    mask[tgt[d_other < d_own]] = True
    return mask


def per_class_counts(flag_mask, y_merged):
    out = {}
    for c in sorted(np.unique(y_merged).tolist()):
        m = y_merged == c
        out[int(c)] = {"n": int(m.sum()), "flagged": int(flag_mask[m].sum())}
    return out


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = duckdb.sql(f"SELECT * FROM '{STABLE_PARQUET}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y_m = merge_classes(df[TARGET].values)
    classes = np.sort(np.unique(y_m))
    cls_to_idx = {int(c): i for i, c in enumerate(classes)}
    y_enc = np.array([cls_to_idx[int(v)] for v in y_m])
    n_classes = len(classes)
    print(f"rows={len(df):,}  classes={classes.tolist()}")

    # 1. centroid heuristic
    cen_mask = centroid_suspect(X, y_m, CLS)
    print(f"centroid: flagged {cen_mask.sum()} (class-{CLS} only)")

    # OOS pred_probs for cleanlab
    print("computing OOS pred_probs (5-fold CatBoost)...")
    P = oos_pred_probs(X, y_enc, n_classes)

    from cleanlab.filter import find_label_issues
    cl_mask = find_label_issues(
        labels=y_enc, pred_probs=P,
        return_indices_ranked_by="self_confidence")
    cl_bool = np.zeros(len(df), dtype=bool)
    cl_bool[cl_mask] = True
    print(f"cleanlab(all): flagged {cl_bool.sum()} across all classes")

    cl12 = cl_bool & (y_m == CLS)
    print(f"cleanlab(cls{CLS}): flagged {int(cl12.sum())} class-{CLS} rows")

    # overlap between centroid and cleanlab on class 12
    cen12 = cen_mask & (y_m == CLS)
    overlap = int((cen12 & cl12).sum())

    summary = {
        "n_rows": len(df), "classes": classes.tolist(),
        "centroid_cls12_flagged": int(cen_mask.sum()),
        "cleanlab_all_flagged": int(cl_bool.sum()),
        "cleanlab_cls12_flagged": int(cl12.sum()),
        "cls12_overlap_centroid_cleanlab": overlap,
        "cls12_total": int((y_m == CLS).sum()),
        "centroid_per_class": per_class_counts(cen_mask, y_m),
        "cleanlab_per_class": per_class_counts(cl_bool, y_m),
    }
    with open(OUT_DIR / "clean_labels_compare.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(OUT_DIR / "clean_labels_masks.npz",
             centroid=cen_mask, cleanlab_all=cl_bool,
             cleanlab_cls12=cl12, y_merged=y_m)

    print("\n=== per-class flag counts (cleanlab all-class scan) ===")
    print(f"{'cls':>4} {'n':>8} {'cl_flag':>8} {'cl_%':>6} {'cen_flag':>9}")
    for c in classes:
        c = int(c)
        n = summary["cleanlab_per_class"][c]["n"]
        clf = summary["cleanlab_per_class"][c]["flagged"]
        cen = summary["centroid_per_class"][c]["flagged"]
        print(f"{c:>4} {n:>8,} {clf:>8} {100*clf/max(n,1):>5.1f}% {cen:>9}")
    print(f"\nclass-{CLS}: centroid={cen12.sum()}  cleanlab={cl12.sum()}  "
          f"overlap={overlap}")
    print(f"Saved -> {OUT_DIR/'clean_labels_compare.json'}")


if __name__ == "__main__":
    run()

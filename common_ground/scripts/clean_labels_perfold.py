"""Per-fold, train-only cleanlab flagging (fully leak-free).

Cleanlab's confident-joint uses DATASET-WIDE per-class thresholds, so a single global
pass over train+test leaks test rows into the flagging decision even if the mask is
later applied train-only. This script removes that leak: for each outer CV fold it runs
cleanlab on the TRAINING rows of that fold ONLY (separate internal CV for pred_probs,
separate thresholds, separate confident joint). The fold's test rows never influence
its own cleaning.

Replicates the pipeline's GroupKFold(cell_id) split EXACTLY so masks align per fold.

Output: clean_labels_perfold.npz with, aligned to full stable row order:
  remove[k]    (n,) bool  rows to drop when fold k is the TEST fold (True only on
                          training rows of fold k; test rows always False)
  corrected[k] (n,) int   training-label to use for fold k (suggested where flagged)
  folds        (n,) int   outer test-fold id per row (for verification)
Also a per-test-pass set for the noise-corrected TEST metric:
  test_inv_noise[k] (C,C) transition matrix estimated on fold k's TEST rows only.

Cost: ~3x the single OOS (one train-only pass per fold) + 3 small test passes.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO / "data"
OUT_DIR = _REPO / "common_ground" / "reports" / "research"
STABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"
MERGE_MAP = {1: 2, 9: 8}
SEED = 0
N_FOLDS = 3
K = 5


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


def oos_pred_probs(X, y_enc, n_classes, tag=""):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    P = np.zeros((len(y_enc), n_classes), dtype=np.float64)
    for i, (tr, va) in enumerate(skf.split(X, y_enc)):
        cb = make_catboost()
        cb.fit(X[tr], y_enc[tr])
        P[np.ix_(va, cb.classes_.astype(int))] = cb.predict_proba(X[va])
        print(f"  {tag} oos fold {i} done")
    return P


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = duckdb.sql(f"SELECT * FROM '{STABLE_PARQUET}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y_m = merge_classes(df[TARGET].values)
    classes = np.sort(np.unique(y_m))
    c2i = {int(c): i for i, c in enumerate(classes)}
    n_classes = len(classes)

    # Replicate pipeline fold assignment EXACTLY.
    groups = df["cell_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = np.empty(len(df), dtype=int)
    for fi, (_, vi) in enumerate(gkf.split(df[FEATURE_COLS], y_m, groups=groups)):
        folds[vi] = fi

    from cleanlab.filter import find_label_issues
    from cleanlab.count import compute_confident_joint, estimate_latent

    remove = np.zeros((N_FOLDS, len(df)), dtype=bool)
    corrected = np.tile(y_m, (N_FOLDS, 1))
    test_inv = np.zeros((N_FOLDS, n_classes, n_classes), dtype=np.float64)

    for k in range(N_FOLDS):
        tr = np.flatnonzero(folds != k)   # training rows for this outer fold
        te = np.flatnonzero(folds == k)   # test rows for this outer fold
        y_tr_enc = np.array([c2i[int(v)] for v in y_m[tr]])
        print(f"=== fold {k}: train pass on {len(tr):,} rows ===")
        P_tr = oos_pred_probs(X[tr], y_tr_enc, n_classes, tag=f"f{k}-train")
        idx = find_label_issues(labels=y_tr_enc, pred_probs=P_tr,
                                return_indices_ranked_by="self_confidence")
        flagged = np.zeros(len(tr), dtype=bool)
        flagged[idx] = True
        remove[k, tr[flagged]] = True
        sug = classes[P_tr.argmax(1)]
        corr = y_m[tr].copy()
        corr[flagged] = sug[flagged]
        corrected[k, tr] = corr

        # Separate TEST pass: transition matrix estimated on test rows only, for the
        # leak-free noise-corrected TEST metric (test judged by other test rows only).
        y_te_enc = np.array([c2i[int(v)] for v in y_m[te]])
        print(f"=== fold {k}: test pass on {len(te):,} rows ===")
        P_te = oos_pred_probs(X[te], y_te_enc, n_classes, tag=f"f{k}-test")
        cj = compute_confident_joint(labels=y_te_enc, pred_probs=P_te)
        _, _, inv = estimate_latent(confident_joint=cj, labels=y_te_enc)
        test_inv[k] = inv

    np.savez(OUT_DIR / "clean_labels_perfold.npz",
             remove=remove, corrected=corrected, folds=folds,
             classes=classes, test_inv_noise=test_inv)
    summary = {
        "n_rows": len(df), "n_folds": N_FOLDS, "classes": classes.tolist(),
        "remove_per_fold": [int(remove[k].sum()) for k in range(N_FOLDS)],
        "relabel_per_fold": [int((corrected[k] != y_m).sum()) for k in range(N_FOLDS)],
    }
    json.dump(summary, open(OUT_DIR / "clean_labels_perfold.json", "w"), indent=2)
    print("remove/fold:", summary["remove_per_fold"],
          " relabel/fold:", summary["relabel_per_fold"])
    print(f"Saved -> {OUT_DIR/'clean_labels_perfold.npz'}")


if __name__ == "__main__":
    run()

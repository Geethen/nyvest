"""Build reusable label-cleaning artifacts from one OOS pred_probs computation.

Produces common_ground/reports/research/clean_labels_full.npz with:
  pred_probs      (n, C) float32  out-of-sample CatBoost probabilities
  y_merged        (n,)   int      observed (merged) labels
  classes         (C,)   int      class order for pred_probs columns
  issue_mask_all  (n,)   bool     cleanlab find_label_issues over ALL classes
  issue_mask_c12  (n,)   bool     issue_mask_all restricted to class 12
  suggested_all   (n,)   int      argmax pred_prob (cleanlab suggested label) for
                                  flagged rows; equals y_merged for unflagged rows
  centroid_c12    (n,)   bool     geometric centroid heuristic, class 12 only

These feed the pipeline's CLEAN_MODE in two treatments (matching cleanlab's tabular
tutorial): REMOVE flagged rows, or CORRECT them to the suggested label. Cleaning is
applied TRAIN-ONLY inside the CV loop; the test fold keeps original labels.

Reuses ~50 min of 5-fold CatBoost OOS; run once.
Output: clean_labels_full.npz  + clean_labels_artifacts.json (summary)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
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
CLS12 = 12


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
        P[np.ix_(va, cb.classes_.astype(int))] = cb.predict_proba(X[va])
        print(f"  oos fold {i} done")
    return P


def centroid_c12(X, y_m):
    cents = {int(c): X[y_m == c].mean(0) for c in np.unique(y_m)}
    tgt = np.flatnonzero(y_m == CLS12)
    Xt = X[tgt]
    d_own = np.linalg.norm(Xt - cents[CLS12], axis=1)
    others = [c for c in cents if c != CLS12]
    OC = np.stack([cents[c] for c in others])
    d_other = np.linalg.norm(Xt[:, None, :] - OC[None, :, :], axis=2).min(1)
    m = np.zeros(len(X), dtype=bool)
    m[tgt[d_other < d_own]] = True
    return m


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = duckdb.sql(f"SELECT * FROM '{STABLE_PARQUET}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y_m = merge_classes(df[TARGET].values)
    classes = np.sort(np.unique(y_m))
    c2i = {int(c): i for i, c in enumerate(classes)}
    y_enc = np.array([c2i[int(v)] for v in y_m])
    n_classes = len(classes)
    print(f"rows={len(df):,}  classes={classes.tolist()}")

    print("computing OOS pred_probs (5-fold CatBoost, ~50 min)...")
    P = oos_pred_probs(X, y_enc, n_classes)

    # Noise transition matrix T for post-hoc test-set noise correction.
    # noise_matrix[j,i] = P(observed=j | true=i); inv_noise[i,j] = P(true=i|obs=j).
    from cleanlab.count import compute_confident_joint, estimate_latent
    cj = compute_confident_joint(labels=y_enc, pred_probs=P)
    py, noise_matrix, inv_noise = estimate_latent(confident_joint=cj, labels=y_enc)

    from cleanlab.filter import find_label_issues
    idx = find_label_issues(labels=y_enc, pred_probs=P,
                            return_indices_ranked_by="self_confidence")
    issue_all = np.zeros(len(df), dtype=bool)
    issue_all[idx] = True
    issue_c12 = issue_all & (y_m == CLS12)

    # suggested label = argmax pred_prob, mapped back to merged class id;
    # for unflagged rows keep the observed label.
    argmax_enc = P.argmax(1)
    suggested = y_m.copy()
    suggested[issue_all] = classes[argmax_enc[issue_all]]

    cen12 = centroid_c12(X, y_m)

    np.savez(
        OUT_DIR / "clean_labels_full.npz",
        pred_probs=P.astype(np.float32), y_merged=y_m, classes=classes,
        issue_mask_all=issue_all, issue_mask_c12=issue_c12,
        suggested_all=suggested, centroid_c12=cen12,
        noise_matrix=noise_matrix, inv_noise=inv_noise, py=py)

    # how many flagged rows actually get a DIFFERENT suggested label
    changed = int((suggested != y_m).sum())
    summary = {
        "n_rows": len(df), "classes": classes.tolist(),
        "issue_all": int(issue_all.sum()),
        "issue_c12": int(issue_c12.sum()),
        "centroid_c12": int(cen12.sum()),
        "suggested_changed": changed,
        "note": "train-only cleaning; remove vs correct treatments",
    }
    with open(OUT_DIR / "clean_labels_artifacts.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"issue_all={issue_all.sum():,}  issue_c12={issue_c12.sum()}  "
          f"centroid_c12={cen12.sum()}  suggested_changed={changed:,}")
    print(f"Saved -> {OUT_DIR/'clean_labels_full.npz'}")


if __name__ == "__main__":
    run()

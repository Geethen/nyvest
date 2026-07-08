"""Shared data loading + CV utilities for the DNN models.

Mirrors the evaluation protocol of common_ground/scripts/llto_schemeB_allyears.py
EXACTLY so DNN macro-F1 is directly comparable to the TabICL reference (0.7139):

  - same stable-allyears parquet (663k rows, 12 classes)
  - same class merge 1->2 and 9->8 (=> 10 effective classes)
  - same 3-fold spatial CV via GroupKFold on cell_id
  - same macro-F1 over all classes (zero_division=0)
  - optional lidar3 extra features (elevation, tri, tch), exact (lon,lat) join,
    median-impute the uncovered rows  (EXTRA_FEATURES=lidar)
  - test fold labels are NEVER cleaned/relabelled (leak-free)

We do NOT use the unstable parquet / conformal pseudo-labelling here — the DNN
trains directly on the stable folds. (Pseudo-labelling can be layered on later.)
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

_REPO = Path(__file__).resolve().parents[1]
DATA_DIR = _REPO / "data"

STABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"
LIDAR_PARQUET = DATA_DIR / "lidar_features.parquet"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
LIDAR_COLS = ["elevation", "tri", "tch"]
TARGET = "class"
MERGE_MAP = {1: 2, 9: 8}
N_FOLDS = 3
SEED = 0


def merge_classes(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype(int)
    for src, tgt in MERGE_MAP.items():
        y[y == src] = tgt
    return y


def load_data(extra_features: str = "lidar"):
    """Load the stable-allyears frame.

    Returns dict with:
      X        float32 [N, F]      feature matrix (embed [+ lidar])
      y        int     [N]         merged class labels (raw values, e.g. 2..12)
      y_enc    int     [N]         label-encoded 0..C-1
      classes  list                sorted unique merged class values (decode map)
      groups   int     [N]         cell_id (for GroupKFold)
      df       DataFrame           the loaded frame (lon/lat/year retained)
      feat_cols list               feature column names in X order
    """
    df, feat_cols, lidar_med = _load_frame(STABLE_PARQUET, extra_features)
    X = df[feat_cols].values.astype(np.float32)
    y = merge_classes(df[TARGET].values)
    classes = sorted(np.unique(y).tolist())
    remap = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([remap[v] for v in y], dtype=np.int64)
    groups = df["cell_id"].values

    return {
        "X": X, "y": y, "y_enc": y_enc, "classes": classes,
        "groups": groups, "df": df, "feat_cols": feat_cols,
        "extra_features": extra_features, "lidar_med": lidar_med,
    }


def _load_frame(parquet, extra_features):
    """Load one parquet -> (df, feat_cols, lidar_median_dict). Shared by
    stable and unstable loaders so feature handling is identical."""
    cols = duckdb.sql(f"DESCRIBE SELECT * FROM '{parquet}'").df()["column_name"]
    meta = [c for c in (TARGET, "cell_id", "lon", "lat", "year") if c in set(cols)]
    sel = ", ".join([f'CAST("{c}" AS FLOAT) AS "{c}"' for c in EMBED_COLS]
                    + [f'"{c}"' for c in meta])
    df = duckdb.sql(f"SELECT {sel} FROM '{parquet}'").df()
    df = df.dropna(subset=EMBED_COLS + [TARGET]).reset_index(drop=True)
    feat_cols = list(EMBED_COLS)
    lidar_med = None
    if extra_features == "lidar":
        extra = (duckdb.sql(
            f"SELECT lon, lat, {', '.join(LIDAR_COLS)} FROM '{LIDAR_PARQUET}'").df()
            .drop_duplicates(subset=["lon", "lat"]))
        lidar_med = {c: float(extra[c].median()) for c in LIDAR_COLS}
        df = df.merge(extra, on=["lon", "lat"], how="left")
        for c in LIDAR_COLS:
            df[c] = df[c].fillna(lidar_med[c]).astype(np.float32)
        feat_cols = feat_cols + LIDAR_COLS
    return df, feat_cols, lidar_med


def load_unstable(classes, extra_features: str = "lidar"):
    """Load the unstable parquet, encoded onto the SAME class set as `classes`
    (the stable label space). Rows whose merged class is absent from `classes`
    are dropped. Returns dict with X (float32), y_enc (true labels, for pseudo-
    label accuracy diagnostics), groups (cell_id), df, feat_cols.
    """
    df, feat_cols, _ = _load_frame(UNSTABLE_PARQUET, extra_features)
    y = merge_classes(df[TARGET].values)
    remap = {c: i for i, c in enumerate(classes)}
    keep = np.array([v in remap for v in y])
    df = df[keep].reset_index(drop=True)
    y = y[keep]
    X = df[feat_cols].values.astype(np.float32)
    y_enc = np.array([remap[v] for v in y], dtype=np.int64)
    return {"X": X, "y_enc": y_enc, "groups": df["cell_id"].values,
            "df": df, "feat_cols": feat_cols}


def fold_indices(y_enc, groups, n_folds: int = N_FOLDS):
    """Yield (fold_k, train_idx, test_idx) using GroupKFold on cell_id —
    identical fold assignment to the reference pipeline."""
    gkf = GroupKFold(n_splits=n_folds)
    folds = np.empty(len(y_enc), dtype=int)
    for k, (_, val_idx) in enumerate(gkf.split(np.zeros(len(y_enc)), y_enc, groups)):
        folds[val_idx] = k
    for k in range(n_folds):
        test = folds == k
        yield k, np.flatnonzero(~test), np.flatnonzero(test)


def clean_stale_class_mask(X, y_enc, df, target_enc, lon, lat):
    """Return a boolean KEEP mask that drops stale `target_enc` rows.

    Port of the reference pipeline's clean_stale_class (geometric, unsupervised):
    a target-class row is 'suspect' if it is nearer (Euclidean, feature space) to
    ANY other class centroid than to its own. Per unique (lon, lat):
      - suspect in ALL years  -> drop the whole location
      - suspect in SOME years -> drop only the suspect rows
      - never suspect          -> keep
    Pass X/y_enc/lon/lat restricted to the TRAIN fold so it stays leak-free.
    """
    classes = np.unique(y_enc)
    cents = {int(c): X[y_enc == c].mean(0) for c in classes}
    keep = np.ones(len(X), dtype=bool)
    if target_enc not in cents:
        return keep
    tgt_idx = np.flatnonzero(y_enc == target_enc)
    Xt = X[tgt_idx]
    d_own = np.linalg.norm(Xt - cents[target_enc], axis=1)
    others = [c for c in cents if c != target_enc]
    OC = np.stack([cents[c] for c in others])
    d_min_other = np.linalg.norm(Xt[:, None, :] - OC[None, :, :], axis=2).min(1)
    suspect = d_min_other < d_own

    sub = pd.DataFrame({"lon": lon[tgt_idx], "lat": lat[tgt_idx],
                        "_suspect": suspect, "_pos": tgt_idx})
    grp = sub.groupby(["lon", "lat"])
    loc_frac = grp["_suspect"].transform("mean")
    drop = ((loc_frac == 1.0) | ((loc_frac < 1.0) & sub["_suspect"].values))
    keep[sub["_pos"].values[drop.values]] = False
    return keep


FULL_CLEANLAB_NPZ = _REPO / "common_ground" / "reports" / "research" / "clean_labels_full.npz"


def apply_cls12_relabel(y_enc, classes, mode):
    """Full-data cls12-targeted relabel from clean_labels_full.npz (stage8's
    'to12_fix'/'cls12_fix' recipe, applied on the whole frame — no folds).

    `suggested_all` holds RAW merged class VALUES (2..12), not enc indices, so
    it's remapped through `classes` before comparing to the encoded cls12
    index. Returns (y_new, n_changed). No-op if mode == 'none' or 12 absent.
    """
    if mode == "none" or 12 not in classes:
        return y_enc.copy(), 0
    z = np.load(FULL_CLEANLAB_NPZ)
    remap = {c: i for i, c in enumerate(classes)}
    suggested = np.vectorize(remap.get)(z["suggested_all"]).astype(y_enc.dtype)
    c12 = classes.index(12)
    y_new = y_enc.copy()
    if mode == "cls12_fix":
        touch = (y_enc == c12)
    elif mode == "to12_fix":
        touch = (y_enc == c12) | (suggested == c12)
    else:
        raise ValueError(f"unknown RELABEL={mode}")
    y_new[touch] = suggested[touch]
    return y_new, int((y_new != y_enc).sum())


def macro_f1(y_true, y_pred, n_classes):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=np.arange(n_classes), zero_division=0))


def per_class_f1(y_true, y_pred, n_classes):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, labels=np.arange(n_classes),
                    average=None, zero_division=0)

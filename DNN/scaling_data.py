"""Load NiN points as leak-free TRAIN-ONLY augmentation for the scaling study.

The stable-grunnkart frame is the ground truth for BOTH train and test (the eval
protocol never touches test labels). NiN (`data/nin_alphaearth.parquet`, 200k
rows over classes 2,4,5,6,7 — exactly the confusion-bound vegetation classes) is
added to the TRAIN pool only, and only for cells NOT in the held-out test fold,
so the test partition stays pure grunnkart and leak-free.

NiN points carry the same 64 AlphaEarth bands + cell_id + lon/lat + year schema
as the stable frame, but have NO lidar coverage (disjoint point locations), so
their 3 lidar features are median-imputed from the stable-frame medians — the
same treatment data_utils gives any lidar-uncovered stable row.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

import data_utils as du

NIN_PARQUET = du.DATA_DIR / "nin_alphaearth.parquet"


def load_nin(classes, feat_cols, lidar_med, extra_features="lidar"):
    """NiN train-augmentation pool encoded onto the stable label space.

    Returns dict: X [M,F] float32, y_enc [M] onto `classes`, groups (cell_id) [M].
    Rows whose merged class is absent from `classes` are dropped. Lidar columns
    are filled from `lidar_med` (NiN has no lidar coverage).
    """
    sel = ", ".join([f'CAST("A{i:02d}" AS FLOAT) AS "A{i:02d}"' for i in range(64)]
                    + ['"class"', '"cell_id"', '"lon"', '"lat"'])
    df = duckdb.sql(f"SELECT {sel} FROM '{NIN_PARQUET}'").df()
    df = df.dropna(subset=du.EMBED_COLS + ["class"]).reset_index(drop=True)
    y = du.merge_classes(df["class"].values)
    remap = {c: i for i, c in enumerate(classes)}
    keep = np.array([v in remap for v in y])
    df, y = df[keep].reset_index(drop=True), y[keep]

    if extra_features == "lidar":
        for c in du.LIDAR_COLS:
            df[c] = np.float32(lidar_med[c])
    X = df[feat_cols].values.astype(np.float32)
    y_enc = np.array([remap[v] for v in y], dtype=np.int64)
    return {"X": X, "y_enc": y_enc, "groups": df["cell_id"].values}


def augment_train(tr_X, tr_y, tr_groups, nin, test_groups):
    """Concatenate leak-free NiN rows onto a train fold.

    Drops NiN rows whose cell_id appears in the test fold (`test_groups`) so the
    spatial hold-out is respected. Returns (X_aug, y_aug).
    """
    test_cells = set(np.unique(test_groups).tolist())
    keep = ~np.isin(nin["groups"], list(test_cells))
    X_aug = np.concatenate([tr_X, nin["X"][keep]], axis=0)
    y_aug = np.concatenate([tr_y, nin["y_enc"][keep]], axis=0)
    return X_aug, y_aug, int(keep.sum())

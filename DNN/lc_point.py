"""One learning-curve point: train best MLP on a stratified FRAC of fold-0 train,
write per-class test F1 to lc_shard_<frac>_<seed>.json. Designed to run MANY in
parallel on the shared GPU (each point is one small MLP, <1 GB).

Usage: lc_point.py <frac> <seed>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402
from dnn_paths import result_path # noqa: E402

DEVICE = s3.DEVICE
FOLD = 0


def stratified_subsample(y, frac, rng):
    if frac >= 1.0:
        return np.arange(len(y))
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        n = max(1, int(round(len(ci) * frac)))
        idx.append(rng.choice(ci, size=n, replace=False))
    return np.concatenate(idx)


def main():
    frac = float(sys.argv[1])
    seed = int(sys.argv[2])
    t0 = time.perf_counter()
    s3.set_seed(seed)
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values

    _, tr, te = list(du.fold_indices(y_enc, groups))[FOLD]
    keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                     cls12_enc, lon[tr], lat[tr])
    tr = tr[keep]
    Xtr_full, ytr_full = X[tr], y_enc[tr]
    Xte, yte = X[te], y_enc[te]

    sub = stratified_subsample(ytr_full, frac, np.random.default_rng(seed))
    Xtr, ytr = Xtr_full[sub], ytr_full[sub]

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    perm = rng.permutation(n)
    n_val = max(1, int(n * s3.VAL_FRAC))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr_t = torch.tensor(Xtr[tr_idx], device=DEVICE)
    ytr_t = torch.tensor(ytr[tr_idx], device=DEVICE)
    Xval_t = torch.tensor(Xtr[val_idx], device=DEVICE)
    w = s3.class_weights(ytr[tr_idx], n_classes, s3.WEIGHT_MODE)
    model, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, ytr[val_idx], Xtr.shape[1],
                            n_classes, w, seed)
    pred = s3.softmax_probs(model, torch.tensor(Xte_s, device=DEVICE),
                            n_classes).argmax(1)
    macro = du.macro_f1(yte, pred, n_classes)
    pc = du.per_class_f1(yte, pred, n_classes)

    out = result_path(f"lc_shard_{frac}_{seed}.json")
    out.write_text(json.dumps({
        "frac": frac, "seed": seed, "n_train": int(len(sub)),
        "macro": round(float(macro), 4),
        "per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc)},
    }))
    print(f"frac={frac} seed={seed} n={len(sub):,} macroF1={macro:.4f} "
          f"{time.perf_counter()-t0:.1f}s -> {out.name}")


if __name__ == "__main__":
    main()

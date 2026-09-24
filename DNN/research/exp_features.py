"""Batch 3 — change the REPRESENTATION, not the architecture.

Rationale from prior work: learning curves flatten by ~70% of the data and every
architecture change loses, but the ONE lever that ever moved the number was
better features (lidar: 0.7074 -> 0.7139). The hard classes (2 sparse-veg,
5 grassland, 6 open-upland, 7 mire) are confusion-bound: their AlphaEarth
embeddings overlap. So construct features that add information the 64-band
embedding does not already carry.

Crucially these are all LEAK-FREE: derived from X/coords of the TRAIN fold only,
or from the row's own values — never from labels of the test fold.

EXP variants:

  neighbor  - SPATIAL CONTEXT features. For each row, aggregate the embeddings of
              its k nearest TRAIN neighbours (excluding itself) via a KD-tree on
              (lon,lat), and append mean+std. Land cover is spatially
              autocorrelated: a grassland pixel in a mire matrix differs from one
              in a forest matrix. The MLP currently sees each pixel in isolation.
              This is the single most information-adding idea available.
  temporal  - TEMPORAL SIGNATURE. Rows are (location, year) pairs across
              2017-2025; per (lon,lat) compute the across-year mean/std of the
              embedding and append the row's deviation from its own location
              mean. Phenology/seasonality separates vegetation types that look
              alike in a single snapshot.
  proto     - CLASS-PROTOTYPE DISTANCES (metric-learning flavoured). Append the
              distance from each row to every TRAIN class centroid. Gives the net
              an explicit read on class geometry rather than making it rediscover
              centroids from raw dims.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec          # noqa: E402
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

EXP = os.environ.get("EXP", "neighbor")
K_NEIGH = int(os.environ.get("K_NEIGH", "8"))
N_PCA = int(os.environ.get("N_PCA", "16"))   # compress embeddings for neighbour stats


def _loc_ids(coords):
    """Integer id per unique (lon,lat), so all years of one location share an id."""
    q = np.round(coords.astype(np.float64), 6)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def neighbor_features(X_tr, coords_tr, X_q, coords_q, k, is_train):
    """Mean/std of the k nearest OTHER-LOCATION train neighbours' embeddings.

    Built ONLY from train-fold rows, so no test information enters the features.

    IMPORTANT: each location contributes ~9 rows (2017-2025) at IDENTICAL
    coordinates. Dropping just the self-match would leave the query's own
    location's other years as its nearest neighbours (distance 0) — the feature
    would then echo the row's own embedding instead of describing its spatial
    surroundings, and measure nothing. So we aggregate over unique LOCATIONS
    (one mean embedding per location) and exclude the query's own location.
    """
    from sklearn.decomposition import PCA
    p = PCA(n_components=N_PCA, random_state=0).fit(X_tr)
    Ztr = p.transform(X_tr).astype(np.float32)

    # collapse train rows to one row per location (mean across years)
    lid = _loc_ids(coords_tr)
    nloc = lid.max() + 1
    sums = np.zeros((nloc, Ztr.shape[1]), dtype=np.float64)
    np.add.at(sums, lid, Ztr)
    cnt = np.bincount(lid, minlength=nloc).astype(np.float64)[:, None]
    loc_z = (sums / np.maximum(cnt, 1)).astype(np.float32)
    loc_xy = np.zeros((nloc, 2), dtype=np.float64)
    loc_xy[lid] = coords_tr

    tree = cKDTree(loc_xy)
    # +1 so we can drop the query's own location when it is present in the tree
    _, idx = tree.query(coords_q, k=k + 1, workers=-1)
    if is_train:
        # the query's own location sits in the tree at distance 0 -> always the
        # first hit, so dropping column 0 excludes it (and all its years).
        idx = idx[:, 1:]
    else:
        idx = idx[:, :k]
    nb = loc_z[idx]                 # [N, k, N_PCA]
    return np.concatenate([nb.mean(1), nb.std(1)], 1).astype(np.float32)


def temporal_features(df, X):
    """Deviation of each row's embedding from its own location's across-year mean,
    plus the location's across-year std. Uses only the row's own location group —
    label-free, so it is leak-free even across folds."""
    key = (df["lon"].values.astype(np.float64) * 1e6).astype(np.int64) * 1000003 \
        + (df["lat"].values.astype(np.float64) * 1e6).astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    gid = np.zeros(len(key), dtype=np.int64)
    gid[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(ks)]))
    ng = gid.max() + 1
    sums = np.zeros((ng, X.shape[1]), dtype=np.float64)
    sqs = np.zeros((ng, X.shape[1]), dtype=np.float64)
    cnt = np.bincount(gid, minlength=ng).astype(np.float64)[:, None]
    np.add.at(sums, gid, X)
    np.add.at(sqs, gid, X.astype(np.float64) ** 2)
    mean = sums / np.maximum(cnt, 1)
    var = np.maximum(sqs / np.maximum(cnt, 1) - mean ** 2, 0)
    std = np.sqrt(var)
    return (X - mean[gid]).astype(np.float32), std[gid].astype(np.float32)


def proto_features(X_tr, y_tr, X_q, n_classes):
    """Distance to each TRAIN class centroid (train-fold labels only)."""
    cents = np.stack([X_tr[y_tr == c].mean(0) if (y_tr == c).any()
                      else np.zeros(X_tr.shape[1], dtype=np.float32)
                      for c in range(n_classes)])
    d = np.linalg.norm(X_q[:, None, :] - cents[None, :, :], axis=2)
    return d.astype(np.float32)


def run():
    t0 = time.perf_counter()
    base_pf, base_mean, base_name = ec.baseline_for(ec.RELABEL)
    name = {"neighbor": f"feat_neighbor_k{K_NEIGH}", "temporal": "feat_temporal",
            "proto": "feat_proto"}[EXP]
    print(f"=== {name} ===")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    coords = np.stack([lon, lat], 1)
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats")

    # temporal features are computed once on the whole frame (label-free)
    if EXP == "temporal":
        dev, lstd = temporal_features(df, X)
        X_aug_all = np.concatenate([X, dev, lstd], 1)
        print(f"temporal: {X.shape[1]} -> {X_aug_all.shape[1]} feats")

    rng = np.random.default_rng(ec.SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if ec.CLEAN_CLS12:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]

        if EXP == "neighbor":
            nf_tr = neighbor_features(X[tr_use], coords[tr_use], X[tr_use],
                                      coords[tr_use], K_NEIGH, True)
            nf_te = neighbor_features(X[tr_use], coords[tr_use], X[te],
                                      coords[te], K_NEIGH, False)
            Xtr_a = np.concatenate([X[tr_use], nf_tr], 1)
            Xte_a = np.concatenate([X[te], nf_te], 1)
        elif EXP == "temporal":
            Xtr_a, Xte_a = X_aug_all[tr_use], X_aug_all[te]
        else:  # proto
            pf_tr = proto_features(X[tr_use], y_enc[tr_use], X[tr_use], n_classes)
            pf_te = proto_features(X[tr_use], y_enc[tr_use], X[te], n_classes)
            Xtr_a = np.concatenate([X[tr_use], pf_tr], 1)
            Xte_a = np.concatenate([X[te], pf_te], 1)

        if k == 0:
            print(f"  feature dim: {X.shape[1]} -> {Xtr_a.shape[1]}")
        P, vf1 = ec.train_fold(lambda i, c: s3.MLP(i, c, ec.HIDDEN, ec.DROPOUT),
                               Xtr_a, y_enc[tr_use], Xte_a, n_classes, rng)
        pred = P.argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  Δ={f1-base_pf[k]:+.4f}  "
              f"(val={vf1:.4f}, {time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    deltas = [f1s[i] - base_pf[i] for i in range(len(f1s))]
    dmean = float(np.mean(deltas))
    all_pos = all(d > 0 for d in deltas)
    verdict = "WIN" if (all_pos and dmean > 0.003) else ("tie" if abs(dmean) <= 0.003 else "LOSS")
    pc_mean = np.mean(pcs, 0)
    print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}  Δ_mean={dmean:+.4f} "
          f"per_fold_Δ={[round(d,4) for d in deltas]}")
    print(f"VERDICT: {verdict}")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")

    rec = {"name": name, "notes": {
               "neighbor": f"k={K_NEIGH} spatial-neighbour embedding mean/std "
                           f"(PCA{N_PCA}); adds spatial context",
               "temporal": "per-location across-year deviation + std; adds phenology",
               "proto": "distance to train class centroids"}[EXP],
           "relabel": ec.RELABEL, "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
           "f1_per_fold": [round(v, 4) for v in f1s], "baseline": base_name,
           "baseline_per_fold": base_pf, "delta_per_fold": [round(d, 4) for d in deltas],
           "delta_mean": round(dmean, 4), "all_folds_positive": all_pos,
           "verdict": verdict,
           "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
           "n_ensemble": ec.N_ENSEMBLE, "runtime_s": round(time.perf_counter() - t0, 1),
           "config": {"idea": "features", "k_neigh": K_NEIGH if EXP == "neighbor" else None}}
    (ec.RESULTS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"saved -> {ec.RESULTS_DIR / f'{name}.json'}")


if __name__ == "__main__":
    run()

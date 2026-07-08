"""Apples-to-apples lidar ablation of the REFERENCE recipe — 3 arms.

The DNN reference best (0.7341) uses 64 AlphaEarth + 3 lidar features + the 5-seed
ensemble + the stage8 `to12_fix` cls12 relabel. Every scaling/reg/spatial cell in
this investigation also carried lidar — BUT the +NiN scaling cells fed NiN rows a
CONSTANT median value in the 3 lidar columns (NiN points have zero real lidar
coverage; scaling_data.py median-fills them). So those cells mixed grunnkart-with-
REAL-lidar and NiN-with-MEDIAN-lidar — not a clean feature space.

To disentangle, run the reference recipe in THREE arms, changing ONLY the lidar
columns:
  1. real   — grunnkart's true per-point elevation/tri/tch (published 0.7341 setup)
  2. median — every row gets the CONSTANT stable-frame median in those 3 columns
              (exactly what NiN experienced; isolates "column present but dead")
  3. none   — drop the 3 columns entirely (64 features)

Reads:
  real − median = value of the REAL lidar signal (vs the column merely existing).
  median − none = whether a dead constant column changes anything (expect ~0).
The `median` arm is the honest baseline to compare the NiN scaling cells against,
since it's the lidar treatment NiN actually got.

Recipe (identical both arms): 256,128 MLP, dropout 0.3, lr 1e-3, wd 1e-4, sqrt
class weights, label smoothing 0.05, 5-seed probability ensemble, per-fold leak-free
`to12_fix` cls12 relabel (from clean_labels_perfold.npz). Protocol: reference 3-fold
GroupKFold(cell_id), macro-F1, leak-free test (test labels never relabelled).

The `to12_fix` relabel_fn reproduces stage8 exactly: for the train fold, any row
whose per-fold cleanlab `corrected` label is 12 OR whose original label was 12 gets
its corrected label; centroid cleaning is skipped (relabel supersedes it for cls12).

Run:
  PY=~/myprojects/recover/.venv/bin/python
  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
    $PY DNN/lidar_ablation.py            # writes lidar_ablation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du            # noqa: E402
import dnn_core as C               # noqa: E402

PERFOLD_NPZ = (du._REPO / "common_ground" / "reports" / "research"
               / "clean_labels_perfold.npz")


def make_to12_relabel_fn(classes):
    """Return a cv_evaluate relabel_fn implementing stage8 `to12_fix` (leak-free).

    Signature matches cv_evaluate: (k, tr, y_enc, classes, X, lon, lat)
    -> (tr_use, ytr, note). Uses the per-fold cleanlab `corrected` (3,N) array,
    remapped to enc indices via `classes`. Only cls12-touching rows change; no
    centroid clean (relabel supersedes it for cls12); test fold never touched.
    """
    z = np.load(PERFOLD_NPZ)
    remap = {c: i for i, c in enumerate(classes)}
    corrected_enc = np.vectorize(remap.get)(z["corrected"])   # (3,N) -> 0..C-1
    c12 = classes.index(12)

    def relabel_fn(k, tr, y_enc, classes_, X, lon, lat):
        ytr = y_enc[tr].copy()
        corr = corrected_enc[k][tr]
        touch = (y_enc[tr] == c12) | (corr == c12)
        ytr[touch] = corr[touch]
        note = f"to12_fix cls12-touched={int(touch.sum())}"
        return tr, ytr, note

    return relabel_fn


def main():
    t0 = time.perf_counter()
    cfg = C.Config(hidden=(256, 128), dropout=0.3, lr=1e-3, weight_decay=1e-4,
                   n_ensemble=5, weight_mode="sqrt", label_smooth=0.05,
                   max_epochs=200, patience=15, seed=0)
    print(f"reference recipe: hidden={cfg.hidden} ensemble={cfg.n_ensemble} "
          f"relabel=to12_fix\n", flush=True)

    def build_arm(arm):
        """Return a data dict for one lidar arm. `real` and `median` share the
        lidar-loaded frame (67 feat); `median` overwrites the 3 lidar columns with
        their column median (constant), matching NiN's median-fill. `none` is the
        64-feature AlphaEarth-only frame."""
        if arm == "none":
            return C.load_cached("none")
        data = dict(C.load_cached("lidar"))     # shallow copy; we replace X only
        if arm == "median":
            X = data["X"].copy()
            feat = data["feat_cols"]
            lid_idx = [feat.index(c) for c in du.LIDAR_COLS]
            for j in lid_idx:
                X[:, j] = np.float32(np.median(X[:, j]))   # constant column
            data["X"] = X
        return data

    results = {}
    for arm in ("real", "median", "none"):
        data = build_arm(arm)
        classes = data["classes"]
        relabel_fn = make_to12_relabel_fn(classes)
        n_feat = data["X"].shape[1]
        print(f"--- lidar={arm}  ({n_feat} features) ---", flush=True)
        res = C.cv_evaluate(data, cfg, relabel_fn=relabel_fn, verbose=True)
        res["n_features"] = n_feat
        results[arm] = res
        print(flush=True)

    real, med, non = (results["real"]["f1_mean"], results["median"]["f1_mean"],
                      results["none"]["f1_mean"])
    out = Path(__file__).resolve().parent / "lidar_ablation.json"
    out.write_text(json.dumps({
        "recipe": {"hidden": [256, 128], "n_ensemble": 5, "relabel": "to12_fix",
                   "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
        "reference_published": 0.7341,
        "arms": {"real": results["real"], "median": results["median"],
                 "none": results["none"]},
        "real_minus_median": round(real - med, 4),     # value of REAL lidar signal
        "median_minus_none": round(med - non, 4),       # dead-column effect (~0?)
        "real_minus_none": round(real - non, 4),        # total lidar column effect
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=2))

    print("=" * 62)
    print("REFERENCE recipe (5-seed + to12_fix) — lidar ablation, apples-to-apples:")
    print(f"  real lidar   (67 feat): {real:.4f} ± {results['real']['f1_std']:.4f}")
    print(f"  median lidar (67 feat): {med:.4f} ± {results['median']['f1_std']:.4f}   <- what NiN saw")
    print(f"  no lidar     (64 feat): {non:.4f} ± {results['none']['f1_std']:.4f}")
    print(f"  real − median (real-signal value): {real-med:+.4f}")
    print(f"  median − none (dead-column effect): {med-non:+.4f}")
    print(f"  (published reference w/ real lidar = 0.7341)")
    print(f"wrote {out.name}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()

"""TEST D — per-class F1 gain from leak-free label correction.

Train the best DNN recipe twice per fold: once on ORIGINAL labels, once on the
leak-free per-fold CORRECTED labels (clean_labels_perfold.npz 'corrected',
cleanlab relabel done train-only per fold). Both scored on the SAME untouched
noisy test fold (the valid comparison — see memory test-label-noise: cleaning is
train-only, test labels never touched).

Per-class interpretation:
  F1 RISES when train labels are corrected  -> that class was NOISE-LIMITED in
    training; relabeling helps -> label noise IS a real problem for it.
  F1 FLAT/î                                  -> noise correction doesn't help
    that class; its ceiling is feature separability, not label noise.

5-seed ensemble, sqrt weights, label smoothing, cls12 centroid clean (same as
best). Run: ~/myprojects/recover/.venv/bin/python DNN/noise_clean_gain.py
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

DEVICE = s3.DEVICE
SEED = 0
N_ENSEMBLE = 5
PERFOLD_NPZ = (Path(__file__).resolve().parents[1] / "common_ground" /
               "reports" / "research" / "clean_labels_perfold.npz")
NAMES = {2: "sparse-veg", 3: "forest", 4: "forest", 5: "GRASSLAND",
         6: "open-upland", 7: "mire/wet", 8: "water", 10: "bare",
         11: "built/infra", 12: "snow/ice"}


def fit_predict(Xtr, ytr, Xte, n_classes, rng):
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    perm = rng.permutation(len(Xtr))
    nv = int(len(Xtr) * s3.VAL_FRAC)
    vi, ti = perm[:nv], perm[nv:]
    Xtr_t = torch.tensor(Xtr[ti], device=DEVICE)
    ytr_t = torch.tensor(ytr[ti], device=DEVICE)
    Xval_t = torch.tensor(Xtr[vi], device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    w = s3.class_weights(ytr[ti], n_classes, s3.WEIGHT_MODE)
    P = np.zeros((len(Xte), n_classes))
    for e in range(N_ENSEMBLE):
        m, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, ytr[vi], Xtr.shape[1],
                            n_classes, w, SEED + 100 * e)
        P += s3.softmax_probs(m, Xte_t, n_classes)
        del m
        torch.cuda.empty_cache()
    return P.argmax(1)


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    z = np.load(PERFOLD_NPZ)
    corrected_raw = z["corrected"]          # (3, N) raw merged labels, per fold
    remap = {c: i for i, c in enumerate(classes)}
    corrected_enc = np.vectorize(remap.get)(corrected_raw)  # (3, N) -> 0..9

    rng = np.random.default_rng(SEED)
    pc_orig, pc_corr = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                         cls12_enc, lon[tr], lat[tr])
        tr = tr[keep]
        yte = y_enc[te]                       # untouched noisy test labels
        # original-label training
        p0 = fit_predict(X[tr], y_enc[tr], X[te], n_classes, rng)
        # corrected-label training (leak-free per-fold relabel for THIS fold)
        p1 = fit_predict(X[tr], corrected_enc[k][tr], X[te], n_classes, rng)
        pc_orig.append(du.per_class_f1(yte, p0, n_classes))
        pc_corr.append(du.per_class_f1(yte, p1, n_classes))
        n_relab = int((corrected_enc[k][tr] != y_enc[tr]).sum())
        print(f"  fold {k}: macro orig={du.macro_f1(yte,p0,n_classes):.4f}  "
              f"corrected={du.macro_f1(yte,p1,n_classes):.4f}  "
              f"(relabeled {n_relab:,} train rows)")

    o = np.mean(pc_orig, axis=0)
    c = np.mean(pc_corr, axis=0)
    print(f"\n=== TEST D: per-class F1, original vs leak-free relabeled train ===")
    print(f"{'class':>14} {'orig':>7} {'relabel':>8} {'Δ':>8}")
    print("-" * 42)
    order = sorted(range(n_classes), key=lambda i: o[i])
    for i in order:
        d = c[i] - o[i]
        flag = "  <- noise-limited" if d >= 0.01 else ("  <- hurt" if d <= -0.01 else "")
        print(f"{NAMES[classes[i]]+'/'+str(classes[i]):>14} {o[i]:>7.3f} "
              f"{c[i]:>8.3f} {d:>+8.3f}{flag}")
    print(f"\n  macro: orig={o.mean():.4f}  relabel={c.mean():.4f}  "
          f"Δ={c.mean()-o.mean():+.4f}")
    print("Reading: Δ>=+0.01 => relabeling that class's train data helps =>\n"
          "label noise is a real, fixable problem for it. Δ~0 => not noise-limited.")
    out = Path(__file__).resolve().parent / "noise_clean_gain.json"
    out.write_text(json.dumps({
        "per_class_orig": {str(classes[i]): round(float(o[i]), 4) for i in range(n_classes)},
        "per_class_relabel": {str(classes[i]): round(float(c[i]), 4) for i in range(n_classes)},
        "per_class_delta": {str(classes[i]): round(float(c[i] - o[i]), 4) for i in range(n_classes)},
        "macro_orig": round(float(o.mean()), 4), "macro_relabel": round(float(c.mean()), 4),
    }, indent=2))
    print(f"saved -> {out}   total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

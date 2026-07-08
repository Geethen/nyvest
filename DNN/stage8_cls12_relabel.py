"""Stage 8 — targeted class-12-ONLY relabel (the only fixable-noise class).

Diagnostics established: cls12 is the one noise-limited class (Test D: full
leak-free relabel lifts cls12 +0.037), but applying that relabel to ALL classes
HURTS macro (-0.0065) because cleanlab miscorrects grassland/mire (confusion, not
noise). The noise itself is STATIC mislabelling (29/146 locations suspect in every
year 2017-2025, no temporal trend) — permanently mislabelled rock/debris polygons.

So: apply the leak-free per-fold CORRECTED labels to class 12 ONLY; every other
class keeps its original label. Goal: bank the cls12 gain without the collateral
macro damage. Compared head-to-head with the current best (original labels +
centroid cls12 clean).

Modes (RELABEL env):
  cls12_fix   - rows ORIGINALLY labeled 12 use their perfold-corrected label
                (so genuinely-not-ice cls12 rows get recoded away from 12).
  to12_fix    - ALSO let perfold reassign OTHER classes' rows TO 12 if cleanlab
                says so (recovers missed ice). Strictly cls12-centric both ways.
  best        - baseline: original labels (+ centroid cls12 clean), == Stage 3.

Same robust recipe (5-seed ensemble, sqrt weights, label smoothing, train-only
cls12 centroid clean, 3-fold spatial CV). Test labels NEVER touched.

Run: RELABEL=cls12_fix ~/myprojects/recover/.venv/bin/python DNN/stage8_cls12_relabel.py
"""

from __future__ import annotations

import json
import os
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
RELABEL = os.environ.get("RELABEL", "cls12_fix")   # cls12_fix | to12_fix | best
PERFOLD_NPZ = (Path(__file__).resolve().parents[1] / "common_ground" /
               "reports" / "research" / "clean_labels_perfold.npz")
NAMES = {2: "sparse-veg", 3: "forest", 4: "forest", 5: "GRASSLAND", 6: "open-upland",
         7: "mire/wet", 8: "water", 10: "bare", 11: "built/infra", 12: "snow/ice"}


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
    print(f"device={DEVICE}  RELABEL={RELABEL}  ensemble={N_ENSEMBLE}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    c12 = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    z = np.load(PERFOLD_NPZ)
    remap = {c: i for i, c in enumerate(classes)}
    corrected_enc = np.vectorize(remap.get)(z["corrected"])    # (3,N) -> 0..9

    rng = np.random.default_rng(SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr], c12,
                                         lon[tr], lat[tr])
        # start from original labels; build the cls12-targeted training labels
        ytr = y_enc[tr].copy()
        corr = corrected_enc[k][tr]
        use_centroid_clean = True
        if RELABEL == "cls12_fix":
            # rows originally 12 -> their corrected label (recode mislabelled ice)
            orig12 = (y_enc[tr] == c12)
            ytr[orig12] = corr[orig12]
            use_centroid_clean = False   # relabel supersedes removal for cls12
        elif RELABEL == "to12_fix":
            # any row whose corrected label is 12 OR was 12 -> corrected label
            touch = (y_enc[tr] == c12) | (corr == c12)
            ytr[touch] = corr[touch]
            use_centroid_clean = False
        # else "best": keep original labels (+ centroid clean below)

        tr_use = tr
        if use_centroid_clean:
            tr_use = tr[keep]
            ytr = y_enc[tr_use].copy()

        pred = fit_predict(X[tr_use], ytr, X[te], n_classes, rng)
        yte = y_enc[te]                       # untouched test labels
        f1 = du.macro_f1(yte, pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(yte, pred, n_classes))
        n_recoded = int((y_enc[tr] != corrected_enc[k][tr])[
            (y_enc[tr] == c12) | (corrected_enc[k][tr] == c12)].sum())
        print(f"  fold {k}: macroF1={f1:.4f}  cls12_F1={pcs[-1][c12]:.4f}  "
              f"(cls12-touched relabels={n_recoded})  {time.perf_counter()-t0:.0f}s")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc = np.mean(pcs, axis=0)
    print(f"\n=== Stage 8  RELABEL={RELABEL} ===")
    print(f"macro F1={f1m:.4f}  std={f1std:.4f}   (best=0.7318, cls12 there=0.702)  "
          f"Δmacro={f1m-0.7318:+.4f}")
    print(f"  cls12 F1={pc[c12]:.4f}  (Δ vs best cls12 0.702 = {pc[c12]-0.702:+.4f})")
    for c, v in zip(classes, pc):
        print(f"    {NAMES[c]+'/'+str(c):>14}: {v:.4f}")
    out = Path(__file__).resolve().parent / f"stage8_{RELABEL}.json"
    out.write_text(json.dumps({
        "stage": "stage8_cls12_relabel", "relabel": RELABEL,
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc)},
        "best_macro": 0.7318, "best_cls12": 0.702,
    }, indent=2))
    print(f"saved -> {out}  total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

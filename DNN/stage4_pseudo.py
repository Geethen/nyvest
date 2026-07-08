"""Stage 4 — DNN + conformal (CC-APS) pseudo-labelling from the unstable parquet.

The reference pipeline lifts F1 by pseudo-labelling confident unstable rows via a
CatBoost CC-APS gate and adding them to the TabICL support. Here we do the same
self-training loop with the DNN itself as the conformal scorer, so the whole
thing stays a single model family.

Per spatial fold (leak-free — test fold never used for calibration or labels):
  1. Split stable-train -> {fit, cal}. Train a 5-seed DNN ensemble on `fit`.
  2. Ensemble probs on `cal` (true labels) -> APS calibration scores -> tau.
  3. Ensemble probs on unstable rows -> APS sets. Keep SINGLETONS as pseudo-labels.
  4. Retrain a fresh 5-seed ensemble on stable-train + pseudo-labelled unstable.
  5. Evaluate on the untouched stable test fold.

We reuse Stage 3's MLP / train_one / softmax_probs so the base learner is
identical to the current best (F1=0.7318). ALPHA matches the pipeline (0.05).
cls12 cleaning is applied train-only as in Stage 3.

Knobs: ALPHA, N_ENSEMBLE, CAL_FRAC, PSEUDO_WEIGHT (sample weight for pseudo rows;
1.0 = same as real). Run:
  ~/myprojects/recover/.venv/bin/python DNN/stage4_pseudo.py
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
import data_utils as du            # noqa: E402
import conformal_utils as cu       # noqa: E402
import stage3_robust_mlp as s3     # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
ALPHA = float(os.environ.get("ALPHA", "0.05"))
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.15"))
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
OUT_JSON = Path(__file__).resolve().parent / "stage4_results.json"


def train_ensemble(Xfit, yfit, n_classes, n_ens, seed0):
    """Train n_ens MLPs on (Xfit,yfit) with an internal val slice for early stop.
    Returns list of models + the fitted scaler. Standardization fit on Xfit only."""
    scaler = StandardScaler().fit(Xfit)
    Xs = scaler.transform(Xfit).astype(np.float32)
    rng = np.random.default_rng(seed0)
    n = len(Xs)
    perm = rng.permutation(n)
    n_val = int(n * s3.VAL_FRAC)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr_t = torch.tensor(Xs[tr_idx], device=DEVICE)
    ytr_t = torch.tensor(yfit[tr_idx], device=DEVICE)
    Xval_t = torch.tensor(Xs[val_idx], device=DEVICE)
    yval_np = yfit[val_idx]
    w = s3.class_weights(yfit[tr_idx], n_classes, s3.WEIGHT_MODE)
    models = []
    for e in range(n_ens):
        m, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, yval_np, Xs.shape[1],
                            n_classes, w, seed0 + 100 * e)
        models.append(m)
    return models, scaler


def ensemble_probs(models, scaler, X, n_classes):
    Xs = scaler.transform(X).astype(np.float32)
    Xt = torch.tensor(Xs, device=DEVICE)
    P = np.zeros((len(X), n_classes), dtype=np.float64)
    for m in models:
        P += s3.softmax_probs(m, Xt, n_classes)
    return P / len(models)


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  alpha={ALPHA}  ensemble={N_ENSEMBLE}  "
          f"cal_frac={CAL_FRAC}  clean_cls12={CLEAN_CLS12}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    uns = du.load_unstable(classes, extra_features="lidar")
    Xu, yu_true = uns["X"], uns["y_enc"]
    print(f"stable {X.shape[0]:,}  unstable {Xu.shape[0]:,}  {X.shape[1]} feats  "
          f"{n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs, stats = [], [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if CLEAN_CLS12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(
                X[tr], y_enc[tr], df.iloc[tr], cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        Xtr, ytr = X[tr_use], y_enc[tr_use]

        # split stable-train -> fit / calibration
        p = rng.permutation(len(Xtr))
        n_cal = int(len(Xtr) * CAL_FRAC)
        cal_i, fit_i = p[:n_cal], p[n_cal:]

        # stage-1 ensemble on fit-portion for the conformal gate
        g1, sc1 = train_ensemble(Xtr[fit_i], ytr[fit_i], n_classes,
                                 N_ENSEMBLE, SEED)
        P_cc = ensemble_probs(g1, sc1, Xtr[cal_i], n_classes)
        P_u = ensemble_probs(g1, sc1, Xu, n_classes)
        keep_mask, pseudo, tau, pct = cu.pseudo_label_singletons(
            P_u, P_cc, ytr[cal_i], ALPHA, SEED + k)
        n_kept = int(keep_mask.sum())
        pseudo_acc = (float((pseudo[keep_mask] == yu_true[keep_mask]).mean())
                      if n_kept else float("nan"))
        del g1
        torch.cuda.empty_cache()

        # stage-2 ensemble on stable-train + pseudo-labelled unstable
        Xp = Xu[keep_mask]
        yp = pseudo[keep_mask]
        X_aug = np.concatenate([Xtr, Xp])
        y_aug = np.concatenate([ytr, yp])
        g2, sc2 = train_ensemble(X_aug, y_aug, n_classes, N_ENSEMBLE, SEED)
        P_te = ensemble_probs(g2, sc2, X[te], n_classes)
        pred = P_te.argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        stats.append({"fold": k, "f1": round(f1, 4), "tau": round(tau, 5),
                      "pct_kept": round(100 * pct, 1), "n_kept": n_kept,
                      "pseudo_acc": round(pseudo_acc, 4)})
        print(f"  fold {k}: F1={f1:.4f}  tau={tau:.4f}  kept={n_kept}"
              f"({100*pct:.1f}%)  pseudo_acc={pseudo_acc:.4f}  "
              f"{time.perf_counter()-ft:.1f}s")
        del g2
        torch.cuda.empty_cache()

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\n=== Stage 4  DNN + CC-APS pseudo-labelling ===")
    print(f"F1 mean={f1m:.4f}  std={f1std:.4f}   (Stage3 best=0.7318, "
          f"target=0.7139)  Δ_vs_best={f1m-0.7318:+.4f}")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")
    print(f"total {time.perf_counter()-t0:.1f}s")

    OUT_JSON.write_text(json.dumps({
        "stage": "stage4_pseudo", "alpha": ALPHA, "n_ensemble": N_ENSEMBLE,
        "cal_frac": CAL_FRAC, "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
        "per_fold_stats": stats, "stage3_best": 0.7318, "target_tabicl": 0.7139,
    }, indent=2))
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

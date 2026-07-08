"""Class-wise learning curves — which classes are data-starved vs confusion-bound.

Train the best MLP recipe (Stage 3) on increasing fractions of the TRAIN fold
and record per-class F1 at each size. Interpretation:
  - F1 still RISING at 100%  -> class would benefit from MORE labelled data.
  - F1 FLAT / saturated      -> ceiling is class confusion / separability, not
                                volume; more data won't help (need better
                                features or a specialist/MoE head).

Stratified subsampling keeps class proportions at each fraction (so rare classes
shrink too — that is the honest "more data of the same distribution" curve). We
also keep a fixed full test fold so F1 is comparable across sizes.

Cheap config: fold 0 only, 2 seeds/point (averaged), no augmentation. Writes
DNN/learning_curve.json and DNN/learning_curve.png.

Run: ~/myprojects/recover/.venv/bin/python DNN/learning_curves.py
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
FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
N_SEEDS = 2
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


def train_eval(Xtr, ytr, Xte, n_classes, seed):
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
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
    P = s3.softmax_probs(model, torch.tensor(Xte, device=DEVICE), n_classes)
    del model
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

    folds = list(du.fold_indices(y_enc, groups))
    _, tr, te = folds[FOLD]
    keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                     cls12_enc, lon[tr], lat[tr])
    tr = tr[keep]
    Xtr_full, ytr_full = X[tr], y_enc[tr]
    Xte, yte = X[te], y_enc[te]
    print(f"fold {FOLD}: train={len(tr):,}  test={len(te):,}  "
          f"{n_classes} classes")

    rng = np.random.default_rng(SEED)
    curve = {"fractions": FRACTIONS, "n_train": [], "macro": [],
             "per_class": {str(c): [] for c in classes}}
    for frac in FRACTIONS:
        macro_s, pc_s = [], []
        n_used = 0
        for s in range(N_SEEDS):
            sub = stratified_subsample(ytr_full, frac, np.random.default_rng(SEED + s))
            n_used = len(sub)
            pred = train_eval(Xtr_full[sub], ytr_full[sub], Xte, n_classes, SEED + s)
            macro_s.append(du.macro_f1(yte, pred, n_classes))
            pc_s.append(du.per_class_f1(yte, pred, n_classes))
        macro = float(np.mean(macro_s))
        pc = np.mean(pc_s, axis=0)
        curve["n_train"].append(int(n_used))
        curve["macro"].append(round(macro, 4))
        for i, c in enumerate(classes):
            curve["per_class"][str(c)].append(round(float(pc[i]), 4))
        print(f"  frac={frac:>4}  n={n_used:>7,}  macroF1={macro:.4f}")

    out = Path(__file__).resolve().parent / "learning_curve.json"
    out.write_text(json.dumps(curve, indent=2))
    print(f"saved -> {out}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = {2: "2 sparse-veg", 3: "3 forest", 4: "4 forest", 5: "5 GRASSLAND",
                 6: "6 open-upland", 7: "7 mire/wet", 8: "8 water", 10: "10 bare",
                 11: "11 built/infra", 12: "12 snow/ice"}
        n = np.array(curve["n_train"])
        fig, ax = plt.subplots(figsize=(9, 6))
        for c in classes:
            ax.plot(n, curve["per_class"][str(c)], marker="o",
                    label=names.get(c, str(c)))
        ax.plot(n, curve["macro"], "k--", lw=2.5, marker="s", label="MACRO")
        ax.set_xscale("log")
        ax.set_xlabel("train rows (stratified, log scale)")
        ax.set_ylabel("test F1 (fold 0)")
        ax.set_title("Class-wise learning curves — rising = wants more data, "
                     "flat = confusion-bound")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        png = Path(__file__).resolve().parent / "learning_curve.png"
        fig.savefig(png, dpi=130)
        print(f"saved -> {png}")
    except Exception as e:
        print(f"plot skipped: {e}")
    print(f"total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

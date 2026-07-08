"""Stage 3 — robust simple MLP for spatial generalization.

Findings driving this stage:
  - Stage 1 plain MLP: F1 0.7112 (val 0.79).  Stage 2 ResMLP+BN: F1 0.7106
    (val 0.965). => Extra capacity just memorises the train fold; the ceiling is
    GENERALISATION across spatial (cell_id) folds, not expressiveness.
So Stage 3 keeps the plain 2-layer MLP and spends the GPU budget on robustness
levers that target the spatial gap, plus domain knowledge:

  1. Input Gaussian noise (feature jitter) — cheap data augmentation in embedding
     space; pushes the net off exact train pixels.
  2. Mixup — convex combinations of (x, y) pairs; a strong, well-evidenced macro
     regulariser for tabular/embedding nets.
  3. Multi-seed probability ensemble — variance reduction, the reliable gain.
  4. cls12 centroid cleaning (TRAIN-ONLY, per fold, leak-free) — the reference
     pipeline's proven label-quality fix for stale snow/ice (see memory).
  5. moderate dropout + weight decay; tempered (sqrt) class weights.

Knobs are env vars. Run:
  ~/myprojects/recover/.venv/bin/python DNN/stage3_robust_mlp.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

HIDDEN = tuple(int(x) for x in os.environ.get("HIDDEN", "256,128").split(","))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "200"))
PATIENCE = int(os.environ.get("PATIENCE", "15"))
BATCH = int(os.environ.get("BATCH", "4096"))
VAL_FRAC = 0.1
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", "sqrt")
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
# Ablation verdict (see README): mixup + input-noise did NOT help on this data —
# dropping both was the best config (0.7318 vs 0.7283). The reliable lever is the
# 5-seed ensemble. So both augmentations default OFF; re-enable via env to probe.
INPUT_NOISE = float(os.environ.get("INPUT_NOISE", "0.0"))   # std in standardized space
MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.0"))   # 0 disables mixup
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
OUT_JSON = Path(__file__).resolve().parent / "stage3_results.json"


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights(y_enc, n_classes, mode):
    if mode == "none":
        return None
    counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    if mode == "inv":
        w = len(y_enc) / (n_classes * counts)
    else:
        w = np.sqrt(1.0 / counts)
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed):
    set_seed(seed)
    model = MLP(in_dim, n_classes, HIDDEN, DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    # mixup needs a soft-target loss; build a weighted KL-style CE manually
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, BATCH):
            b = order[i:i + BATCH]
            xb, yb = Xtr_t[b], ytr_t[b]
            if INPUT_NOISE > 0:
                xb = xb + INPUT_NOISE * torch.randn(xb.shape, device=DEVICE, generator=g)
            opt.zero_grad()
            if MIXUP_ALPHA > 0 and xb.shape[0] > 1:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(xb.shape[0], device=DEVICE, generator=g)
                xm = lam * xb + (1 - lam) * xb[perm]
                logits = model(xm)
                loss = lam * crit(logits, yb) + (1 - lam) * crit(logits, yb[perm])
            else:
                loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t).argmax(1).cpu().numpy()
        f1 = du.macro_f1(yval_np, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def softmax_probs(model, X_t, n_classes):
    model.eval()
    out = np.zeros((X_t.shape[0], n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], 16384):
            out[i:i + 16384] = F.softmax(model(X_t[i:i + 16384]), 1).cpu().numpy()
    return out


def train_fold(Xtr, ytr, Xte, n_classes, rng):
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    in_dim = Xtr.shape[1]
    n = len(Xtr)
    perm = rng.permutation(n)
    n_val = int(n * VAL_FRAC)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr_t = torch.tensor(Xtr[tr_idx], device=DEVICE)
    ytr_t = torch.tensor(ytr[tr_idx], device=DEVICE)
    Xval_t = torch.tensor(Xtr[val_idx], device=DEVICE)
    yval_np = ytr[val_idx]
    Xte_t = torch.tensor(Xte, device=DEVICE)
    w = class_weights(ytr[tr_idx], n_classes, WEIGHT_MODE)
    probs = np.zeros((len(Xte), n_classes), dtype=np.float64)
    vf1s = []
    for e in range(N_ENSEMBLE):
        model, vf1 = train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim,
                               n_classes, w, SEED + 100 * e)
        probs += softmax_probs(model, Xte_t, n_classes)
        vf1s.append(vf1)
        del model
        torch.cuda.empty_cache()
    return probs.argmax(1), float(np.mean(vf1s))


def main():
    set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  hidden={HIDDEN} dropout={DROPOUT} lr={LR} "
          f"weights={WEIGHT_MODE} ls={LABEL_SMOOTH} ensemble={N_ENSEMBLE} "
          f"noise={INPUT_NOISE} mixup={MIXUP_ALPHA} clean_cls12={CLEAN_CLS12}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    print(f"loaded: {X.shape[0]:,} rows  {X.shape[1]} features  {n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        n_clean = 0
        if CLEAN_CLS12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(
                X[tr], y_enc[tr], df.iloc[tr], cls12_enc, lon[tr], lat[tr])
            n_clean = int((~keep).sum())
            tr_use = tr[keep]
        pred, val_f1 = train_fold(X[tr_use], y_enc[tr_use], X[te], n_classes, rng)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        f1s.append(f1)
        print(f"  fold {k}: F1={f1:.4f}  (val_f1={val_f1:.4f}, "
              f"cls12_dropped={n_clean}, {time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\n=== Stage 3 Robust MLP ===")
    print(f"F1 mean={f1m:.4f}  std={f1std:.4f}   (target TabICL=0.7139)  "
          f"Δ={f1m-0.7139:+.4f}")
    print("per-class F1:")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")
    print(f"total {time.perf_counter()-t0:.1f}s")

    summary = {
        "stage": "stage3_robust_mlp",
        "config": {"hidden": list(HIDDEN), "dropout": DROPOUT, "lr": LR,
                   "weight_mode": WEIGHT_MODE, "label_smooth": LABEL_SMOOTH,
                   "n_ensemble": N_ENSEMBLE, "input_noise": INPUT_NOISE,
                   "mixup_alpha": MIXUP_ALPHA, "clean_cls12": CLEAN_CLS12},
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
        "target_tabicl": 0.7139,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

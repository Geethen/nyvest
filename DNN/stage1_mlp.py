"""Stage 1 — simple MLP baseline.

Goal: get a clean, honest macro-F1 with the simplest reasonable DNN, on the same
3-fold spatial CV as the TabICL reference (target to beat: 0.7139).

Design (deliberately plain):
  - StandardScaler on features (fit on TRAIN fold only — no leak)
  - 2 hidden layers, ReLU, dropout
  - class-weighted cross-entropy (inverse-frequency) to handle the strong
    imbalance (cls12 ~1.3k rows vs cls6 ~106k)
  - Adam, early stopping on a held-out slice of the train fold
  - all on GPU; batch size keeps memory trivial (<1 GB)

Run:
  ~/myprojects/recover/.venv/bin/python DNN/stage1_mlp.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
HIDDEN = (256, 128)
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 15
BATCH = 4096
VAL_FRAC = 0.1


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights(y_enc, n_classes):
    counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y_enc) / (n_classes * counts)  # inverse-frequency, mean ~1
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_fold(Xtr, ytr, Xte, n_classes, rng):
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    # carve a validation slice out of train for early stopping
    n = len(Xtr)
    perm = rng.permutation(n)
    n_val = int(n * VAL_FRAC)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr_t = torch.tensor(Xtr[tr_idx], device=DEVICE)
    ytr_t = torch.tensor(ytr[tr_idx], device=DEVICE)
    Xval_t = torch.tensor(Xtr[val_idx], device=DEVICE)
    yval_np = ytr[val_idx]

    model = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    w = class_weights(ytr[tr_idx], n_classes)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, best_state, bad = -1.0, None, 0
    n_tr = len(tr_idx)
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = crit(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        # validation macro-F1
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t).argmax(1).cpu().numpy()
        f1 = du.macro_f1(yval_np, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1, best_state, bad = f1, {k: v.detach().cpu().clone()
                                            for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        Xte_t = torch.tensor(Xte, device=DEVICE)
        for i in range(0, len(Xte_t), 16384):
            preds.append(model(Xte_t[i:i + 16384]).argmax(1).cpu().numpy())
    return np.concatenate(preds), best_f1, epoch + 1


def main():
    set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    n_classes = len(data["classes"])
    print(f"loaded: {X.shape[0]:,} rows  {X.shape[1]} features  "
          f"{n_classes} classes {data['classes']}")

    rng = np.random.default_rng(SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        pred, val_f1, n_ep = train_fold(X[tr], y_enc[tr], X[te], n_classes, rng)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        pc = du.per_class_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(pc)
        print(f"  fold {k}: F1={f1:.4f}  (val_f1={val_f1:.4f}, {n_ep} epochs, "
              f"{time.perf_counter()-ft:.1f}s)")

    f1m, f1s_std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\n=== Stage 1 MLP ===")
    print(f"F1 mean={f1m:.4f}  std={f1s_std:.4f}   (target TabICL=0.7139)  "
          f"Δ={f1m-0.7139:+.4f}")
    print("per-class F1:")
    for c, v in zip(data["classes"], pc_mean):
        print(f"  class {c:2d}: {v:.4f}")
    print(f"total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

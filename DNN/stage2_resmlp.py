"""Stage 2 — residual MLP with stronger regularization.

Stage 1 (plain 2-layer MLP) already hit F1=0.7112 vs TabICL 0.7139, and its
val_f1 (~0.79) >> test_f1 (~0.71): the gap is spatial GENERALIZATION, not model
capacity. So Stage 2 adds regularization-oriented complexity, not just depth:

  - BatchNorm + residual (pre-activation) blocks: stabler/deeper optimisation
    without the spatial overfit that raw stacking gives.
  - Label smoothing (0.05): softens the noisy grunnkart labels (see test-label
    noise memory) — a cheap, leak-free robustness lever.
  - Softer class weighting: full inverse-frequency over-weights the 146-location
    cls12; use sqrt-inverse-freq (tempered) which empirically trades a little
    rare-class recall for better macro stability.
  - Cosine LR schedule + warmup, AdamW.
  - SWA-style weight averaging over the last good epochs (flatter minima
    generalise better across spatial folds).
  - Ensemble of N seeds (probability average) — the single cheapest macro-F1
    gain for a fast model, and we have the GPU headroom.

Knobs are env vars so Stage 3 can sweep them.

Run:
  ~/myprojects/recover/.venv/bin/python DNN/stage2_resmlp.py
"""

from __future__ import annotations

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

WIDTH = int(os.environ.get("WIDTH", "256"))
N_BLOCKS = int(os.environ.get("N_BLOCKS", "3"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
LR = float(os.environ.get("LR", "2e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "150"))
PATIENCE = int(os.environ.get("PATIENCE", "20"))
BATCH = int(os.environ.get("BATCH", "4096"))
VAL_FRAC = 0.1
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", "sqrt")  # none|inv|sqrt
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "3"))
SWA_START_FRAC = 0.7  # start averaging weights after this frac of (best) epochs


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class ResBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Linear, twice, + skip."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(F.relu(self.bn1(x)))
        h = self.drop(h)
        h = self.fc2(F.relu(self.bn2(h)))
        return x + h


class ResMLP(nn.Module):
    def __init__(self, in_dim, n_classes, width, n_blocks, dropout):
        super().__init__()
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, dropout) for _ in range(n_blocks)])
        self.head_bn = nn.BatchNorm1d(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(F.relu(self.head_bn(x)))


def class_weights(y_enc, n_classes, mode):
    if mode == "none":
        return None
    counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    if mode == "inv":
        w = len(y_enc) / (n_classes * counts)
    else:  # sqrt-tempered inverse frequency
        inv = 1.0 / counts
        w = np.sqrt(inv)
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed):
    set_seed(seed)
    model = ResMLP(in_dim, n_classes, WIDTH, N_BLOCKS, DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    swa_state, swa_n = None, 0
    swa_start = int(MAX_EPOCHS * SWA_START_FRAC)
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = crit(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        sched.step()
        # accumulate SWA weights in the back portion of training
        if epoch >= swa_start:
            cur = {k: v.detach().float() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in cur.items()}
            else:
                for k in swa_state:
                    swa_state[k] = (swa_state[k] * swa_n + cur[k]) / (swa_n + 1)
            swa_n += 1
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t).argmax(1).cpu().numpy()
        f1 = du.macro_f1(yval_np, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    # Pick the better of (best-checkpoint) vs (SWA-averaged) on the val slice.
    candidates = [("best", best_state)]
    if swa_state is not None:
        candidates.append(("swa", {k: v.cpu() for k, v in swa_state.items()}))
    best_choice, best_cf1 = best_state, best_f1
    for name, st in candidates:
        model.load_state_dict(st)
        # BN running stats from SWA averaging can be stale; recompute on a pass.
        if name == "swa":
            model.train()
            with torch.no_grad():
                for i in range(0, n_tr, BATCH):
                    model(Xtr_t[i:i + BATCH])
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t).argmax(1).cpu().numpy()
        cf1 = du.macro_f1(yval_np, vp, n_classes)
        if cf1 > best_cf1:
            best_cf1, best_choice = cf1, {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_choice)
    return model, best_cf1


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
    print(f"device={DEVICE}  width={WIDTH} blocks={N_BLOCKS} dropout={DROPOUT} "
          f"lr={LR} wd={WEIGHT_DECAY} ls={LABEL_SMOOTH} weights={WEIGHT_MODE} "
          f"ensemble={N_ENSEMBLE}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    n_classes = len(data["classes"])
    print(f"loaded: {X.shape[0]:,} rows  {X.shape[1]} features  {n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        pred, val_f1 = train_fold(X[tr], y_enc[tr], X[te], n_classes, rng)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        f1s.append(f1)
        print(f"  fold {k}: F1={f1:.4f}  (val_f1={val_f1:.4f}, "
              f"{time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\n=== Stage 2 ResMLP ===")
    print(f"F1 mean={f1m:.4f}  std={f1std:.4f}   (target TabICL=0.7139)  "
          f"Δ={f1m-0.7139:+.4f}")
    print("per-class F1:")
    for c, v in zip(data["classes"], pc_mean):
        print(f"  class {c:2d}: {v:.4f}")
    print(f"total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

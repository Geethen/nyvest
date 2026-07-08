"""Stage 5 — newer NN architectures vs the plain-MLP best (F1=0.7318).

Same robust recipe as Stage 3 (5-seed ensemble, sqrt class weights, label
smoothing, train-only cls12 cleaning, identical spatial-CV protocol) so any
delta is purely the ARCHITECTURE. Swap with ARCH env var:

  glu   - GLU/GEGLU gated MLP. Gated linear units are a consistently strong
          tabular-DL block (gating gives multiplicative feature interactions a
          plain ReLU-MLP lacks).
  snn   - Self-Normalizing Network: SELU + AlphaDropout + lecun_normal init.
          A self-regularizing MLP; good when the lever is generalization.
   residual_glu - residual GLU blocks (a touch more depth, still gated).
  ft    - FT-Transformer-lite: tokenize each scalar feature into a d-dim token
          (per-feature weight+bias), prepend a [CLS] token, a few Transformer
          encoder layers, classify from [CLS]. The current SOTA-ish tabular DL
          architecture; per-feature attention is the novel inductive bias to
          test on the 67 AlphaEarth+lidar features.

Run:  ARCH=glu ~/myprojects/recover/.venv/bin/python DNN/stage5_arch.py
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
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402
from dnn_paths import result_path # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
ARCH = os.environ.get("ARCH", "glu")
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
WIDTH = int(os.environ.get("WIDTH", "256"))
# FT-Transformer knobs
FT_DIM = int(os.environ.get("FT_DIM", "64"))
FT_HEADS = int(os.environ.get("FT_HEADS", "8"))
FT_LAYERS = int(os.environ.get("FT_LAYERS", "3"))
LR = float(os.environ.get("LR", "1e-3"))
OUT_JSON = result_path(f"stage5_{ARCH}_results.json")


# ------------------------------- architectures -------------------------------
class GLUBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.drop(a * torch.sigmoid(b))  # GLU gating


class GLUMLP(nn.Module):
    def __init__(self, in_dim, n_classes, width, dropout):
        super().__init__()
        self.b1 = GLUBlock(in_dim, width, dropout)
        self.b2 = GLUBlock(width, width // 2, dropout)
        self.head = nn.Linear(width // 2, n_classes)

    def forward(self, x):
        return self.head(self.b2(self.b1(x)))


class ResidualGLU(nn.Module):
    def __init__(self, in_dim, n_classes, width, dropout, n_blocks=3):
        super().__init__()
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([GLUBlock(width, width, dropout)
                                     for _ in range(n_blocks)])
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        x = F.relu(self.stem(x))
        for b in self.blocks:
            x = x + b(x)
        return self.head(x)


class SNN(nn.Module):
    """Self-normalizing net: SELU + AlphaDropout + lecun_normal init."""

    def __init__(self, in_dim, n_classes, width, dropout):
        super().__init__()
        dims = [in_dim, width, width, width // 2]
        self.lins = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.drops = nn.ModuleList(
            [nn.AlphaDropout(dropout) for _ in range(len(self.lins))])
        self.head = nn.Linear(dims[-1], n_classes)
        for lin in self.lins:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        for lin, drop in zip(self.lins, self.drops):
            x = drop(F.selu(lin(x)))
        return self.head(x)


class FTTransformer(nn.Module):
    """FT-Transformer-lite: per-feature scalar tokenizer + [CLS] + encoder."""

    def __init__(self, in_dim, n_classes, dim, heads, layers, dropout):
        super().__init__()
        # each scalar feature -> dim-vector: x_j * w_j + b_j
        self.feat_w = nn.Parameter(torch.randn(in_dim, dim) * 0.02)
        self.feat_b = nn.Parameter(torch.zeros(in_dim, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        # x: [B, F] -> tokens [B, F, dim]
        tok = x.unsqueeze(-1) * self.feat_w + self.feat_b
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))


def build(in_dim, n_classes):
    if ARCH == "glu":
        return GLUMLP(in_dim, n_classes, WIDTH, DROPOUT)
    if ARCH == "residual_glu":
        return ResidualGLU(in_dim, n_classes, WIDTH, DROPOUT)
    if ARCH == "snn":
        return SNN(in_dim, n_classes, WIDTH, DROPOUT)
    if ARCH == "ft":
        return FTTransformer(in_dim, n_classes, FT_DIM, FT_HEADS, FT_LAYERS, DROPOUT)
    raise ValueError(f"unknown ARCH={ARCH}")


# ------------------------------- training -------------------------------
def train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed):
    s3.set_seed(seed)
    model = build(in_dim, n_classes).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=s3.LABEL_SMOOTH)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=s3.WEIGHT_DECAY)
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(s3.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, s3.BATCH):
            b = order[i:i + s3.BATCH]
            opt.zero_grad()
            loss = crit(model(Xtr_t[b]), ytr_t[b])
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
            if bad >= s3.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def train_fold(Xtr, ytr, Xte, n_classes, rng):
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    in_dim = Xtr.shape[1]
    n = len(Xtr)
    perm = rng.permutation(n)
    n_val = int(n * s3.VAL_FRAC)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr_t = torch.tensor(Xtr[tr_idx], device=DEVICE)
    ytr_t = torch.tensor(ytr[tr_idx], device=DEVICE)
    Xval_t = torch.tensor(Xtr[val_idx], device=DEVICE)
    yval_np = ytr[val_idx]
    Xte_t = torch.tensor(Xte, device=DEVICE)
    w = s3.class_weights(ytr[tr_idx], n_classes, s3.WEIGHT_MODE)
    probs = np.zeros((len(Xte), n_classes), dtype=np.float64)
    vf1s = []
    for e in range(N_ENSEMBLE):
        model, vf1 = train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim,
                               n_classes, w, SEED + 100 * e)
        probs += s3.softmax_probs(model, Xte_t, n_classes)
        vf1s.append(vf1)
        del model
        torch.cuda.empty_cache()
    return probs.argmax(1), float(np.mean(vf1s))


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  ARCH={ARCH}  ensemble={N_ENSEMBLE}  "
          f"clean_cls12={CLEAN_CLS12}  width={WIDTH} dropout={DROPOUT} lr={LR}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    print(f"loaded: {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if CLEAN_CLS12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(
                X[tr], y_enc[tr], df.iloc[tr], cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        pred, vf1 = train_fold(X[tr_use], y_enc[tr_use], X[te], n_classes, rng)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  (val_f1={vf1:.4f}, "
              f"{time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\n=== Stage 5  ARCH={ARCH} ===")
    print(f"F1 mean={f1m:.4f}  std={f1std:.4f}   (MLP best=0.7318, target=0.7139)  "
          f"Δ_vs_best={f1m-0.7318:+.4f}")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")
    print(f"total {time.perf_counter()-t0:.1f}s")

    OUT_JSON.write_text(json.dumps({
        "stage": "stage5_arch", "arch": ARCH, "n_ensemble": N_ENSEMBLE,
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
        "mlp_best": 0.7318, "target_tabicl": 0.7139,
    }, indent=2))
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

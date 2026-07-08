"""Stage 9 — lightweight FEATURE attention in front of the best MLP.

Rationale for the design choice: we have 67 covariates, so attention is tempting.
But FT-Transformer-style feature SELF-attention (per-feature token + transformer)
already underperformed the plain MLP (partial run ~0.71-0.73 < 0.7318) and is
slow — and the 64 AlphaEarth bands are ALREADY a learned embedding, so attending
across them is largely redundant. The lighter, better-motivated bet is
input-dependent feature GATING (squeeze-excite / channel attention): learn a
per-feature soft importance weight that depends on the input, reweight the
features, then run the SAME winning 2-layer MLP. Cheap (no token blow-up), keeps
the model that works, and lets the net softly select among many covariates
per-sample.

Variants (ATTN env):
  se      - squeeze-excite gate: w = sigmoid(MLP(x)) in [0,1]^F, x' = x * w.
  softmax - competitive attention: w = F * softmax(MLP(x)) (sums to F; forces
            the net to allocate a fixed budget across features).
  none    - control = plain best MLP (== Stage 3), for an apples-to-apples Δ.

Everything else identical to the best recipe (5-seed ensemble, sqrt weights,
label smoothing, train-only cls12 centroid clean, 3-fold spatial CV). So any Δ is
purely the attention block.

Run: ATTN=se ~/myprojects/recover/.venv/bin/python DNN/stage9_feat_attn.py
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

DEVICE = s3.DEVICE
SEED = 0
ATTN = os.environ.get("ATTN", "se")          # se | softmax | none
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
HIDDEN = tuple(int(x) for x in os.environ.get("HIDDEN", "256,128").split(","))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
ATTN_HIDDEN = int(os.environ.get("ATTN_HIDDEN", "32"))   # bottleneck of the gate
LR = float(os.environ.get("LR", "1e-3"))
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
OUT_JSON = Path(__file__).resolve().parent / f"stage9_attn_{ATTN}.json"
NAMES = {2: "sparse-veg", 3: "forest", 4: "forest", 5: "GRASSLAND", 6: "open-upland",
         7: "mire/wet", 8: "water", 10: "bare", 11: "built/infra", 12: "snow/ice"}


class FeatAttnMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, dropout, attn, attn_hidden):
        super().__init__()
        self.attn = attn
        if attn != "none":
            self.gate = nn.Sequential(
                nn.Linear(in_dim, attn_hidden), nn.ReLU(),
                nn.Linear(attn_hidden, in_dim))
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)
        self._last_w = None

    def forward(self, x):
        if self.attn == "se":
            w = torch.sigmoid(self.gate(x))            # [0,1]^F per-feature gate
            x = x * w
            self._last_w = w
        elif self.attn == "softmax":
            w = x.shape[1] * F.softmax(self.gate(x), dim=-1)  # sums to F
            x = x * w
            self._last_w = w
        return self.net(x)


def train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed):
    s3.set_seed(seed)
    model = FeatAttnMLP(in_dim, n_classes, HIDDEN, DROPOUT, ATTN, ATTN_HIDDEN).to(DEVICE)
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
    return model


def probs(model, Xt, n_classes):
    model.eval()
    out = np.zeros((Xt.shape[0], n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, Xt.shape[0], 16384):
            out[i:i + 16384] = F.softmax(model(Xt[i:i + 16384]), 1).cpu().numpy()
    return out


def train_fold(Xtr, ytr, Xte, n_classes, rng, feat_names):
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
    attn_w = np.zeros(Xtr.shape[1])
    for e in range(N_ENSEMBLE):
        m = train_one(Xtr_t, ytr_t, Xval_t, ytr[vi], Xtr.shape[1],
                      n_classes, w, SEED + 100 * e)
        P += probs(m, Xte_t, n_classes)
        if ATTN != "none":
            m.eval()
            with torch.no_grad():
                _ = m(Xte_t[:20000]); attn_w += m._last_w.mean(0).cpu().numpy()
        del m
        torch.cuda.empty_cache()
    return P.argmax(1), attn_w / N_ENSEMBLE


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  ATTN={ATTN}  ensemble={N_ENSEMBLE}  attn_hidden={ATTN_HIDDEN}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]; n_classes = len(classes)
    c12 = classes.index(12); lon, lat = df["lon"].values, df["lat"].values
    feat_names = data["feat_cols"]
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs, attn_acc = [], [], np.zeros(X.shape[1])
    for k, tr, te in du.fold_indices(y_enc, groups):
        tr_use = tr
        if CLEAN_CLS12:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr], c12,
                                             lon[tr], lat[tr])
            tr_use = tr[keep]
        pred, aw = train_fold(X[tr_use], y_enc[tr_use], X[te], n_classes, rng, feat_names)
        attn_acc += aw
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1); pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  {time.perf_counter()-t0:.0f}s")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc = np.mean(pcs, axis=0)
    print(f"\n=== Stage 9 feature-attention  ATTN={ATTN} ===")
    print(f"macro F1={f1m:.4f}  std={f1std:.4f}   (best=0.7341, plain-MLP=0.7318)  "
          f"Δ_vs_best={f1m-0.7341:+.4f}")
    for c, v in zip(classes, pc):
        print(f"    {NAMES[c]+'/'+str(c):>14}: {v:.4f}")
    if ATTN != "none":
        aw = attn_acc / len(f1s)
        topi = np.argsort(-aw)[:8]; boti = np.argsort(aw)[:5]
        print(f"  top-weighted feats: {[(feat_names[i], round(float(aw[i]),3)) for i in topi]}")
        print(f"  low-weighted feats: {[(feat_names[i], round(float(aw[i]),3)) for i in boti]}")
    OUT_JSON.write_text(json.dumps({
        "stage": "stage9_feat_attn", "attn": ATTN, "n_ensemble": N_ENSEMBLE,
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc)},
        "best": 0.7341, "plain_mlp": 0.7318,
    }, indent=2))
    print(f"saved -> {OUT_JSON}  total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

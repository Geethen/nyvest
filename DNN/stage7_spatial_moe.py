"""Stage 7 — SPATIAL Mixture-of-Experts: each expert specializes by GEOGRAPHY.

Motivation: the bottleneck here is spatial generalization (Stages 2/5/6), and the
AOI spans distinct land-cover regimes (coast / fjord / valley / alpine). So tie
experts to REGIONS, not classes, and let geography pick the expert. Experts still
see the 67 features; only the GATING is spatial.

Two routing modes (MODE env var):
  soft  - gate is an MLP on (lon,lat) -> softmax over E experts. Smooth, learned,
          differentiable geographic specialization; border rows blend experts.
  hard  - KMeans(E) on (lon,lat) of the TRAIN fold only (leak-free); each row is
          routed to its region's expert (one-hot). Test rows assigned to the
          nearest train centroid. Pure regional sub-models, fixed boundaries.

LEAK NOTE: coords are not labels, so gating on them is leak-free. KMeans centroids
are fit on TRAIN coords only. BUT the CV folds are themselves geographic
(GroupKFold on cell_id), so a hard region can line up with a held-out fold — an
expert may then train on little data resembling the test fold. That is the honest
failure mode of hard spatial experts (not leakage); the soft mode blends past it.
The comparison surfaces exactly this.

Held identical to the best recipe: 5-seed ensemble, sqrt weights, label smoothing,
train-only cls12 cleaning, same 3-fold spatial CV, StandardScaler, early stopping.
Only the architecture/routing changes.

Run:  MODE=soft N_EXPERTS=4 ~/myprojects/recover/.venv/bin/python DNN/stage7_spatial_moe.py
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402
from stage6_moe import ExpertMLP, importance_loss  # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
MODE = os.environ.get("MODE", "soft")          # soft | hard
N_EXPERTS = int(os.environ.get("N_EXPERTS", "4"))
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
GATE_HIDDEN = int(os.environ.get("GATE_HIDDEN", "64"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
HIDDEN = tuple(int(x) for x in os.environ.get("HIDDEN", "256,128").split(","))
LR = float(os.environ.get("LR", "1e-3"))
LB_COEF = float(os.environ.get("LB_COEF", "0.01"))
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
OUT_JSON = Path(__file__).resolve().parent / f"stage7_spatial_{MODE}_e{N_EXPERTS}.json"
NAMES = {2: "sparse-veg", 3: "forest", 4: "forest", 5: "GRASSLAND", 6: "open-upland",
         7: "mire/wet", 8: "water", 10: "bare", 11: "built/infra", 12: "snow/ice"}


class SpatialMoE(nn.Module):
    """Experts route by geography. Input is [features | coords]; the gate sees
    ONLY the coords (last 2 dims), experts see ONLY the features.

    soft: gate = softmax(MLP(coords)).
    hard: gate = one-hot(region_id) supplied as the coord slot (region passed in
          as an integer in coords[:,0]); see forward.
    """

    def __init__(self, feat_dim, n_classes, n_experts, hidden, dropout,
                 gate_hidden, mode):
        super().__init__()
        self.mode = mode
        self.n_experts = n_experts
        self.feat_dim = feat_dim
        self.experts = nn.ModuleList(
            [ExpertMLP(feat_dim, n_classes, hidden, dropout) for _ in range(n_experts)])
        if mode == "soft":
            self.gate = nn.Sequential(
                nn.Linear(2, gate_hidden), nn.ReLU(),
                nn.Linear(gate_hidden, n_experts))

    def forward(self, x):
        feats = x[:, :self.feat_dim]
        tail = x[:, self.feat_dim:]
        if self.mode == "soft":
            gate = F.softmax(self.gate(tail), dim=-1)              # [B,E] on coords
        else:  # hard: tail[:,0] holds the region id
            rid = tail[:, 0].long().clamp(0, self.n_experts - 1)
            gate = F.one_hot(rid, self.n_experts).float()
        ex = torch.stack([e(feats) for e in self.experts], 1)     # [B,E,C]
        logits = (gate.unsqueeze(-1) * ex).sum(1)
        return logits, gate


def train_one(Xtr_t, ytr_t, Xval_t, yval_np, feat_dim, n_classes, w, seed):
    s3.set_seed(seed)
    model = SpatialMoE(feat_dim, n_classes, N_EXPERTS, HIDDEN, DROPOUT,
                       GATE_HIDDEN, MODE).to(DEVICE)
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
            logits, gate = model(Xtr_t[b])
            loss = crit(logits, ytr_t[b])
            if MODE == "soft" and LB_COEF > 0:
                loss = loss + LB_COEF * importance_loss(gate)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t)[0].argmax(1).cpu().numpy()
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


def probs(model, Xt, n_classes):
    model.eval()
    out = np.zeros((Xt.shape[0], n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, Xt.shape[0], 16384):
            out[i:i + 16384] = F.softmax(model(Xt[i:i + 16384])[0], 1).cpu().numpy()
    return out


def gate_usage(model, Xt):
    """Mean gate weight per expert over a sample (specialization diagnostic)."""
    model.eval()
    with torch.no_grad():
        g = model(Xt[:20000])[1].mean(0).cpu().numpy()
    return g


def train_fold(Xtr, ytr, coord_tr, Xte, coord_te, n_classes, rng):
    feat_dim = Xtr.shape[1]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    # build the gate "tail": soft -> standardized coords; hard -> region id
    if MODE == "soft":
        cs = StandardScaler().fit(coord_tr)
        tail_tr = cs.transform(coord_tr).astype(np.float32)
        tail_te = cs.transform(coord_te).astype(np.float32)
    else:
        km = KMeans(n_clusters=N_EXPERTS, n_init=4, random_state=SEED).fit(coord_tr)
        rid_tr = km.predict(coord_tr).astype(np.float32)[:, None]
        rid_te = km.predict(coord_te).astype(np.float32)[:, None]
        tail_tr = np.repeat(rid_tr, 2, axis=1)   # pad to width-2 tail
        tail_te = np.repeat(rid_te, 2, axis=1)

    Xtr_aug = np.concatenate([Xtr_s, tail_tr], axis=1)
    Xte_aug = np.concatenate([Xte_s, tail_te], axis=1)

    n = len(Xtr_aug)
    perm = rng.permutation(n)
    nv = int(n * s3.VAL_FRAC)
    vi, ti = perm[:nv], perm[nv:]
    Xtr_t = torch.tensor(Xtr_aug[ti], device=DEVICE)
    ytr_t = torch.tensor(ytr[ti], device=DEVICE)
    Xval_t = torch.tensor(Xtr_aug[vi], device=DEVICE)
    Xte_t = torch.tensor(Xte_aug, device=DEVICE)
    w = s3.class_weights(ytr[ti], n_classes, s3.WEIGHT_MODE)

    P = np.zeros((len(Xte), n_classes))
    vf1s, gates = [], []
    for e in range(N_ENSEMBLE):
        m, vf1 = train_one(Xtr_t, ytr_t, Xval_t, ytr[vi], feat_dim,
                           n_classes, w, SEED + 100 * e)
        P += probs(m, Xte_t, n_classes)
        vf1s.append(vf1)
        gates.append(gate_usage(m, Xte_t))
        del m
        torch.cuda.empty_cache()
    return P.argmax(1), float(np.mean(vf1s)), np.mean(gates, axis=0)


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  MODE={MODE}  N_EXPERTS={N_EXPERTS}  "
          f"ensemble={N_ENSEMBLE}  lb={LB_COEF}  clean_cls12={CLEAN_CLS12}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    coords = np.stack([lon, lat], axis=1).astype(np.float64)
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs, gate_log = [], [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if CLEAN_CLS12:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        pred, vf1, gate = train_fold(X[tr_use], y_enc[tr_use], coords[tr_use],
                                     X[te], coords[te], n_classes, rng)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        gate_log.append([round(float(v), 3) for v in gate])
        print(f"  fold {k}: F1={f1:.4f}  (val={vf1:.4f})  "
              f"gate_usage={gate_log[-1]}  {time.perf_counter()-ft:.1f}s")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\n=== Stage 7 spatial MoE  MODE={MODE} E={N_EXPERTS} ===")
    print(f"F1 mean={f1m:.4f}  std={f1std:.4f}   (MLP best=0.7318, "
          f"class-MoE=0.7294, target=0.7139)  Δ_vs_best={f1m-0.7318:+.4f}")
    for c, v in zip(classes, pc_mean):
        print(f"  class {NAMES[c]+'/'+str(c):>14}: {v:.4f}")
    print(f"total {time.perf_counter()-t0:.1f}s")
    OUT_JSON.write_text(json.dumps({
        "stage": "stage7_spatial_moe", "mode": MODE, "n_experts": N_EXPERTS,
        "n_ensemble": N_ENSEMBLE, "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
        "gate_usage_per_fold": gate_log, "mlp_best": 0.7318, "class_moe": 0.7294,
    }, indent=2))
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

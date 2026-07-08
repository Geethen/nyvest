"""Stage 6 — Mixture-of-Experts (MoE) vs the plain-MLP best (F1=0.7318).

Hypothesis: a soft (differentiable) MoE lets experts SPECIALIZE on the hard /
confusable land-cover classes (2 sparse-veg, 5 grassland, 6 open-upland,
7 mire/wet, 12 snow/ice) while easy classes (8 water, 10 bare) are handled
trivially — buying macro-F1 over a single shared MLP.

Everything except the ARCHITECTURE is held identical to Stage 3 / Stage 5:
5-seed probability ensemble, sqrt class weights, label smoothing, train-only
cls12 cleaning, the same 3-fold spatial CV (du.fold_indices), the same
StandardScaler + early-stopping training loop. So any delta is purely the MoE.

Soft MoE: a gating net produces a softmax over E experts; each expert is the
same 2-layer MLP (256,128) head as the best model. Output mixes expert LOGITS:
    logits = sum_e gate_e(x) * expert_e(x)
A load-balancing (importance) loss keeps experts from collapsing onto one.

Variants (VARIANT env var):
  moe4   - plain soft MoE, E=4 experts.
  moe8   - more experts, E=8.
  moe_hard - capacity biased toward hard classes: a dedicated "specialist"
             expert is supervised with an auxiliary CE on the hard-class subset
             so it learns to discriminate 2/5/6/7/12; the gate is free to route
             hard-class rows to it.

Run:  VARIANT=moe4 ~/myprojects/recover/.venv/bin/python DNN/stage6_moe.py
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
VARIANT = os.environ.get("VARIANT", "moe4")
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", "5"))
CLEAN_CLS12 = os.environ.get("CLEAN_CLS12", "1") == "1"
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
HIDDEN = tuple(int(x) for x in os.environ.get("HIDDEN", "256,128").split(","))
LR = float(os.environ.get("LR", "1e-3"))
# MoE knobs
N_EXPERTS = int(os.environ.get("N_EXPERTS", "4"))
GATE_HIDDEN = int(os.environ.get("GATE_HIDDEN", "64"))
LB_COEF = float(os.environ.get("LB_COEF", "0.01"))     # load-balance loss weight
AUX_COEF = float(os.environ.get("AUX_COEF", "0.3"))    # moe_hard specialist aux loss
# encoded hard classes are resolved at runtime from the class list
HARD_CLASSES = [2, 5, 6, 7, 12]

# per-variant overrides
if VARIANT == "moe8":
    N_EXPERTS = int(os.environ.get("N_EXPERTS", "8"))

OUT_JSON = result_path(f"stage6_moe_{VARIANT}_results.json")


# ------------------------------- architecture -------------------------------
class ExpertMLP(nn.Module):
    """Same 2-layer MLP shape as the best model, emitting class logits."""

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


class SoftMoE(nn.Module):
    """Soft mixture of E expert MLPs with a softmax gating network.

    Mixes expert logits by the gate weights. Also exposes the gate weights and
    per-expert logits so we can (a) add a load-balancing loss and (b) measure
    expert specialization cheaply at eval time.
    """

    def __init__(self, in_dim, n_classes, n_experts, hidden, dropout, gate_hidden):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [ExpertMLP(in_dim, n_classes, hidden, dropout) for _ in range(n_experts)])
        self.gate = nn.Sequential(
            nn.Linear(in_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, n_experts))

    def forward(self, x):
        gate = F.softmax(self.gate(x), dim=-1)              # [B, E]
        ex = torch.stack([e(x) for e in self.experts], 1)   # [B, E, C]
        logits = (gate.unsqueeze(-1) * ex).sum(1)           # [B, C]
        return logits, gate, ex


def importance_loss(gate):
    """Coefficient-of-variation^2 of per-batch expert importance (Shazeer et al).
    Encourages the gate to use all experts -> prevents expert collapse."""
    importance = gate.sum(0)                       # [E]
    eps = 1e-8
    cv2 = importance.var(unbiased=False) / (importance.mean() ** 2 + eps)
    return cv2


def build(in_dim, n_classes):
    return SoftMoE(in_dim, n_classes, N_EXPERTS, HIDDEN, DROPOUT, GATE_HIDDEN)


# ------------------------------- training -------------------------------
def train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed, hard_mask_t):
    s3.set_seed(seed)
    model = build(in_dim, n_classes).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=s3.LABEL_SMOOTH)
    # specialist aux loss (moe_hard): unweighted CE on the last expert, evaluated
    # only on rows whose label is a hard class, so expert -1 becomes the
    # hard-class discriminator the gate can route to.
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=s3.WEIGHT_DECAY)
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    use_aux = VARIANT == "moe_hard"
    for epoch in range(s3.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, s3.BATCH):
            b = order[i:i + s3.BATCH]
            opt.zero_grad()
            logits, gate, ex = model(Xtr_t[b])
            loss = crit(logits, ytr_t[b]) + LB_COEF * importance_loss(gate)
            if use_aux:
                yb = ytr_t[b]
                hm = hard_mask_t[yb]                       # [B] bool
                if hm.any():
                    spec_logits = ex[:, -1, :][hm]         # last expert
                    loss = loss + AUX_COEF * F.cross_entropy(spec_logits, yb[hm])
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


def softmax_probs(model, X_t, n_classes):
    model.eval()
    out = np.zeros((X_t.shape[0], n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], 16384):
            out[i:i + 16384] = F.softmax(model(X_t[i:i + 16384])[0], 1).cpu().numpy()
    return out


def expert_assignment(model, X_t):
    """argmax gate expert per row -> [N] int. For specialization diagnostics."""
    model.eval()
    out = np.zeros(X_t.shape[0], dtype=np.int64)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], 16384):
            out[i:i + 16384] = model(X_t[i:i + 16384])[1].argmax(1).cpu().numpy()
    return out


def train_fold(Xtr, ytr, Xte, n_classes, rng, hard_mask_t):
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
    assign = None  # gate argmax on the TEST fold from the first ensemble member
    for e in range(N_ENSEMBLE):
        model, vf1 = train_one(Xtr_t, ytr_t, Xval_t, yval_np, in_dim,
                               n_classes, w, SEED + 100 * e, hard_mask_t)
        probs += softmax_probs(model, Xte_t, n_classes)
        vf1s.append(vf1)
        if e == 0:
            assign = expert_assignment(model, Xte_t)
        del model
        torch.cuda.empty_cache()
    return probs.argmax(1), float(np.mean(vf1s)), assign


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  VARIANT={VARIANT}  n_experts={N_EXPERTS}  "
          f"ensemble={N_ENSEMBLE}  clean_cls12={CLEAN_CLS12}  hidden={HIDDEN} "
          f"lb={LB_COEF} aux={AUX_COEF if VARIANT=='moe_hard' else 0}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    # hard-class mask in encoded space
    hard_enc = [classes.index(c) for c in HARD_CLASSES if c in classes]
    hard_mask = np.zeros(n_classes, dtype=bool)
    hard_mask[hard_enc] = True
    hard_mask_t = torch.tensor(hard_mask, device=DEVICE)
    print(f"loaded: {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes")

    rng = np.random.default_rng(SEED)
    f1s, pcs = [], []
    # accumulate expert-by-class assignment counts across folds (test rows)
    expert_class_counts = np.zeros((n_classes, N_EXPERTS), dtype=np.int64)
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        n_clean = 0
        if CLEAN_CLS12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(
                X[tr], y_enc[tr], df.iloc[tr], cls12_enc, lon[tr], lat[tr])
            n_clean = int((~keep).sum())
            tr_use = tr[keep]
        pred, vf1, assign = train_fold(X[tr_use], y_enc[tr_use], X[te],
                                       n_classes, rng, hard_mask_t)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        for c in range(n_classes):
            cm = y_enc[te] == c
            if cm.any():
                expert_class_counts[c] += np.bincount(assign[cm], minlength=N_EXPERTS)
        print(f"  fold {k}: F1={f1:.4f}  (val_f1={vf1:.4f}, "
              f"cls12_dropped={n_clean}, {time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\n=== Stage 6 MoE  VARIANT={VARIANT} ===")
    print(f"F1 mean={f1m:.4f}  std={f1std:.4f}   (MLP best=0.7318, target=0.7139)  "
          f"Δ_vs_best={f1m-0.7318:+.4f}")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")

    # specialization: dominant expert per class (fraction of that class routed there)
    print("\nexpert specialization (test rows, gate argmax, member 0):")
    row_frac = expert_class_counts / np.maximum(expert_class_counts.sum(1, keepdims=True), 1)
    dominant = {}
    for ci, c in enumerate(classes):
        e = int(row_frac[ci].argmax())
        dominant[str(c)] = {"expert": e, "frac": round(float(row_frac[ci, e]), 3)}
        print(f"  class {c:2d}: expert {e}  ({row_frac[ci, e]*100:.0f}% of rows)  "
              f"dist={np.round(row_frac[ci], 2).tolist()}")
    print(f"total {time.perf_counter()-t0:.1f}s")

    OUT_JSON.write_text(json.dumps({
        "stage": "stage6_moe", "arch": VARIANT, "n_experts": N_EXPERTS,
        "n_ensemble": N_ENSEMBLE,
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
        "expert_dominant_by_class": dominant,
        "mlp_best": 0.7318, "target_tabicl": 0.7139,
    }, indent=2))
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

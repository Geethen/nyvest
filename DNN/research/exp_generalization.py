"""Batch 2 — attack SPATIAL GENERALIZATION, the established bottleneck.

Prior work is unambiguous: capacity is not the lever (residual nets reach 0.965
val / 0.71 test), and the hard classes are confusion-bound. Every architecture
tried has lost. So batch 2 stops redesigning the network and instead changes the
TRAINING OBJECTIVE / OPTIMIZATION to favour features that transfer across
geography. These are the levers that plausibly move a spatial-generalization
ceiling.

EXP variants:

  sam       - Sharpness-Aware Minimization (Foret et al. 2021). Seeks flat minima;
              flat minima are the standard tool for distribution shift, and our
              train->test gap IS a shift (different geography). The single most
              mechanism-matched idea in this whole search.
  irm       - IRM-v1 penalty (Arjovsky et al. 2019) with TRAIN-fold spatial
              KMeans regions as environments. Penalizes predictors whose optimal
              classifier differs per region -> keeps only geography-invariant
              features. Directly targets "learns region-specific shortcuts".
  groupdro  - Group-DRO (Sagawa et al. 2020): optimize the WORST-region loss
              rather than the average. Same environment split as irm; robustness
              to the worst geography instead of invariance.
  ema       - EMA of weights (Polyak averaging; standard in modern LLM training).
              Cheap flat-minimum proxy. SWA lost in stage2 but was entangled with
              BN+residual; EMA on the plain MLP isolates the averaging effect.
  logitadj  - Logit adjustment (Menon et al. 2021): principled long-tail
              correction applied at inference via class priors, replacing the
              heuristic sqrt class weights. Targets macro-F1 on rare classes
              (cls12, cls2) which are what drag the macro average.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

EXP = os.environ.get("EXP", "sam")
RHO = float(os.environ.get("RHO", "0.05"))         # SAM neighbourhood radius
IRM_LAMBDA = float(os.environ.get("IRM_LAMBDA", "100.0"))
N_ENV = int(os.environ.get("N_ENV", "4"))
DRO_ETA = float(os.environ.get("DRO_ETA", "0.01"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))
TAU = float(os.environ.get("TAU", "1.0"))          # logit-adjustment strength


def mlp(i, c):
    return s3.MLP(i, c, ec.HIDDEN, ec.DROPOUT)


# ------------------------------------------------------------------ SAM
class SAMTrainer:
    """SAM needs two forward/backward passes per step, so it can't ride the
    stock loss_fn hook — it gets a bespoke train_one below."""


def train_one_sam(build_fn, Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed,
                  loss_fn=None):
    import data_utils as du
    s3.set_seed(seed)
    model = build_fn(in_dim, n_classes).to(ec.DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=s3.LABEL_SMOOTH)
    opt = torch.optim.Adam(model.parameters(), lr=ec.LR, weight_decay=s3.WEIGHT_DECAY)
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=ec.DEVICE).manual_seed(seed)
    for epoch in range(s3.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=ec.DEVICE, generator=g)
        for i in range(0, n_tr, s3.BATCH):
            b = order[i:i + s3.BATCH]
            xb, yb = Xtr_t[b], ytr_t[b]
            # --- 1st pass: gradient at w
            opt.zero_grad()
            crit(model(xb), yb).backward()
            # --- ascend to w + e(w) : the sharpness probe
            with torch.no_grad():
                gn = torch.norm(torch.stack([
                    p.grad.norm() for p in model.parameters() if p.grad is not None]))
                scale = RHO / (gn + 1e-12)
                eps = []
                for p in model.parameters():
                    if p.grad is None:
                        eps.append(None); continue
                    e = p.grad * scale
                    p.add_(e); eps.append(e)
            # --- 2nd pass: gradient at the perturbed point, applied at w
            opt.zero_grad()
            crit(model(xb), yb).backward()
            with torch.no_grad():
                for p, e in zip(model.parameters(), eps):
                    if e is not None:
                        p.sub_(e)
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


# ------------------------------------------------------------------ EMA
def train_one_ema(build_fn, Xtr_t, ytr_t, Xval_t, yval_np, in_dim, n_classes, w, seed,
                  loss_fn=None):
    import data_utils as du
    s3.set_seed(seed)
    model = build_fn(in_dim, n_classes).to(ec.DEVICE)
    ema = build_fn(in_dim, n_classes).to(ec.DEVICE)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=s3.LABEL_SMOOTH)
    opt = torch.optim.Adam(model.parameters(), lr=ec.LR, weight_decay=s3.WEIGHT_DECAY)
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=ec.DEVICE).manual_seed(seed)
    for epoch in range(s3.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=ec.DEVICE, generator=g)
        for i in range(0, n_tr, s3.BATCH):
            b = order[i:i + s3.BATCH]
            opt.zero_grad()
            crit(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA_DECAY).add_(pm, alpha=1 - EMA_DECAY)
        ema.eval()
        with torch.no_grad():
            vp = ema(Xval_t).argmax(1).cpu().numpy()
        f1 = du.macro_f1(yval_np, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
        else:
            bad += 1
            if bad >= s3.PATIENCE:
                break
    ema.load_state_dict(best_state)
    return ema, best_f1


# ------------------------------------- environment-based losses (IRM / DRO)
_ENV_T = {}   # filled per fold by the patched train_fold


def _irm_penalty(logits, y, crit):
    """IRM-v1: squared grad of the loss wrt a dummy scale of 1.0."""
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = crit(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def make_env_loss(kind):
    state = {"q": None}

    def loss_fn(model, xb, yb, crit):
        env = _ENV_T["env"]
        # env ids for this batch are carried in the last feature column slot
        eb = xb[:, -1].long()
        xf = xb[:, :-1]
        logits = model(xf)
        n_env = int(_ENV_T["n_env"])
        if kind == "irm":
            total, pen = 0.0, 0.0
            cnt = 0
            for e in range(n_env):
                m = eb == e
                if m.sum() < 2:
                    continue
                le = crit(logits[m], yb[m])
                total = total + le
                pen = pen + _irm_penalty(logits[m], yb[m], crit)
                cnt += 1
            if cnt == 0:
                return crit(logits, yb)
            return total / cnt + IRM_LAMBDA * pen / cnt
        else:  # group DRO
            if state["q"] is None:
                state["q"] = torch.ones(n_env, device=xb.device) / n_env
            losses = torch.zeros(n_env, device=xb.device)
            present = torch.zeros(n_env, dtype=torch.bool, device=xb.device)
            for e in range(n_env):
                m = eb == e
                if m.sum() < 2:
                    continue
                losses[e] = crit(logits[m], yb[m])
                present[e] = True
            with torch.no_grad():
                q = state["q"]
                q = q * torch.exp(DRO_ETA * losses.detach())
                q = torch.where(present, q, torch.zeros_like(q))
                s = q.sum()
                state["q"] = q / s if s > 0 else torch.ones_like(q) / n_env
            return (state["q"] * losses).sum()
    return loss_fn


class EnvMLP(nn.Module):
    """Wrapper so eval-time forward ignores the trailing env column."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = s3.MLP(in_dim - 1, n_classes, ec.HIDDEN, ec.DROPOUT)

    def forward(self, x):
        if x.shape[1] == self.net.net[0].in_features + 1:
            x = x[:, :-1]
        return self.net(x)


def run_env_experiment(kind):
    """IRM / GroupDRO need per-row environment ids that survive into the batch.
    We append the env id as an extra column of X (train AND test; the model
    slices it off), and set spatial KMeans envs from TRAIN coords only."""
    import data_utils as du
    import json
    import time

    t0 = time.perf_counter()
    base_pf, base_mean, base_name = ec.baseline_for(ec.RELABEL)
    name = f"{kind}_env{N_ENV}"
    print(f"=== {name} ===")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    coords = np.stack([lon, lat], 1)
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats")

    rng = np.random.default_rng(ec.SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if ec.CLEAN_CLS12:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        km = KMeans(n_clusters=N_ENV, n_init=4, random_state=ec.SEED).fit(coords[tr_use])
        env_tr = km.predict(coords[tr_use]).astype(np.float32)[:, None]
        env_te = km.predict(coords[te]).astype(np.float32)[:, None]
        Xtr_e = np.concatenate([X[tr_use], env_tr], 1)
        Xte_e = np.concatenate([X[te], env_te], 1)
        _ENV_T["n_env"] = N_ENV
        _ENV_T["env"] = True
        P, vf1 = ec.train_fold(lambda i, c: EnvMLP(i, c), Xtr_e, y_enc[tr_use],
                               Xte_e, n_classes, rng, make_env_loss(kind))
        pred = P.argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1); pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  Δ={f1-base_pf[k]:+.4f}  "
              f"(val={vf1:.4f}, {time.perf_counter()-ft:.1f}s)")

    f1m = float(np.mean(f1s)); f1std = float(np.std(f1s))
    deltas = [f1s[k] - base_pf[k] for k in range(len(f1s))]
    dmean = float(np.mean(deltas)); all_pos = all(d > 0 for d in deltas)
    verdict = "WIN" if (all_pos and dmean > 0.003) else ("tie" if abs(dmean) <= 0.003 else "LOSS")
    pc_mean = np.mean(pcs, 0)
    print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}  Δ_mean={dmean:+.4f} "
          f"per_fold_Δ={[round(d,4) for d in deltas]}\nVERDICT: {verdict}")
    rec = {"name": name, "notes": f"{kind.upper()} with {N_ENV} spatial KMeans environments",
           "relabel": ec.RELABEL, "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
           "f1_per_fold": [round(v, 4) for v in f1s], "baseline": base_name,
           "baseline_per_fold": base_pf, "delta_per_fold": [round(d, 4) for d in deltas],
           "delta_mean": round(dmean, 4), "all_folds_positive": all_pos, "verdict": verdict,
           "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
           "n_ensemble": ec.N_ENSEMBLE, "runtime_s": round(time.perf_counter() - t0, 1),
           "config": {"idea": "invariance", "n_env": N_ENV,
                      "irm_lambda": IRM_LAMBDA if kind == "irm" else None}}
    (ec.RESULTS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"saved -> {ec.RESULTS_DIR / f'{name}.json'}")


def main():
    if EXP == "sam":
        orig = ec.train_one
        ec.train_one = train_one_sam
        try:
            ec.run_experiment("sam", mlp,
                notes=f"Sharpness-Aware Minimization rho={RHO}; flat minima for "
                      "geographic distribution shift",
                extra={"idea": "flat_minima", "rho": RHO})
        finally:
            ec.train_one = orig
    elif EXP == "ema":
        orig = ec.train_one
        ec.train_one = train_one_ema
        try:
            ec.run_experiment("ema", mlp,
                notes=f"EMA/Polyak weight averaging decay={EMA_DECAY}",
                extra={"idea": "flat_minima", "ema_decay": EMA_DECAY})
        finally:
            ec.train_one = orig
    elif EXP in ("irm", "groupdro"):
        run_env_experiment(EXP)
    elif EXP == "logitadj":
        import data_utils as du
        data = du.load_data(extra_features="lidar")
        # logit adjustment: train UNWEIGHTED, correct with priors at inference
        raise SystemExit("logitadj handled in exp_logitadj.py")
    else:
        raise SystemExit(f"unknown EXP={EXP}")


if __name__ == "__main__":
    main()

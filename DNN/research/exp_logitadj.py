"""Logit adjustment (Menon et al. 2021) vs the heuristic sqrt class weights.

Macro-F1 averages over classes, so the rare/hard classes (cls12 snow/ice, cls2
sparse-veg) dominate the metric's headroom. The current recipe handles imbalance
with sqrt class weights in the loss — a heuristic. Logit adjustment is the
principled alternative: train with plain CE, then subtract tau*log(prior) from
the logits at inference, which is Bayes-optimal for the balanced error rate.

Two variants — note they use OPPOSITE signs, which is the crux of the method:

  post  - train unweighted, then predict argmax(logits - tau*log(prior)).
          SUBTRACTING the prior up-weights rare classes: the balanced-error
          -optimal rule is argmax p(y|x)/p(y), i.e. log p(y|x) - log p(y).
  loss  - ADD tau*log(prior) inside the softmax DURING TRAINING, then predict
          with PLAIN logits. The additive term makes the model learn a margin
          that already compensates for the prior, so no inference-time shift.

Getting these signs backwards silently destroys the rare classes rather than
erroring (cls12 F1 -> 0.045), so `_sign_selftest()` asserts the direction on a
synthetic imbalanced problem before any real training runs.

Both keep the 5-seed ensemble, cls12 cleaning, folds, and scaler identical.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec          # noqa: E402
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

MODE = os.environ.get("MODE", "post")     # post | loss
TAU = float(os.environ.get("TAU", "1.0"))


class AdjustedMLP(nn.Module):
    """MLP that can shift its logits by `self.adj`.

    `self.adj` is a buffer so it moves with .to(device) and is saved/restored by
    state_dict alongside the weights (early stopping copies state_dict).

    The CALLER sets adj's sign: +tau*log(prior) for the training-time adjusted
    loss, -tau*log(prior) for post-hoc inference. See module docstring."""

    def __init__(self, in_dim, n_classes, hidden, dropout):
        super().__init__()
        self.net = s3.MLP(in_dim, n_classes, hidden, dropout)
        self.register_buffer("adj", torch.zeros(n_classes))
        self.apply_adj = False

    def forward(self, x):
        z = self.net(x)
        return z + self.adj if self.apply_adj else z


def _sign_selftest():
    """Assert the adjustment direction on a synthetic imbalanced problem.

    A sign error here is silent and catastrophic (it suppresses exactly the rare
    classes the method exists to help), so verify before burning GPU hours.
    """
    from scipy.stats import norm
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    prior = np.array([0.9, 0.1])
    y = rng.choice(2, size=20000, p=prior)
    x = rng.normal(loc=np.where(y == 1, 1.0, 0.0), scale=1.0)
    lik = np.stack([norm.pdf(x, 0, 1), norm.pdf(x, 1, 1)], 1)
    post = lik * prior
    post /= post.sum(1, keepdims=True)
    logits = np.log(post + 1e-12)
    lp = np.log(prior + 1e-12)
    f_minus = f1_score(y, (logits - lp).argmax(1), average="macro")
    f_plus = f1_score(y, (logits + lp).argmax(1), average="macro")
    f_none = f1_score(y, logits.argmax(1), average="macro")
    assert f_minus > f_none > f_plus, (
        f"logit-adjustment sign self-test failed: minus={f_minus:.3f} "
        f"none={f_none:.3f} plus={f_plus:.3f}")
    print(f"sign self-test OK: macro-F1 minus={f_minus:.3f} > none={f_none:.3f} "
          f"> plus={f_plus:.3f}  -> post-hoc must SUBTRACT tau*log(prior)")


def run():
    t0 = time.perf_counter()
    _sign_selftest()
    base_pf, base_mean, base_name = ec.baseline_for(ec.RELABEL)
    name = f"logitadj_{MODE}_tau{TAU:g}"
    print(f"=== {name} ===  mode={MODE} tau={TAU}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes")

    rng = np.random.default_rng(ec.SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if ec.CLEAN_CLS12:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        Xtr, ytr, Xte = X[tr_use], y_enc[tr_use], X[te]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)
        n = len(Xtr_s)
        perm = rng.permutation(n)
        nv = int(n * s3.VAL_FRAC)
        vi, ti = perm[:nv], perm[nv:]
        Xtr_t = torch.tensor(Xtr_s[ti], device=ec.DEVICE)
        ytr_t = torch.tensor(ytr[ti], device=ec.DEVICE)
        Xval_t = torch.tensor(Xtr_s[vi], device=ec.DEVICE)
        Xte_t = torch.tensor(Xte_s, device=ec.DEVICE)
        yval_np = ytr[vi]

        # class priors from the TRAIN split only (leak-free)
        cnt = np.bincount(ytr[ti], minlength=n_classes).astype(np.float64)
        prior = cnt / cnt.sum()
        log_prior = TAU * np.log(prior + 1e-12)
        # post-hoc SUBTRACTS the prior at inference (up-weights rare classes);
        # the adjusted LOSS ADDS it during training and predicts unshifted.
        adj_vec = (+log_prior) if MODE == "loss" else (-log_prior)
        adj = torch.tensor(adj_vec, dtype=torch.float32, device=ec.DEVICE)

        P = np.zeros((len(Xte_s), n_classes))
        vf1s = []
        for e in range(ec.N_ENSEMBLE):
            s3.set_seed(ec.SEED + 100 * e)
            model = AdjustedMLP(Xtr_s.shape[1], n_classes, ec.HIDDEN, ec.DROPOUT).to(ec.DEVICE)
            model.adj.copy_(adj)
            # logit-adjusted LOSS trains with the prior inside the softmax;
            # post-hoc trains plain and only adjusts at inference.
            model.apply_adj = (MODE == "loss")
            crit = nn.CrossEntropyLoss(label_smoothing=s3.LABEL_SMOOTH)  # NO class weights
            opt = torch.optim.Adam(model.parameters(), lr=ec.LR,
                                   weight_decay=s3.WEIGHT_DECAY)
            best_f1, best_state, bad = -1.0, None, 0
            g = torch.Generator(device=ec.DEVICE).manual_seed(ec.SEED + 100 * e)
            for epoch in range(s3.MAX_EPOCHS):
                model.train()
                order = torch.randperm(len(Xtr_t), device=ec.DEVICE, generator=g)
                for i in range(0, len(Xtr_t), s3.BATCH):
                    b = order[i:i + s3.BATCH]
                    opt.zero_grad()
                    crit(model(Xtr_t[b]), ytr_t[b]).backward()
                    opt.step()
                # Validate exactly the way we will predict (see PREDICT below):
                #   post -> shift ON  (subtract prior at inference)
                #   loss -> shift OFF (the trained margin already compensates)
                model.eval()
                was = model.apply_adj
                model.apply_adj = (MODE == "post")
                with torch.no_grad():
                    vp = model(Xval_t).argmax(1).cpu().numpy()
                model.apply_adj = was
                f1 = du.macro_f1(yval_np, vp, n_classes)
                if f1 > best_f1 + 1e-4:
                    best_f1, bad = f1, 0
                    best_state = {kk: v.detach().cpu().clone()
                                  for kk, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= s3.PATIENCE:
                        break
            model.load_state_dict(best_state)
            # PREDICT: post-hoc applies the -log(prior) shift now; the adjusted
            # loss already baked the correction into the weights, so it predicts
            # with plain logits (shifting again would double-count the prior).
            model.apply_adj = (MODE == "post")
            P += ec.probs(model, Xte_t, n_classes)
            vf1s.append(best_f1)
            del model
            torch.cuda.empty_cache()

        pred = (P / ec.N_ENSEMBLE).argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  Δ={f1-base_pf[k]:+.4f}  "
              f"(val={np.mean(vf1s):.4f}, {time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    deltas = [f1s[k] - base_pf[k] for k in range(len(f1s))]
    dmean = float(np.mean(deltas))
    all_pos = all(d > 0 for d in deltas)
    verdict = "WIN" if (all_pos and dmean > 0.003) else ("tie" if abs(dmean) <= 0.003 else "LOSS")
    pc_mean = np.mean(pcs, 0)
    print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}  Δ_mean={dmean:+.4f} "
          f"per_fold_Δ={[round(d,4) for d in deltas]}")
    print(f"VERDICT: {verdict}")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")

    rec = {"name": name,
           "notes": f"logit adjustment ({MODE}) tau={TAU}, replaces sqrt class weights",
           "relabel": ec.RELABEL, "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
           "f1_per_fold": [round(v, 4) for v in f1s], "baseline": base_name,
           "baseline_per_fold": base_pf, "delta_per_fold": [round(d, 4) for d in deltas],
           "delta_mean": round(dmean, 4), "all_folds_positive": all_pos,
           "verdict": verdict,
           "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc_mean)},
           "n_ensemble": ec.N_ENSEMBLE, "runtime_s": round(time.perf_counter() - t0, 1),
           "config": {"idea": "long_tail", "mode": MODE, "tau": TAU}}
    (ec.RESULTS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"saved -> {ec.RESULTS_DIR / f'{name}.json'}")


if __name__ == "__main__":
    run()

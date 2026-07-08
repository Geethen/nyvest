"""Investigation 2 — train on ALL pseudo-labels, no uncertainty removal.

Stage 4 only added unstable rows whose CC-APS prediction set was a SINGLETON
(the confident ones) and it was a no-op (0.7320 vs 0.7318). The question here is
the opposite: what if we keep EVERY unstable row, uncertain ones included, each
tagged with the ensemble's argmax pseudo-label? Two competing effects:
  + more data / more coverage of the unstable (transition/edge) manifold
  - the uncertain rows carry wrong labels, injecting noise

We run three arms head-to-head, identical base learner (headline recipe), same
leak-free spatial 3-fold CV, test labels never touched:

  none       : stable-train only (== headline, the control)
  singleton  : stable-train + CC-APS singleton unstable rows (== stage 4)
  all        : stable-train + ALL unstable rows, argmax pseudo-label, NO filter

For `all` we optionally down-weight the pseudo rows (PSEUDO_WEIGHT) — default 1.0
so it's a clean "no removal, no discount" test; set <1 to soften uncertain rows.
We log pseudo-label accuracy against the unstable rows' own (noisy) grunnkart
labels for each arm so we can see how much wrong-label noise `all` really adds.

Run: ~/myprojects/recover/.venv/bin/python DNN/pseudo_all.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
import conformal_utils as cu     # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
N_ENSEMBLE = 5
ALPHA = float(os.environ.get("ALPHA", "0.05"))
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.15"))
PSEUDO_WEIGHT = float(os.environ.get("PSEUDO_WEIGHT", "1.0"))
OUT_JSON = Path(__file__).resolve().parent / "pseudo_all.json"


def train_ensemble(Xfit, yfit, n_classes, seed0, sample_w=None):
    """5-seed MLP ensemble on (Xfit,yfit). Optional per-row sample_w scales the
    loss (used to down-weight pseudo rows). Returns (models, scaler)."""
    scaler = StandardScaler().fit(Xfit)
    Xs = scaler.transform(Xfit).astype(np.float32)
    rng = np.random.default_rng(seed0)
    perm = rng.permutation(len(Xs))
    n_val = int(len(Xs) * s3.VAL_FRAC)
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr_t = torch.tensor(Xs[ti], device=DEVICE)
    ytr_t = torch.tensor(yfit[ti], device=DEVICE)
    Xval_t = torch.tensor(Xs[vi], device=DEVICE)
    w = s3.class_weights(yfit[ti], n_classes, s3.WEIGHT_MODE)
    sw_t = None if sample_w is None else torch.tensor(
        sample_w[ti].astype(np.float32), device=DEVICE)
    models = []
    for e in range(N_ENSEMBLE):
        if sw_t is None:
            m, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, yfit[vi], Xs.shape[1],
                                n_classes, w, seed0 + 100 * e)
        else:
            m = train_one_weighted(Xtr_t, ytr_t, sw_t, Xval_t, yfit[vi],
                                   Xs.shape[1], n_classes, w, seed0 + 100 * e)
        models.append(m)
    return models, scaler


def train_one_weighted(Xtr_t, ytr_t, sw_t, Xval_t, yval_np, in_dim, n_classes,
                       w, seed):
    """Copy of s3.train_one but with per-sample loss weights (reduction='none').
    Only used when PSEUDO_WEIGHT != 1.0."""
    import torch.nn as nn
    s3.set_seed(seed)
    model = s3.MLP(in_dim, n_classes, s3.HIDDEN, s3.DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=s3.LABEL_SMOOTH,
                               reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=s3.LR,
                           weight_decay=s3.WEIGHT_DECAY)
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(s3.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, s3.BATCH):
            b = order[i:i + s3.BATCH]
            opt.zero_grad()
            loss = (crit(model(Xtr_t[b]), ytr_t[b]) * sw_t[b]).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t).argmax(1).cpu().numpy()
        f1 = du.macro_f1(yval_np, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= s3.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def ensemble_probs(models, scaler, X, n_classes):
    Xt = torch.tensor(scaler.transform(X).astype(np.float32), device=DEVICE)
    P = np.zeros((len(X), n_classes), dtype=np.float64)
    for m in models:
        P += s3.softmax_probs(m, Xt, n_classes)
    return P / len(models)


def run_arm(arm, X, y_enc, groups, df, classes, cls12_enc, lon, lat,
            Xu, yu_true, rng):
    """Return (f1_mean, f1_std, per_class_mean, stats) for one pseudo arm.

    arm: none      -> stable-train only (headline control)
         singleton -> + CC-APS singleton unstable rows (stage 4)
         all       -> + ALL unstable rows, argmax pseudo-label, no filter
    """
    n_classes = len(classes)
    f1s, pcs, stats = [], [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                         cls12_enc, lon[tr], lat[tr])
        tr_use = tr[keep]
        Xtr, ytr = X[tr_use], y_enc[tr_use]
        st = {"fold": k}
        sample_w = None

        if arm == "none":
            X_aug, y_aug = Xtr, ytr
        else:
            p = rng.permutation(len(Xtr))
            n_cal = int(len(Xtr) * CAL_FRAC)
            cal_i, fit_i = p[:n_cal], p[n_cal:]
            g1, sc1 = train_ensemble(Xtr[fit_i], ytr[fit_i], n_classes, SEED)
            P_cc = ensemble_probs(g1, sc1, Xtr[cal_i], n_classes)
            P_u = ensemble_probs(g1, sc1, Xu, n_classes)
            del g1
            torch.cuda.empty_cache()
            pseudo_all = P_u.argmax(1)
            if arm == "singleton":
                keep_mask, pseudo, tau, pct = cu.pseudo_label_singletons(
                    P_u, P_cc, ytr[cal_i], ALPHA, SEED + k)
                Xp, yp = Xu[keep_mask], pseudo[keep_mask]
                st["n_pseudo"] = int(keep_mask.sum())
                st["pct_kept"] = round(100 * float(pct), 1)
                st["pseudo_acc"] = round(float(
                    (yp == yu_true[keep_mask]).mean()) if keep_mask.sum() else float("nan"), 4)
            else:  # "all"
                Xp, yp = Xu, pseudo_all
                st["n_pseudo"] = int(len(Xu))
                st["pct_kept"] = 100.0
                st["pseudo_acc"] = round(float((pseudo_all == yu_true).mean()), 4)
            X_aug = np.concatenate([Xtr, Xp])
            y_aug = np.concatenate([ytr, yp])
            if PSEUDO_WEIGHT != 1.0:
                sample_w = np.concatenate([
                    np.ones(len(Xtr)), np.full(len(Xp), PSEUDO_WEIGHT)])

        g2, sc2 = train_ensemble(X_aug, y_aug, n_classes, SEED, sample_w=sample_w)
        P_te = ensemble_probs(g2, sc2, X[te], n_classes)
        pred = P_te.argmax(1)
        del g2
        torch.cuda.empty_cache()
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        st["f1"] = round(f1, 4)
        stats.append(st)
        print(f"  [{arm}] fold {k}: F1={f1:.4f}  "
              + (f"pseudo={st.get('n_pseudo')} acc={st.get('pseudo_acc')}"
                 if arm != "none" else ""))
    return (float(np.mean(f1s)), float(np.std(f1s)),
            np.mean(pcs, axis=0), stats)


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  ensemble={N_ENSEMBLE}  alpha={ALPHA}  "
          f"cal_frac={CAL_FRAC}  pseudo_weight={PSEUDO_WEIGHT}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    uns = du.load_unstable(classes, extra_features="lidar")
    Xu, yu_true = uns["X"], uns["y_enc"]
    print(f"stable {X.shape[0]:,}  unstable {Xu.shape[0]:,}")

    results = {}
    for arm in ["none", "singleton", "all"]:
        rng = np.random.default_rng(SEED)
        f1m, f1std, pc, stats = run_arm(
            arm, X, y_enc, groups, df, classes, cls12_enc, lon, lat,
            Xu, yu_true, rng)
        results[arm] = {
            "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
            "per_class_f1": {str(c): round(float(v), 4)
                             for c, v in zip(classes, pc)},
            "per_fold_stats": stats,
        }
        print(f"=== arm={arm}: F1={f1m:.4f} ± {f1std:.4f}  "
              f"{time.perf_counter()-t0:.0f}s ===\n")

    base = results["none"]["f1_mean"]
    for arm in ["singleton", "all"]:
        results[arm]["delta_vs_none"] = round(results[arm]["f1_mean"] - base, 4)
    print("Summary (macro-F1):")
    for arm in ["none", "singleton", "all"]:
        d = f"  Δ={results[arm].get('delta_vs_none'):+.4f}" if arm != "none" else ""
        print(f"  {arm:>10}: {results[arm]['f1_mean']:.4f}{d}")

    OUT_JSON.write_text(json.dumps({
        "investigation": "pseudo_all",
        "arms": ["none (headline)", "singleton (stage4 CC-APS)",
                 "all (no removal, argmax)"],
        "alpha": ALPHA, "cal_frac": CAL_FRAC, "pseudo_weight": PSEUDO_WEIGHT,
        "n_unstable": int(len(Xu)),
        "results": results,
    }, indent=2))
    print(f"saved -> {OUT_JSON}  total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()

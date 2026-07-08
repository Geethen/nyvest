"""Does adding ALL pseudo-labels improve TEMPORAL robustness? (LLTO x pseudo)

Two prior results, never crossed:
  - pseudo_all.py: adding all 16.8k unstable rows (argmax, no filter) is a no-op
    under SPATIAL 3-fold CV (+0.0003).
  - temporal_llto.py: the model loses only -0.009 under leave-year-and-block-out,
    almost all of it class-12 snow/ice.

But the unstable pool is exactly the transition/edge pixels, which could plausibly
help generalize to an UNSEEN YEAR even if it doesn't help spatially. This script
runs the LLTO protocol with three pseudo arms so the comparison is apples-to-apples
on the hard temporal hold-out:

  none      : stable-train only (== temporal_llto.py control)
  singleton : + CC-APS singleton unstable rows (confident only)
  all       : + ALL unstable rows, argmax pseudo-label, NO uncertainty removal

LEAK CONTROL (critical): the unstable parquet has a `year` column. For held-out
year Y we DROP every unstable row whose year == Y, so pseudo-labelled data from
the test year never enters training. Spatial block leakage is already handled by
training only on cell_id blocks != k -- but pseudo rows come from lon/lat we don't
map to blocks, so to stay strictly leak-free we ALSO gate the unstable rows to the
train years only (year != Y). (Unstable lon/lat are disjoint from stable cell grid
by construction, but excluding the test year is the load-bearing guard.)

Per (year Y, spatial block k):
  test  = stable rows in (year==Y AND block==k)
  train = stable rows in (year!=Y AND block!=k)  [+ pseudo from unstable year!=Y]
  gate (for singleton): trained on a fit-slice of train, calibrated on the rest.

Base learner = headline recipe (5-seed MLP ensemble, sqrt weights, label smooth,
train-only cls12 centroid clean). Test labels never touched.

Run: ~/myprojects/recover/.venv/bin/python DNN/temporal_pseudo.py
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
import data_utils as du            # noqa: E402
import conformal_utils as cu       # noqa: E402
import stage3_robust_mlp as s3     # noqa: E402
from dnn_paths import result_path   # noqa: E402
import temporal_llto as tl         # noqa: E402  (spatial_blocks, clean_train)

DEVICE = s3.DEVICE
SEED = 0
N_ENSEMBLE = 5
N_SPATIAL = du.N_FOLDS
ALPHA = float(os.environ.get("ALPHA", "0.05"))
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.15"))
ARMS = os.environ.get("ARMS", "none,all").split(",")   # default: control vs all
OUT_JSON = result_path("temporal_pseudo.json")


def train_ensemble(Xfit, yfit, n_classes, seed0):
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
    models = []
    for e in range(N_ENSEMBLE):
        m, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, yfit[vi], Xs.shape[1],
                            n_classes, w, seed0 + 100 * e)
        models.append(m)
    return models, scaler


def ensemble_probs(models, scaler, X, n_classes):
    Xt = torch.tensor(scaler.transform(X).astype(np.float32), device=DEVICE)
    P = np.zeros((len(X), n_classes), dtype=np.float64)
    for m in models:
        P += s3.softmax_probs(m, Xt, n_classes)
    return P / len(models)


def build_train(arm, Xtr, ytr, Xu, yu_true, n_classes, rng, seed_k):
    """Return (X_aug, y_aug, stat) for one arm. Xu/yu_true already restricted to
    the allowed (train-year) unstable rows."""
    if arm == "none":
        return Xtr, ytr, {"n_pseudo": 0}
    # gate ensemble on a fit-slice; calibrate on the rest of the stable train
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
            P_u, P_cc, ytr[cal_i], ALPHA, seed_k)
        Xp, yp = Xu[keep_mask], pseudo[keep_mask]
        acc = float((yp == yu_true[keep_mask]).mean()) if keep_mask.sum() else float("nan")
        stat = {"n_pseudo": int(keep_mask.sum()), "pseudo_acc": round(acc, 4)}
    else:  # all
        Xp, yp = Xu, pseudo_all
        stat = {"n_pseudo": int(len(Xu)),
                "pseudo_acc": round(float((pseudo_all == yu_true).mean()), 4)}
    return (np.concatenate([Xtr, Xp]), np.concatenate([ytr, yp]), stat)


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  arms={ARMS}  alpha={ALPHA}  ensemble={N_ENSEMBLE}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    years = df["year"].values.astype(int)
    uyears = sorted(np.unique(years).tolist())
    blocks = tl.spatial_blocks(groups)

    uns = du.load_unstable(classes, extra_features="lidar")
    Xu_all, yu_all = uns["X"], uns["y_enc"]
    uyear = uns["df"]["year"].values.astype(int)
    print(f"stable {X.shape[0]:,}  unstable {Xu_all.shape[0]:,}  years={uyears}")

    results = {}
    for arm in ARMS:
        rng = np.random.default_rng(SEED)
        per_year = {}
        for yr in uyears:
            # unstable rows from TRAIN years only (drop the held-out year)
            um = uyear != yr
            Xu, yu = Xu_all[um], yu_all[um]
            f1s, pcs = [], []
            for k in range(N_SPATIAL):
                te = np.flatnonzero((years == yr) & (blocks == k))
                tr = np.flatnonzero((years != yr) & (blocks != k))
                tr = tl.clean_train(X, y_enc, df, tr, cls12_enc, lon, lat)
                Xtr, ytr = X[tr], y_enc[tr]
                X_aug, y_aug, stat = build_train(
                    arm, Xtr, ytr, Xu, yu, n_classes, rng, SEED + k)
                models, sc = train_ensemble(X_aug, y_aug, n_classes, SEED)
                pred = ensemble_probs(models, sc, X[te], n_classes).argmax(1)
                del models
                torch.cuda.empty_cache()
                f1s.append(du.macro_f1(y_enc[te], pred, n_classes))
                pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
            f1m = float(np.mean(f1s))
            pc = np.mean(pcs, axis=0)
            per_year[str(yr)] = {
                "macro_f1": round(f1m, 4),
                "macro_f1_std": round(float(np.std(f1s)), 4),
                "n_pseudo": stat["n_pseudo"],
                "pseudo_acc": stat.get("pseudo_acc"),
                "per_class_f1": {str(c): round(float(v), 4)
                                 for c, v in zip(classes, pc)},
            }
            print(f"  [{arm}] year={yr}: macroF1={f1m:.4f} "
                  f"(pseudo={stat['n_pseudo']}, acc={stat.get('pseudo_acc')})  "
                  f"{time.perf_counter()-t0:.0f}s")
        llto = float(np.mean([v["macro_f1"] for v in per_year.values()]))
        # per-class LLTO mean + cls12 spread
        pc_mean = {str(c): round(float(np.mean(
            [per_year[y]["per_class_f1"][str(c)] for y in per_year])), 4)
            for c in classes}
        cls12_vals = [per_year[y]["per_class_f1"]["12"] for y in per_year]
        results[arm] = {
            "llto_macro_f1": round(llto, 4),
            "per_class_llto_mean": pc_mean,
            "cls12_min": round(min(cls12_vals), 4),
            "cls12_max": round(max(cls12_vals), 4),
            "cls12_spread": round(max(cls12_vals) - min(cls12_vals), 4),
            "per_year": per_year,
        }
        print(f"=== arm={arm}: LLTO macroF1={llto:.4f}  "
              f"cls12 [{min(cls12_vals):.3f},{max(cls12_vals):.3f}]  "
              f"{time.perf_counter()-t0:.0f}s ===\n")

    if "none" in results:
        base = results["none"]["llto_macro_f1"]
        for arm in results:
            results[arm]["delta_vs_none"] = round(
                results[arm]["llto_macro_f1"] - base, 4)

    print("Summary — LLTO macro-F1 (does pseudo help temporal robustness?):")
    for arm in ARMS:
        d = results[arm].get("delta_vs_none")
        ds = f"  Δ={d:+.4f}" if d is not None else ""
        print(f"  {arm:>10}: {results[arm]['llto_macro_f1']:.4f}{ds}  "
              f"(cls12 spread {results[arm]['cls12_spread']:.3f})")

    OUT_JSON.write_text(json.dumps({
        "investigation": "temporal_pseudo",
        "question": "does adding all pseudo-labels improve temporal (LLTO) robustness?",
        "scheme": "leave-year-and-block-out x pseudo-label arm",
        "leak_control": "unstable rows from held-out year dropped",
        "arms": ARMS, "alpha": ALPHA, "cal_frac": CAL_FRAC,
        "n_ensemble": N_ENSEMBLE, "results": results,
    }, indent=2))
    print(f"saved -> {OUT_JSON}  total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()

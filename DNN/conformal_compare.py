"""Investigation 3 — compare conformal / uncertainty-quantification methods.

The pipeline uses APS (Adaptive Prediction Sets) both for pseudo-labelling and as
its notion of predictive uncertainty. This benchmarks alternative conformal score
functions and predictors on the headline DNN ensemble, all leak-free and at the
same target coverage 1-alpha, so we can pick the best UQ for downstream use
(active learning, pseudo-labelling, flagging low-confidence pixels).

Methods (self-contained in conformal_methods.py; mirror TorchCP's catalogue):
  score functions : LAC, Margin, APS, RAPS, SAPS
  predictors      : split (global tau, marginal coverage)
                    classcond / Mondrian (per-class tau, class-conditional cov)
  point UQ        : temperature scaling (ECE before/after) + entropy/max-prob

Protocol (leak-free): for each spatial fold k, train the headline 5-seed ensemble
on the OTHER folds; predict probs on fold k. Split fold k into a calibration half
and an evaluation half (random, seeded) — calibrate tau on the cal half, score
sets on the eval half. Aggregate metrics over the 3 folds.

We report, per method: marginal coverage (should ~= 1-alpha), average set size
(smaller = more informative at fixed coverage), singleton rate, and per-class
coverage spread (max-min) — the key weakness APS has on imbalanced classes and
the thing Mondrian is meant to fix. Point-UQ block reports ECE reduction from
temperature scaling.

Run: ALPHA=0.1 ~/myprojects/recover/.venv/bin/python DNN/conformal_compare.py
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
import conformal_methods as cm     # noqa: E402
import stage3_robust_mlp as s3     # noqa: E402
from dnn_paths import result_path   # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
N_ENSEMBLE = 5
ALPHA = float(os.environ.get("ALPHA", "0.1"))     # target miscoverage (90% cov)
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.5"))
OUT_JSON = result_path("conformal_compare.json")


def train_and_prob(Xtr, ytr, Xte, n_classes):
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xtr))
    nv = int(len(Xtr) * s3.VAL_FRAC)
    vi, ti = perm[:nv], perm[nv:]
    Xtr_t = torch.tensor(Xtr[ti], device=DEVICE)
    ytr_t = torch.tensor(ytr[ti], device=DEVICE)
    Xval_t = torch.tensor(Xtr[vi], device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    w = s3.class_weights(ytr[ti], n_classes, s3.WEIGHT_MODE)
    P = np.zeros((len(Xte), n_classes))
    for e in range(N_ENSEMBLE):
        m, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, ytr[vi], Xtr.shape[1],
                            n_classes, w, SEED + 100 * e)
        P += s3.softmax_probs(m, Xte_t, n_classes)
        del m
        torch.cuda.empty_cache()
    return P / N_ENSEMBLE


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    print(f"device={DEVICE}  alpha={ALPHA} (target cov {1-ALPHA:.0%})  "
          f"cal_frac={CAL_FRAC}  ensemble={N_ENSEMBLE}")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values

    # collect per-fold {method: metrics} then average
    fold_metrics = []          # list over folds of {method_key: metrics}
    temp_stats = []            # temperature-scaling stats per fold
    accs = []
    for k, tr, te in du.fold_indices(y_enc, groups):
        keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                         cls12_enc, lon[tr], lat[tr])
        P = train_and_prob(X[tr[keep]], y_enc[tr[keep]], X[te], n_classes)
        yte = y_enc[te]
        accs.append(float((P.argmax(1) == yte).mean()))

        # split the test fold into cal / eval halves (leak-free: model never saw te)
        rng = np.random.default_rng(SEED + k)
        perm = rng.permutation(len(te))
        n_cal = int(len(te) * CAL_FRAC)
        cal_i, ev_i = perm[:n_cal], perm[n_cal:]
        P_cal, y_cal = P[cal_i], yte[cal_i]
        P_ev, y_ev = P[ev_i], yte[ev_i]

        # independent uniform draws for randomized scores
        u_cal = rng.uniform(size=len(P_cal))
        u_ev = rng.uniform(size=len(P_ev))

        m_fold = {}
        for sname, sfn in cm.SCORES.items():
            args_cal = (P_cal, u_cal) if sname in cm.RANDOMIZED else (P_cal,)
            args_ev = (P_ev, u_ev) if sname in cm.RANDOMIZED else (P_ev,)
            S_cal = sfn(*args_cal)
            S_ev = sfn(*args_ev)
            # split predictor
            tau = cm.calibrate_split(S_cal, y_cal, ALPHA)
            m_fold[f"{sname}+split"] = cm.set_metrics(
                cm.sets_from_tau(S_ev, tau), y_ev, n_classes)
            # class-conditional (Mondrian) predictor
            taus = cm.calibrate_classcond(S_cal, y_cal, ALPHA, n_classes)
            m_fold[f"{sname}+mondrian"] = cm.set_metrics(
                cm.sets_from_tau(S_ev, taus), y_ev, n_classes)
        fold_metrics.append(m_fold)

        # point UQ: temperature scaling on cal, eval on ev
        T, P_ev_ts = cm.temperature_scale(P_cal, y_cal, P_ev)
        temp_stats.append({
            "fold": k, "T": round(float(T), 3),
            "ece_before": round(cm.ece(P_ev, y_ev), 4),
            "ece_after": round(cm.ece(P_ev_ts, y_ev), 4),
            "mean_entropy": round(float(cm.entropy(P_ev).mean()), 4),
        })
        print(f"  fold {k}: acc={accs[-1]:.4f}  T={T:.2f}  "
              f"ECE {cm.ece(P_ev, y_ev):.4f}->{cm.ece(P_ev_ts, y_ev):.4f}  "
              f"{time.perf_counter()-t0:.0f}s")

    # ---- aggregate over folds ----
    method_keys = list(fold_metrics[0].keys())
    agg = {}
    for mk in method_keys:
        cov = np.mean([fm[mk]["coverage"] for fm in fold_metrics])
        size = np.mean([fm[mk]["avg_set_size"] for fm in fold_metrics])
        single = np.mean([fm[mk]["singleton_rate"] for fm in fold_metrics])
        empty = np.mean([fm[mk]["empty_rate"] for fm in fold_metrics])
        # per-class coverage spread: avg over folds of (max-min over classes)
        spreads = []
        for fm in fold_metrics:
            pcc = [v for v in fm[mk]["per_class_coverage"].values() if v is not None]
            spreads.append(max(pcc) - min(pcc))
        agg[mk] = {
            "coverage": round(float(cov), 4),
            "avg_set_size": round(float(size), 3),
            "singleton_rate": round(float(single), 4),
            "empty_rate": round(float(empty), 4),
            "per_class_cov_spread": round(float(np.mean(spreads)), 4),
        }

    print(f"\n=== Conformal method comparison (alpha={ALPHA}, target cov "
          f"{1-ALPHA:.0%}) ===")
    print(f"{'method':<18}{'cov':>7}{'set_size':>10}{'single':>9}"
          f"{'empty':>8}{'cls_cov_spread':>16}")
    for mk in method_keys:
        a = agg[mk]
        print(f"{mk:<18}{a['coverage']:>7.3f}{a['avg_set_size']:>10.2f}"
              f"{a['singleton_rate']:>9.3f}{a['empty_rate']:>8.3f}"
              f"{a['per_class_cov_spread']:>16.3f}")
    ece_b = np.mean([t["ece_before"] for t in temp_stats])
    ece_a = np.mean([t["ece_after"] for t in temp_stats])
    print(f"\ntemperature scaling: mean T={np.mean([t['T'] for t in temp_stats]):.2f}  "
          f"ECE {ece_b:.4f} -> {ece_a:.4f}  (mean acc {np.mean(accs):.4f})")

    OUT_JSON.write_text(json.dumps({
        "investigation": "conformal_compare",
        "alpha": ALPHA, "target_coverage": round(1 - ALPHA, 3),
        "cal_frac": CAL_FRAC, "n_ensemble": N_ENSEMBLE,
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "methods": agg,
        "temperature_scaling": {
            "per_fold": temp_stats,
            "ece_before_mean": round(float(ece_b), 4),
            "ece_after_mean": round(float(ece_a), 4),
        },
        "note": ("scores LAC/Margin/APS/RAPS/SAPS x predictors split/mondrian; "
                 "leak-free: model trained on other spatial folds, cal/eval split "
                 "within the held-out fold"),
    }, indent=2))
    print(f"saved -> {OUT_JSON}  total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()

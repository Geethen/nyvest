"""Diagnostic: WHY is the seed-ensemble the only lever that works, and is it
saturated?

Not a leaderboard entry — this explains the mechanism instead of ranking another
config. Every architecture/objective idea has tied, while 1-seed -> 5-seed buys
+0.007. Two candidate explanations:

  (a) variance reduction — members make INDEPENDENT errors and averaging cancels
      them. Then gains follow the classic ~1/sqrt(M) curve and saturate.
  (b) the members are nearly identical and the gain is a fluke of the metric.

We measure it directly on ONE fold:
  - macro-F1 as a function of ensemble size M = 1..15 (with error bars over
    which subset of members is used) -> shows the saturation curve.
  - pairwise DISAGREEMENT between members (fraction of rows where argmax
    differs) -> the diversity that averaging exploits.
  - per-class disagreement -> whether the hard classes are where members differ.

Runs on fold 0 only (the diagnostic is about the mechanism, not the headline
number), 15 members, ~5 min.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec          # noqa: E402
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

N_MEMBERS = 15
OUT = ec.RESULTS_DIR / "diag_ensemble.json"


def main():
    t0 = time.perf_counter()
    print("=== ensemble diagnostic (fold 0, 15 members) ===")
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values

    k, tr, te = next(iter(du.fold_indices(y_enc, groups)))
    keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr], cls12_enc,
                                     lon[tr], lat[tr])
    tr = tr[keep]
    scaler = StandardScaler().fit(X[tr])
    Xtr_s = scaler.transform(X[tr]).astype(np.float32)
    Xte_s = scaler.transform(X[te]).astype(np.float32)
    rng = np.random.default_rng(ec.SEED)
    n = len(Xtr_s)
    perm = rng.permutation(n)
    nv = int(n * s3.VAL_FRAC)
    vi, ti = perm[:nv], perm[nv:]
    Xtr_t = torch.tensor(Xtr_s[ti], device=ec.DEVICE)
    ytr_t = torch.tensor(y_enc[tr][ti], device=ec.DEVICE)
    Xval_t = torch.tensor(Xtr_s[vi], device=ec.DEVICE)
    Xte_t = torch.tensor(Xte_s, device=ec.DEVICE)
    yval = y_enc[tr][vi]
    yte = y_enc[te]
    w = s3.class_weights(y_enc[tr][ti], n_classes, s3.WEIGHT_MODE)

    # train N members, keep each one's test probabilities
    Ps = []
    for e in range(N_MEMBERS):
        model, vf1 = ec.train_one(lambda i, c: s3.MLP(i, c, ec.HIDDEN, ec.DROPOUT),
                                  Xtr_t, ytr_t, Xval_t, yval, Xtr_s.shape[1],
                                  n_classes, w, ec.SEED + 100 * e)
        Ps.append(ec.probs(model, Xte_t, n_classes))
        del model
        torch.cuda.empty_cache()
        print(f"  member {e}: val_f1={vf1:.4f}")
    Ps = np.stack(Ps)                      # [M, N, C]
    preds = Ps.argmax(2)                   # [M, N]

    # --- saturation curve: F1 vs M, averaged over random member subsets
    curve = {}
    for M in [1, 2, 3, 5, 7, 10, 15]:
        combos = list(itertools.combinations(range(N_MEMBERS), M))
        rs = np.random.default_rng(0)
        pick = combos if len(combos) <= 8 else [combos[i] for i in
                                                rs.choice(len(combos), 8, replace=False)]
        f1s = [du.macro_f1(yte, Ps[list(c)].mean(0).argmax(1), n_classes) for c in pick]
        curve[M] = {"mean": round(float(np.mean(f1s)), 4),
                    "std": round(float(np.std(f1s)), 4)}
        print(f"  M={M:2d}: F1={curve[M]['mean']:.4f} ±{curve[M]['std']:.4f}")

    # --- pairwise disagreement
    dis = []
    for a, b in itertools.combinations(range(N_MEMBERS), 2):
        dis.append(float((preds[a] != preds[b]).mean()))
    print(f"\npairwise disagreement: mean={np.mean(dis):.4f} "
          f"min={np.min(dis):.4f} max={np.max(dis):.4f}")

    # --- per-class disagreement (where does diversity live?)
    per_class = {}
    for ci, c in enumerate(classes):
        m = yte == ci
        if not m.any():
            continue
        d = [float((preds[a][m] != preds[b][m]).mean())
             for a, b in itertools.combinations(range(N_MEMBERS), 2)]
        per_class[str(c)] = round(float(np.mean(d)), 4)
    print("per-class mean disagreement:")
    for c, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  class {c:>2}: {v:.4f}")

    # --- oracle: best achievable if we could pick the right member per row
    oracle = float((preds == yte[None, :]).any(0).mean())
    single_acc = float((preds[0] == yte).mean())
    ens_acc = float((Ps.mean(0).argmax(1) == yte).mean())
    print(f"\naccuracy: single={single_acc:.4f}  ensemble={ens_acc:.4f}  "
          f"ORACLE(any member right)={oracle:.4f}")
    print(f"=> headroom if routing were perfect: {oracle-ens_acc:+.4f}")

    OUT.write_text(json.dumps({
        "name": "diag_ensemble", "fold": 0, "n_members": N_MEMBERS,
        "f1_vs_M": curve, "pairwise_disagreement": {
            "mean": round(float(np.mean(dis)), 4),
            "min": round(float(np.min(dis)), 4),
            "max": round(float(np.max(dis)), 4)},
        "per_class_disagreement": per_class,
        "single_acc": round(single_acc, 4), "ensemble_acc": round(ens_acc, 4),
        "oracle_acc": round(oracle, 4),
        "runtime_s": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()

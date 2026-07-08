"""Per-class label-noise diagnostics (leak-free).

Three cheap tests, all reusing the existing leak-free per-fold cleanlab artifact
(common_ground/reports/research/clean_labels_perfold.npz) + one DNN OOS pass.
The 4th test (clean-vs-noisy F1 gain) is a training run in noise_clean_gain.py.

TEST A — Noise transition matrix (instant, no training):
  test_inv_noise[k] is the per-fold inverse noise matrix estimated on TEST rows
  only (cleanlab estimate_latent). inv_noise[i,j] = P(true=i | observed=j). So
  the diagonal P(true=c|observed=c) is the per-class label TRUSTWORTHINESS;
  1 - diagonal = estimated noise rate of that class's labels. The off-diagonal
  argmax says which class the noise most often really is.

TEST B — Confident-disagreement rate (cheap, one OOS pass):
  fraction of each class's rows where the DNN ensemble's top prediction (with
  prob >= THRESH) disagrees with the given label. High rate = labels likely wrong
  OR systematic confusion. Combined with A it separates the two.

TEST C — Noise-corrected F1 ceiling (instant):
  using the transition matrix, estimate how much of a class's low F1 is just
  noisy SCORING (a correct pred marked wrong because the test label is noise) vs
  genuine error. We report the per-class estimated label-noise rate alongside the
  observed F1 so you can see which low-F1 classes are noise-suspect.

Run: ~/myprojects/recover/.venv/bin/python DNN/noise_diagnostics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
THRESH = 0.70  # "confident" prob threshold for disagreement test
PERFOLD_NPZ = (Path(__file__).resolve().parents[1] / "common_ground" /
               "reports" / "research" / "clean_labels_perfold.npz")
NAMES = {2: "sparse-veg", 3: "forest", 4: "forest", 5: "GRASSLAND",
         6: "open-upland", 7: "mire/wet", 8: "water", 10: "bare",
         11: "built/infra", 12: "snow/ice"}


def main():
    s3.set_seed(SEED)
    t0 = time.perf_counter()
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    z = np.load(PERFOLD_NPZ)
    inv_noise = z["test_inv_noise"]          # (3, 10, 10), P(true=i | obs=j)
    folds = z["folds"]

    # ---- TEST A + C: transition matrix -> per-class noise rate ----
    # diag of inv_noise[:, c, c] = P(true=c | observed=c); mean over folds.
    diag = np.array([[inv_noise[k][c, c] for c in range(n_classes)]
                     for k in range(inv_noise.shape[0])])
    trust = diag.mean(0)              # label trustworthiness per class
    noise_rate = 1.0 - trust         # estimated label-noise rate per class
    # where does the noise go: for observed=c, the most likely TRUE class != c
    off = inv_noise.mean(0).copy()   # P(true=i | obs=j) averaged
    confused_with = {}
    for c in range(n_classes):
        col = off[:, c].copy()
        col[c] = -1
        j = int(np.argmax(col))
        confused_with[c] = (classes[j], float(off[j, c]))

    # ---- TEST B: confident-disagreement from a leak-free OOS DNN pass ----
    # one MLP per fold trained on the OTHER folds; predict its own test rows ->
    # OOS probs for the whole dataset (no row scored by a model that trained on it)
    oos_prob = np.zeros((len(X), n_classes), dtype=np.float32)
    for k, tr, te in du.fold_indices(y_enc, groups):
        keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                         cls12_enc, lon[tr], lat[tr])
        tr = tr[keep]
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]).astype(np.float32)
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(Xtr))
        nv = int(len(Xtr) * s3.VAL_FRAC)
        vi, ti = perm[:nv], perm[nv:]
        w = s3.class_weights(y_enc[tr][ti], n_classes, s3.WEIGHT_MODE)
        m, _ = s3.train_one(torch.tensor(Xtr[ti], device=DEVICE),
                            torch.tensor(y_enc[tr][ti], device=DEVICE),
                            torch.tensor(Xtr[vi], device=DEVICE),
                            y_enc[tr][vi], Xtr.shape[1], n_classes, w, SEED)
        Xte = sc.transform(X[te]).astype(np.float32)
        oos_prob[te] = s3.softmax_probs(m, torch.tensor(Xte, device=DEVICE), n_classes)
        del m
        torch.cuda.empty_cache()
    top = oos_prob.argmax(1)
    topp = oos_prob.max(1)
    confident = topp >= THRESH
    disagree = confident & (top != y_enc)
    conf_dis_rate = np.array([
        disagree[y_enc == c].sum() / max(1, (y_enc == c).sum())
        for c in range(n_classes)])
    # where the confident disagreement points instead (the model's vote)
    dis_target = {}
    for c in range(n_classes):
        m = (y_enc == c) & disagree
        if m.sum():
            tgt = np.bincount(top[m], minlength=n_classes)
            tgt[c] = 0
            dis_target[c] = (classes[int(tgt.argmax())],
                             float(tgt.max() / m.sum()))
        else:
            dis_target[c] = (None, 0.0)

    # known per-class F1 on the best MLP (from stage3_results.json)
    import json
    f1pc = json.loads((Path(__file__).resolve().parent /
                       "stage3_results.json").read_text())["f1_per_class"]

    print(f"\n=== Per-class label-noise diagnostics (leak-free) ===")
    print(f"{'class':>14} {'F1':>6} {'noise%':>7} {'confDis%':>9} "
          f"{'noise→':>16} {'modelVote→':>16}")
    print("-" * 76)
    order = sorted(range(n_classes), key=lambda i: f1pc[str(classes[i])])
    for i in order:
        c = classes[i]
        nw_cls, nw_p = confused_with[i]
        dv_cls, dv_p = dis_target[i]
        print(f"{NAMES[c]+'/'+str(c):>14} {f1pc[str(c)]:>6.3f} "
              f"{100*noise_rate[i]:>6.1f}% {100*conf_dis_rate[i]:>8.1f}% "
              f"{str(nw_cls)+f'({nw_p:.2f})':>16} "
              f"{str(dv_cls)+f'({dv_p:.2f})':>16}")
    print("\nReading: high noise% AND/OR high confDis% with BOTH arrows pointing\n"
          "to the SAME class => systematic label confusion with that class\n"
          "(fixable by relabeling). High confDis% but low estimated noise% =>\n"
          "genuine model error / feature overlap (relabel won't help).")
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""Build the aggregated confusion matrix for the best DNN (cls12 to12_fix).

Leak-free: each test fold predicted by an ensemble trained only on the other
folds (the same protocol as the model). Aggregated over all 3 folds (663,740
test rows), row-normalized to recall %. Emits an HTML <table class='cm'> block +
a one-line legend, matching the styling of schemeB_allyears_best_report.html so
it can be pasted straight into dnn_replacement_report.html.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

DEVICE = s3.DEVICE
SEED = 0
N_ENSEMBLE = 5
PERFOLD_NPZ = (Path(__file__).resolve().parents[1] / "common_ground" /
               "reports" / "research" / "clean_labels_perfold.npz")
LABELS = {2: "bare", 3: "cropland", 4: "forest", 5: "grassland", 6: "scrub",
          7: "wetland", 8: "water", 10: "settle", 11: "infra", 12: "snow/ice"}


def fit_predict(Xtr, ytr, Xte, n_classes, rng):
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
    for e in range(N_ENSEMBLE):
        m, _ = s3.train_one(Xtr_t, ytr_t, Xval_t, ytr[vi], Xtr.shape[1],
                            n_classes, w, SEED + 100 * e)
        P += s3.softmax_probs(m, Xte_t, n_classes)
        del m
        torch.cuda.empty_cache()
    return P.argmax(1)


def cell_color(pct, diag):
    """Blue scale matching the original report (rgb interpolation toward #0969da)."""
    if pct < 0.05:
        return "#ffffff", "#d0d7de"
    t = min(pct / 100.0, 1.0)
    r = int(round(249 - (249 - 18) * t))
    g = int(round(251 - (251 - 110) * t))
    b = int(round(254 - (254 - 219) * t))
    fg = "#ffffff" if t > 0.55 else "#1f2328"
    return f"rgb({r},{g},{b})", fg


def main():
    s3.set_seed(SEED)
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    c12 = classes.index(12)
    lon, lat = df["lon"].values, df["lat"].values
    z = np.load(PERFOLD_NPZ)
    remap = {c: i for i, c in enumerate(classes)}
    corrected_enc = np.vectorize(remap.get)(z["corrected"])

    y_true_all = np.empty(len(X), dtype=int)
    y_pred_all = np.empty(len(X), dtype=int)
    rng = np.random.default_rng(SEED)
    for k, tr, te in du.fold_indices(y_enc, groups):
        # winning config: targeted to12_fix relabel on the training fold
        ytr = y_enc[tr].copy()
        corr = corrected_enc[k][tr]
        touch = (y_enc[tr] == c12) | (corr == c12)
        ytr[touch] = corr[touch]
        pred = fit_predict(X[tr], ytr, X[te], n_classes, rng)
        y_true_all[te] = y_enc[te]
        y_pred_all[te] = pred
        print(f"fold {k} done")

    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes))
    rec = 100.0 * cm / cm.sum(axis=1, keepdims=True)

    # ---- HTML table ----
    head_cols = "".join(f"<th>{c}</th>" for c in classes)
    rows = []
    for i, c in enumerate(classes):
        cells = []
        for j in range(n_classes):
            pct = rec[i, j]
            bg, fg = cell_color(pct, i == j)
            cls = "diag" if i == j else ""
            txt = "·" if pct < 0.05 else f"{int(round(pct))}"
            cells.append(f"<td class='{cls}' style='background:{bg};color:{fg}'>{txt}</td>")
        rows.append(f"<tr><th class='rowlab'>{c} {LABELS[c]}</th>{''.join(cells)}</tr>")
    rowlab_first = rows[0].replace("<tr>", "<tr><th class='rowlab' rowspan='10'>True</th>", 1)
    rows[0] = rowlab_first
    html = (
        "<div class=\"cm-wrap\">\n"
        "<table class='cm'><thead><tr><th class='corner'></th><th class='corner'></th>"
        f"<th colspan='{n_classes}'>Predicted</th></tr>"
        f"<tr><th class='corner'></th><th class='corner'></th>{head_cols}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
    )
    out = Path(__file__).resolve().parent / "confusion_matrix_fragment.html"
    out.write_text(html)
    print(f"\nsaved -> {out}")
    # also dump recall diagonal + top off-diagonals for the legend
    print("recall (diag):", {classes[i]: round(rec[i, i], 0) for i in range(n_classes)})
    for i in range(n_classes):
        off = [(classes[j], round(rec[i, j])) for j in range(n_classes)
               if j != i and rec[i, j] >= 8]
        if off:
            print(f"  {classes[i]} {LABELS[classes[i]]} leaks ->", off)


if __name__ == "__main__":
    main()

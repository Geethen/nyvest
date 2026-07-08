"""Joint (parameters x data) scaling study for the nyvest DNN — done carefully to
AVOID the classic scaling-law bugs that the earlier width-only sweep (stage5/6/7)
and data-only learning curve (lc_point) each fell into.

Why the earlier "capacity/data don't help" conclusions are not clean scaling tests
----------------------------------------------------------------------------------
1. FIXED HYPERPARAMETERS ACROSS SCALES. stage5/6/7 changed width/depth but kept
   lr=1e-3, wd=1e-4, dropout=0.3, patience=15 constant. A bigger model at a small
   model's LR/regularization overfits and looks worse — that measures tuning, not
   capacity. => here we RE-TUNE LR per (width, data, source) cell on the val split.
2. CAPACITY/DATA CONFOUND. lc_point subsampled data but held the model at 256,128.
   "More data doesn't help" was measured at FIXED small capacity — exactly where a
   capacity-limited model is expected to saturate. Chinchilla's point is that data
   and params must scale TOGETHER. => here we sweep the full N x P grid.
3. THE ACTUAL EXTRA DATA WAS NEVER IN THE LOOP. The LC subsampled existing
   grunnkart; it never added the NiN pool the question is really about. => here
   every grid cell is run twice: grunnkart-only vs +NiN (leak-free, train-only).
4. EARLY-STOP INTERACTS WITH SIZE. Fixed patience stops big nets before their (later)
   optimum. => patience/epochs bumped; LR search covers the effective-LR axis.

Diagnostics: every cell records val-F1 (fit/optimization) AND test-F1
(spatial generalization). The GAP tells us which ceiling a cell hits:
  val high & test low  => spatial-generalization bound (the known wall)
  val low  & test low  => under-fit / optimization bound (fixable by tuning)
  val ~ test, both rise with N,P => genuine scaling headroom.

Protocol is otherwise IDENTICAL to the reference (data_utils): 3-fold spatial CV
on cell_id, merge 1->2/9->8, macro-F1, 64 AE + 3 lidar, leak-free test, train-only
cls12 centroid clean. Baseline to beat: DNN best 0.7341 (to12_fix), plain-clean
0.7318, TabICL 0.7139.

Run:
  PY=~/myprojects/recover/.venv/bin/python
  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
    $PY DNN/scaling_grid.py            # writes scaling_grid.json

Env knobs:
  WIDTHS   "64|256,128|512,256|1024,512,256"   pipe-separated hidden tuples
  FRACS    "0.25,0.5,1.0,2.0"  data fractions (>1.0 needs +NiN to have effect)
  LRS      "3e-4,1e-3,3e-3"    per-cell LR search set
  SOURCES  "base,nin"          which train pools to run
  SEEDS    3                   ensemble size per cell (variance control)
  FOLDS    3                   spatial CV folds
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du            # noqa: E402
import dnn_core as C               # noqa: E402
from dnn_paths import result_path   # noqa: E402
import scaling_data as sd          # noqa: E402


def parse_widths(s):
    return [tuple(int(x) for x in grp.split(",")) for grp in s.split("|")]


WIDTHS = parse_widths(os.environ.get("WIDTHS", "64|256,128|512,256|1024,512,256"))
FRACS = [float(x) for x in os.environ.get("FRACS", "0.25,0.5,1.0").split(",")]
LRS = [float(x) for x in os.environ.get("LRS", "3e-4,1e-3,3e-3").split(",")]
SOURCES = os.environ.get("SOURCES", "base,nin").split(",")
SEEDS = int(os.environ.get("SEEDS", "3"))
FOLDS = int(os.environ.get("FOLDS", "3"))
CLS12_CLEAN = os.environ.get("CLS12_CLEAN", "1") == "1"


def stratified_subsample(y, frac, rng):
    """Down-sample (<1) the train pool stratified by class. frac>=1 keeps all."""
    if frac >= 1.0:
        return np.arange(len(y))
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        n = max(1, int(round(len(ci) * frac)))
        idx.append(rng.choice(ci, size=n, replace=False))
    return np.concatenate(idx)


def n_params(hidden, in_dim, n_classes):
    d, tot = in_dim, 0
    for h in hidden:
        tot += d * h + h
        d = h
    tot += d * n_classes + n_classes
    return tot


def cell_config(hidden, lr):
    """Per-cell config. Capacity-aware training budget so big nets aren't
    early-stopped before their optimum (bug #4). Everything else is the winning
    recipe. Ensemble size and val-based early stop unchanged."""
    return C.Config(
        hidden=hidden, lr=lr,
        dropout=0.3, weight_decay=1e-4,
        max_epochs=300, patience=25,          # bumped vs the 200/15 default
        n_ensemble=SEEDS, weight_mode="sqrt", label_smooth=0.05,
        input_noise=0.0, mixup_alpha=0.0, seed=0,
    )


def fit_one_fold(Xtr, ytr, Xte, yte, hidden, n_classes, feat_cols, classes,
                 lidar_med, rng):
    """Per-cell LR search (fix for bug #1): fit the ensemble at each LR, keep the
    one with the best ENSEMBLE val-F1, report its test-F1. Returns
    (test_f1, val_f1, best_lr, per_class_test_f1)."""
    best = None
    for lr in LRS:
        cfg = cell_config(hidden, lr)
        ens = C.fit_ensemble(Xtr, ytr, n_classes, cfg, feat_cols, classes,
                             lidar_med, rng)
        vf1 = ens.val_f1
        if best is None or vf1 > best[0]:
            pred = ens.predict_proba(Xte).argmax(1)
            tf1 = du.macro_f1(yte, pred, n_classes)
            pc = du.per_class_f1(yte, pred, n_classes)
            best = (vf1, tf1, lr, pc)
    return best[1], best[0], best[2], best[3]


def _write(results, t0, done, total):
    """Write scaling_grid.json (incremental during the run, final at the end)."""
    out = result_path("scaling_grid.json")
    payload = {
        "baseline": {"dnn_best_to12fix": 0.7341, "dnn_plain": 0.7318,
                     "tabicl": 0.7139},
        "grid_axes": {"widths": [list(w) for w in WIDTHS], "fracs": FRACS,
                      "lrs": LRS, "sources": SOURCES, "seeds": SEEDS,
                      "folds": FOLDS},
        "progress": {"done": done, "total": total},
        "results": results,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def main():
    t0 = time.perf_counter()
    data = C.load_cached("lidar")
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    lon, lat = data["lon"], data["lat"]
    classes, feat_cols = data["classes"], data["feat_cols"]
    lidar_med = data.get("lidar_med")
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    in_dim = X.shape[1]

    nin = sd.load_nin(classes, feat_cols, lidar_med, "lidar")
    print(f"NiN pool: {len(nin['y_enc']):,} rows, classes "
          f"{sorted(set(classes[i] for i in np.unique(nin['y_enc'])))}", flush=True)

    folds = list(du.fold_indices(y_enc, groups, FOLDS))
    results = []
    grid = [(w, f, s) for w in WIDTHS for f in FRACS for s in SOURCES]
    print(f"grid: {len(grid)} cells x {FOLDS} folds x {SEEDS} seeds x "
          f"{len(LRS)} LRs\n", flush=True)

    for hidden, frac, source in grid:
        if source == "nin" and frac < 1.0:
            # NiN is extra data ON TOP of full grunnkart; sub-fraction+NiN is a
            # confusing mixture. Only run NiN at frac>=1.0 (full base + NiN).
            continue
        P = n_params(hidden, in_dim, n_classes)
        f1s, vf1s, lrs_used, pcs, ntrs = [], [], [], [], []
        for k, tr, te in folds:
            rng = np.random.default_rng(k)
            # train-only cls12 centroid clean (leak-free), same as reference
            if CLS12_CLEAN and cls12_enc >= 0:
                keep = du.clean_stale_class_mask(
                    X[tr], y_enc[tr], None, cls12_enc, lon[tr], lat[tr])
                tr = tr[keep]
            Xtr, ytr = X[tr], y_enc[tr]
            # data fraction (subsample the grunnkart train pool, stratified)
            if frac < 1.0:
                sub = stratified_subsample(ytr, frac, rng)
                Xtr, ytr = Xtr[sub], ytr[sub]
            # NiN augmentation (leak-free: drop NiN cells in the test fold)
            if source == "nin":
                Xtr, ytr, n_add = sd.augment_train(Xtr, ytr, groups[tr], nin,
                                                   groups[te])
            tf1, vf1, lr, pc = fit_one_fold(
                Xtr, ytr, X[te], y_enc[te], hidden, n_classes, feat_cols,
                classes, lidar_med, rng)
            f1s.append(tf1); vf1s.append(vf1); lrs_used.append(lr)
            pcs.append(pc); ntrs.append(len(ytr))
        rec = {
            "hidden": list(hidden), "params": int(P), "frac": frac,
            "source": source, "n_train_mean": int(np.mean(ntrs)),
            "test_f1": round(float(np.mean(f1s)), 4),
            "test_f1_std": round(float(np.std(f1s)), 4),
            "val_f1": round(float(np.mean(vf1s)), 4),
            "gap": round(float(np.mean(vf1s) - np.mean(f1s)), 4),
            "lrs": lrs_used,
            "per_class": {str(c): round(float(v), 4)
                          for c, v in zip(classes, np.mean(pcs, axis=0))},
        }
        results.append(rec)
        print(f"h={str(hidden):<16} P={P:>9,} frac={frac:<4} src={source:<4} "
              f"N={rec['n_train_mean']:>7,}  test={rec['test_f1']:.4f}"
              f"±{rec['test_f1_std']:.3f}  val={rec['val_f1']:.4f}  "
              f"gap={rec['gap']:+.3f}  lr~{max(set(lrs_used),key=lrs_used.count):.0e}",
              flush=True)
        # incremental write so a live artifact can track progress mid-run
        _write(results, t0, done=len(results), total=len([
            (w, f, s) for w in WIDTHS for f in FRACS for s in SOURCES
            if not (s == "nin" and f < 1.0)]))

    total = len([(w, f, s) for w in WIDTHS for f in FRACS for s in SOURCES
                 if not (s == "nin" and f < 1.0)])
    out = _write(results, t0, done=len(results), total=total)
    print(f"\nwrote {out.name}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # quick verdict
    base_best = max((r for r in results if r["source"] == "base"),
                    key=lambda r: r["test_f1"])
    overall = max(results, key=lambda r: r["test_f1"])
    print(f"\nbest base cell: {base_best['test_f1']:.4f} "
          f"h={base_best['hidden']} frac={base_best['frac']}")
    print(f"best overall  : {overall['test_f1']:.4f} h={overall['hidden']} "
          f"frac={overall['frac']} src={overall['source']}")
    print(f"vs DNN best 0.7341: {overall['test_f1']-0.7341:+.4f}")


if __name__ == "__main__":
    main()

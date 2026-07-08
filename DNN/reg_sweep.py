"""Regularization sweep — the follow-up to scaling_grid.py.

The scaling grid showed every over-capacity cell fails the SAME way: val-F1 races
to 0.86-0.92 while test-F1 stalls at ~0.72, a val-test gap that grows monotonically
with width (+0.02 at 64-wide -> +0.20 at 1024-wide). That is the signature of a
REGULARIZATION-STARVED model, not a capacity-tapped one. The grid was deliberately
run at fixed light reg (dropout 0.3, wd 1e-4) to keep the capacity axis clean, so
reg is the obvious untested lever.

CAVEAT this sweep is built to expose: the gap is SPATIAL (GroupKFold on cell_id),
i.e. geographic memorization, not ordinary sample overfitting. Dropout / weight
decay target sample memorization. So heavier reg may shrink the gap by dragging
VAL down toward test WITHOUT lifting test. => the decisive column is `test_f1`
RISING, not `gap` shrinking. We print both and a `reg_helped_test` flag.

Two fronts (per the chosen design):
  * 256,128  — the best-capacity net. Q: can reg push the PEAK past 0.7304?
  * 512,256  — an over-capacity net. Q: can reg RESCUE a big net to competitive?

Grid per net: dropout {0.3,0.5,0.7} x weight_decay {1e-4,1e-3,1e-2} = 9 cells.
Full data, base source only (NiN is a settled negative in scaling_grid). LR fixed
at 1e-3 (both widths already preferred it in the scaling grid; reg is the axis
under test). 3-seed ensemble, cls12 clean, 3-fold spatial CV — same protocol.

Baselines from scaling_grid.json (full data, base, lr~1e-3, reg=0.3/1e-4):
  256,128 -> test 0.7304 (gap +0.068)   512,256 -> test 0.7281 (gap +0.130)
Reference DNN best 0.7341 (5-seed + cls12 relabel); this sweep is 3-seed, no relabel.

Run:
  PY=~/myprojects/recover/.venv/bin/python
  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
    $PY DNN/reg_sweep.py            # writes reg_sweep.json (incremental)

Env: WIDTHS "256,128|512,256"  DROPOUTS "0.3,0.5,0.7"  WDS "1e-4,1e-3,1e-2"
     LR 1e-3  SEEDS 3  FOLDS 3
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du            # noqa: E402
import dnn_core as C               # noqa: E402
from dnn_paths import result_path   # noqa: E402


def parse_widths(s):
    return [tuple(int(x) for x in grp.split(",")) for grp in s.split("|")]


WIDTHS = parse_widths(os.environ.get("WIDTHS", "256,128|512,256"))
DROPOUTS = [float(x) for x in os.environ.get("DROPOUTS", "0.3,0.5,0.7").split(",")]
WDS = [float(x) for x in os.environ.get("WDS", "1e-4,1e-3,1e-2").split(",")]
LR = float(os.environ.get("LR", "1e-3"))
SEEDS = int(os.environ.get("SEEDS", "3"))
FOLDS = int(os.environ.get("FOLDS", "3"))

# per-(width) scaling-grid baseline at reg 0.3/1e-4, full data, base
BASELINE = {(256, 128): 0.7304, (512, 256): 0.7281}


def _write(results, t0, done, total):
    out = result_path("reg_sweep.json")
    out.write_text(json.dumps({
        "baseline": {"dnn_best_to12fix": 0.7341,
                     "scaling_256_128": 0.7304, "scaling_512_256": 0.7281},
        "grid_axes": {"widths": [list(w) for w in WIDTHS], "dropouts": DROPOUTS,
                      "wds": WDS, "lr": LR, "seeds": SEEDS, "folds": FOLDS},
        "progress": {"done": done, "total": total},
        "results": results,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=2))
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

    folds = list(du.fold_indices(y_enc, groups, FOLDS))
    grid = [(w, dp, wd) for w in WIDTHS for dp in DROPOUTS for wd in WDS]
    total = len(grid)
    print(f"reg sweep: {total} cells x {FOLDS} folds x {SEEDS} seeds "
          f"(LR={LR:.0e})\n", flush=True)

    results = []
    for hidden, dp, wd in grid:
        cfg = C.Config(hidden=hidden, lr=LR, dropout=dp, weight_decay=wd,
                       max_epochs=300, patience=25, n_ensemble=SEEDS,
                       weight_mode="sqrt", label_smooth=0.05,
                       input_noise=0.0, mixup_alpha=0.0, seed=0)
        f1s, vf1s, pcs = [], [], []
        for k, tr, te in folds:
            rng = np.random.default_rng(k)
            if cls12_enc >= 0:
                keep = du.clean_stale_class_mask(
                    X[tr], y_enc[tr], None, cls12_enc, lon[tr], lat[tr])
                tr = tr[keep]
            ens = C.fit_ensemble(X[tr], y_enc[tr], n_classes, cfg, feat_cols,
                                 classes, lidar_med, rng)
            pred = ens.predict_proba(X[te]).argmax(1)
            f1s.append(du.macro_f1(y_enc[te], pred, n_classes))
            vf1s.append(ens.val_f1)
            pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        tf1, vf1 = float(np.mean(f1s)), float(np.mean(vf1s))
        base = BASELINE.get(hidden, tf1)
        rec = {
            "hidden": list(hidden), "dropout": dp, "weight_decay": wd,
            "test_f1": round(tf1, 4), "test_f1_std": round(float(np.std(f1s)), 4),
            "val_f1": round(vf1, 4), "gap": round(vf1 - tf1, 4),
            "vs_scaling_base": round(tf1 - base, 4),
            "reg_helped_test": bool(tf1 > base + 0.001),
            "per_class": {str(c): round(float(v), 4)
                          for c, v in zip(classes, np.mean(pcs, axis=0))},
        }
        results.append(rec)
        flag = "TEST↑" if rec["reg_helped_test"] else ("test↓" if rec["vs_scaling_base"] < -0.001 else "flat")
        print(f"h={str(hidden):<12} drop={dp} wd={wd:<6.0e}  test={tf1:.4f}"
              f"±{rec['test_f1_std']:.3f}  val={vf1:.4f}  gap={rec['gap']:+.3f}  "
              f"Δbase={rec['vs_scaling_base']:+.4f} [{flag}]", flush=True)
        _write(results, t0, len(results), total)

    _write(results, t0, len(results), total)
    best = max(results, key=lambda r: r["test_f1"])
    print(f"\nbest: {best['test_f1']:.4f}  h={best['hidden']} "
          f"drop={best['dropout']} wd={best['weight_decay']:.0e} "
          f"gap={best['gap']:+.3f}", flush=True)
    print(f"vs scaling-grid best 0.7304: {best['test_f1']-0.7304:+.4f}   "
          f"vs DNN best 0.7341: {best['test_f1']-0.7341:+.4f}", flush=True)
    helped = [r for r in results if r["reg_helped_test"]]
    print(f"cells where reg LIFTED test over its scaling baseline: "
          f"{len(helped)}/{total}", flush=True)


if __name__ == "__main__":
    main()

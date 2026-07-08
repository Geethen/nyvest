"""Micro-benchmark for DNN training speedups — REAL data, REAL model.

Times single-model and full 5-seed-ensemble fit on fold-0 train rows of the
actual stable-allyears frame (cached), reporting wall time and the fold-0 test
macro-F1 so we can confirm a speed change did NOT move the numbers.

Every run prints one JSON line so a sweep is greppable. Levers come straight
from dnn_core.Config / env, plus this harness sets the process-wide toggles
(TF32, fused Adam) that Config does not own.

Usage:
  PY=~/myprojects/recover/.venv/bin/python
  $PY DNN/bench_train.py --tag baseline
  DNN_TF32=1 DNN_FUSED=1 $PY DNN/bench_train.py --tag tf32_fused
  DNN_STREAMS=1 $PY DNN/bench_train.py --tag streams
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_core as C  # noqa: E402
import data_utils as du  # noqa: E402


def synchronize():
    if C.DEVICE == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="repeat full-ensemble fits")
    args = ap.parse_args()

    # process-wide toggles the Config dataclass does not own
    if os.environ.get("DNN_TF32", "0") == "1":
        torch.set_float32_matmul_precision("high")
    cfg = C.Config()

    data = C.load_cached("lidar")
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    lon, lat = data["lon"], data["lat"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1

    # fold-0 split with the default train-only cls12 clean (matches cv_evaluate)
    folds = list(du.fold_indices(y_enc, groups))
    k, tr, te = folds[args.fold]
    keep = du.clean_stale_class_mask(X[tr], y_enc[tr], None, cls12_enc, lon[tr], lat[tr])
    tr_use = tr[keep]
    ytr = y_enc[tr_use]

    # warm up CUDA context / caches so the first timed fit isn't skewed
    _ = C.fit_ensemble(X[tr_use][:2000], ytr[:2000], n_classes, cfg,
                       data["feat_cols"], classes, data.get("lidar_med"))
    synchronize()

    fit_times, f1s = [], []
    for r in range(args.repeat):
        t0 = time.perf_counter()
        ens = C.fit_ensemble(X[tr_use], ytr, n_classes, cfg,
                             data["feat_cols"], classes, data.get("lidar_med"))
        synchronize()
        fit_times.append(time.perf_counter() - t0)
        pred = ens.predict_proba(X[te]).argmax(1)
        f1s.append(du.macro_f1(y_enc[te], pred, n_classes))

    rec = {
        "tag": args.tag,
        "device": C.DEVICE,
        "tf32": os.environ.get("DNN_TF32", "0"),
        "fused": os.environ.get("DNN_FUSED", "0"),
        "streams": os.environ.get("DNN_STREAMS", "0"),
        "amp": os.environ.get("DNN_AMP", "0"),
        "bf16": os.environ.get("DNN_BF16", "0"),
        "compile": os.environ.get("DNN_COMPILE", "0"),
        "n_ensemble": cfg.n_ensemble,
        "n_train": int(len(tr_use)),
        "fit_s_mean": round(float(np.mean(fit_times)), 2),
        "fit_s_min": round(float(np.min(fit_times)), 2),
        "fold0_f1": round(float(np.mean(f1s)), 4),
    }
    print("BENCH " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

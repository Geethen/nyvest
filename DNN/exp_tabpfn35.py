"""Head-to-head: TabPFN-3.5 (in-context, GPU, full per-fold context) vs the
deployed DNN (models/dnn_final_moe8_merged.pt, moe_shared@8, macro F1 0.7453).

Same protocol as the deployed model's own evaluation
(autoresearch/results/moe_shared__n_experts8_top_k2.json):
  - stable-allyears parquet, 64 AlphaEarth + 3 lidar features (67 total)
  - 3-fold spatial CV via GroupKFold on cell_id (data_utils.fold_indices)
  - class merge 1->2 / 9->8, macro-F1 over all classes (zero_division=0)
  - to12_fix relabel: per-fold leak-free cleanlab correction touches ONLY
    rows originally class 12 or corrected to class 12 (exact port of
    stage8_cls12_relabel.py's RELABEL=to12_fix branch)
  - test-fold labels never touched

TabPFN-3.5 gets the ENTIRE per-fold train partition as its in-context support
set (no subsampling) -- this is the point of the run: the old v3/2.5 test
([[tabpfn-v3]] memory) was capped at ~25-32k context rows; 3.5 raises the
context limit to 1M rows on GPU, so this checks whether removing that ceiling
changes the outcome. Runs the local checkpoint downloaded from
https://huggingface.co/Prior-Labs/tabpfn_3_5 (tabpfn-v3.5-20260909.safetensors)
directly from ~/.cache/tabpfn -- no network/license call at runtime.

n_estimators starts at 16 and backs off (16->8->4->2) on CUDA OOM, per fold,
since 24GB can't hold the full ~440k-row context at 16 estimators (measured:
200k rows/16 est = 13.5GB, so ~440k/16 est would exceed the 24GB card). Rows
are NEVER subsampled -- estimators is the only knob touched under memory
pressure, to honor "full context".

Run:
  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
    ~/myprojects/recover/.venv/bin/python DNN/exp_tabpfn35.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
from dnn_paths import result_path  # noqa: E402

from tabpfn import TabPFNClassifier

MODEL_PATH = "tabpfn-v3.5-20260909.safetensors"
PERFOLD_NPZ = (Path(__file__).resolve().parents[1] / "common_ground" /
               "reports" / "research" / "clean_labels_perfold.npz")
PRED_CHUNK = 10_000
# Measured directly on this GPU/checkpoint at full 442k-row context:
# n_estimators=8 -> fit 519s, predict(10k) 11.6s, peak 15.8GB (comfortable on
# 24GB). n_estimators=16 blew past GPU memory AND, in the retry, the host
# 40G systemd cgroup cap (SIGKILL, exit 137) -- do not retry through 16.
N_EST_TRY = [8, 4, 2, 1]
OUT = result_path("tabpfn35_comparison.json")

DEPLOYED_REF = {  # autoresearch/results/moe_shared__n_experts8_top_k2.json
    "f1_mean": 0.7453,
    "f1_per_fold": [0.747, 0.7463, 0.7426],
    "f1_per_class": {"2": 0.5507, "3": 0.7869, "4": 0.7422, "5": 0.596,
                      "6": 0.6652, "7": 0.691, "8": 0.9536, "10": 0.8487,
                      "11": 0.7763, "12": 0.8426},
}


def to12_fix_labels(y_enc, tr, corrected_enc_fold, c12):
    """Exact port of stage8_cls12_relabel.py RELABEL='to12_fix' for one fold."""
    ytr = y_enc[tr].copy()
    corr = corrected_enc_fold[tr]
    touch = (y_enc[tr] == c12) | (corr == c12)
    ytr[touch] = corr[touch]
    return ytr


def predict_chunked(clf, Xte, n_classes, chunk=PRED_CHUNK):
    # predict_proba's columns follow clf.classes_, which may not be a sorted
    # 0..n_classes-1 run (e.g. if a class is entirely absent from this fold's
    # train partition) -- remap explicitly rather than assume column order.
    col_for_class = {int(c): j for j, c in enumerate(clf.classes_)}
    out = np.zeros((len(Xte), n_classes), dtype=np.float32)
    for i in range(0, len(Xte), chunk):
        p = clf.predict_proba(Xte[i:i + chunk])
        for cls_val, col in col_for_class.items():
            out[i:i + chunk, cls_val] = p[:, col]
    return out


def fit_with_backoff(Xtr, ytr):
    last_err = None
    for n_est in N_EST_TRY:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        clf = None
        print(f"    trying n_estimators={n_est}...", flush=True)
        try:
            clf = TabPFNClassifier(
                model_path=MODEL_PATH, device="cuda", n_estimators=n_est,
                ignore_pretraining_limits=True, fit_mode="fit_with_cache",
            )
            t0 = time.perf_counter()
            clf.fit(Xtr, ytr)
            torch.cuda.synchronize()
            fit_s = time.perf_counter() - t0
            return clf, n_est, fit_s
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print(f"    n_estimators={n_est} OOM, backing off...", flush=True)
            if clf is not None:
                del clf
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"OOM at every n_estimators in {N_EST_TRY}") from last_err


def main():
    print(f"tabpfn version 9.0.0 (TabPFN-3.5), checkpoint={MODEL_PATH}", flush=True)
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    c12 = classes.index(12)

    z = np.load(PERFOLD_NPZ)
    remap = {c: i for i, c in enumerate(classes)}
    corrected_enc = np.vectorize(remap.get)(z["corrected"])   # (3, N) -> 0..9

    t0 = time.perf_counter()
    f1s, pcs, fold_meta = [], [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        ytr = to12_fix_labels(y_enc, tr, corrected_enc[k], c12)
        Xtr, Xte = X[tr], X[te]
        print(f"fold {k}: context={len(tr):,} rows  test={len(te):,} rows", flush=True)

        clf, n_est_used, fit_s = fit_with_backoff(Xtr, ytr)
        peak_fit_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  fit done: n_estimators={n_est_used}  {fit_s:.1f}s  "
              f"peak_gpu={peak_fit_mb:.0f}MB", flush=True)

        tp0 = time.perf_counter()
        proba = predict_chunked(clf, Xte, n_classes)
        pred_s = time.perf_counter() - tp0
        peak_total_mb = torch.cuda.max_memory_allocated() / 1e6
        pred_enc = proba.argmax(1)

        yte = y_enc[te]
        f1 = du.macro_f1(yte, pred_enc, n_classes)
        pc = du.per_class_f1(yte, pred_enc, n_classes)
        f1s.append(f1)
        pcs.append(pc)
        fold_meta.append({
            "fold": k, "n_estimators": n_est_used, "context_rows": int(len(tr)),
            "test_rows": int(len(te)), "fit_s": round(fit_s, 1),
            "predict_s": round(pred_s, 1), "peak_gpu_mb": round(peak_total_mb, 0),
        })
        print(f"  fold {k}: macroF1={f1:.4f}  cls12_F1={pc[c12]:.4f}  "
              f"predict={pred_s:.1f}s  total_fold={time.perf_counter()-ft:.1f}s  "
              f"elapsed={time.perf_counter()-t0:.0f}s", flush=True)

        del clf
        torch.cuda.empty_cache()

        # Write partial results after every fold so a late OOM/crash doesn't
        # lose earlier folds.
        _write(f1s, pcs, classes, fold_meta, t0, done=(k == 2))

    print(f"\n=== TabPFN-3.5 vs deployed moe_shared@8 (0.7453) ===", flush=True)
    print(f"TabPFN-3.5 macro F1={np.mean(f1s):.4f} (+/-{np.std(f1s):.4f})  "
          f"deployed=0.7453  delta={np.mean(f1s)-0.7453:+.4f}", flush=True)


def _write(f1s, pcs, classes, fold_meta, t0, done):
    f1m = float(np.mean(f1s))
    f1std = float(np.std(f1s)) if len(f1s) > 1 else 0.0
    pc = np.mean(pcs, axis=0)
    OUT.write_text(json.dumps({
        "model": "TabPFN-3.5 (Prior-Labs/tabpfn_3_5, tabpfn-v3.5-20260909.safetensors)",
        "protocol": "3-fold spatial CV, GroupKFold cell_id, to12_fix relabel, "
                    "64 AlphaEarth + 3 lidar features, full per-fold context "
                    "(no subsampling)",
        "done": done,
        "f1_mean": round(f1m, 4),
        "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4)
                          for c, v in zip(classes[:len(pc)], pc)} if len(pc) else {},
        "fold_meta": fold_meta,
        "deployed_reference": DEPLOYED_REF,
        "delta_vs_deployed": round(f1m - DEPLOYED_REF["f1_mean"], 4),
        "wall_s_so_far": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"  [checkpoint written -> {OUT}]", flush=True)


if __name__ == "__main__":
    main()

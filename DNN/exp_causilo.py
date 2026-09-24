"""Head-to-head: Causilo v1.0.2 (in-context tabular foundation model, GPU)
vs the deployed DNN (models/dnn_final_moe8_merged.pt, moe_shared@8,
macro F1 0.7453).

Same protocol as the deployed model's own evaluation
(autoresearch/results/moe_shared__n_experts8_top_k2.json):
  - stable-allyears parquet, 64 AlphaEarth + 3 lidar features (67 total)
  - 3-fold spatial CV via GroupKFold on cell_id (data_utils.fold_indices)
  - class merge 1->2 / 9->8, macro-F1 over all classes (zero_division=0)
  - to12_fix relabel: per-fold leak-free cleanlab correction touches ONLY
    rows originally class 12 or corrected to class 12 (exact port of
    stage8_cls12_relabel.py's RELABEL=to12_fix branch)
  - test-fold labels never touched

Causilo gets the ENTIRE per-fold train partition as its in-context support
set (no subsampling). Causilo is a pretrained tabular foundation model from
Nums AI Inc. that performs in-context learning (no gradient training during
fit). It's ranked #1 on TabArena and has a native 10-class classification
head, making it ideal for this task.

n_estimators backs off (8->4->2->1) on CUDA OOM, per fold.
Rows are NEVER subsampled.

Run:
  systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 \
    ~/myprojects/recover/.venv/bin/python DNN/exp_causilo.py 2>&1 \
    | tee DNN/reports/logs/causilo_comparison.log
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

from causilo import CausiloClassifier

PERFOLD_NPZ = (Path(__file__).resolve().parents[1] / "common_ground" /
               "reports" / "research" / "clean_labels_perfold.npz")
PRED_CHUNK = 10_000
# Back off n_estimators on OOM. Memory measured during scale testing:
# 5k->9.4GB, 20k->10.5GB, 50k->11.6GB, 100k->~13GB, 200k->~16GB.
# Full 442k fold should fit at n_estimators=8 on 24GB with headroom.
N_EST_TRY = [8, 4, 2, 1]
OUT = result_path("causilo_comparison.json")

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


def predict_chunked(clf, Xte, chunk=PRED_CHUNK):
    """Predict in chunks to limit GPU peak memory during prediction."""
    preds = []
    for i in range(0, len(Xte), chunk):
        p = clf.predict(Xte[i:i + chunk])
        preds.append(p)
    return np.concatenate(preds)


def fit_with_backoff(Xtr, ytr):
    last_err = None
    for n_est in N_EST_TRY:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        clf = None
        print(f"    trying n_estimators={n_est}...", flush=True)
        try:
            clf = CausiloClassifier(
                n_estimators=n_est, device="cuda",
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
    import causilo
    ver = getattr(causilo, "__version__", "unknown")
    print(f"causilo version {ver}", flush=True)
    print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}, "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
          flush=True)

    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    c12 = classes.index(12)
    print(f"Data: {X.shape[0]:,} rows, {X.shape[1]} features, {n_classes} classes",
          flush=True)

    z = np.load(PERFOLD_NPZ)
    remap = {c: i for i, c in enumerate(classes)}
    corrected_enc = np.vectorize(remap.get)(z["corrected"])   # (3, N) -> 0..9

    t0 = time.perf_counter()
    f1s, pcs, fold_meta = [], [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        ytr = to12_fix_labels(y_enc, tr, corrected_enc[k], c12)
        Xtr, Xte = X[tr], X[te]
        print(f"\nfold {k}: context={len(tr):,} rows  test={len(te):,} rows", flush=True)

        clf, n_est_used, fit_s = fit_with_backoff(Xtr, ytr)
        peak_fit_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  fit done: n_estimators={n_est_used}  {fit_s:.1f}s  "
              f"peak_gpu={peak_fit_mb:.0f}MB", flush=True)

        tp0 = time.perf_counter()
        pred_enc = predict_chunked(clf, Xte)
        torch.cuda.synchronize()
        pred_s = time.perf_counter() - tp0
        peak_total_mb = torch.cuda.max_memory_allocated() / 1e6

        yte = y_enc[te]
        f1 = du.macro_f1(yte, pred_enc, n_classes)
        pc = du.per_class_f1(yte, pred_enc, n_classes)
        f1s.append(f1)
        pcs.append(pc)
        rows_per_sec = len(te) / pred_s if pred_s > 0 else float("inf")
        fold_meta.append({
            "fold": k, "n_estimators": n_est_used, "context_rows": int(len(tr)),
            "test_rows": int(len(te)), "fit_s": round(fit_s, 1),
            "predict_s": round(pred_s, 1), "peak_gpu_mb": round(peak_total_mb, 0),
            "rows_per_sec_predict": round(rows_per_sec, 0),
        })
        print(f"  fold {k}: macroF1={f1:.4f}  cls12_F1={pc[c12]:.4f}  "
              f"predict={pred_s:.1f}s ({rows_per_sec:,.0f} rows/s)  "
              f"total_fold={time.perf_counter()-ft:.1f}s  "
              f"elapsed={time.perf_counter()-t0:.0f}s", flush=True)
        print(f"  per-class F1: {dict(zip(classes, [round(float(v), 4) for v in pc]))}",
              flush=True)

        del clf
        gc.collect()
        torch.cuda.empty_cache()

        # Write partial results after every fold so a late OOM/crash doesn't
        # lose earlier folds.
        _write(f1s, pcs, classes, fold_meta, t0, done=(k == 2))

    # Final summary
    mean_f1 = np.mean(f1s)
    mean_pc = np.mean(pcs, axis=0)
    mean_rows_per_sec = np.mean([m["rows_per_sec_predict"] for m in fold_meta])
    print(f"\n{'='*60}", flush=True)
    print(f"=== Causilo v{ver} vs deployed moe_shared@8 (0.7453) ===", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Causilo macro F1={mean_f1:.4f} (+/-{np.std(f1s):.4f})  "
          f"deployed=0.7453  delta={mean_f1-0.7453:+.4f}", flush=True)
    print(f"Causilo predict throughput: {mean_rows_per_sec:,.0f} rows/sec  "
          f"deployed GPU: 9,360,000 px/sec", flush=True)
    for i, c in enumerate(classes):
        delta = float(mean_pc[i]) - DEPLOYED_REF["f1_per_class"].get(str(c), 0)
        if abs(delta) > 0.01:
            print(f"  class {c}: causilo={mean_pc[i]:.4f}  deployed="
                  f"{DEPLOYED_REF['f1_per_class'].get(str(c), '?')}  "
                  f"delta={delta:+.4f}", flush=True)

    print(f"\nLicense: Code Apache-2.0, weights Causilo License v1.0 "
          f"(non-commercial only; commercial/production use requires "
          f"separate license from Nums AI Inc.)", flush=True)


def _write(f1s, pcs, classes, fold_meta, t0, done):
    import causilo
    ver = getattr(causilo, "__version__", "unknown")
    f1m = float(np.mean(f1s))
    f1std = float(np.std(f1s)) if len(f1s) > 1 else 0.0
    pc = np.mean(pcs, axis=0)

    mean_rows_per_sec = np.mean([m["rows_per_sec_predict"] for m in fold_meta])
    total_pred_rows = sum(m["test_rows"] for m in fold_meta)

    pc_dict = {str(c): round(float(v), 4)
               for c, v in zip(classes[:len(pc)], pc)} if len(pc) else {}
    delta_pc = {}
    for c_str, v in pc_dict.items():
        ref = DEPLOYED_REF["f1_per_class"].get(c_str, 0)
        delta_pc[c_str] = round(v - ref, 4)

    OUT.write_text(json.dumps({
        "model": f"Causilo v{ver} (nums-ai/causilo, CausiloClassifier)",
        "protocol": "3-fold spatial CV, GroupKFold cell_id, to12_fix relabel, "
                    "64 AlphaEarth + 3 lidar features, full per-fold context "
                    "(no subsampling)",
        "done": done,
        "f1_mean": round(f1m, 4),
        "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": pc_dict,
        "f1_per_class_per_fold": [[round(float(x), 4) for x in fold_pc]
                                   for fold_pc in pcs],
        "timing": {
            "fit_seconds_per_fold": [m["fit_s"] for m in fold_meta],
            "predict_seconds_per_fold": [m["predict_s"] for m in fold_meta],
            "rows_per_sec_predict": [m["rows_per_sec_predict"] for m in fold_meta],
            "total_predict_rows": total_pred_rows,
            "mean_rows_per_sec": round(mean_rows_per_sec, 0),
        },
        "fold_meta": fold_meta,
        "deployed_reference": DEPLOYED_REF,
        "delta_vs_deployed": round(f1m - DEPLOYED_REF["f1_mean"], 4),
        "delta_per_class_vs_deployed": delta_pc,
        "license": {
            "code": "Apache-2.0",
            "weights": "Causilo License v1.0 (non-commercial only; "
                       "commercial/production use requires separate license "
                       "from Nums AI Inc.)",
        },
        "wall_s_so_far": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"  [checkpoint written -> {OUT}]", flush=True)


if __name__ == "__main__":
    main()

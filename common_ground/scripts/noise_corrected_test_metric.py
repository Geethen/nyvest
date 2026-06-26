"""Post-hoc noise-corrected test metric (leak-free).

The test partition keeps its ORIGINAL noisy grunnkart labels (no cleaning, no leakage).
Measured F1 against those noisy labels is a pessimistic lower bound. This script
estimates the CLEAN-label F1 using the noise transition matrix T (from cleanlab's
confident-joint), adjusting only the aggregate confusion — never inspecting which
individual test rows are wrong.

Method:
  cm_noisy[obs, pred]  : confusion vs noisy test labels (summed over folds; saved by
                         the pipeline as per-fold "confusion_noisy").
  inv_noise[true, obs] = P(true=true | observed=obs)  (from estimate_latent).
  Each noisy-label row is a mixture of true-label rows, so estimate
     cm_true[true, pred] = sum_obs inv_noise[true, obs] * cm_noisy[obs, pred].
  Report macro-F1 and class-12 F1 from cm_true, alongside the noisy-label F1.

This assumes predictions are conditionally independent of the label noise given the
features (reasonable: the model never sees test labels). It is an ESTIMATE, reported
next to the leak-free noisy-label lower bound.

Usage:
  python noise_corrected_test_metric.py <results_json> [<results_json> ...]
Output: prints a table; writes *_noise_corrected.json next to each input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
OUT_DIR = _REPO / "common_ground" / "reports" / "research"
ART = OUT_DIR / "clean_labels_full.npz"


def f1_from_confusion(cm):
    # cm[true, pred]; macro-F1 and per-class F1
    cm = np.asarray(cm, dtype=np.float64)
    cm = np.clip(cm, 0, None)  # corrected counts can go slightly negative
    tp = np.diag(cm)
    pred_sum = cm.sum(0)  # predicted as class j
    true_sum = cm.sum(1)  # actually class i
    prec = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    rec = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1


def run(paths):
    npz = np.load(ART)
    inv_noise = npz["inv_noise"]          # [true, obs] = P(true|obs)
    classes = npz["classes"].tolist()     # merged class ids, pipeline order
    cls12_idx = classes.index(12)
    print(f"transition matrix loaded; classes={classes}")
    # diagnostic: P(true=12 | observed=12) and main leak-out target
    print(f"P(true=12 | obs=12) = {inv_noise[cls12_idx, cls12_idx]:.3f}")

    for p in paths:
        d = json.load(open(p))
        folds = [f for f in d.get("per_fold", []) if "confusion_noisy" in f]
        if not folds:
            print(f"{Path(p).name}: no confusion_noisy saved (rerun pipeline)")
            continue
        cm_noisy = np.sum([np.array(f["confusion_noisy"]) for f in folds], axis=0)
        # cm_noisy[obs, pred] -> cm_true[true, pred] = inv_noise @ cm_noisy
        cm_true = inv_noise @ cm_noisy

        f1_noisy = f1_from_confusion(cm_noisy)
        f1_corr = f1_from_confusion(cm_true)
        macro_noisy = float(np.mean(f1_noisy))
        macro_corr = float(np.mean(f1_corr))
        out = {
            "clean_mode": d.get("clean_mode"),
            "macro_f1_noisy_lowerbound": round(macro_noisy, 4),
            "macro_f1_noise_corrected": round(macro_corr, 4),
            "cls12_f1_noisy": round(float(f1_noisy[cls12_idx]), 4),
            "cls12_f1_corrected": round(float(f1_corr[cls12_idx]), 4),
            "per_class_noisy": {str(classes[i]): round(float(f1_noisy[i]), 4)
                                for i in range(len(classes))},
            "per_class_corrected": {str(classes[i]): round(float(f1_corr[i]), 4)
                                    for i in range(len(classes))},
        }
        outp = Path(p).with_name(Path(p).stem + "_noise_corrected.json")
        json.dump(out, open(outp, "w"), indent=2)
        print(f"\n{Path(p).name}  [clean_mode={out['clean_mode']}]")
        print(f"  macro-F1  noisy(lower bound) = {macro_noisy:.4f}"
              f"   noise-corrected = {macro_corr:.4f}  Δ={macro_corr-macro_noisy:+.4f}")
        print(f"  class-12  noisy = {f1_noisy[cls12_idx]:.4f}"
              f"   noise-corrected = {f1_corr[cls12_idx]:.4f}"
              f"  Δ={f1_corr[cls12_idx]-f1_noisy[cls12_idx]:+.4f}")


if __name__ == "__main__":
    args = sys.argv[1:] or [str(OUT_DIR / "schemeB_allyears_results.json")]
    run(args)

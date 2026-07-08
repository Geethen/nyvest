"""Fit uncertainty-quantification calibrators for the final DNN ensemble.

Companion to `train_final.py`: that script bakes the winning recipe into
`models/dnn_final.pt` trained on the FULL stable frame (no held-out fold, so
its own training probabilities are optimistic and unusable for calibration).
This script instead runs the reference leak-free 3-fold spatial CV (same
protocol as `conformal_compare.py`) to produce out-of-fold (OOF) probabilities
covering the entire stable dataset — every row's probability comes from a
model that never saw it during training — then fits:

  1. LAC + Mondrian (class-conditional) conformal prediction sets at
     alpha=0.1 (90% target coverage). This is the method Investigation 3
     (`conformal_compare.py`) found best: tightest sets among valid methods,
     no empty sets, and per-class coverage spread collapses from 0.29 (plain
     split) to 0.035 (Mondrian) — see DNN/README.md and the
     `dnn-robustness-uq` memory.
  2. A point-calibration method for the "calibrated probability" output band,
     chosen head-to-head between temperature scaling (the method
     Investigation 3 used) and Venn-Abers (one-vs-rest binary calibrators,
     since the `venn-abers` package is binary-only). Both are fit on the same
     cal/eval split of the OOF set and compared by ECE on the eval half; the
     lower-ECE method wins and is saved. Both numbers are recorded regardless
     so the comparison is auditable later.

Output: models/dnn_final_calib.npz — loaded by dnn_core.Calibration and
consumed by predict.py / predict_raster.py to emit calibrated proba +
prediction-set size + per-class inclusion bands alongside the class map.

Run:
  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
    ~/myprojects/recover/.venv/bin/python DNN/fit_calibration.py
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
import conformal_methods as cm     # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
EXTRA = os.environ.get("EXTRA", "lidar")
RELABEL = os.environ.get("RELABEL", "to12_fix")
ALPHA = float(os.environ.get("ALPHA", "0.1"))
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.5"))
SEED = int(os.environ.get("SEED", "0"))
OUT = Path(os.environ.get("OUT", _REPO / "models" / "dnn_final_calib.npz"))
TIMING_ROWS = int(os.environ.get("TIMING_ROWS", "1_000_000"))


fit_venn_abers_ovr = cm.fit_venn_abers_ovr
apply_venn_abers_ovr = cm.apply_venn_abers_ovr


def main():
    t0 = time.perf_counter()
    cfg = C.Config()
    print(f"device={C.DEVICE}  extra={EXTRA}  relabel={RELABEL}  alpha={ALPHA} "
          f"(target cov {1-ALPHA:.0%})  cal_frac={CAL_FRAC}  ensemble={cfg.n_ensemble}",
          flush=True)

    data = C.load_cached(EXTRA)
    X, y_enc, groups, classes = data["X"], data["y_enc"], data["groups"], data["classes"]
    n_classes = len(classes)
    ytr_full, n_changed = du.apply_cls12_relabel(y_enc, classes, RELABEL)
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes  "
          f"relabel changed {n_changed}", flush=True)

    # ---- leak-free 3-fold spatial CV -> OOF probabilities over ALL rows ----
    # Mirrors stage8_cls12_relabel.py / train_final.py exactly: a cls12 relabel
    # mode (cls12_fix / to12_fix) SUPERSEDES centroid-cleaning for that fold —
    # they are mutually exclusive, never both applied. Only "none" falls back
    # to plain centroid cleaning (dnn_core.cv_evaluate's default path).
    P_oof = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    f1s = []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        if RELABEL in ("cls12_fix", "to12_fix"):
            tr_use, ytr = tr, ytr_full[tr]
        else:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], None,
                                             classes.index(12) if 12 in classes else -1,
                                             data["lon"][tr], data["lat"][tr])
            tr_use, ytr = tr[keep], y_enc[tr[keep]]
        ens = C.fit_ensemble(X[tr_use], ytr, n_classes, cfg,
                             data["feat_cols"], classes, data.get("lidar_med"),
                             rng=np.random.default_rng(SEED + k))
        P = ens.predict_proba(X[te])
        P_oof[te] = P
        f1 = du.macro_f1(y_enc[te], P.argmax(1), n_classes)
        f1s.append(f1)
        print(f"  fold {k}: F1={f1:.4f}  val_f1={ens.val_f1:.4f}  "
              f"{time.perf_counter()-ft:.1f}s", flush=True)
    oof_f1 = float(np.mean(f1s))
    print(f"OOF macro-F1 = {oof_f1:.4f} (CV reference 0.7341)", flush=True)

    # ---- LAC + Mondrian conformal thresholds (fit on the FULL OOF set) ----
    y_true = ytr_full  # calibrate against the same labels the final model trains on
    S_full = cm.score_lac(P_oof)
    lac_taus = cm.calibrate_classcond(S_full, y_true, ALPHA, n_classes)
    sets_full = cm.sets_from_tau(S_full, lac_taus)
    conf_metrics = cm.set_metrics(sets_full, y_true, n_classes)
    print(f"LAC+Mondrian (in-sample check): coverage={conf_metrics['coverage']} "
          f"avg_set_size={conf_metrics['avg_set_size']} "
          f"(conformal_compare.json reference: 0.8999 / 1.644)", flush=True)

    # ---- cal/eval split of the OOF set for the point-calibration comparison ----
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_true))
    n_cal = int(len(y_true) * CAL_FRAC)
    cal_i, ev_i = perm[:n_cal], perm[n_cal:]
    P_cal, y_cal = P_oof[cal_i], y_true[cal_i]
    P_ev, y_ev = P_oof[ev_i], y_true[ev_i]

    ece_before = cm.ece(P_ev, y_ev)

    # temperature scaling
    t_ts0 = time.perf_counter()
    T, P_ev_ts = cm.temperature_scale(P_cal, y_cal, P_ev)
    ece_ts = cm.ece(P_ev_ts, y_ev)
    ts_fit_s = time.perf_counter() - t_ts0
    t_ts1 = time.perf_counter()
    idx = rng.integers(0, len(P_ev), size=min(TIMING_ROWS, len(P_ev)))
    _ = cm.temperature_scale(P_cal, y_cal, P_ev[idx])[1]
    ts_apply_s = time.perf_counter() - t_ts1

    # venn-abers OvR
    t_va0 = time.perf_counter()
    va_calibrators = fit_venn_abers_ovr(P_cal, y_cal)
    P_ev_va = apply_venn_abers_ovr(va_calibrators, P_ev)
    ece_va = cm.ece(P_ev_va, y_ev)
    va_fit_s = time.perf_counter() - t_va0
    t_va1 = time.perf_counter()
    _ = apply_venn_abers_ovr(va_calibrators, P_ev[idx])
    va_apply_s = time.perf_counter() - t_va1

    print(f"\n=== point-calibration comparison (eval n={len(y_ev):,}) ===", flush=True)
    print(f"  ECE before      : {ece_before:.4f}", flush=True)
    print(f"  temp-scale (T={T:.3f}): ECE {ece_ts:.4f}  fit {ts_fit_s:.1f}s  "
          f"apply/{len(idx):,} rows {ts_apply_s*1000:.1f}ms", flush=True)
    print(f"  venn-abers OvR  : ECE {ece_va:.4f}  fit {va_fit_s:.1f}s  "
          f"apply/{len(idx):,} rows {va_apply_s*1000:.1f}ms", flush=True)

    winner = "venn_abers" if ece_va < ece_ts else "temp_scale"
    print(f"  winner: {winner} (lower ECE)", flush=True)

    # ---- save artifact ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        calib_method=winner,
        alpha=ALPHA,
        classes=np.array(classes),
        lac_mondrian_taus=lac_taus,
        T=T,
        ece_before=ece_before,
        ece_temp_scale=ece_ts,
        ece_venn_abers=ece_va,
        ts_fit_s=ts_fit_s, ts_apply_s=ts_apply_s,
        va_fit_s=va_fit_s, va_apply_s=va_apply_s,
        oof_macro_f1=oof_f1,
        conformal_coverage=conf_metrics["coverage"],
        conformal_avg_set_size=conf_metrics["avg_set_size"],
    )
    # Always persist the VA breakpoints regardless of which method won — they're
    # already computed, and discarding the loser means a later switch (or an
    # audit of "what would VA have looked like") needs a full CV re-run to
    # recover them.
    for c, (p0, p1, cpts) in va_calibrators.items():
        save_kwargs[f"va_p0_{c}"] = p0
        save_kwargs[f"va_p1_{c}"] = p1
        save_kwargs[f"va_c_{c}"] = cpts
    np.savez(OUT, **save_kwargs)

    meta = {
        "artifact": str(OUT), "alpha": ALPHA, "classes": classes,
        "oof_macro_f1": round(oof_f1, 4),
        "conformal": conf_metrics,
        "calib_method": winner,
        "temp_scale": {"T": round(float(T), 4), "ece_before": round(ece_before, 4),
                       "ece_after": round(ece_ts, 4), "fit_s": round(ts_fit_s, 2),
                       "apply_ms_per_batch": round(ts_apply_s * 1000, 2), "batch": len(idx)},
        "venn_abers": {"ece_before": round(ece_before, 4), "ece_after": round(ece_va, 4),
                       "fit_s": round(va_fit_s, 2),
                       "apply_ms_per_batch": round(va_apply_s * 1000, 2), "batch": len(idx)},
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved -> {OUT}  ({meta['wall_s']}s)", flush=True)
    print(f"meta  -> {OUT.with_suffix('.meta.json')}", flush=True)


if __name__ == "__main__":
    main()

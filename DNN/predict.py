"""Batch inference on a tabular parquet of AlphaEarth points (P-drive / GEE).

For point/tabular inputs shaped like the training data: a parquet with columns
A00..A63 (+ optional elevation,tri,tch if the model uses lidar). Writes a
parquet with the predicted class code plus uncertainty-quantification columns:
calibrated per-class probability (temperature scaling or Venn-Abers,
whichever `fit_calibration.py` found better-calibrated — see
models/dnn_final_calib.meta.json), LAC+Mondrian conformal prediction-set size,
and a 0/1 inclusion column per class. For raster tiles use `predict_raster.py`
instead.

Run:
  ~/myprojects/recover/.venv/bin/python DNN/predict.py \
    --in points.parquet --out predictions.parquet \
    [--model models/dnn_final.pt] [--calib models/dnn_final_calib.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_core as C  # noqa: E402

LIDAR_COLS = ["elevation", "tri", "tch"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] /
                                           "models" / "dnn_final.pt"))
    ap.add_argument("--calib", default=str(Path(__file__).resolve().parents[1] /
                                           "models" / "dnn_final_calib.npz"))
    args = ap.parse_args()

    ens = C.Ensemble.load(args.model)
    calib = C.Calibration.load(args.calib)
    if calib.classes != ens.classes:
        raise ValueError(
            f"--model/--calib class mismatch: ens.classes={ens.classes} vs "
            f"calib.classes={calib.classes} — regenerate dnn_final_calib.npz "
            f"(fit_calibration.py) against the current model, they are indexed "
            f"positionally and a mismatch silently mislabels output columns.")
    df = pd.read_parquet(args.inp)
    lidar_med = ens.lidar_med or {}
    missing = []
    for col in ens.feat_cols:
        if col not in df.columns:
            if col in LIDAR_COLS:
                df[col] = lidar_med.get(col, 0.0)
                missing.append(col)
            else:
                raise ValueError(f"input parquet is missing required feature {col!r}")
    if missing:
        print(f"filled {missing} with training medians", flush=True)

    X = df[ens.feat_cols].to_numpy(np.float32)
    result = ens.predict_full(X, calib)
    out = df.copy()
    out["pred_class"] = result["pred_class"]
    for j, c in enumerate(ens.classes):
        out[f"pcal_{c}"] = result["proba_calibrated"][:, j]
    out["set_size"] = result["set_size"]
    for j, c in enumerate(ens.classes):
        out[f"inset_{c}"] = result["included"][:, j].astype(np.int8)
    out.to_parquet(args.out, index=False)
    print(f"predicted {len(out):,} rows -> {args.out}  "
          f"(calib_method={calib.calib_method})", flush=True)


if __name__ == "__main__":
    main()

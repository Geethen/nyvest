"""Train the FINAL production ensemble on ALL stable data and save one artifact.

Unlike the stageN scripts (which do 3-fold spatial CV to *measure* macro-F1),
this trains the winning recipe on the entire stable-allyears frame and writes a
single portable file `models/dnn_final.pt` that `predict.py` / `predict_raster.py`
load standalone. Use CV (stageN / `dnn_core.cv_evaluate`) to pick a recipe; use
this to bake the chosen recipe into a deployable model.

Recipe = the DNN best (README: 0.7341): 5-seed class-weighted MLP ensemble,
sqrt weights, label smoothing 0.05, + the targeted class-12 relabel (`to12_fix`)
which is the only label fix that helped. Because we train on the WHOLE dataset
(no held-out test fold), the relabel uses the full-data cleanlab artifact
`clean_labels_full.npz` — there is no test partition to leak into here.

Knobs (env): all `dnn_core.Config` fields, plus:
  RELABEL   = to12_fix | cls12_fix | none        (default to12_fix)
  OUT       = models/dnn_final.pt
  EXTRA     = lidar | none                        (feature set; default lidar)

Run:
  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
    ~/myprojects/recover/.venv/bin/python DNN/train_final.py

This writes the POINT-prediction model only. For uncertainty-quantified
inference (calibrated probabilities + conformal prediction sets), also run
the companion calibration step, which fits its own leak-free out-of-fold CV
(separate from this full-data fit, since calibration needs held-out
probabilities):

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
import dnn_core as C  # noqa: E402
import data_utils as du  # noqa: E402
from data_utils import apply_cls12_relabel  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
RELABEL = os.environ.get("RELABEL", "to12_fix")
EXTRA = os.environ.get("EXTRA", "lidar")
OUT = Path(os.environ.get("OUT", _REPO / "models" / "dnn_final.pt"))


def main():
    t0 = time.perf_counter()
    cfg = C.Config()
    print(f"device={C.DEVICE}  extra={EXTRA}  relabel={RELABEL}  ensemble={cfg.n_ensemble}",
          flush=True)
    data = C.load_cached(EXTRA)
    X, y_enc, classes = data["X"], data["y_enc"], data["classes"]
    n_classes = len(classes)
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes", flush=True)

    ytr, n_changed = apply_cls12_relabel(y_enc, classes, RELABEL)
    print(f"relabel {RELABEL}: {n_changed} labels changed", flush=True)

    ens = C.fit_ensemble(X, ytr, n_classes, cfg, data["feat_cols"], classes,
                         data.get("lidar_med"), verbose=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ens.save(OUT)
    meta = {
        "artifact": str(OUT), "extra_features": EXTRA, "relabel": RELABEL,
        "n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
        "feat_cols": ens.feat_cols, "classes": classes,
        "labels_changed": n_changed, "val_f1": round(ens.val_f1, 4),
        "config": cfg.to_dict(), "wall_s": round(time.perf_counter() - t0, 1),
        "cv_reference_macro_f1": 0.7341,
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {OUT}  (val_f1={ens.val_f1:.4f}, {meta['wall_s']}s)", flush=True)
    print(f"meta  -> {OUT.with_suffix('.meta.json')}", flush=True)


if __name__ == "__main__":
    main()

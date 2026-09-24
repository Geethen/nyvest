"""Batch 4 — change the CLASS ONTOLOGY, not the model or the features.

Two questions, one protocol (identical to exp_common: fixed folds, cls12 clean,
to12_fix relabel, plain 256,128 MLP, 5-seed ensemble, StandardScaler). The ONLY
thing that varies is how raw merged labels map to the trained/scored class set.

  baseline    - the standard 10-class set (2,3,4,5,6,7,8,10,11,12). Reproduces
                the stage8 to12_fix best (~0.7341) as a sanity check.

  onto_merge  - collapse the two symmetric, feature-overlapping boundaries:
                  grassland(5) + crop(3)      -> one class
                  scrub(6)     + sparse-veg(11)-> one class
                => 8 classes. Macro-F1 will RISE mechanically (two hard
                boundaries deleted). This does NOT prove the ceiling is
                definitional — a pure sensing limit predicts the same rise. What
                it DOES give is a QUANTIFIED upper bound on how much macro-F1
                those two boundaries cost, i.e. the most any intervention on them
                (sensor, label, ontology) could ever buy back.

  unmerge12   - undo the 1->2 merge (sand `Snaumark_skrinn` back out of rock
                `Snaumark_impediment`). data_utils bakes 1->2 in at load, so we
                reload the RAW class column and re-derive labels without it.
                Tests whether merging thin-soil-sand (which grades into heath)
                into rock manufactures the weak class-2 (F1 0.554). A genuine
                two-sided question: sand may just be small (7.5k rows), not
                confounded.

Env: EXP={baseline|onto_merge|unmerge12}. Uses RELABEL from env (default
to12_fix here so it matches the reported best, not the stage3 0.7318).

Run: EXP=onto_merge ~/myprojects/recover/.venv/bin/python DNN/research/exp_ontology.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec          # noqa: E402
import data_utils as du          # noqa: E402
import stage3_robust_mlp as s3   # noqa: E402

EXP = os.environ.get("EXP", "baseline")


def build_mlp(in_dim, n_classes):
    """The exact stage-3 plain MLP (256,128) that every batch here holds fixed."""
    h1, h2 = ec.HIDDEN
    return nn.Sequential(
        nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(ec.DROPOUT),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(ec.DROPOUT),
        nn.Linear(h2, n_classes),
    )


def load_with_ontology(exp):
    """Return (X, y_enc, groups, df, classes, cls12_enc) for the chosen ontology.

    For baseline/onto_merge we start from the standard merged 10-class labels and
    optionally collapse pairs. For unmerge12 we re-derive labels WITHOUT the 1->2
    merge so sand(1) is its own class again.
    """
    if exp == "unmerge12":
        # Re-load the frame and merge only 9->8, leaving 1 separate.
        df, feat_cols, _ = du._load_frame(du.STABLE_PARQUET, "lidar")
        y = df[du.TARGET].values.astype(int).copy()
        y[y == 9] = 8                       # keep the marine->freshwater merge
        X = df[feat_cols].values.astype(np.float32)
    else:
        data = du.load_data(extra_features="lidar")
        X, df = data["X"], data["df"]
        y = du.merge_classes(df[du.TARGET].values)   # 1->2, 9->8
        if exp == "onto_merge":
            y = y.copy()
            y[y == 3] = 5                   # crop -> grassland  (grass/crop pair)
            y[y == 11] = 6                  # sparse-veg -> scrub (scrub/sparse pair)

    classes = sorted(np.unique(y).tolist())
    remap = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([remap[v] for v in y], dtype=np.int64)
    groups = df["cell_id"].values
    cls12_enc = classes.index(12) if 12 in classes else -1
    return X, y_enc, groups, df, classes, cls12_enc


def main():
    t0 = time.perf_counter()
    print(f"=== ontology experiment: EXP={EXP} ===")
    print(f"device={ec.DEVICE} relabel={ec.RELABEL} ensemble={ec.N_ENSEMBLE} "
          f"hidden={ec.HIDDEN} dropout={ec.DROPOUT} lr={ec.LR}")

    X, y_enc, groups, df, classes, cls12_enc = load_with_ontology(EXP)
    n_classes = len(classes)
    lon, lat = df["lon"].values, df["lat"].values
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {n_classes} classes "
          f"({classes})")

    # to12_fix relabel is defined on the STANDARD class set. It is meaningful only
    # for baseline/onto_merge (both contain cls12). For unmerge12 the raw label
    # space differs, so we skip it and compare unmerge12 to a no-relabel baseline
    # separately (its point is class-2, which relabel never touches).
    relabel = ec.RELABEL
    if EXP == "onto_merge" and relabel == "to12_fix":
        # apply to12_fix on the standard 10-class encoding, THEN collapse pairs,
        # so cls12 cleaning stays identical to baseline. Reload standard encoding
        # to run the relabel, map suggestions through, then fold the merges in.
        std = du.load_data(extra_features="lidar")
        std_classes = std["classes"]
        y_std, _ = du.apply_cls12_relabel(std["y_enc"], std_classes, "to12_fix")
        # decode std enc -> raw, apply the same pair-merge, re-encode to `classes`
        dec = np.array(std_classes)[y_std]
        dec[dec == 3] = 5
        dec[dec == 11] = 6
        remap = {c: i for i, c in enumerate(classes)}
        y_enc = np.vectorize(remap.get)(dec).astype(np.int64)
        print("applied to12_fix (pre-merge) then ontology collapse")
    elif EXP == "baseline" and relabel != "none":
        y_enc, n_ch = du.apply_cls12_relabel(y_enc, classes, relabel)
        print(f"relabel={relabel}: {n_ch} labels changed")
    elif EXP == "unmerge12":
        print("relabel skipped (raw label space); comparing on class-2 recovery")

    rng = np.random.default_rng(ec.SEED)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        tr_use = tr
        if ec.CLEAN_CLS12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr],
                                             cls12_enc, lon[tr], lat[tr])
            tr_use = tr[keep]
        P, vf1 = ec.train_fold(build_mlp, X[tr_use], y_enc[tr_use], X[te],
                               n_classes, rng)
        pred = P.argmax(1)
        f1 = du.macro_f1(y_enc[te], pred, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        print(f"  fold {k}: F1={f1:.4f}  (val={vf1:.4f}, "
              f"{time.perf_counter()-ft:.1f}s)")

    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc_mean = np.mean(pcs, axis=0)
    print(f"\nF1 mean={f1m:.4f} std={f1std:.4f}   ({n_classes} classes)")
    print("per-class F1:")
    for c, v in zip(classes, pc_mean):
        print(f"  class {c:2d}: {v:.4f}")

    rec = {
        "name": f"ontology_{EXP}", "exp": EXP, "relabel": relabel,
        "n_classes": n_classes, "classes": classes,
        "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4)
                         for c, v in zip(classes, pc_mean)},
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    out = ec.RESULTS_DIR / f"ontology_{EXP}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

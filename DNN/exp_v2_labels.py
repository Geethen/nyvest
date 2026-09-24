"""Key training experiments for the grunnkart v2 (FKB Bygning 2018) relabel.

Two questions, both scored on the reference 3-fold spatial CV with test labels
left untouched (leak-free), paired across seeds so deltas are read per-seed
rather than from means.

  relabel   Swap the v1 label for the v2 label at the training points.
            Only 3 of 74,639 points change (the v2 edit is 86% class-13
            "other" -> built, and class 13 is excluded from FSCS sampling),
            so this is expected to be a no-op. Run to confirm empirically.

  augment   Add the newly-built pixels (v1=13 -> v2=10) as EXTRA class-10
            training rows. This is the only route by which v2 injects label
            information the training set never had. Test folds keep their
            original points and v1 labels, so the comparison is honest.

Usage
  ~/myprojects/recover/.venv/bin/python DNN/exp_v2_labels.py --exp relabel
  ~/myprojects/recover/.venv/bin/python DNN/exp_v2_labels.py --exp augment --n_aug 5000
  ~/myprojects/recover/.venv/bin/python DNN/exp_v2_labels.py --exp both --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402
import dnn_core as C  # noqa: E402
from data_utils import apply_cls12_relabel  # noqa: E402
from dnn_paths import result_path  # noqa: E402
from extract_v2_changes import EMBED_COLS, LIDAR_COLS  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
LABELS_V2 = _REPO / "data" / "labels_v2.parquet"
V2_CHANGES = _REPO / "data" / "v2_changes.parquet"
BUILT = 10


def v2_label_vector(data):
    """Map the per-point v2 class onto the cached frame's row order.

    labels_v2.parquet is one row per unique (lon, lat); the frame has one row
    per (point, year). Returns (y_enc_v2, n_changed).
    """
    lut = pd.read_parquet(LABELS_V2)
    key = lambda lon, lat: (np.round(lon, 9).astype(str) + "_"
                            + np.round(lat, 9).astype(str))
    m = dict(zip(key(lut["lon"].values, lut["lat"].values),
                 lut["class_v2"].values))
    raw = np.array([m.get(k, -1) for k in key(data["lon"], data["lat"])])
    if (raw < 0).any():
        raise RuntimeError(f"{(raw < 0).sum()} frame rows had no v2 label")
    merged = du.merge_classes(raw)
    classes = data["classes"]
    remap = {c: i for i, c in enumerate(classes)}
    if not set(np.unique(merged)).issubset(remap):
        bad = set(np.unique(merged)) - set(remap)
        raise RuntimeError(f"v2 introduces classes outside the label space: {bad}")
    y2 = np.array([remap[v] for v in merged], dtype=data["y_enc"].dtype)
    return y2, int((y2 != data["y_enc"]).sum())


def load_augmentation(data, n_aug, seed, source_class=13):
    """Newly-built pixels as extra class-10 rows, in the frame's feature order."""
    ch = pd.read_parquet(V2_CHANGES)
    ch = ch[(ch["class_v1"] == source_class) & (ch["class_v2"] == BUILT)]
    ch = ch.dropna(subset=EMBED_COLS)
    if n_aug and len(ch) > n_aug:
        ch = ch.sample(n_aug, random_state=seed)
    med = data.get("lidar_med") or {}
    for c in LIDAR_COLS:
        if c in ch.columns:
            ch[c] = ch[c].fillna(med.get(c, 0.0))
    Xa = ch[data["feat_cols"]].to_numpy(np.float32)
    ya = np.full(len(ch), data["classes"].index(BUILT), dtype=np.int64)
    return Xa, ya


def run_cv(data, cfg, y_train_all, Xa=None, ya=None, verbose=True):
    """Reference 3-fold spatial CV. `y_train_all` supplies TRAIN labels; test
    always scores against data['y_enc']. (Xa, ya) are appended to train only."""
    C.set_seed(cfg.seed)
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    lon, lat, classes = data["lon"], data["lat"], data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    rng = np.random.default_rng(cfg.seed)
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        t0 = time.perf_counter()
        keep = du.clean_stale_class_mask(
            X[tr], y_enc[tr], None, cls12_enc, lon[tr], lat[tr])
        tr_use = tr[keep]
        Xtr, ytr = X[tr_use], y_train_all[tr_use]
        if Xa is not None and len(Xa):
            Xtr = np.concatenate([Xtr, Xa], 0)
            ytr = np.concatenate([ytr, ya], 0)
        ens = C.fit_ensemble(Xtr, ytr, n_classes, cfg, data["feat_cols"],
                             classes, data.get("lidar_med"), rng)
        pred = ens.predict_proba(X[te]).argmax(1)
        f1s.append(du.macro_f1(y_enc[te], pred, n_classes))
        pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        if verbose:
            print(f"    fold {k}: F1={f1s[-1]:.4f}  n_train={len(Xtr):,}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    return {"f1_mean": float(np.mean(f1s)),
            "f1_per_fold": [round(v, 4) for v in f1s],
            "f1_per_class": {str(c): round(float(v), 4)
                             for c, v in zip(classes, np.mean(pcs, 0))}}


def paired(label_a, label_b, results):
    """Per-seed deltas — means hide pairing (see README: use PAIRED deltas)."""
    d = np.array([results[label_b][s]["f1_mean"] - results[label_a][s]["f1_mean"]
                  for s in results[label_a]])
    return {"per_seed": [round(float(v), 4) for v in d],
            "mean": round(float(d.mean()), 4),
            "sd": round(float(d.std(ddof=1)) if len(d) > 1 else 0.0, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="both",
                    choices=["relabel", "augment", "both"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n_aug", type=int, default=5000)
    ap.add_argument("--extra", default="lidar")
    ap.add_argument("--relabel_cls12", default="to12_fix")
    ap.add_argument("--out", default=str(result_path("v2_experiments.json")))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    data = C.load_cached(args.extra)
    print(f"frame: {data['X'].shape[0]:,} rows x {data['X'].shape[1]} feats, "
          f"{len(data['classes'])} classes", flush=True)

    y_v1, n12 = apply_cls12_relabel(data["y_enc"], data["classes"],
                                    args.relabel_cls12)
    print(f"cls12 relabel ({args.relabel_cls12}): {n12} labels changed")
    y_v2_raw, n_ch = v2_label_vector(data)
    print(f"v2 relabel: {n_ch} of {len(y_v2_raw):,} frame rows change "
          f"({n_ch/len(y_v2_raw):.5%})", flush=True)
    # v2 labels also get the cls12 fix, so the arms differ ONLY by the v2 edit
    y_v2, _ = apply_cls12_relabel(y_v2_raw, data["classes"], args.relabel_cls12)
    print(f"arms differ at {int((y_v1 != y_v2).sum())} rows", flush=True)

    arms = {"v1": (y_v1, None, None)}
    if args.exp in ("relabel", "both"):
        arms["v2"] = (y_v2, None, None)
    if args.exp in ("augment", "both"):
        Xa, ya = load_augmentation(data, args.n_aug, seeds[0])
        print(f"augmentation: {len(Xa):,} newly-built (13->10) rows", flush=True)
        arms["v1+aug"] = (y_v1, Xa, ya)

    results = {k: {} for k in arms}
    for seed in seeds:
        for name, (yt, Xa, ya) in arms.items():
            cfg = C.Config()
            cfg.seed = seed
            print(f"\n[seed {seed}] arm={name}", flush=True)
            results[name][str(seed)] = run_cv(data, cfg, yt, Xa, ya)
            print(f"  -> F1 {results[name][str(seed)]['f1_mean']:.4f}", flush=True)

    report = {"seeds": seeds, "n_rows_changed_by_v2": n_ch,
              "n_aug": int(len(arms["v1+aug"][1])) if "v1+aug" in arms else 0,
              "arms": results, "paired_deltas": {}}
    for name in arms:
        if name == "v1":
            continue
        report["paired_deltas"][f"{name} - v1"] = paired("v1", name, results)

    print("\n" + "=" * 62)
    for name in arms:
        f1 = [results[name][str(s)]["f1_mean"] for s in seeds]
        print(f"{name:<10} F1 per seed: "
              + "  ".join(f"{v:.4f}" for v in f1)
              + f"   mean {np.mean(f1):.4f}")
    print("-" * 62)
    for k, v in report["paired_deltas"].items():
        print(f"paired Δ {k:<16} {v['mean']:+.4f} (sd {v['sd']:.4f})  "
              f"per-seed {v['per_seed']}")
    print("noise floor from prior work: sd ~0.0013 — treat |Δ| < 0.003 as null")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[OK] wrote {args.out}")


if __name__ == "__main__":
    main()

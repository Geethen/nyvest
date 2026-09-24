"""Audit the deployed DNN against the grunnkart v2 label correction.

The v2 relabel (FKB Bygning 2018) moves 118,968 pixels into class 10 (built);
86% of them came from class 13 ("other"), which FSCS sampling excludes, so
only 3 of the 74,639 training points change label. Retraining is therefore a
no-op by construction. The informative question is the opposite one:

    On the pixels v2 newly calls built, what did the model already predict?

High built-rate => the model was already right and v1 was wrong (v2 confirms
the model; the v1-scored metric understated built accuracy). Low built-rate
=> these are genuinely missed scattered rural buildings, i.e. real headroom.

Three strata, all scored with the same deployed ensemble:
  changed_13to10  v1=13 -> v2=10   (the correction itself)
  changed_other   v1 in 1..12 -> v2=10
  ctrl_13         v1=13, unchanged (what "other" looks like to the model)
  ctrl_10         v1=10, unchanged (model's rate on established built)

Usage
  ~/myprojects/recover/.venv/bin/python DNN/audit_v2.py
  ~/myprojects/recover/.venv/bin/python DNN/audit_v2.py --n_ctrl 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_core as C  # noqa: E402
from dnn_paths import result_path  # noqa: E402
from extract_v2_changes import (AEF_SCALE, AEF_VRT, EMBED_COLS, LIDAR_COLS,  # noqa: E402
                                LIDAR_TIF, RASTER_V1, RASTER_V2,
                                attach_embeddings, check_embedding_scale)

_REPO = Path(__file__).resolve().parents[1]
LABELS = {2: "rock+sand", 3: "crop", 4: "forest", 5: "grassland", 6: "scrub",
          7: "wetland", 8: "water", 10: "built", 11: "sparse-veg",
          12: "snow/ice"}
BUILT = 10


def sample_strata(n_ctrl, seed, step=4000):
    """One pass over v1/v2 collecting changed pixels plus reservoir-style
    control samples of unchanged class-13 and class-10 pixels."""
    rng = np.random.default_rng(seed)
    changed, c13, c10 = [], [], []
    with rasterio.open(RASTER_V1) as s1, rasterio.open(RASTER_V2) as s2:
        T = s1.transform
        n_blocks = int(np.ceil(s1.height / step))
        per_blk = max(1, int(np.ceil(n_ctrl / n_blocks)))
        for r0 in range(0, s1.height, step):
            h = min(step, s1.height - r0)
            w = Window(0, r0, s1.width, h)
            a = s1.read(1, window=w)
            b = s2.read(1, window=w)
            same = a == b
            rr, cc = np.nonzero(~same)
            if len(rr):
                changed.append(pd.DataFrame({
                    "row": rr + r0, "col": cc,
                    "class_v1": a[rr, cc], "class_v2": b[rr, cc]}))
            for code, sink in ((13, c13), (BUILT, c10)):
                rr, cc = np.nonzero(same & (a == code))
                if not len(rr):
                    continue
                take = rng.choice(len(rr), size=min(per_blk, len(rr)),
                                  replace=False)
                sink.append(pd.DataFrame({
                    "row": rr[take] + r0, "col": cc[take],
                    "class_v1": code, "class_v2": code}))

    def finish(parts, stratum):
        df = pd.concat(parts, ignore_index=True)
        df["stratum"] = stratum
        return df

    ch = finish(changed, "changed")
    ch["stratum"] = np.where(ch["class_v1"] == 13,
                             "changed_13to10", "changed_other")
    out = pd.concat([ch, finish(c13, "ctrl_13"), finish(c10, "ctrl_10")],
                    ignore_index=True)
    out["x"] = T.c + (out["col"].values + 0.5) * T.a
    out["y"] = T.f + (out["row"].values + 0.5) * T.e
    return out


def built_density(df, k=5, batch=8192):
    """Fraction of v2 class-10 pixels in a k x k neighbourhood of each point.

    A scattered rural building occupies one or two 10 m pixels and is
    spectrally dominated by whatever surrounds it, so an isolated built pixel
    may simply not be recognisable from a 10 m embedding. Splitting the audit
    by local built density separates 'model missed it' from 'not learnable at
    this resolution'.
    """
    half = k // 2
    out = np.full(len(df), np.nan, dtype=np.float32)
    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()
    # row-band major, column-band minor, so each batch reads a compact window
    order = np.lexsort((cols // 512, rows // 512))
    with rasterio.open(RASTER_V2) as src:
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            r0 = max(0, rows[sel].min() - half)
            r1 = min(src.height, rows[sel].max() + half + 1)
            c0 = max(0, cols[sel].min() - half)
            c1 = min(src.width, cols[sel].max() + half + 1)
            m = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)) == BUILT
            rr = rows[sel] - r0
            cc = cols[sel] - c0
            # k*k direct gathers: exact integer counts, no summed-area table
            # (a float32 SAT over a window this size loses the low bits, and
            # the box sums are differences of large numbers).
            tot = np.zeros(len(sel), dtype=np.int32)
            for di in range(-half, half + 1):
                ri = np.clip(rr + di, 0, m.shape[0] - 1)
                for dj in range(-half, half + 1):
                    cj = np.clip(cc + dj, 0, m.shape[1] - 1)
                    tot += m[ri, cj]
            out[sel] = tot / float(k * k)
            print(f"    density {min(i + batch, len(order)):,}/{len(order):,}",
                  flush=True)
    return out


def subsample(df, n_per, seed):
    picks = []
    for _, g in df.groupby("stratum"):
        picks.append(g.sample(min(len(g), n_per), random_state=seed).index)
    return df.loc[np.concatenate(picks)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_ctrl", type=int, default=20000,
                    help="control pixels sampled per control stratum")
    ap.add_argument("--n_per", type=int, default=20000,
                    help="cap per stratum after sampling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=str(_REPO / "models" / "dnn_final.pt"))
    ap.add_argument("--out", default=str(result_path("v2_audit.json")))
    args = ap.parse_args()

    print("scanning rasters for strata...", flush=True)
    df = sample_strata(args.n_ctrl, args.seed)
    print(df["stratum"].value_counts().to_string())
    df = subsample(df, args.n_per, args.seed)
    print(f"  after cap: {len(df):,} rows", flush=True)

    print("computing local built density...", flush=True)
    df["built_frac"] = built_density(df)
    print("sampling AlphaEarth 2024...", flush=True)
    emb = attach_embeddings(df, AEF_VRT, 64, EMBED_COLS, AEF_SCALE)
    check_embedding_scale(emb)
    print("sampling lidar...", flush=True)
    lid = attach_embeddings(df, LIDAR_TIF, 3, LIDAR_COLS, 1.0,
                            zero_is_gap=False)
    df = pd.concat([df.reset_index(drop=True), emb, lid], axis=1)

    ok = df[EMBED_COLS].notna().all(axis=1)
    print(f"  usable (valid embeddings): {ok.sum():,} / {len(df):,}")
    df = df[ok].reset_index(drop=True)

    ens = C.Ensemble.load(args.model)
    med = ens.lidar_med or {}
    for c in LIDAR_COLS:
        n_fill = int(df[c].isna().sum())
        df[c] = df[c].fillna(med.get(c, 0.0))
        if n_fill:
            print(f"  median-filled {c}: {n_fill:,} rows")

    X = df[ens.feat_cols].to_numpy(np.float32)
    print(f"predicting {len(X):,} rows...", flush=True)
    proba = ens.predict_proba(X)
    pred = np.array(ens.classes)[proba.argmax(1)]
    df["pred"] = pred
    df["p_built"] = proba[:, ens.classes.index(BUILT)]

    report = {"model": args.model, "n_ctrl": args.n_ctrl, "seed": args.seed,
              "strata": {}}
    print(f"\n{'stratum':<16}{'n':>8}{'built%':>9}{'p_built':>9}   top predictions")
    print("-" * 78)
    for s, g in df.groupby("stratum"):
        vc = g["pred"].value_counts(normalize=True)
        top = ", ".join(f"{LABELS.get(int(k), k)} {v:.1%}"
                        for k, v in vc.head(4).items())
        built_rate = float((g["pred"] == BUILT).mean())
        print(f"{s:<16}{len(g):>8,}{built_rate:>8.1%}"
              f"{g['p_built'].mean():>9.3f}   {top}")
        report["strata"][s] = {
            "n": int(len(g)), "built_rate": round(built_rate, 4),
            "mean_p_built": round(float(g["p_built"].mean()), 4),
            "pred_dist": {LABELS.get(int(k), str(k)): round(float(v), 4)
                          for k, v in vc.items()},
        }

    # Is a missed pixel a model error, or a building too small to see at 10 m?
    print(f"\nbuilt-rate by local built density (5x5 neighbourhood)")
    print(f"{'stratum':<16}{'density bin':<14}{'n':>8}{'built%':>9}")
    print("-" * 50)
    bins = [0, 0.12, 0.28, 0.52, 1.01]
    names = ["isolated (<3/25)", "sparse (3-6/25)", "clustered (7-12/25)",
             "dense (>12/25)"]
    df["dens_bin"] = pd.cut(df["built_frac"], bins=bins, labels=names,
                            right=False, include_lowest=True)
    dens = {}
    for s in ("changed_13to10", "changed_other", "ctrl_10"):
        g0 = df[df["stratum"] == s]
        if not len(g0):
            continue
        dens[s] = {}
        for b, g in g0.groupby("dens_bin", observed=True):
            r = float((g["pred"] == BUILT).mean())
            print(f"{s:<16}{str(b):<14}{len(g):>8,}{r:>8.1%}")
            dens[s][str(b)] = {"n": int(len(g)), "built_rate": round(r, 4)}
    report["built_rate_by_density"] = dens

    pts = str(Path(args.out).with_name("v2_audit_points.parquet"))
    keep = ["stratum", "class_v1", "class_v2", "x", "y", "built_frac",
            "pred", "p_built"]
    df[keep].to_parquet(pts, index=False)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[OK] wrote {args.out}")
    print(f"[OK] wrote {pts}")


if __name__ == "__main__":
    main()

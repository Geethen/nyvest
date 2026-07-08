"""Probe: does adding field-mapped NiN_v2 points to TRAINING lift macro-F1?

Mirrors feature_probe_lidar.py's CV contract exactly (same source parquet, label
merge 1->2 / 9->8, 3-fold GroupKFold on cell_id, macro-F1) but injects the NiN
training points (scripts/extraction/nin_make_points.py +
nin_extract_alphaearth.py -> data/nin_alphaearth.parquet) into the TRAIN side of
each fold only.

LEAK-SAFETY: NiN points share the same 25 km cell_id grid as the grunnkart data.
GroupKFold holds out whole cells, so the test set is ALWAYS pure grunnkart. NiN
points whose cell_id falls in the held-out fold are DROPPED from that fold's train
(they belong to a test cell) — they are never evaluated and never leak across the
spatial block. This makes the comparison honest: same grunnkart test set, only the
training data differs.

NiN only labels vegetation classes (4,5,6,7 + a little 2), so we expect movement
on those; report per-class deltas there.

Usage:
  python scripts/feature_probe_nin.py                 # embed + embed+nin
  python scripts/feature_probe_nin.py --features lidar3  # + lidar3 on both sides
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
REPORTS = REPO / "reports"
STABLE = DATA / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"
NIN = DATA / "nin_alphaearth.parquet"
LIDAR = DATA / "lidar_features.parquet"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
LIDAR3 = ["elevation", "tri", "tch"]
MERGE_MAP = {1: 2, 9: 8}
N_FOLDS = 3
SEED = 0
NIN_CLASSES = (2, 4, 5, 6, 7)  # classes NiN actually labels


def _load_embed(path: Path) -> pd.DataFrame:
    sel = ", ".join([f'CAST("{c}" AS FLOAT) AS "{c}"' for c in EMBED_COLS]
                    + ['"class"', '"cell_id"', '"lon"', '"lat"'])
    df = duckdb.sql(f"SELECT {sel} FROM '{path}'").df()
    return df.dropna(subset=EMBED_COLS + ["class"]).reset_index(drop=True)


def _join_lidar(df: pd.DataFrame) -> pd.DataFrame:
    lid = (duckdb.sql(f"SELECT lon,lat,{','.join(LIDAR3)} FROM '{LIDAR}'")
           .df().drop_duplicates(["lon", "lat"]))
    df = df.merge(lid, on=["lon", "lat"], how="left")
    for c in LIDAR3:
        df[c] = df[c].fillna(df[c].median()).astype(np.float32)
    return df


def load(with_lidar: bool):
    stable = _load_embed(STABLE)
    stable["_src"] = "grunnkart"
    nin = _load_embed(NIN)
    nin["_src"] = "nin"
    both = pd.concat([stable, nin], ignore_index=True)
    if with_lidar:
        both = _join_lidar(both)
    both["_y"] = pd.Series(both["class"]).replace(MERGE_MAP).astype(int).values
    return both


def run(df, feat_cols, le, use_nin: bool, nin_keep_classes=None, nin_cap=None):
    """3-fold GroupKFold on cell_id. Test = grunnkart only. Train += NiN
    iff use_nin and the NiN point's cell is NOT in the held-out fold.

    nin_keep_classes: if set, only add NiN points whose FSCS class is in this set
    (e.g. drop the label-mismatched cls6/cls7). nin_cap: max NiN points per class
    per fold (class-balancing so cls6 doesn't flood)."""
    grunn = df[df["_src"] == "grunnkart"].reset_index(drop=True)
    nin = df[df["_src"] == "nin"].reset_index(drop=True)
    if nin_keep_classes is not None:
        nin = nin[nin["_y"].isin(nin_keep_classes)].reset_index(drop=True)
    y_g = le.transform(grunn["_y"].values)
    groups = grunn["cell_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(grunn[feat_cols], y_g, groups)):
        test_cells = set(grunn.iloc[te]["cell_id"].unique())
        X_tr = [grunn.iloc[tr][feat_cols].values]
        y_tr = [y_g[tr]]
        if use_nin:
            nin_tr = nin[~nin["cell_id"].isin(test_cells)]
            if nin_cap is not None and len(nin_tr):
                idx = (nin_tr.groupby("_y")
                       .sample(frac=1.0, random_state=SEED)  # shuffle in-group
                       .groupby("_y").head(nin_cap).index)
                nin_tr = nin_tr.loc[idx]
            if len(nin_tr):
                X_tr.append(nin_tr[feat_cols].values)
                y_tr.append(le.transform(nin_tr["_y"].values))
        X_tr = np.vstack(X_tr)
        y_tr = np.concatenate(y_tr)
        m = CatBoostClassifier(
            iterations=600, depth=8, learning_rate=0.06,
            loss_function="MultiClass", task_type="GPU", devices="0",
            random_seed=SEED, verbose=False)
        m.fit(X_tr, y_tr)
        pred = m.predict(grunn.iloc[te][feat_cols].values).ravel().astype(int)
        per = f1_score(y_g[te], pred, average=None,
                       labels=np.arange(len(le.classes_)))
        rows.append({
            "macro_f1": f1_score(y_g[te], pred, average="macro"),
            "bal_acc": balanced_accuracy_score(y_g[te], pred),
            "per_class": per.tolist(),
            "n_nin_train": int(len(nin_tr)) if use_nin else 0,
        })
    return rows


def summarize(rows, le):
    macro = np.array([r["macro_f1"] for r in rows])
    per = np.array([r["per_class"] for r in rows]).mean(0)
    return {
        "macro_f1_mean": float(macro.mean()), "macro_f1_std": float(macro.std()),
        "bal_acc_mean": float(np.mean([r["bal_acc"] for r in rows])),
        "n_nin_train_mean": float(np.mean([r["n_nin_train"] for r in rows])),
        "per_class_f1": {int(c): round(float(v), 4)
                         for c, v in zip(le.classes_, per)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="embed",
                    choices=["embed", "lidar3"],
                    help="embed = 64 bands; lidar3 = + elevation,tri,tch")
    args = ap.parse_args()
    with_lidar = args.features == "lidar3"
    feat_cols = EMBED_COLS + (LIDAR3 if with_lidar else [])

    print("loading grunnkart + NiN ...", flush=True)
    df = load(with_lidar)
    le = LabelEncoder().fit(df["_y"].values)
    ng = (df["_src"] == "grunnkart").sum()
    nn = (df["_src"] == "nin").sum()
    print(f"  grunnkart={ng:,}  nin={nn:,}  classes={len(le.classes_)} "
          f"feat={len(feat_cols)}")

    # NiN cls6/cls7 sit nearer grunnkart's cls5 centroid than their own (label-
    # definition mismatch), so try restricting to the agreeing classes and
    # class-balancing the flood of cls6 heathland points.
    strategies = {
        "grunnkart": dict(use_nin=False),
        "+nin_all": dict(use_nin=True),
        "+nin_cap2k": dict(use_nin=True, nin_cap=2000),
        "+nin_245": dict(use_nin=True, nin_keep_classes={2, 4, 5}),
        "+nin_245_cap2k": dict(use_nin=True, nin_keep_classes={2, 4, 5},
                               nin_cap=2000),
        "+nin_2457_cap2k": dict(use_nin=True, nin_keep_classes={2, 4, 5, 7},
                                nin_cap=2000),
    }
    results = {}
    for name, kw in strategies.items():
        results[name] = summarize(run(df, feat_cols, le, **kw), le)
        s = results[name]
        print(f"{name:18s} macroF1={s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f}"
              f"  bal_acc={s['bal_acc_mean']:.4f}  nin_train={s['n_nin_train_mean']:.0f}",
              flush=True)

    sb = results["grunnkart"]
    print("\nmacro-F1 delta vs grunnkart baseline:")
    for name, s in results.items():
        if name == "grunnkart":
            continue
        print(f"  {name:18s} {s['macro_f1_mean'] - sb['macro_f1_mean']:+.4f}")

    # per-class table across all strategies for the NiN-labelled classes
    print("\nper-class F1 by strategy (NiN-labelled classes):")
    hdr = "  cls " + "".join(f"{n[:14]:>15s}" for n in results)
    print(hdr)
    for c in NIN_CLASSES:
        row = f"  {c:<4d}" + "".join(
            f"{results[n]['per_class_f1'].get(c, float('nan')):>15.4f}"
            for n in results)
        print(row)

    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame([{"set": n, **s} for n, s in results.items()]).to_csv(
        REPORTS / f"feature_probe_nin_{args.features}.csv", index=False)
    print(f"\nsaved {REPORTS / f'feature_probe_nin_{args.features}.csv'}")


if __name__ == "__main__":
    main()

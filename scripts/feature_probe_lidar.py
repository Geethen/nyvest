"""Fast CatBoost probe for lidar/dem feature selection on stable-allyears.

Mirrors the CV contract of the real TabICL pipeline
(common_ground/scripts/llto_schemeB_allyears.py) so feature comparisons transfer,
but swaps the expensive CC-APS + TabICL stage for a single CatBoost classifier
that trains in ~1 min/fold on the A40. This is the cheap-probe step from the
dem-features workflow: search feature sets here, then run only the winner through
the full TabICL pipeline.

What is held identical to the pipeline:
  * source: data/grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet
  * label merge 1->2, 9->8
  * 3-fold GroupKFold on cell_id (spatial blocking)
  * macro-F1 (+ per-class F1, balanced accuracy)

What differs (kept simple on purpose):
  * model = CatBoost (GPU), no CC-APS noise correction, no stable-allyears dedup,
    no TabICL. Absolute F1 will sit below the 0.7074 TabICL number; we only use
    DELTAS between feature sets to rank them.

Feature sets (column groups joined by exact (lon, lat)):
  embed   : A00..A63                                  (64 AlphaEarth bands)
  lidar   : elevation, tch, slope, aspect_sin, aspect_cos, tri
  dem     : elevation_dem, slope_dem                  (Copernicus GLO30)

Usage:
  python scripts/feature_probe_lidar.py --sets embed,embed+lidar,embed+dem,embed+lidar+dem
  python scripts/feature_probe_lidar.py --importance embed+lidar   # ranking only
  python scripts/feature_probe_lidar.py --subsets embed+lidar --topk 16,32,48
Outputs: reports/feature_probe_lidar.csv (one row per feature set x fold + means)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
REPORTS = REPO / "reports"
STABLE = DATA / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"
LIDAR = DATA / "lidar_features.parquet"
DEM = DATA / "dem_features.parquet"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
LIDAR_COLS = ["elevation", "tch", "slope", "aspect_sin", "aspect_cos", "tri"]
DEM_COLS = ["elevation_dem", "slope_dem"]
MERGE_MAP = {1: 2, 9: 8}
N_FOLDS = 3
SEED = 0


def load() -> pd.DataFrame:
    sel = ", ".join([f'CAST("{c}" AS FLOAT) AS "{c}"' for c in EMBED_COLS]
                    + ['"class"', '"cell_id"', '"lon"', '"lat"'])
    df = duckdb.sql(f"SELECT {sel} FROM '{STABLE}'").df()
    df = df.dropna(subset=EMBED_COLS + ["class"]).reset_index(drop=True)
    # lidar join (median-impute the ~24% uncovered rows, as the pipeline does)
    lid = (duckdb.sql(f"SELECT lon,lat,{','.join(LIDAR_COLS)} FROM '{LIDAR}'")
           .df().drop_duplicates(["lon", "lat"]))
    df = df.merge(lid, on=["lon", "lat"], how="left")
    for c in LIDAR_COLS:
        df[c] = df[c].fillna(df[c].median()).astype(np.float32)
    # dem join, renamed to avoid colliding with lidar elevation/slope
    dem = (duckdb.sql(f"SELECT lon,lat,elevation AS elevation_dem,"
                      f"slope AS slope_dem FROM '{DEM}'")
           .df().drop_duplicates(["lon", "lat"]))
    df = df.merge(dem, on=["lon", "lat"], how="left")
    for c in DEM_COLS:
        df[c] = df[c].fillna(df[c].median()).astype(np.float32)
    df["_y"] = pd.Series(df["class"]).replace(MERGE_MAP).astype(int).values
    return df


def cols_for(spec: str) -> list[str]:
    """spec like 'embed+lidar' -> column list."""
    groups = {"embed": EMBED_COLS, "lidar": LIDAR_COLS, "dem": DEM_COLS,
              # lean lidar: drop aspect_sin/aspect_cos (≈0 importance) and slope
              "lidar3": ["elevation", "tri", "tch"],
              "lidar4": ["elevation", "tri", "tch", "slope"]}
    out: list[str] = []
    for part in spec.split("+"):
        out += groups[part]
    return out


def run_set(df: pd.DataFrame, feat_cols: list[str], le: LabelEncoder,
            return_importance: bool = False):
    y = le.transform(df["_y"].values)
    groups = df["cell_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []
    imp_accum = np.zeros(len(feat_cols))
    for fold, (tr, te) in enumerate(gkf.split(df[feat_cols], y, groups)):
        m = CatBoostClassifier(
            iterations=600, depth=8, learning_rate=0.06,
            loss_function="MultiClass", task_type="GPU", devices="0",
            random_seed=SEED, verbose=False,
        )
        m.fit(df.iloc[tr][feat_cols].values, y[tr])
        pred = m.predict(df.iloc[te][feat_cols].values).ravel().astype(int)
        macro = f1_score(y[te], pred, average="macro")
        bal = balanced_accuracy_score(y[te], pred)
        per = f1_score(y[te], pred, average=None,
                       labels=np.arange(len(le.classes_)))
        rows.append({"fold": fold, "macro_f1": macro, "bal_acc": bal,
                     "per_class": per.tolist()})
        if return_importance:
            imp_accum += m.get_feature_importance(
                Pool(df.iloc[te][feat_cols].values, y[te]),
                type="LossFunctionChange")
    if return_importance:
        return rows, imp_accum / N_FOLDS
    return rows


def summarize(rows, le) -> dict:
    macro = np.array([r["macro_f1"] for r in rows])
    per = np.array([r["per_class"] for r in rows]).mean(0)
    return {
        "macro_f1_mean": float(macro.mean()), "macro_f1_std": float(macro.std()),
        "bal_acc_mean": float(np.mean([r["bal_acc"] for r in rows])),
        "per_class_f1": {int(c): round(float(v), 4)
                         for c, v in zip(le.classes_, per)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="embed,embed+lidar,embed+dem,embed+lidar+dem")
    ap.add_argument("--importance", default="",
                    help="feature spec to rank by LossFunctionChange importance")
    ap.add_argument("--subsets", default="",
                    help="feature spec to subset by top-k embedding importance")
    ap.add_argument("--topk", default="16,32,48")
    args = ap.parse_args()

    print("loading + joining ...", flush=True)
    t0 = time.time()
    df = load()
    le = LabelEncoder().fit(df["_y"].values)
    print(f"  {len(df):,} rows, {len(le.classes_)} classes "
          f"({time.time()-t0:.0f}s)", flush=True)

    if args.importance:
        cols = cols_for(args.importance)
        rows, imp = run_set(df, cols, le, return_importance=True)
        order = np.argsort(imp)[::-1]
        print(f"\n=== importance ({args.importance}), "
              f"macro-F1={summarize(rows, le)['macro_f1_mean']:.4f} ===")
        rank = [{"feature": cols[i], "importance": round(float(imp[i]), 4)}
                for i in order]
        for r in rank:
            print(f"  {r['feature']:14s} {r['importance']:+.4f}")
        (REPORTS / "feature_probe_importance.json").write_text(
            json.dumps(rank, indent=2))
        return

    if args.subsets:
        base = cols_for(args.subsets)
        # rank embeddings within this set, then build top-k embedding subsets
        rows, imp = run_set(df, base, le, return_importance=True)
        emb_idx = [i for i, c in enumerate(base) if c in EMBED_COLS]
        extra = [c for c in base if c not in EMBED_COLS]
        emb_ranked = [base[i] for i in np.argsort(imp)[::-1] if i in emb_idx]
        results = []
        for k in [int(x) for x in args.topk.split(",")]:
            cols = emb_ranked[:k] + extra
            rr = run_set(df, cols, le)
            s = summarize(rr, le)
            print(f"top{k}_embed+{'+'.join(set(c for c in extra))[:20]}: "
                  f"macro-F1={s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f}")
            results.append({"set": f"top{k}_{args.subsets}", "k": k, **s})
        pd.DataFrame(results).to_csv(
            REPORTS / "feature_probe_subsets.csv", index=False)
        return

    out = []
    for spec in args.sets.split(","):
        cols = cols_for(spec)
        t = time.time()
        rows = run_set(df, cols, le)
        s = summarize(rows, le)
        print(f"{spec:22s} macro-F1={s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f} "
              f"bal_acc={s['bal_acc_mean']:.4f}  ({time.time()-t:.0f}s, {len(cols)} feat)",
              flush=True)
        out.append({"set": spec, "n_feat": len(cols), **s})
    df_out = pd.DataFrame(out)
    df_out.to_csv(REPORTS / "feature_probe_lidar.csv", index=False)
    print(f"\nsaved {REPORTS / 'feature_probe_lidar.csv'}")
    # quick per-class delta vs embed baseline on the confusable classes
    if "embed" in df_out["set"].values:
        base = df_out[df_out["set"] == "embed"].iloc[0]["per_class_f1"]
        print("\nper-class F1 delta vs embed (classes 3,5,6,7,11,12):")
        for _, r in df_out.iterrows():
            if r["set"] == "embed":
                continue
            d = {c: round(r["per_class_f1"][c] - base[c], 4)
                 for c in (3, 5, 6, 7, 11, 12) if c in base}
            print(f"  {r['set']:22s} {d}")


if __name__ == "__main__":
    main()

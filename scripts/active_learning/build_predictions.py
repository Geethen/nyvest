"""Score the seed model over all unique sample locations -> predictions.parquet.

Produces the table that seeds the active-learning queue and the map's point layer:
one row per unique (lon, lat) sample location with the grunnkart label, the seed
model's prediction, the APS prediction-set size (uncertainty / acquisition score),
and the max class probability. Also callable from the app to refresh after a retrain.

Output: data/active_learning/predictions.parquet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_core as mc

PRED_PARQUET = mc.AL_DATA_DIR / "predictions.parquet"


def build_predictions(seed: mc.SeedModel | None = None,
                      out_path: Path = PRED_PARQUET,
                      batch: int = 50_000) -> pd.DataFrame:
    """Score seed model over deduped sample locations; write + return the table."""
    if seed is None:
        seed = mc.load_seed_model()
    df = mc.dedup_to_latest_year(mc.load_stable())
    grunnkart = mc.merge_classes(df[mc.TARGET].values)
    X = df[mc.FEATURE_COLS].values.astype(np.float32)

    preds = np.empty(len(df), dtype=int)
    sizes = np.empty(len(df), dtype=int)
    pmax = np.empty(len(df), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(0, len(X), batch):
        sl = slice(i, i + batch)
        p, s, probs = seed.predict(X[sl])
        preds[sl] = p
        sizes[sl] = s
        pmax[sl] = probs.max(axis=1)
    dt = time.perf_counter() - t0

    out = pd.DataFrame({
        "lon": df["lon"].values, "lat": df["lat"].values,
        "grunnkart": grunnkart.astype(int),
        "pred": preds, "set_size": sizes, "p_max": np.round(pmax, 4),
        "disagree": (preds != grunnkart),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"scored {len(out):,} locations in {dt:.1f}s -> {out_path}")
    print(f"  mean set-size={out.set_size.mean():.2f}  "
          f"singletons={100*(out.set_size==1).mean():.1f}%  "
          f"model-vs-grunnkart disagree={100*out.disagree.mean():.1f}%")
    return out


if __name__ == "__main__":
    build_predictions()

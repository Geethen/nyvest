"""Extract the grunnkart v1->v2 changed pixels and attach AlphaEarth embeddings.

Every changed pixel becomes class 10 (built) in v2; 86% of them were class 13
("other" = settlements & artificial areas), which FSCS sampling excludes
entirely. So these pixels carry label information the training set has never
seen, and they are the only place the v2 relabel can be measured.

Writes data/v2_changes.parquet: x, y (EPSG:25832), lon, lat, class_v1,
class_v2, A00..A63 (2024 AlphaEarth, NaN where the VRT has no data).

Usage
  ~/myprojects/recover/.venv/bin/python DNN/extract_v2_changes.py
  ~/myprojects/recover/.venv/bin/python DNN/extract_v2_changes.py --max_points 20000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

_REPO = Path(__file__).resolve().parents[1]
DATA_DIR = _REPO / "data"
OUT_PARQUET = DATA_DIR / "v2_changes.parquet"

_ROOT = Path(os.environ.get("NYVEST_DATA_DIR", "/data/P-Prosjekter2/154001_nyvest"))
_GIS = _ROOT / "GIS" / "NIBIO" / "Version_2" / "rasterized_10m"
RASTER_V1 = _GIS / "grunnkart_nyvest_10m.tif"
RASTER_V2 = _GIS / "grunnkart_nyvest_10m_v2.tif"
AEF_VRT = _ROOT / "landcover_Geethen" / "data" / "aef_2024.vrt"
LIDAR_TIF = _ROOT / "landcover_Geethen" / "data" / "lidar_3band.tif"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
LIDAR_COLS = ["elevation", "tri", "tch"]
# aef_2024.vrt is float32 and ALREADY unit-normalised (measured L2/px = 1.0001,
# identical to the training parquet), so no rescale. Do not confuse this with
# the int-scaled source.coop / GEE-export tiles that predict_raster.py takes
# `--scale 1000` for — applying that here drives every feature to ~0.
AEF_SCALE = 1.0


def find_changed_pixels(step=4000):
    """Scan v1/v2 in row blocks, return DataFrame of changed pixels."""
    recs = []
    with rasterio.open(RASTER_V1) as s1, rasterio.open(RASTER_V2) as s2:
        if s1.transform != s2.transform or s1.shape != s2.shape:
            raise RuntimeError("v1/v2 grids differ — cannot diff pixelwise")
        T = s1.transform
        for r0 in range(0, s1.height, step):
            h = min(step, s1.height - r0)
            w = Window(0, r0, s1.width, h)
            a = s1.read(1, window=w)
            b = s2.read(1, window=w)
            rr, cc = np.nonzero(a != b)
            if len(rr) == 0:
                continue
            recs.append(pd.DataFrame({
                "row": rr + r0, "col": cc,
                "class_v1": a[rr, cc], "class_v2": b[rr, cc]}))
    df = pd.concat(recs, ignore_index=True)
    # pixel centres in EPSG:25832
    df["x"] = T.c + (df["col"].values + 0.5) * T.a
    df["y"] = T.f + (df["row"].values + 0.5) * T.e
    return df


def attach_embeddings(df, path=AEF_VRT, n_bands=64, col_names=None, scale=1.0,
                      zero_is_gap=True, batch=4096):
    """Sample a multi-band raster at each point (reprojected to the raster's
    CRS). Rows outside bounds / on nodata come back as NaN."""
    col_names = col_names or EMBED_COLS
    with rasterio.open(path) as src:
        tf = Transformer.from_crs("EPSG:25832", src.crs, always_xy=True)
        xs, ys = tf.transform(df["x"].values, df["y"].values)
        inv = ~src.transform
        cols, rows = inv * (xs, ys)
        rows = np.floor(rows).astype(np.int64)
        cols = np.floor(cols).astype(np.int64)
        inside = ((rows >= 0) & (rows < src.height)
                  & (cols >= 0) & (cols < src.width))
        print(f"  inside VRT bounds: {inside.sum():,} / {len(df):,}")

        out = np.full((len(df), n_bands), np.nan, dtype=np.float32)
        idx = np.flatnonzero(inside)
        # Order by 2-D tile, not by row: sorting on row alone makes every batch
        # span the raster's full width, so each read pulls a huge window and
        # the sampler crawls. Blocking on (row, col) keeps windows compact.
        tile = 512
        order = idx[np.lexsort((cols[idx] // tile, rows[idx] // tile))]
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            r0, r1 = rows[sel].min(), rows[sel].max() + 1
            c0, c1 = cols[sel].min(), cols[sel].max() + 1
            # guard against a pathologically wide window
            if (r1 - r0) * (c1 - c0) > 60_000_000:
                for j in sel:
                    win = Window(cols[j], rows[j], 1, 1)
                    out[j] = src.read(window=win).reshape(n_bands)
                continue
            blk = src.read(window=Window(c0, r0, c1 - c0, r1 - r0))
            out[sel] = blk[:, rows[sel] - r0, cols[sel] - c0].T
            if (i // batch) % 20 == 0:
                print(f"    {min(i + batch, len(order)):,}/{len(order):,}",
                      flush=True)
        nod = src.nodata
    if nod is not None and np.isfinite(nod):
        out[out == nod] = np.nan
    if zero_is_gap:
        # all-zero reads across every band are VRT gaps, not real embeddings
        out[np.all(out == 0, axis=1)] = np.nan
    out /= scale
    return pd.DataFrame(out, columns=col_names, index=df.index)


def check_embedding_scale(emb, tol=0.15):
    """AlphaEarth embeddings are unit-norm per pixel; the training parquet
    measures L2 = 1.000. Anything else means a rescale bug, which silently
    produces near-zero features and meaningless predictions."""
    l2 = np.linalg.norm(emb[EMBED_COLS].to_numpy(np.float64), axis=1)
    l2 = l2[np.isfinite(l2) & (l2 > 0)]
    med = float(np.median(l2)) if len(l2) else float("nan")
    print(f"  embedding L2/px median = {med:.4f} (training = 1.000)")
    if not (1 - tol) <= med <= (1 + tol):
        raise RuntimeError(
            f"embedding scale looks wrong: median L2/px = {med:.4g}, expected "
            f"~1.0. Check AEF_SCALE ({AEF_SCALE}) against the raster's units.")
    return med


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_points", type=int, default=0,
                    help="subsample changed pixels (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = find_changed_pixels()
    print(f"  changed pixels: {len(df):,}")
    print(df.groupby(["class_v1", "class_v2"]).size()
          .sort_values(ascending=False).head(15).to_string())

    if args.max_points and len(df) > args.max_points:
        # stratify by source class so rare transitions survive subsampling
        n_tot = len(df)
        picks = []
        for cls, g in df.groupby("class_v1"):
            n = min(len(g), max(200, int(args.max_points * len(g) / n_tot)))
            picks.append(g.sample(n, random_state=args.seed).index)
        df = df.loc[np.concatenate(picks)].reset_index(drop=True)
        print(f"  subsampled to {len(df):,}")

    tf = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    df["lon"], df["lat"] = tf.transform(df["x"].values, df["y"].values)

    df = df.reset_index(drop=True)
    print("  sampling AlphaEarth 2024...")
    emb = attach_embeddings(df, AEF_VRT, 64, EMBED_COLS, AEF_SCALE)
    check_embedding_scale(emb)
    print("  sampling lidar (elevation, tri, tch)...")
    lid = attach_embeddings(df, LIDAR_TIF, 3, LIDAR_COLS, 1.0,
                            zero_is_gap=False)
    df = pd.concat([df, emb, lid], axis=1)

    has = df[EMBED_COLS].notna().all(axis=1)
    print(f"  with valid 64-band embeddings: {has.sum():,} / {len(df):,} "
          f"({has.mean():.1%})")
    hl = df[LIDAR_COLS].notna().all(axis=1)
    print(f"  with valid lidar:              {hl.sum():,} / {len(df):,} "
          f"({hl.mean():.1%})")
    print("  coverage by source class:")
    print(df.assign(has=has).groupby("class_v1")["has"]
          .agg(["size", "sum", "mean"]).to_string())

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[OK] wrote {OUT_PARQUET}  ({os.path.getsize(OUT_PARQUET)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

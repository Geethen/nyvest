"""Dense per-pixel prediction over a single AlphaEarth embedding tile.

Reads one 64-band tile (A00..A63, EPSG:32633, 10 m), runs the seed CatBoost +
APS uncertainty per pixel, and writes two GeoTIFFs:
  - <stem>_class.tif        uint8   predicted merged class code (0 = nodata)
  - <stem>_uncertainty.tif  uint8   APS prediction-set size 1..n_classes (0 = nodata)
then reprojects both to EPSG:3857 (Web Mercator) as <stem>_class_3857.tif /
<stem>_uncertainty_3857.tif so Folium can overlay them as image layers.

Scope is intentionally ONE tile (the user's "test on a single tile for now"). The
`--tiles-glob` path loops the same code over many tiles later with no logic change.

Backends (`--backend`):
  native  (default) native CatBoost predict_proba — ~0.7 s per 954k-px tile (symmetric
          oblivious trees vectorise well; CPU predict, no GPU needed).
  timber  placeholder; raises (timber-compiled inference is a net loss on this batched
          workload — see reports/active_learning/timber_report.md). Seam kept for a
          future single-sample / edge deployment.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/active_learning/predict_raster.py
  ~/myprojects/recover/.venv/bin/python scripts/active_learning/predict_raster.py \
      --tile /path/to/embeddings_XXXX.tif --out-dir reports/active_learning
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_core as mc

EMBED_DIR = Path("/data/P-Prosjekter2/154001_nyvest/Nature_types_mapping/Features/embeddings")
DEFAULT_OUT = mc.REPO / "reports" / "active_learning"


def pick_default_tile() -> Path:
    """A median-size tile: small enough to be quick, big enough to be representative."""
    tiles = sorted(glob.glob(str(EMBED_DIR / "embeddings_*.tif")))
    if not tiles:
        raise FileNotFoundError(f"no embedding tiles under {EMBED_DIR}")
    tiles.sort(key=lambda t: Path(t).stat().st_size)
    return Path(tiles[len(tiles) // 2])


def predict_tile(tile: Path, seed: mc.SeedModel, backend: str,
                 out_dir: Path, blocksize: int = 512) -> dict:
    """Predict one tile block-windowed; write class + uncertainty GeoTIFFs (UTM)."""
    import rasterio
    from rasterio.windows import Window

    if backend == "timber":
        raise NotImplementedError(
            "timber backend unavailable: timber-compiled inference is a net loss on "
            "this batched workload, and won't compile multiclass CatBoost at all "
            "(see reports/active_learning/timber_report.md). Use --backend native.")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tile.stem
    class_path = out_dir / f"{stem}_class.tif"
    unc_path = out_dir / f"{stem}_uncertainty.tif"

    with rasterio.open(tile) as src:
        assert src.count == len(mc.FEATURE_COLS), \
            f"expected {len(mc.FEATURE_COLS)} bands, got {src.count}"
        profile = src.profile.copy()
        H, W = src.height, src.width
        # output rasters: uint8, 0 = nodata. Predicted codes (2..12) and set-sizes
        # (1..10) both fit in 1..255, so 0 is a safe nodata sentinel.
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")

        n_pix = n_valid = 0
        size_hist = np.zeros(seed.n_classes + 1, dtype=np.int64)  # index = set size
        t0 = time.perf_counter()

        with rasterio.open(class_path, "w", **out_profile) as dst_c, \
             rasterio.open(unc_path, "w", **out_profile) as dst_u:
            for row in range(0, H, blocksize):
                h = min(blocksize, H - row)
                for col in range(0, W, blocksize):
                    w = min(blocksize, W - col)
                    win = Window(col, row, w, h)
                    block = src.read(window=win).astype(np.float32)  # (64, h, w)
                    feats = block.reshape(src.count, -1).T            # (h*w, 64)
                    # NaN in any band => nodata pixel (tile header nodata is None but
                    # the data carries NaN, confirmed in the Phase-0 smoke test).
                    valid = ~np.isnan(feats).any(axis=1)
                    cls_blk = np.zeros(feats.shape[0], dtype=np.uint8)
                    unc_blk = np.zeros(feats.shape[0], dtype=np.uint8)
                    if valid.any():
                        pred, sizes, _ = seed.predict(feats[valid])
                        cls_blk[valid] = pred.astype(np.uint8)
                        unc_blk[valid] = sizes.astype(np.uint8)
                        size_hist += np.bincount(sizes, minlength=len(size_hist))
                    dst_c.write(cls_blk.reshape(h, w), 1, window=win)
                    dst_u.write(unc_blk.reshape(h, w), 1, window=win)
                    n_pix += feats.shape[0]
                    n_valid += int(valid.sum())
        dt = time.perf_counter() - t0

    print(f"  predicted {n_valid:,}/{n_pix:,} valid px in {dt:.1f}s "
          f"({n_valid/max(dt,1e-9):,.0f} px/s)")
    nz = size_hist[1:]
    if nz.sum():
        mean_sz = float((np.arange(1, len(nz) + 1) * nz).sum() / nz.sum())
        print(f"  set-size: mean={mean_sz:.2f}  singletons={100*nz[0]/nz.sum():.1f}%")
    return {"class_utm": class_path, "unc_utm": unc_path,
            "n_valid": n_valid, "n_pix": n_pix}


def reproject_to_3857(src_path: Path) -> Path:
    """Reproject a UTM raster to EPSG:3857 (Web Mercator) for Folium overlay."""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    dst_path = src_path.with_name(src_path.stem + "_3857.tif")
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        prof = src.profile.copy()
        prof.update(crs="EPSG:3857", transform=transform, width=w, height=h,
                    compress="deflate")
        with rasterio.open(dst_path, "w", **prof) as dst:
            reproject(
                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs="EPSG:3857",
                resampling=Resampling.nearest,  # categorical: never interpolate
            )
    return dst_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=Path, default=None,
                    help="single embedding tile (default: a median-size one)")
    ap.add_argument("--tiles-glob", type=str, default=None,
                    help="glob for multiple tiles (future scale-up; loops the same code)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--backend", choices=["native", "timber"], default="native")
    ap.add_argument("--blocksize", type=int, default=512,
                    help="raster window size; larger = bigger predict batches, fewer reads")
    args = ap.parse_args()

    seed = mc.load_seed_model()
    print(f"seed model: {seed.n_classes} classes, tau={seed.tau:.4f}, "
          f"backend={args.backend}")

    if args.tiles_glob:
        tiles = [Path(p) for p in sorted(glob.glob(args.tiles_glob))]
    elif args.tile:
        tiles = [args.tile]
    else:
        tiles = [pick_default_tile()]
    print(f"tiles: {len(tiles)} -> {[t.name for t in tiles]}")

    for tile in tiles:
        print(f"\n=== {tile.name} ===")
        r = predict_tile(tile, seed, args.backend, args.out_dir, args.blocksize)
        for key in ("class_utm", "unc_utm"):
            p3857 = reproject_to_3857(r[key])
            print(f"  wrote {r[key].name}  +  {p3857.name}")
    print("\ndone.")


if __name__ == "__main__":
    main()

"""Preprocess raw source.coop AEF tiles into model-ready AE rasters.

This is the "predict AOI once" preprocessing step. Raw AEF tiles downloaded
from source.coop (with s5cmd — see DATA_INFERENCE.md) are NOT directly
consumable by predict_raster.py: they are

  * int8-quantized  (dequantize with ((v/127.5)**2)*sign(v); -128 = nodata),
  * band-interleaved COGs in native UTM zones 31N/32N (the nyvest AOI spans
    both), not the project's working CRS EPSG:32633, and
  * bottom-up (row 0 = south).

This script fixes all three per tile: dequantize to float32 (nodata -> NaN),
then reproject/flip onto EPSG:32633 north-up at 10 m with a WarpedVRT. The
output is a 64-band float32 GeoTIFF whose embeddings are already unit-norm
(NO --scale needed downstream, unlike the P-drive tiles). Feed the results —
or a build_vrt.py mosaic of them — straight to predict_raster.py.

Order matters: dequantize BEFORE warping. Warping raw int8 would let the
resampler interpolate across the -128 nodata sentinel and across the nonlinear
quantization curve, both of which corrupt the embedding. Dequantizing first,
with nodata as NaN, makes the (nearest, by default) resampler nodata-aware.

Design constraints (match build_vrt.py):
  * pure rasterio — no gdalwarp/gdal_translate CLI in this venv.
  * writes GeoTIFF with BIGTIFF=IF_SAFER (a 64-band float32 8192^2 tile is
    ~17 GB uncompressed; deflate can't predict size and would overflow the
    4 GB classic-TIFF limit mid-write otherwise).

Run:
  PY=~/myprojects/recover/.venv/bin/python

  # one tile
  $PY DNN/prep_aef_tiles.py --in ~/aef_2024/32N/xtlm...-0-0.tiff \
      --out-dir ~/aef_2024_32633/

  # a whole download dir (both zones), 4 tiles at a time
  $PY DNN/prep_aef_tiles.py --glob '~/aef_2024/**/*.tiff' \
      --out-dir ~/aef_2024_32633/ --workers 4

Then:
  $PY DNN/build_vrt.py --glob '~/aef_2024_32633/*.tif' --out aef_2024.vrt
  $PY DNN/predict_raster.py --in aef_2024.vrt --out classified.tif \
      --readers 6 --mask-allzero            # NB: no --scale (already unit-norm)
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_aef_sourcecoop import AEF_NODATA  # -128  # noqa: E402

N_AE = 64
DST_CRS = "EPSG:32633"
DST_RES = 10.0

# int8 -> float32 dequant lookup table, indexed by (raw + 128). The AEF spec's
# elementwise ((v/127.5)**2)*sign(v) recomputed per-pixel is ~1.37x slower than
# one LUT gather (measured); a tile is 8192^2*64 = 4.3e9 elements, so it matters.
# -128 (nodata) maps to NaN so the warp resampler treats it as nodata.
_lut_in = np.arange(-128, 128, dtype=np.float32)
_DEQUANT_LUT = ((_lut_in / 127.5) ** 2) * np.sign(_lut_in)
_DEQUANT_LUT[0] = np.nan   # index 0 == raw -128 == nodata


def dequantize_lut(raw: np.ndarray) -> np.ndarray:
    """int8 [-128,127] -> float32 via LUT; nodata (-128) -> NaN. See module doc."""
    # raw is int8/int16; shift into [0,255] to index the 256-entry LUT.
    return _DEQUANT_LUT[raw.astype(np.int16) + 128]


def _dst_blocks(width, height, bx, by):
    from rasterio.windows import Window
    for row in range(0, height, by):
        h = min(by, height - row)
        for col in range(0, width, bx):
            w = min(bx, width - col)
            yield Window(col, row, w, h)


def prep_one(inp: str, out_dir: str, overwrite: bool = False,
             block: int = 2048) -> tuple[str, str, float]:
    """Dequantize + reproject one AEF tile to EPSG:32633 north-up float32.

    Returns (input_path, output_path, seconds). Idempotent: skips if the output
    already exists and --overwrite was not passed.

    Streams in `block`-sized WINDOWS on the destination grid rather than loading
    the whole tile: a full 8192^2 x 64 tile is ~17 GB as float32 and the naive
    read-all-dequant-warp-write path peaks near ~50 GB (3 simultaneous copies),
    which exceeds the box. Windowed, peak is a few GB regardless of tile size.

    Warp order: wrap the raw int8 source in a nodata-aware WarpedVRT
    (src_nodata=-128), read int8 from the WARPED grid, then dequantize each
    block to float32. Nearest resampling picks source pixels (never interpolates
    values), so dequant-after-nearest-warp is exact and the -128 sentinel is
    honoured by the resampler (warp margins come back as -128 -> NaN).
    """
    t0 = time.perf_counter()
    out_path = str(Path(out_dir) / (Path(inp).stem + "_32633.tif"))
    if os.path.exists(out_path) and not overwrite:
        return inp, out_path, 0.0

    with rasterio.open(inp) as src:
        if src.count < N_AE:
            raise ValueError(f"{inp}: {src.count} bands, expected >= {N_AE}")

        # Destination grid: the tile's own footprint reprojected to 32633 @ 10 m.
        # calculate_default_transform gives a north-up transform (negative e),
        # which also flips the source's bottom-up rows for free.
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds,
            resolution=DST_RES)

        # Snap the origin to the global 10 m lattice (multiples of DST_RES,
        # anchored at 0,0). calculate_default_transform picks each tile's origin
        # independently, so tiles reprojected separately would NOT share a pixel
        # grid — and build_vrt.py assumes a common grid (offsets 0). Snapping the
        # top-left DOWN/UP to the lattice and growing the size by one pixel of
        # slack keeps full coverage and makes every tile grid-aligned.
        c = np.floor(dst_transform.c / DST_RES) * DST_RES     # west edge -> lattice
        f = np.ceil(dst_transform.f / DST_RES) * DST_RES      # north edge -> lattice
        dst_transform = rasterio.Affine(DST_RES, 0.0, c, 0.0, -DST_RES, f)
        dst_w += 1
        dst_h += 1

        prof = src.profile.copy()
        prof.update(driver="GTiff", crs=DST_CRS, transform=dst_transform,
                    width=dst_w, height=dst_h, count=N_AE, dtype="float32",
                    nodata=np.nan, compress="deflate", tiled=True,
                    blockxsize=512, blockysize=512, bigtiff="IF_SAFER")

        tmp_out = out_path + ".tmp"
        with WarpedVRT(src, crs=DST_CRS, transform=dst_transform,
                       width=dst_w, height=dst_h,
                       resampling=Resampling.nearest,
                       src_nodata=AEF_NODATA, nodata=AEF_NODATA) as vrt, \
             rasterio.open(tmp_out, "w", **prof) as dst:
            for b in range(1, N_AE + 1):
                dst.set_band_description(b, f"A{b-1:02d}")
            for win in _dst_blocks(dst_w, dst_h, block, block):
                raw = vrt.read(range(1, N_AE + 1), window=win,
                               out_dtype="int16")     # warped int8 (-128 margins)
                dst.write(dequantize_lut(raw), window=win)   # -> float32, NaN nodata
        os.replace(tmp_out, out_path)                  # atomic: no half-written tile

    return inp, out_path, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="inp", help="single raw AEF .tiff")
    g.add_argument("--glob", help="glob for many raw AEF tiles (both zones OK)")
    ap.add_argument("--out-dir", required=True, help="output dir for *_32633.tif")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel tiles (windowed, so ~0.5 GB/worker at the "
                         "default --block, independent of tile size)")
    ap.add_argument("--block", type=int, default=2048,
                    help="dest-grid window size (px); ~block^2 x 64 x int16 RAM "
                         "per read (2048 -> ~0.5 GB)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-process tiles whose output already exists")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.inp:
        files = [args.inp]
    else:
        files = sorted(globmod.glob(os.path.expanduser(args.glob), recursive=True))
        if not files:
            raise SystemExit(f"no files matched {args.glob!r}")
    print(f"prepping {len(files)} tile(s) -> {args.out_dir}  (workers={args.workers})",
          flush=True)

    t0 = time.perf_counter()
    done = 0
    if args.workers == 1:
        for f in files:
            inp, out, dt = prep_one(f, args.out_dir, args.overwrite, args.block)
            done += 1
            print(f"  [{done}/{len(files)}] {Path(out).name}  "
                  f"{'skipped (exists)' if dt == 0 else f'{dt:.1f}s'}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(prep_one, f, args.out_dir, args.overwrite, args.block): f
                    for f in files}
            for fut in as_completed(futs):
                inp, out, dt = fut.result()
                done += 1
                print(f"  [{done}/{len(files)}] {Path(out).name}  "
                      f"{'skipped (exists)' if dt == 0 else f'{dt:.1f}s'}", flush=True)
    print(f"done: {done} tile(s) in {time.perf_counter()-t0:.1f}s -> {args.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()

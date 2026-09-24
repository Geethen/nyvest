"""Is the raster read wall a property of the data, or of how we chose to store it?

`predict_raster.py` is I/O bound — README.md measures 1.4 M px/s end-to-end on
the 8-core VDI with 6 readers, decompression-limited, GPU idle. With the model
side now fused (`autoresearch/moe_fast.py`) that read ceiling is the only wall
left, so it is worth asking what sets it.

THE OBSERVATION THIS TESTS

`prep_aef_tiles.py` takes source.coop AEF tiles that are **int8 + ZSTD** and
writes **float32 + DEFLATE**. The dequantisation it applies is a 256-entry
lookup table (`_DEQUANT_LUT`), so every value in the prepped tile is one of 256
floats, and the warp uses nearest resampling, which cannot introduce a 257th.
The prepped raster therefore carries exactly 8 bits of information per value and
spends 32 bits storing it. DEFLATE hides that on disk — it compresses to under
0.1 bytes/value — but decompression still has to MATERIALISE 4 bytes/value into
memory, with zlib rather than zstd, and that is CPU the pipeline cannot overlap
away.

So the question is not "can we read bytes faster" (the files are already tiny)
but "how much of the read wall is us inflating 8-bit data to 32-bit with a slow
codec".

WHAT IS MEASURED

The read exactly as `predict_raster.py`'s reader threads do it: one rasterio
handle per thread, `read(range(1, 65), window=...)` over `--block`-sized
windows, all 64 bands. Variants differ only in on-disk encoding; every one is
checked to decode back to the ORIGINAL float32 values (via the LUT for the int8
variants) before it is timed, so a fast lossy encoding cannot win.

Both warm and cold caches are reported. Warm isolates the codec+widening CPU,
which is the axis under our control; cold (`POSIX_FADV_DONTNEED` between reps,
no root needed) adds the real disk term. Warm is the fair comparison between
encodings, cold is the one that resembles a first pass over a county.

WHAT THESE NUMBERS ARE NOT

They are much higher than README.md's end-to-end 1.4 M px/s, and that gap is
real, not a contradiction. This measures ONE stage — decompress a window of one
local tile — with no warp, no GPU inference, no write, and on a window that is
still ~25% nodata (this bench tile is a single warped UTM tile, so a fully
covered window does not exist in it; production reads a VRT mosaic where
interior windows are dense). Treat the RATIOS BETWEEN ROWS as the result and the
absolute M px/s as an upper bound on the read stage alone.

Run:
    ~/myprojects/recover/.venv/bin/python bench_io_formats.py
    ~/myprojects/recover/.venv/bin/python bench_io_formats.py --row 7144 --col 6900
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# BEFORE rasterio/GDAL initialises. Both of these were caught making this
# benchmark lie, so they are set here rather than left to the environment:
#
# GDAL_CACHEMAX  GDAL's block cache defaults to 5% of RAM, which is larger than
#   a whole variant here. With it, the timed reads were served as memcpy out of
#   already-decompressed blocks and the table was measuring dtype width, not
#   codec cost. `predict_raster.py` scans each window exactly once and gets no
#   such reuse, so a small cache is the honest setting, not a handicap.
# GDAL_NUM_THREADS  the GTiff default is single-threaded decompression. That is
#   worth ~4x on its own, but it is NOT a free 4x on top of `predict_raster.py`,
#   which already runs 6 reader threads: the two are substitutes for the same
#   cores and saturate together (`--thread-sweep`). Pinned here so the variant
#   table compares variants rather than thread placement.
os.environ.setdefault("GDAL_CACHEMAX", "32")
# Assignment, NOT setdefault: sweep cells are subprocesses that inherit this very
# variable from their parent, and setdefault silently made every cell run the
# parent's value — the sweep reported a flat table until that was caught.
os.environ["GDAL_NUM_THREADS"] = (
    sys.argv[6] if len(sys.argv) > 6 and sys.argv[1] == "--_cell"
    else os.environ.get("GDAL_NUM_THREADS", "ALL_CPUS"))

import rasterio                                             # noqa: E402
from rasterio.windows import Window                         # noqa: E402

from prep_aef_tiles import _DEQUANT_LUT, N_AE               # noqa: E402

DEFAULT_SRC = "/home/geethen.singh/myprojects/nyvest/aef_bench_prepped/" \
              "xnkqw3fftuuts84jo-0000008192-0000008192_32633.tif"


def to_int8(arr):
    """float32 (LUT-valued, NaN nodata) -> the int8 it was dequantised from.

    Exact, not nearest-match-approximate: the values came from a 256-entry LUT
    and nearest-neighbour warping, so every finite value IS a LUT entry.
    """
    lut = _DEQUANT_LUT.copy()
    lut[0] = np.inf                      # keep NaN from matching entry 0
    flat = arr.ravel()
    idx = np.abs(flat[:, None] - lut[None, :]).argmin(1).astype(np.int16) - 128
    idx[~np.isfinite(flat)] = -128       # nodata
    return idx.astype(np.int8).reshape(arr.shape)


def to_int8_chunked(arr, rows=64):
    out = np.empty(arr.shape, np.int8)
    for b in range(arr.shape[0]):
        for i in range(0, arr.shape[1], rows):
            out[b, i:i + rows] = to_int8(arr[b, i:i + rows])
    return out


# ------------------------------------------------------------------- variants
def variants():
    """(key, label, dtype, creation options). Block size in the options."""
    def opts(compress, blk, **kw):
        d = dict(driver="GTiff", tiled=True, blockxsize=blk, blockysize=blk,
                 compress=compress, interleave="band", BIGTIFF="IF_SAFER")
        d.update(kw)
        return d
    return [
        ("f32_deflate_512", "float32 DEFLATE 512   (what prep writes today)",
         "float32", opts("deflate", 512)),
        ("f32_deflate_p3", "float32 DEFLATE 512 +predictor=3",
         "float32", opts("deflate", 512, predictor=3)),
        ("f32_zstd_512", "float32 ZSTD 512",
         "float32", opts("zstd", 512, ZSTD_LEVEL=1)),
        ("f32_zstd_p3", "float32 ZSTD 512 +predictor=3",
         "float32", opts("zstd", 512, ZSTD_LEVEL=1, predictor=3)),
        ("i8_deflate_512", "int8 DEFLATE 512",
         "int8", opts("deflate", 512)),
        ("i8_zstd_512", "int8 ZSTD 512         (source.coop's own encoding)",
         "int8", opts("zstd", 512, ZSTD_LEVEL=1)),
        ("i8_zstd_1024", "int8 ZSTD 1024",
         "int8", opts("zstd", 1024, ZSTD_LEVEL=1)),
        ("i8_lzw_512", "int8 LZW 512",
         "int8", opts("lzw", 512)),
    ]


def write_variant(path, data_f32, data_i8, dtype, opts, size):
    arr = data_f32 if dtype == "float32" else data_i8
    prof = dict(opts)
    prof.update(width=size, height=size, count=N_AE, dtype=dtype,
                nodata=(np.nan if dtype == "float32" else -128), crs=None,
                transform=rasterio.transform.from_origin(0, 0, 10, 10))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)
    return os.path.getsize(path)


# ---------------------------------------------------------------- measurement
def windows_of(size, block):
    return [Window(c, r, min(block, size - c), min(block, size - r))
            for r in range(0, size, block) for c in range(0, size, block)]


def read_all(path, size, block, readers, dequant):
    """The reader-thread pattern from predict_raster.py, timed end to end.

    `dequant` decides who pays for the int8 -> float32 LUT expansion:

      True   on the CPU, in the reader thread, as a numpy fancy-index gather.
             This is what an int8 tile costs if `predict_raster.py` is left
             alone, and it is NOT cheap — a 64-band 2048 block is 268M gathers.
      False  not at all here, because the pipeline already ships every block to
             the GPU and a 256-entry LUT gather is one `torch.take` on device.
             This is the row that describes the change worth making, and it is
             only honest because the dequant genuinely moves rather than
             vanishes — see the GPU-side timing in the summary.
    """
    wins = windows_of(size, block)
    local = __import__("threading").local()

    def one(win):
        if not hasattr(local, "src"):
            local.src = rasterio.open(path)
        a = local.src.read(range(1, N_AE + 1), window=win)
        if dequant:
            a = _DEQUANT_LUT[a.astype(np.int16) + 128]
        return int(a.shape[1]) * int(a.shape[2])

    t0 = time.perf_counter()
    if readers == 1:
        px = sum(one(w) for w in wins)
    else:
        with ThreadPoolExecutor(readers) as ex:
            px = sum(ex.map(one, wins))
    return px / (time.perf_counter() - t0)


def evict(path):
    """Drop this file from the page cache. No root needed — POSIX_FADV_DONTNEED
    evicts clean pages, and these were just written and closed."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def gpu_dequant_rate(i8, reps=5):
    """M px/s of the LUT expansion done on-GPU, the cost the int8-raw rows defer.

    Reported next to them so "the dequant moved to the GPU" is a measurement
    rather than an assertion.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        dev = torch.device("cuda")
        lut = torch.from_numpy(np.nan_to_num(_DEQUANT_LUT, nan=0.0)).to(dev)
        blk = torch.from_numpy(i8[:, :1024, :1024].copy()).to(dev)
        px = blk.shape[1] * blk.shape[2]

        def once():
            idx = (blk.to(torch.int16) + 128).long()
            return lut[idx]
        for _ in range(2):
            once()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            once()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return px / float(np.median(ts))
    except Exception:
        return None


def verify(path, ref, dequant):
    with rasterio.open(path) as s:
        a = s.read()
    a = _DEQUANT_LUT[a.astype(np.int16) + 128] if dequant else a
    same = np.array_equal(np.nan_to_num(a, nan=-9e9),
                          np.nan_to_num(ref, nan=-9e9))
    return same


def thread_sweep(args, wd, ref, i8):
    """Reader threads vs GDAL's internal decompression threads.

    Worth its own mode because the two are SUBSTITUTES, not multipliers — both
    are ways of spending the same 8 cores on the same zlib/zstd work, and they
    saturate at the same ceiling. `predict_raster.py` already runs 6 reader
    threads, so GDAL_NUM_THREADS is not a free multiple on top of it; the number
    that matters is the ceiling each ENCODING saturates to.

    GDAL_NUM_THREADS is read at GDAL init, so each cell is a subprocess.
    """
    import subprocess
    picks = [("f32_deflate_512", "float32 DEFLATE (today)"),
             ("i8_zstd_512", "int8 ZSTD")]
    for key, _, dtype, opts in variants():
        if key in dict(picks):
            write_variant(wd / f"{key}.tif", ref, i8, dtype, opts, args.size)
    block = args.size // 3                      # 9 windows, so readers have work
    print(f"block={block} -> {len(windows_of(args.size, block))} windows\n")
    print(f"{'encoding':26s} {'GDAL_NUM_THREADS':>17s} "
          + "".join(f"{'rd=%d' % r:>9s}" for r in (1, 3, 6)))
    rows = []
    for key, label in picks:
        for nt in ("1", "ALL_CPUS"):
            cells = []
            for rd in (1, 3, 6):
                out = subprocess.run(
                    [sys.executable, __file__, "--_cell",
                     str(wd / f"{key}.tif"), str(args.size), str(block),
                     str(rd), nt],
                    capture_output=True, text=True)
                cells.append(float(out.stdout.strip() or "nan"))
            rows.append({"encoding": key, "gdal_num_threads": nt,
                         "px_per_s": [round(c) for c in cells]})
            print(f"{label:26s} {nt:>17s} "
                  + "".join(f"{c/1e6:9.2f}" for c in cells))
    print("\nBoth knobs spend the same cores: each encoding saturates at a "
          "ceiling\nthat the other knob cannot raise. The ceiling is set by the "
          "ENCODING.")
    op = Path("DNN/reports/results/io_thread_sweep.json")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps({"kind": "io_thread_sweep", "size": args.size,
                              "block": block, "cores": os.cpu_count(),
                              "rows": rows}, indent=2))
    print(f"saved -> {op}")
    shutil.rmtree(wd, ignore_errors=True)


def _cell():
    """One sweep cell, in its own process so GDAL_NUM_THREADS takes effect."""
    path, size, block, readers, _ = sys.argv[2:7]
    for _ in range(2):
        read_all(path, int(size), int(block), int(readers), False)
    r = np.median([read_all(path, int(size), int(block), int(readers), False)
                   for _ in range(3)])
    print(f"{r:.0f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        return _cell()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--size", type=int, default=1536, help="square window of the tile")
    # Default offset is NOT (0,0): this tile is a warped UTM-zone product and is
    # ~90% NaN margin, so the origin window is pure nodata and every codec looks
    # infinitely fast on it. --min-finite is the guard that makes that loud.
    ap.add_argument("--row", type=int, default=7144, help="window row offset")
    ap.add_argument("--col", type=int, default=6900, help="window col offset")
    ap.add_argument("--min-finite", type=float, default=0.5,
                    help="refuse to benchmark a window this empty")
    ap.add_argument("--thread-sweep", action="store_true",
                    help="reader threads x GDAL_NUM_THREADS, on two encodings")
    ap.add_argument("--block", type=int, default=512,
                    help="read window (predict_raster.py uses 2048, but needs "
                         "size >> block for the reader-thread column to mean "
                         "anything; this tile has no window that large)")
    ap.add_argument("--readers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workdir", default="/home/geethen.singh/myprojects/nyvest/.io_bench_tmp")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default="DNN/reports/results/io_format_bench.json")
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    nwin = len(windows_of(args.size, args.block))
    print(f"source: {args.src}\nwindow: {args.size}x{args.size} x {N_AE} bands   "
          f"block={args.block}  readers={args.readers}  reps={args.reps}  "
          f"({nwin} read windows)\n")
    # A --size/--block pair that yields one window silently turns the reader-thread
    # column into a copy of the single-thread column. That happened here and was
    # only caught by an outside cross-check, so it is an error now, not a footnote.
    if nwin < args.readers and not args.thread_sweep:   # the sweep picks its own
        raise SystemExit(
            f"{args.size}/{args.block} gives {nwin} window(s) but --readers is "
            f"{args.readers}: the reader-thread column would measure nothing. "
            f"Use --block {args.size // max(2, args.readers)} or fewer readers.")

    with rasterio.open(args.src) as s:
        ref = s.read(range(1, N_AE + 1),
                     window=Window(args.col, args.row, args.size, args.size))
    ref = ref.astype(np.float32)
    finite = float(np.isfinite(ref).mean())
    ndist = len(np.unique(ref[np.isfinite(ref)]))
    print(f"window finite fraction: {finite:.3f}   distinct finite values: "
          f"{ndist} (the LUT has 255 + NaN)")
    if finite < args.min_finite:
        raise SystemExit(
            f"window at row={args.row} col={args.col} is {1-finite:.1%} nodata — "
            f"codecs compress nodata to nothing and the benchmark would be "
            f"meaningless. Move --row/--col onto real data.")
    i8 = to_int8_chunked(ref)
    raw_gb = ref.nbytes / 1e9
    gpu_deq = gpu_dequant_rate(i8)
    print(f"uncompressed float32: {raw_gb:.2f} GB\n")

    if args.thread_sweep:
        return thread_sweep(args, wd, ref, i8)

    rows, base, base_cold = [], None, None
    print(f"{'variant':46s} {'MB':>7} {'B/val':>6} "
          f"{'1rdr/w':>8} {'%drdr/w' % args.readers:>8} "
          f"{'%drdr/c' % args.readers:>8} {'warm':>7} {'cold':>7}")
    for key, label, dtype, opts in variants():
        p = wd / f"{key}.tif"
        mb = write_variant(p, ref, i8, dtype, opts, args.size) / 1e6
        is_i8 = dtype == "int8"
        if not verify(p, ref, is_i8):
            raise SystemExit(f"{key}: does not round-trip to the original values")
        # int8 is timed twice: paying the LUT on the CPU, and deferring it to the
        # GPU. Reporting only one of the two would be picking the answer.
        for deq, suffix in ([(True, " [CPU dequant]"), (False, " [GPU dequant]")]
                            if is_i8 else [(False, "")]):
            def med(readers, cold):
                ts = []
                for _ in range(args.reps):
                    if cold:
                        evict(p)
                    ts.append(read_all(p, args.size, args.block, readers, deq))
                return float(np.median(ts))
            r1 = med(1, False)
            rn = med(args.readers, False)
            rc = med(args.readers, True)
            if base is None:
                base, base_cold = rn, rc
            rows.append({"name": key + ("_cpudeq" if deq else ""),
                         "label": (label + suffix).strip(), "dtype": dtype,
                         "dequant": "cpu" if deq else ("gpu" if is_i8 else "none"),
                         "file_mb": round(mb, 2),
                         "bytes_per_value": round(mb * 1e6 / ref.size, 4),
                         "px_per_s_1reader_warm": round(r1),
                         f"px_per_s_{args.readers}readers_warm": round(rn),
                         f"px_per_s_{args.readers}readers_cold": round(rc),
                         "vs_current_warm": round(rn / base, 2),
                         "vs_current_cold": round(rc / base_cold, 2),
                         "roundtrip_exact": True})
            print(f"{label+suffix:46s} {mb:7.2f} {mb*1e6/ref.size:6.3f} "
                  f"{r1/1e6:8.2f} {rn/1e6:8.2f} {rc/1e6:8.2f} "
                  f"{rn/base:6.2f}x {rc/base_cold:6.2f}x")

    if gpu_deq:
        print(f"\non-GPU LUT dequant (what the [GPU dequant] rows defer): "
              f"{gpu_deq/1e6:.1f} M px/s")
    out = {"kind": "io_format_bench", "src": args.src, "size": args.size,
           "window_row": args.row, "window_col": args.col,
           "window_finite_frac": round(finite, 4), "distinct_values": ndist,
           "block": args.block, "readers": args.readers, "reps": args.reps,
           "n_bands": N_AE, "cores": os.cpu_count(),
           "gpu_dequant_px_per_s": None if gpu_deq is None else round(gpu_deq),
           "note": "M px/s counts pixels once, not per band; warm cache",
           "io_ceiling_readme_px_per_s": 1.4e6,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": rows}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {op}")
    if not args.keep:
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()

"""Build a 3-band (elevation, tri, tch) lidar raster on a target AE grid.

`predict_raster.py --lidar-raster` needs a raster whose bands are exactly
`elevation, tri, tch` (LIDAR_COLS order) on the EXACT SAME grid (transform / CRS
/ size) as the AlphaEarth `--in` raster it predicts on — it reads lidar with the
same pixel Window as the AE input and does NO internal resampling.

The raw P-drive lidar tiles are 2-band (DTM, chm) float64 at 3 m in EPSG:32633,
millimetre-scaled (x0.001 -> m; verified vs GLO30, see
scripts/extraction/extract_lidar_features.py). This script mirrors that script's
point-level math but generalized to a full raster, per DATA_INFERENCE.md:

  1. Mosaic the lidar tiles from BOTH dirs (Features/lidar + the larger
     Vestland_Moreromsdal_features/lidar, the latter winning a name collision;
     809 tiles deduped) into a VRT.
  2. TRI (Riley 1999: mean abs diff of a pixel vs its 8 neighbours) is computed
     at the lidar's NATIVE 3 m resolution — matches training, which derives TRI
     from the native-resolution neighbourhood, not a coarsened one. So we build
     a native-3 m intermediate (elevation, tri, tch), then...
  3. ...reproject/resample all 3 bands onto the target AE grid with
     Resampling.average (down-sampling 3 m -> 10 m).

Parallel + windowed: the target grid is split into independent blocks, computed
across a ProcessPoolExecutor (each worker reads its own small 3 m source
footprint + a halo for the TRI stencil), and the parent process serializes the
GeoTIFF writes (GDAL is not safe for concurrent writes to one dataset). Peak RAM
stays bounded regardless of AOI size. Only blocks that overlap the (~22%-valid)
lidar footprint do work; the rest are left NaN and median-filled by
predict_raster.py.

TRI is the hot spot (~73% of per-block time) and is memory-bandwidth-bound over 8
neighbour passes, so it runs in float32 — the elevation differences are metres on
values <3000 m, and float32's ~1e-4 m error vs the f64 reference is negligible
(verified: 0 NaN mismatches, max abs diff 2.4e-4 m over 207M pixels). Net: a
30454x35692 build takes ~2.7 min on 8 cores vs ~36 min single-threaded f64.

Run:
  PY=~/myprojects/recover/.venv/bin/python
  $PY DNN/build_lidar_raster.py --like aef_2024.vrt --out lidar_3band.tif --workers 8
"""
from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
import rasterio.windows as rw
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

MM_TO_M = 0.001            # both bands stored as mm (extract_lidar_features.py)
DTM_BAND, CHM_BAND = 1, 2  # descriptions == ('DTM', 'chm')
NATIVE_RES = 3.0
OUT_BANDS = ["elevation", "tri", "tch"]   # must match LIDAR_COLS order

DATA_ROOT_CANDIDATES = (
    os.environ.get("NYVEST_DATA_DIR"),
    "/data/P-Prosjekter2/154001_nyvest",
    "P:/154001_nyvest",
)


def _data_root() -> str:
    for c in DATA_ROOT_CANDIDATES:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("set NYVEST_DATA_DIR to the nyvest data root")


def lidar_tiles() -> list[str]:
    """All lidar tile paths across both dirs; on a basename collision the Vestland
    dir wins (listed last so it overwrites), matching extract_lidar_features.py."""
    root = _data_root()
    dirs = [
        os.path.join(root, "Nature_types_mapping/Features/lidar"),
        os.path.join(root, "Nature_types_mapping/Vestland_Moreromsdal_features/lidar"),
    ]
    by_name: dict[str, str] = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "lidar_*.tif"))):
            by_name[os.path.basename(f)] = f
    return list(by_name.values())


def build_lidar_vrt(out_vrt: str) -> str:
    """Merge the lidar tiles (both dirs) into one plain rasterio-written VRT
    mosaic. All tiles share CRS/res/grid (3 m EPSG:32633), same as build_vrt.py."""
    files = lidar_tiles()
    minx = miny = np.inf
    maxx = maxy = -np.inf
    infos = []
    res_x = res_y = crs_wkt = None
    for f in files:
        with rasterio.open(f) as s:
            t = s.transform
            if res_x is None:
                res_x, res_y, crs_wkt = t.a, t.e, s.crs.to_wkt()
            b = s.bounds
            minx, miny = min(minx, b.left), min(miny, b.bottom)
            maxx, maxy = max(maxx, b.right), max(maxy, b.top)
            infos.append((f, s.width, s.height, t.c, t.f))
    width = int(round((maxx - minx) / res_x))
    height = int(round((maxy - miny) / (-res_y)))
    lines = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
             f'  <SRS>{escape(crs_wkt)}</SRS>',
             f'  <GeoTransform>{minx}, {res_x}, 0.0, {maxy}, 0.0, {res_y}</GeoTransform>']
    for band in (DTM_BAND, CHM_BAND):
        lines.append(f'  <VRTRasterBand dataType="Float64" band="{band}">')
        for (f, w, h, cx, fy) in infos:
            xoff = int(round((cx - minx) / res_x))
            yoff = int(round((maxy - fy) / (-res_y)))
            rel = os.path.relpath(f, start=Path(out_vrt).resolve().parent)
            lines.append(
                f'    <SimpleSource>'
                f'<SourceFilename relativeToVRT="1">{escape(rel)}</SourceFilename>'
                f'<SourceBand>{band}</SourceBand>'
                f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
                f'<DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}"/>'
                f'</SimpleSource>')
        lines.append('  </VRTRasterBand>')
    lines.append('</VRTDataset>')
    Path(out_vrt).write_text("\n".join(lines))
    return out_vrt


def _tri_native(dtm: np.ndarray) -> np.ndarray:
    """Riley 1999 TRI at native res: mean abs diff of each pixel vs its 8
    neighbours, in float32. NaN-aware (edges/nodata divide by the count of finite
    neighbours). `dtm` includes a 1 px halo; returns the halo-trimmed interior."""
    e = dtm[1:-1, 1:-1]
    acc = np.zeros_like(e)
    cnt = np.zeros(e.shape, dtype=np.int16)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neigh = dtm[1 + dr: 1 + dr + e.shape[0], 1 + dc: 1 + dc + e.shape[1]]
            d = np.abs(neigh - e)
            m = np.isfinite(d)
            acc += np.where(m, d, np.float32(0))
            cnt += m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


# Per-worker globals: opened once via the pool initializer so each block doesn't
# re-open the datasets. Workers only READ; the parent owns the single writer.
_LIKE = None
_LSRC = None


def _init(like_path: str, lid_vrt: str):
    global _LIKE, _LSRC
    _LIKE = rasterio.open(like_path)
    _LSRC = rasterio.open(lid_vrt)


def _compute_block(win_tuple):
    """Compute one target block; return (win_tuple, out_block) or (win_tuple, None)
    if the block has no lidar. Runs in a worker on the shared open datasets."""
    col, row, w, h = win_tuple
    win = Window(col, row, w, h)
    bnds = rw.bounds(win, _LIKE.transform)
    pad = NATIVE_RES * 4          # target-block halo for the 3x3 TRI stencil
    src_win = from_bounds(bnds[0] - pad, bnds[1] - pad, bnds[2] + pad, bnds[3] + pad,
                          _LSRC.transform).round_offsets().round_lengths()
    # Clip to source extent by hand: Window.intersection RAISES on a disjoint
    # window, and most AOI blocks fall outside the lidar footprint.
    c0 = max(0, src_win.col_off)
    r0 = max(0, src_win.row_off)
    c1 = min(_LSRC.width, src_win.col_off + src_win.width)
    r1 = min(_LSRC.height, src_win.row_off + src_win.height)
    if c1 - c0 <= 2 or r1 - r0 <= 2:
        return win_tuple, None
    src_win = Window(c0, r0, c1 - c0, r1 - r0)

    dtm = _LSRC.read(DTM_BAND, window=src_win).astype(np.float32)
    if not np.any(dtm != 0):
        return win_tuple, None            # window inside footprint but all-gap
    chm = _LSRC.read(CHM_BAND, window=src_win).astype(np.float32)
    # VRT gaps read as 0 (no nodata on the float lidar); treat exact-0 as nodata
    # so seams don't inject fake sea-level pixels.
    dtm[dtm == 0] = np.nan
    chm[chm == 0] = np.nan
    dtm *= MM_TO_M
    chm *= MM_TO_M

    tri = np.full_like(dtm, np.nan)
    if dtm.shape[0] >= 3 and dtm.shape[1] >= 3:
        tri[1:-1, 1:-1] = _tri_native(dtm)
    src_stack = np.stack([dtm, tri, chm]).astype(np.float32)

    src_transform = rw.transform(src_win, _LSRC.transform)
    block_transform = rw.transform(win, _LIKE.transform)
    out_block = np.full((3, h, w), np.nan, dtype=np.float32)
    reproject(source=src_stack, destination=out_block,
              src_transform=src_transform, src_crs=_LSRC.crs,
              dst_transform=block_transform, dst_crs=_LIKE.crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.average)
    return win_tuple, out_block


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", required=True,
                    help="target AE raster/VRT whose grid to match exactly")
    ap.add_argument("--out", required=True, help="output 3-band lidar GeoTIFF")
    ap.add_argument("--block", type=int, default=2048, help="target-grid block px")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    lid_vrt = args.out + ".lidar_src.vrt"
    build_lidar_vrt(lid_vrt)

    with rasterio.open(args.like) as like:
        dst_w, dst_h = like.width, like.height
        prof = dict(driver="GTiff", crs=like.crs, transform=like.transform,
                    width=dst_w, height=dst_h, count=3, dtype="float32",
                    nodata=np.nan, compress="deflate", tiled=True,
                    blockxsize=512, blockysize=512, bigtiff="IF_SAFER",
                    predictor=3)

    blocks = []
    for row in range(0, dst_h, args.block):
        h = min(args.block, dst_h - row)
        for col in range(0, dst_w, args.block):
            w = min(args.block, dst_w - col)
            blocks.append((col, row, w, h))

    n_total = len(blocks)
    n_written = n_empty = 0
    with rasterio.open(args.out, "w", **prof) as dst, \
         ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init, initargs=(args.like, lid_vrt)) as ex:
        for b, name in enumerate(OUT_BANDS, start=1):
            dst.set_band_description(b, name)
        futs = {ex.submit(_compute_block, blk): blk for blk in blocks}
        done = 0
        for fut in as_completed(futs):
            win_tuple, out_block = fut.result()
            done += 1
            if out_block is None:
                n_empty += 1
            else:
                col, row, w, h = win_tuple
                dst.write(out_block, window=Window(col, row, w, h))
                n_written += 1
            if done % 20 == 0 or done == n_total:
                print(f"  [{done}/{n_total}] written={n_written} empty={n_empty}",
                      flush=True)
    os.remove(lid_vrt)
    print(f"wrote {args.out}: {dst_w}x{dst_h}, 3 bands {OUT_BANDS}; "
          f"{n_written} blocks with lidar, {n_empty} empty (median-filled downstream)",
          flush=True)


if __name__ == "__main__":
    main()

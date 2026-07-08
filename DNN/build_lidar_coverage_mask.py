"""Build a lidar-coverage mask on a target grid, for visual/QA investigation.

Answers "where does real lidar exist vs. where does predict_raster.py fall back
to training-set medians?" as a single-band raster you can load straight into
QGIS/rasterio alongside classified_2024.tif. Byte-valued:

  1   = lidar-covered (elevation/tri/tch band 1 is finite in lidar_3band.tif)
  0   = no lidar (median-filled at inference)
  255 = nodata (outside --aoi, if one was given)

Reuses the exact tile discovery + AOI logic from build_lidar_raster.py so the
mask always matches what that script actually produced — computed straight from
the ALREADY-BUILT lidar_3band.tif (band 1, elevation) rather than re-deriving
coverage from the raw tiles, so it reflects the real output, not an estimate.

Run:
  PY=~/myprojects/recover/.venv/bin/python
  $PY DNN/build_lidar_coverage_mask.py --lidar lidar_3band.tif --out lidar_coverage.tif \
      --aoi /path/to/nyvest_fylker.shp
"""
from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

NODATA = 255


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lidar", required=True,
                    help="lidar_3band.tif (from build_lidar_raster.py); band 1 "
                         "(elevation) finite/NaN defines coverage")
    ap.add_argument("--out", required=True, help="output uint8 coverage mask")
    ap.add_argument("--aoi", default=None,
                    help="optional AOI vector to also clip to (matches "
                         "predict_raster.py --aoi); outside -> nodata (255)")
    ap.add_argument("--block", type=int, default=4096)
    args = ap.parse_args()

    with rasterio.open(args.lidar) as lid:
        prof = dict(driver="GTiff", crs=lid.crs, transform=lid.transform,
                    width=lid.width, height=lid.height, count=1, dtype="uint8",
                    nodata=NODATA, compress="deflate", tiled=True,
                    blockxsize=512, blockysize=512)

        aoi_geoms = None
        if args.aoi:
            import geopandas as gpd
            gdf = gpd.read_file(args.aoi).to_crs(lid.crs)
            aoi_geoms = list(gdf.geometry.values)

        n_lidar = n_no_lidar = n_outside_aoi = 0
        with rasterio.open(args.out, "w", **prof) as dst:
            for row in range(0, lid.height, args.block):
                h = min(args.block, lid.height - row)
                for col in range(0, lid.width, args.block):
                    w = min(args.block, lid.width - col)
                    win = Window(col, row, w, h)
                    elev = lid.read(1, window=win)
                    covered = np.isfinite(elev)
                    out = np.where(covered, np.uint8(1), np.uint8(0))
                    if aoi_geoms is not None:
                        win_transform = lid.window_transform(win)
                        inside = geometry_mask(aoi_geoms, out_shape=(h, w),
                                               transform=win_transform,
                                               invert=True, all_touched=True)
                        n_outside_aoi += int((~inside).sum())
                        out = np.where(inside, out, np.uint8(NODATA))
                    n_lidar += int((out == 1).sum())
                    n_no_lidar += int((out == 0).sum())
                    dst.write(out, 1, window=win)

    denom = n_lidar + n_no_lidar
    pct = 100 * n_lidar / denom if denom else float("nan")
    print(f"wrote {args.out}: {lid.width}x{lid.height}", flush=True)
    print(f"  lidar-covered:    {n_lidar:,} px", flush=True)
    print(f"  no-lidar (median-filled at inference): {n_no_lidar:,} px", flush=True)
    if aoi_geoms is not None:
        print(f"  outside AOI (nodata): {n_outside_aoi:,} px", flush=True)
    print(f"  coverage within predicted area: {pct:.1f}%", flush=True)


if __name__ == "__main__":
    main()

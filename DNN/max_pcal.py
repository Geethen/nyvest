"""Single-band max calibrated-probability (winning-class confidence) raster.

Reduces the C-band uq_*_pcal.tif (per-class calibrated probability, uint16 scaled
x60000) to one band = the per-pixel MAX probability, i.e. how confident the model
is in whichever class it picked. A quick, viewable confidence layer to drop next
to classified_2024.tif without carrying all C probability bands.

Same encoding as the source (uint16, ÷60000 -> [0,1], nodata 65535), so it decodes
identically. Windowed on the source's native block grid -> bounded memory on the
44k x 75k county grid.

Run:
  PY DNN/max_pcal.py --pcal uq_2024_pcal.tif --out uq_2024_maxpcal.tif
"""
from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.windows import Window

PCAL_NODATA = 65535


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcal", required=True, help="uq_*_pcal.tif (C-band uint16)")
    ap.add_argument("--out", required=True, help="single-band max-proba GeoTIFF")
    ap.add_argument("--block", type=int, default=4096, help="window size (px)")
    args = ap.parse_args()

    with rasterio.open(args.pcal) as src:
        prof = dict(driver="GTiff", crs=src.crs, transform=src.transform,
                    width=src.width, height=src.height, count=1,
                    dtype="uint16", nodata=PCAL_NODATA, compress="deflate",
                    predictor=2, tiled=True, blockxsize=512, blockysize=512,
                    bigtiff="IF_SAFER")
        with rasterio.open(args.out, "w", **prof) as dst:
            dst.set_band_description(1, "max_pcal")
            for row in range(0, src.height, args.block):
                h = min(args.block, src.height - row)
                for col in range(0, src.width, args.block):
                    w = min(args.block, src.width - col)
                    win = Window(col, row, w, h)
                    block = src.read(window=win)          # (C, h, w) uint16
                    # nodata pixels are nodata in every band; take the max over
                    # bands, but keep genuine nodata as nodata (a valid pixel has
                    # at least one non-nodata band).
                    valid = (block != PCAL_NODATA).any(axis=0)
                    m = np.where(block == PCAL_NODATA, 0, block).max(axis=0)
                    out = np.where(valid, m, PCAL_NODATA).astype(np.uint16)
                    dst.write(out, 1, window=win)
    print(f"wrote {args.out}: 1-band uint16 max calibrated proba "
          f"(÷60000 -> [0,1]), nodata {PCAL_NODATA}", flush=True)


if __name__ == "__main__":
    main()

"""Take one downloaded AEF year from raw tiles to an inference-ready VRT.

Wraps steps 5-7 of DNN/DATA_INFERENCE.md for a single year and — the point of
the script — REFUSES to hand on a mosaic that does not actually cover the AOI.
Two independent failures have silently truncated this pipeline before, and
neither raised anything:

  * wrong bounds at fetch time    -> the AEF rectangle misses part of the AOI
                                     (61% of the AOI was never downloaded once)
  * zone-overlap nodata clobber   -> the rectangle is right but adjacent-zone
                                     all-nodata tiles paint over real data along
                                     every 31N/32N seam (~23% of pixels lost)

The first is caught by comparing the AOI to the VRT's extent, the second only by
counting VALID PIXELS inside the AOI. Both checks run here, at full resolution
for the extent test and on a fixed decimated grid for the coverage estimate —
decimation is fine for a go/no-go threshold but NOT for a reported number, so
the printed coverage is labelled as an estimate (see DATA_INFERENCE.md, "Do not
estimate coverage from a decimated raster read").

Run:
  ~/…/python DNN/prep_year.py --year 2024
  ~/…/python DNN/prep_year.py --year 2018 --workers 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import rasterio                                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
BASE = Path(os.environ.get("NYVEST_DATA_DIR", "/data/P-Prosjekter2/154001_nyvest"))
AOI = BASE / "GIS" / "Boundaries" / "nyvest_fylker.shp"
N_TILES_EXPECTED = 46
# 7b threshold. The 3-county AOI measured 88.4% AE-valid in the 2024 reference
# run; the remaining ~11% is genuine water (one large fjord system plus
# speckle). The buggy-VRT failure mode produced 65%, so anything below ~80% is
# the bug, not the coastline.
MIN_COVERAGE = 0.80


def sh(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd)}")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--float32", action="store_true",
                    help="write dequantised float32 tiles instead of int8 "
                         "(int8 is lossless here and ~4x faster to read)")
    ap.add_argument("--skip-prep", action="store_true")
    args = ap.parse_args()

    raw = BASE / f"aef_{args.year}"
    prepped = BASE / f"aef_{args.year}_32633"
    vrt = BASE / f"aef_{args.year}.vrt"
    t0 = time.perf_counter()

    tiles = sorted(raw.rglob("*.tiff"))
    print(f"=== {args.year}: {len(tiles)} raw tiles in {raw}", flush=True)
    if len(tiles) != N_TILES_EXPECTED:
        raise SystemExit(
            f"expected {N_TILES_EXPECTED} raw tiles for the 3-county AOI, found "
            f"{len(tiles)}. Prepping a partial download silently produces a "
            f"partial map — finish the s5cmd run first.")
    # A truncated download leaves a short file that prep would read as valid
    # until it hits the missing bytes. Cheap guard: every tile must be openable
    # and carry 64 bands.
    for t in tiles:
        with rasterio.open(t) as s:
            if s.count < 64:
                raise SystemExit(f"{t} has {s.count} bands — truncated download?")

    if not args.skip_prep:
        print(f"--- prep -> {prepped}", flush=True)
        cmd = [PY, REPO / "DNN" / "prep_aef_tiles.py", "--glob", f"{raw}/**/*.tiff",
               "--out-dir", prepped, "--workers", args.workers]
        if not args.float32:
            cmd.append("--int8")
        sh(cmd)

    print(f"--- mosaic -> {vrt}", flush=True)
    sh([PY, REPO / "DNN" / "build_vrt.py", "--glob", f"{prepped}/*.tif", "--out", vrt])

    # ---- 7a extent + 7b per-pixel coverage --------------------------------
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.features import geometry_mask

    gdf = gpd.read_file(AOI)
    with rasterio.open(vrt) as ae:
        gdf = gdf.to_crs(ae.crs)
        aoi = gdf.union_all()
        outside = aoi.difference(box(*ae.bounds)).area / aoi.area
        print(f"\n  7a AOI area outside the AEF rectangle: {100*outside:.2f}%  (want ~0)")
        if outside > 0.005:
            raise SystemExit(
                f"7a FAILED: {100*outside:.1f}% of the AOI is outside the mosaic. "
                f"The tile list did not cover the AOI — re-derive bounds from "
                f"{AOI} and re-fetch (DATA_INFERENCE.md 'AOI-bounds pitfall').")
        shp = (2000, 1200)
        b1 = ae.read(1, out_shape=shp)
        valid = np.isfinite(b1) & (b1 != 0)
        if ae.nodata is not None:
            valid &= b1 != ae.nodata
        dt = ae.transform * rasterio.Affine.scale(ae.width / shp[1], ae.height / shp[0])
        inside = geometry_mask(list(gdf.geometry.values), out_shape=shp,
                               transform=dt, invert=True, all_touched=True)
        cov = (valid & inside).sum() / max(inside.sum(), 1)
        print(f"  7b AE-valid within AOI (decimated ESTIMATE): {100*cov:.1f}%  "
              f"(want ~88%; <{100*MIN_COVERAGE:.0f}% means the VRT clobbered data)")
        print(f"  grid: {ae.width} x {ae.height}  {ae.dtypes[0]}  "
              f"{ae.width*ae.height/1e9:.2f}B px  crs={ae.crs}")
        if cov < MIN_COVERAGE:
            raise SystemExit(
                f"7b FAILED: only {100*cov:.1f}% of the AOI has valid AE data. "
                f"This is the zone-overlap nodata bug — build_vrt.py must emit "
                f"ComplexSource+NODATA (it does when tiles carry nodata; check "
                f"the prepped tiles have one).")
    print(f"\n{args.year} READY -> {vrt}   ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

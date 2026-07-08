"""Locate + stream AlphaEarth Foundations (AEF) embeddings from source.coop.

Investigated as an alternative to the incomplete P-drive embeddings mosaic
(only 154 tiles exist locally, covering a subset of the 3-county AOI) and to
GEE's synchronous pixel API. See DNN/README.md "Inference data sourcing" for
the full write-up. Summary of the 3 options tested:

  - **source.coop -- winner.** Free HTTPS, no auth, no billing account. Data
    is int8-quantized (`dequantize()` below) and bottom-up-oriented COGs in
    native UTM zones (the nyvest AOI spans zones 31N/32N, not the project's
    working 32633/33N). ~140 MB/s per connection once HTTP/2 multiplexing is
    enabled -- the default vsicurl config serializes the 64 per-band block
    range-requests on these band-interleaved COGs and is ~5x slower
    (measured: 41s -> 8s for a 1024x1024x64 int8 block). The dataset ships a
    per-tile `.vrt` that fixes the bottom-up flip, but it hardcodes a
    `/vsis3/...` source and its VRTWarpedDataset warp is itself slow even
    over vsicurl (measured 41s for a 512x512x64 read that takes 2-4s off the
    raw `.tiff` -- the warp machinery, not the transport, is the cost).
    **Fastest correct path found:** read the raw `.tiff` directly via
    `/vsicurl` (no VRT), flip axis 1 (row) in numpy, dequantize in numpy.
    Reprojecting UTM31N/32N -> 32633 needs `rasterio.vrt.WarpedVRT` in
    Python at read time (measured 0.09s/window after GDAL's block cache
    warms) rather than a static warped-VRT file -- GDAL's serialized warp
    VRT XML is what's slow, not the warp computation itself.
  - GCS (`gs://alphaearth_foundations`): identical data, but the bucket is
    requester-pays -- needs a billing-enabled GCP project + real credentials
    (GS_NO_SIGN_REQUEST does not work here, unlike source.coop's S3 mirror).
  - GEE `ee.data.computePixels`: hard-capped at 48 MB/request (~290x290 px
    at float64 x 64 bands), ~12k px/s single-threaded -> full AOI (270M px)
    is ~6 hours single-threaded, needs ~40 parallel threads to match
    source.coop's single-thread throughput. `Export.image.toCloudStorage`
    avoids the cap (server-side tiling) but is an async batch job (minutes-
    hours queue latency), not usable interactively.

Usage:
  # 1. one-time: download the spatial index (~78 MB parquet)
  curl -o aef_index.parquet https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet

  # 2. find which raw tiles cover an AOI/year
  PY DNN/fetch_aef_sourcecoop.py --index aef_index.parquet \\
      --bounds -80000 6489990 70000 6670000 --bounds-crs EPSG:32633 --year 2024

  # 3. read a window from one tile, reprojected + dequantized (see open_tile_window)
"""
from __future__ import annotations

import argparse
import os

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import box

# HTTP/2 multiplexing is the single biggest lever: without it, GDAL issues the
# 64 per-band block range-requests serially on these band-interleaved COGs,
# ~5x slower (measured: 41s -> 8s for a 1024x1024x64 int8 block).
GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    # source.coop's Cloudflare proxy returns sporadic HTTP 500s on the scattered
    # per-band-block range pattern; without retry, GDAL crashes on the first one
    # with a misleading "ZSTDDecode: Unknown frame descriptor". Mandatory.
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "1",
}
for k, v in GDAL_ENV.items():
    os.environ.setdefault(k, v)

BASE_URL = "https://data.source.coop/tge-labs/aef/v1/annual"
AEF_NODATA = -128


def dequantize(raw: np.ndarray) -> np.ndarray:
    """int8 [-127,127] (-128=nodata) -> float32 unit-norm embedding, per AEF spec.

    Nodata (-128) pixels are passed through as -128 in the output (NOT run
    through the dequantization formula, which has no special case for it and
    would silently produce a bogus non-zero "valid-looking" value).
    """
    f = raw.astype(np.float32)
    out = ((f / 127.5) ** 2) * np.sign(f)
    out[raw == AEF_NODATA] = AEF_NODATA
    return out


def find_tiles(index_path: str, bounds, bounds_crs: str, year: int) -> gpd.GeoDataFrame:
    """Tiles from the AEF index intersecting `bounds` (in `bounds_crs`) for `year`."""
    gdf = gpd.read_parquet(index_path)
    # transform_bounds densifies the edges before reprojecting, so the lon/lat
    # envelope covers the projected bbox's edge bulge (real at Norwegian
    # latitudes) — transforming only the 4 corners under-covers and can miss
    # edge tiles.
    t = Transformer.from_crs(bounds_crs, "EPSG:4326", always_xy=True)
    minlon, minlat, maxlon, maxlat = t.transform_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], densify_pts=21)
    aoi = box(minlon, minlat, maxlon, maxlat)
    return gdf[(gdf["year"] == year) & gdf.intersects(aoi)]


def tile_url(s3_path: str) -> str:
    """index `path` is an s3:// URI; source.coop also mirrors it over free HTTPS."""
    key = s3_path.split("/v1/annual/", 1)[1]
    return f"/vsicurl/{BASE_URL}/{key}"


def open_tile_window(s3_path: str, dst_crs: str, bounds, resolution=10.0):
    """Open one AEF tile reprojected to `dst_crs`, read `bounds`, return a
    dequantized float32 [64, h, w] array (north-up, ready for model input).

    `bounds` is (minx, miny, maxx, maxy) in `dst_crs`. Uses WarpedVRT (fast,
    in-process warp) rather than a static warped-VRT file (measured much
    slower -- see module docstring).
    """
    src = rasterio.open(tile_url(s3_path))
    vrt = WarpedVRT(src, crs=dst_crs, resampling=Resampling.nearest,
                     nodata=AEF_NODATA)
    window = vrt.window(*bounds)
    raw = vrt.read(range(1, 65), window=window, out_dtype="int16")
    vrt.close()
    src.close()
    return dequantize(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="local path to aef_index.parquet")
    ap.add_argument("--bounds", type=float, nargs=4, required=True,
                     metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--bounds-crs", default="EPSG:32633")
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()

    tiles = find_tiles(args.index, args.bounds, args.bounds_crs, args.year)
    print(f"{len(tiles)} AEF tiles cover the AOI for {args.year} "
          f"(zones: {sorted(tiles['utm_zone'].unique())})")
    for _, row in tiles.iterrows():
        print(f"  {row['utm_zone']}  {tile_url(row['path'])}")


if __name__ == "__main__":
    main()

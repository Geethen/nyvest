"""Build a GDAL .vrt mosaic over a folder of aligned single-grid GeoTIFF tiles.

Written with rasterio only (no osgeo/gdalbuildvrt CLI in this venv). All tiles
must share CRS, resolution and pixel grid (verified true for the nyvest
embeddings: EPSG:32633 @ 10 m, offsets 0). Emits a plain VRT that rasterio /
predict_raster.py open like any raster, so 154 tiles become one virtual mosaic
and inference runs ONCE instead of per-tile.

Usage:
  PY DNN/build_vrt.py --glob '/data/.../embeddings/*.tif' --out embeddings.vrt
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for input tiles")
    ap.add_argument("--out", required=True, help="output .vrt path")
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob!r}")

    # Read grid geometry from every tile; derive the mosaic extent.
    infos, res_x, res_y, crs_wkt, nbands, dtype = [], None, None, None, None, None
    nodata = None
    minx = miny = np.inf
    maxx = maxy = -np.inf
    for f in files:
        with rasterio.open(f) as s:
            t = s.transform
            if res_x is None:
                res_x, res_y = t.a, t.e            # e is negative (north-up)
                crs_wkt = s.crs.to_wkt()
                nbands = s.count
                dtype = s.dtypes[0]
                nodata = s.nodata
            b = s.bounds
            minx, miny = min(minx, b.left), min(miny, b.bottom)
            maxx, maxy = max(maxx, b.right), max(maxy, b.top)
            infos.append((f, s.width, s.height, t.c, t.f, s.descriptions))

    width = int(round((maxx - minx) / res_x))
    height = int(round((maxy - miny) / (-res_y)))
    gdal_dtype = {"float64": "Float64", "float32": "Float32",
                  "int16": "Int16", "uint8": "Byte"}[dtype]

    lines = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
             f'  <SRS>{escape(crs_wkt)}</SRS>',
             f'  <GeoTransform>{minx}, {res_x}, 0.0, {maxy}, 0.0, {res_y}</GeoTransform>']
    # When tiles carry a nodata value, adjacent tiles OVERLAP (AEF ships one tile
    # per UTM zone, and a tile's footprint spills into the neighbouring zone's
    # area where it is all-nodata). A plain <SimpleSource> paints sources in
    # order, so a later all-nodata tile OVERWRITES an earlier tile's real data
    # along every zone seam — silent holes. <ComplexSource> + <NODATA> makes GDAL
    # skip nodata source pixels so valid data always wins the overlap. Emit a
    # band-level <NoDataValue> too so the mosaic reports nodata correctly.
    # If the tiles have no nodata (e.g. P-drive embeddings use all-zero gaps,
    # masked downstream by --mask-allzero), fall back to the plain SimpleSource.
    src_tag = "ComplexSource" if nodata is not None else "SimpleSource"
    nd_str = repr(float(nodata)) if nodata is not None else None
    descs = infos[0][5]
    for band in range(1, nbands + 1):
        desc = descs[band - 1] if descs and descs[band - 1] else ""
        lines.append(f'  <VRTRasterBand dataType="{gdal_dtype}" band="{band}">')
        if nd_str is not None:
            lines.append(f'    <NoDataValue>{nd_str}</NoDataValue>')
        if desc:
            lines.append(f'    <Description>{escape(desc)}</Description>')
        for (f, w, h, cx, fy, _d) in infos:
            xoff = int(round((cx - minx) / res_x))
            yoff = int(round((maxy - fy) / (-res_y)))
            rel = os.path.relpath(f, start=Path(args.out).resolve().parent)
            nd_elem = f'<NODATA>{nd_str}</NODATA>' if nd_str is not None else ""
            lines.append(
                f'    <{src_tag}>'
                f'<SourceFilename relativeToVRT="1">{escape(rel)}</SourceFilename>'
                f'<SourceBand>{band}</SourceBand>'
                f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
                f'<DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}"/>'
                f'{nd_elem}'
                f'</{src_tag}>')
        lines.append('  </VRTRasterBand>')
    lines.append('</VRTDataset>')
    Path(args.out).write_text("\n".join(lines))

    with rasterio.open(args.out) as v:
        print(f"VRT {args.out}: {v.width} x {v.height}, {v.count} bands, {v.dtypes[0]}, "
              f"{v.width*v.height/1e6:.1f}M px, from {len(files)} tiles", flush=True)


if __name__ == "__main__":
    main()

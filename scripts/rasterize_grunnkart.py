"""Rasterize NIBIO Grunnkart for the three NYVEST counties into one 10 m GeoTIFF.

Output:
  <data-root>/GIS/NIBIO/Version_2/rasterized_10m/grunnkart_nyvest_10m.tif

A single sparse, tiled, zstd-compressed UInt8 raster in EPSG:25832 covering the
union bbox of Rogaland + Vestland + Møre og Romsdal. Empty blocks between
counties are not materialised on disk (GDAL SPARSE_OK).

Strategy (vector-first reclassify, window-wise burn):
  1. Each county's Grunnkart layer is read once, reclassified to a uint8 class
     code in vector space, and clipped to its county boundary.
  2. The polygons are written into the global grid block-by-block via an
     R-tree spatial index — only blocks touched by polygons are read/written.
  3. FKB grønnstruktur (built/road/grey) is read with bbox + SQL filter so
     only built features for the three counties enter memory, then burned on
     top of the grunnkart pass.

Class codes:
   0  nodata / outside study area
   1  sand        (arealdekke=Snaumark_skrinn | grunnforhold=Jorddekt
                   | okosystemtype_3=Mineral extraction sites)
   2  rock        (arealdekke in Snaumark_impediment, Snaumark_uspesifisert)
   3  crop        (okosystemtype_1=Cropland)
   4  forest      (okosystemtype_1=Forest and woodlands)
   5  grassland   (okosystemtype_1=Grassland)
   6  scrub       (okosystemtype_1=Heathland and shrub)
   7  wetland     (okosystemtype_1=Inland wetlands)
   8  freshwater  (okosystemtype_1 in Rivers and canals, Lakes and reservoirs)
   9  marine      (okosystemtype_1=Marine ecosystems)
  10  built       (FKB klasse_navn in FKB_bygg, FKB_vei, greyArea)
  11  sparse      (okosystemtype_1=Sparsely vegetated ecosystems)
  12  snow        (arealdekke=Snoisbre)
  13  other       (grunnkart polygon that matched no rule)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from shapely.geometry import box


DATA_ROOT_CANDIDATES = (
    Path("/data/P-Prosjekter2/154001_nyvest"),
    Path("P:/154001_nyvest"),
)

# R: drive (shared GeoSpatialData) — auto-detection. Used for v2 built sources.
R_ROOT_CANDIDATES = (
    Path("/data/R/GeoSpatialData"),
    Path("R:/GeoSpatialData"),
)

# Pre-2020 kommune-code prefixes per study county. Vestland (current code 46)
# was created in 2020 from Hordaland (12) + Sogn og Fjordane (14); the FKB
# Bygning 2018 dataset is keyed by the old codes.
COUNTY_KOMM_PREFIXES = {
    "rogaland": ("11",),
    "more_og_romsdal": ("15",),
    "vestland": ("12", "14"),
}

TARGET_CRS = "EPSG:25832"
GREY_CRS = "EPSG:25833"
RES = 10.0
BLOCK = 512


@dataclass(frozen=True)
class County:
    name: str
    gdb_dir: str
    gdb_inner: str
    boundary_navn: str


COUNTIES = (
    County(
        name="rogaland",
        gdb_dir="Basisdata_11_Rogaland_25832_GrunnkartArealregnskap",
        gdb_inner="11_25832_arealregnskap_gdb.gdb",
        boundary_navn="Rogaland",
    ),
    County(
        name="more_og_romsdal",
        gdb_dir="Basisdata_15_More_og_Romsdal_25832",
        gdb_inner="15_25832_arealregnskap_gdb.gdb",
        boundary_navn="Møre og Romsdal",
    ),
    County(
        name="vestland",
        gdb_dir="Basisdata_46_Vestland_25832_GrunnkartArealregnskap",
        gdb_inner="46_25832_arealregnskap_gdb.gdb",
        boundary_navn="Vestland",
    ),
)


def detect_data_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit)
    else:
        env = os.environ.get("NYVEST_DATA_DIR")
        root = Path(env) if env else next((p for p in DATA_ROOT_CANDIDATES if p.exists()), None)
    if root is None or not root.exists():
        tried = ", ".join(str(p) for p in DATA_ROOT_CANDIDATES)
        raise FileNotFoundError(f"No NYVEST data root. Set --data-root or NYVEST_DATA_DIR. Tried: {tried}")
    return root


def pick_grunnkart_layer(gdb: Path) -> str:
    layers = pyogrio.list_layers(gdb)[:, 0].tolist()
    for name in layers:
        if "arealregnskap" in name.lower():
            return name
    if len(layers) == 1:
        return layers[0]
    raise ValueError(f"Cannot pick grunnkart layer in {gdb}. Found: {layers}")


def norm(s) -> np.ndarray:
    return pd.Series(np.asarray(s)).fillna("").astype(str).str.strip().str.lower().to_numpy()


def reclass_grunnkart(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    required = ["arealdekke", "grunnforhold", "okosystemtype_3", "okosystemtype_1", "geometry"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing grunnkart fields: {missing}")

    arealdekke = norm(gdf["arealdekke"].to_numpy())
    grunnforhold = norm(gdf["grunnforhold"].to_numpy())
    okosys3 = norm(gdf["okosystemtype_3"].to_numpy())
    okosys1 = norm(gdf["okosystemtype_1"].to_numpy())

    cls = np.full(len(gdf), 13, dtype=np.uint8)

    cls[
        (arealdekke == "snaumark_skrinn")
        | (grunnforhold == "jorddekt")
        | (okosys3 == "mineral extraction sites")
    ] = 1
    cls[(arealdekke == "snaumark_impediment") | (arealdekke == "snaumark_uspesifisert")] = 2
    cls[okosys1 == "cropland"] = 3
    cls[okosys1 == "forest and woodlands"] = 4
    cls[okosys1 == "grassland"] = 5
    cls[okosys1 == "heathland and shrub"] = 6
    cls[okosys1 == "inland wetlands"] = 7
    cls[(okosys1 == "rivers and canals") | (okosys1 == "lakes and reservoirs")] = 8
    cls[okosys1 == "marine ecosystems"] = 9
    cls[okosys1 == "sparsely vegetated ecosystems"] = 11
    cls[arealdekke == "snoisbre"] = 12

    out = gpd.GeoDataFrame(
        {"class_code": cls},
        geometry=gdf.geometry.values,
        crs=gdf.crs,
    )
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    return out


def load_boundary(boundaries_shp: Path) -> gpd.GeoDataFrame:
    bnd = gpd.read_file(boundaries_shp, encoding="latin1")
    if "navn" not in bnd.columns:
        raise KeyError(f"Boundary shapefile {boundaries_shp} has no 'navn' field. Cols: {bnd.columns.tolist()}")
    return bnd.to_crs(TARGET_CRS)


def aligned_grid(bounds: tuple[float, float, float, float], res: float):
    minx, miny, maxx, maxy = bounds
    left = math.floor(minx / res) * res
    bottom = math.floor(miny / res) * res
    right = math.ceil(maxx / res) * res
    top = math.ceil(maxy / res) * res
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    transform = from_origin(left, top, res, res)
    return transform, width, height


def rasterize_to_subgrid(gdf: gpd.GeoDataFrame, res: float) -> tuple[np.ndarray, rasterio.Affine]:
    """Rasterize all features in gdf to a tight, block-aligned subgrid."""
    transform, width, height = aligned_grid(tuple(gdf.total_bounds), res)
    arr = rasterize(
        ((geom, int(code)) for geom, code in zip(gdf.geometry.values, gdf["class_code"].to_numpy())),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return arr, transform


def merge_into_dst(dst, arr: np.ndarray, src_transform: rasterio.Affine, *, label: str) -> None:
    """Read-modify-write each touched destination block with the non-zero pixels of arr.

    Uses block-aligned windows on dst so we only pay one read per block.
    """
    if arr.size == 0:
        return
    h_src, w_src = arr.shape
    # Compute the destination window covering the entire arr footprint.
    src_xmin = src_transform.c
    src_ymax = src_transform.f
    src_xmax = src_xmin + w_src * src_transform.a
    src_ymin = src_ymax + h_src * src_transform.e  # e is negative
    full_win = from_bounds(src_xmin, src_ymin, src_xmax, src_ymax, dst.transform)
    col0 = int(round(full_win.col_off))
    row0 = int(round(full_win.row_off))

    by, bx = dst.block_shapes[0]
    # Block-align: first block start <= row0/col0; clamp to 0 so we never
    # issue a read/write at a negative offset if polygons slightly overshoot
    # the dst grid edge.
    r_start = max(0, (row0 // by) * by)
    c_start = max(0, (col0 // bx) * bx)

    n_blocks_total = 0
    n_blocks_written = 0
    for r in range(r_start, row0 + h_src, by):
        for c in range(c_start, col0 + w_src, bx):
            n_blocks_total += 1
            wh = min(by, dst.height - r)
            ww = min(bx, dst.width - c)
            if wh <= 0 or ww <= 0:
                continue
            # arr indices for this block
            ar_r0 = r - row0
            ar_c0 = c - col0
            ar_r1 = ar_r0 + wh
            ar_c1 = ar_c0 + ww
            # Clip into arr bounds
            sr0 = max(ar_r0, 0)
            sc0 = max(ar_c0, 0)
            sr1 = min(ar_r1, h_src)
            sc1 = min(ar_c1, w_src)
            if sr0 >= sr1 or sc0 >= sc1:
                continue
            tile = arr[sr0:sr1, sc0:sc1]
            if not tile.any():
                continue
            # Where in the destination block this tile lands
            dr0 = sr0 - ar_r0
            dc0 = sc0 - ar_c0
            dr1 = dr0 + (sr1 - sr0)
            dc1 = dc0 + (sc1 - sc0)

            window = Window(c, r, ww, wh)
            base = dst.read(1, window=window)
            mask = tile != 0
            base[dr0:dr1, dc0:dc1][mask] = tile[mask]
            dst.write(base, 1, window=window)
            n_blocks_written += 1
    print(f"  [{label}] blocks scanned: {n_blocks_total}, written: {n_blocks_written}")


def block_windows_in_bounds(dst, bounds: tuple[float, float, float, float]):
    """Yield block-aligned Windows that intersect bounds."""
    minx, miny, maxx, maxy = bounds
    full = from_bounds(minx, miny, maxx, maxy, dst.transform)
    by, bx = dst.block_shapes[0]
    c0 = (max(0, int(math.floor(full.col_off))) // bx) * bx
    r0 = (max(0, int(math.floor(full.row_off))) // by) * by
    c_end = min(dst.width, int(math.ceil(full.col_off + full.width)))
    r_end = min(dst.height, int(math.ceil(full.row_off + full.height)))
    for r in range(r0, r_end, by):
        for c in range(c0, c_end, bx):
            yield Window(c, r, min(bx, dst.width - c), min(by, dst.height - r))


def burn_sindex(dst, gdf: gpd.GeoDataFrame, *, label: str) -> None:
    """Block-by-block overwrite burn using R-tree spatial index.

    Used for the built pass where polygons are sparse so most blocks hit 0 index
    results and are skipped cheaply.
    """
    if gdf.empty:
        return
    sindex = gdf.sindex
    geoms = gdf.geometry.values
    codes = gdf["class_code"].to_numpy()
    bounds = tuple(gdf.total_bounds)
    n_scanned = 0
    n_written = 0
    for window in block_windows_in_bounds(dst, bounds):
        n_scanned += 1
        win_bounds = rasterio.windows.bounds(window, dst.transform)
        win_poly = box(*win_bounds)
        idx = list(sindex.query(win_poly, predicate="intersects"))
        if not idx:
            continue
        h, w = int(window.height), int(window.width)
        wtransform = dst.window_transform(window)
        burnt = rasterize(
            ((geoms[i], int(codes[i])) for i in idx),
            out_shape=(h, w),
            transform=wtransform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        mask = burnt != 0
        if not mask.any():
            continue
        base = dst.read(1, window=window)
        base[mask] = burnt[mask]
        dst.write(base, 1, window=window)
        n_written += 1
    print(f"  [{label}] blocks scanned: {n_scanned}, written: {n_written}")


def _rasterize_county(
    root: Path, county: County, res: float
) -> tuple[np.ndarray, rasterio.Affine, str]:
    """Read, reclassify, and rasterize one county to a numpy subgrid.

    No dst access — safe to run in a thread alongside other counties.
    Returns (arr, src_transform, label) for the caller to merge into dst.
    """
    gdb = root / "GIS" / "NIBIO" / "Version_2" / county.gdb_dir / county.gdb_inner
    if not gdb.exists():
        raise FileNotFoundError(f"Missing gdb: {gdb}")
    layer = pick_grunnkart_layer(gdb)
    print(f"[{county.name}] reading {gdb.name}::{layer}")
    t0 = time.time()
    gdf = gpd.read_file(gdb, layer=layer, columns=["arealdekke", "grunnforhold", "okosystemtype_3", "okosystemtype_1"])
    print(f"[{county.name}] read {len(gdf):,} polygons in {time.time() - t0:.1f}s")

    if gdf.crs is None or str(gdf.crs).lower() != TARGET_CRS.lower():
        gdf = gdf.to_crs(TARGET_CRS)

    # Grunnkart layers are already published per-county; no clipping needed.
    gdf = reclass_grunnkart(gdf)
    print(f"[{county.name}] rasterizing {len(gdf):,} polygons to subgrid")
    t0 = time.time()
    arr, src_transform = rasterize_to_subgrid(gdf, res)
    print(f"[{county.name}] subgrid {arr.shape[1]:,} x {arr.shape[0]:,} px in {time.time() - t0:.1f}s")
    return arr, src_transform, county.name


def detect_r_root(explicit: str | None) -> Path:
    """Locate the R:/GeoSpatialData mount used for v2 built sources."""
    if explicit:
        root = Path(explicit)
    else:
        env = os.environ.get("NYVEST_R_DIR")
        root = Path(env) if env else next((p for p in R_ROOT_CANDIDATES if p.exists()), None)
    if root is None or not root.exists():
        tried = ", ".join(str(p) for p in R_ROOT_CANDIDATES)
        raise FileNotFoundError(f"No R:/GeoSpatialData mount. Set --r-root or NYVEST_R_DIR. Tried: {tried}")
    return root


def _read_one_bygning_gdb(gdb_dir: Path) -> gpd.GeoDataFrame | None:
    """Read fkb_bygning_omrade from one per-kommune FKB Bygning 2018 GDB."""
    # The .gdb is nested one level inside the Basisdata_*_FGDB directory.
    inner = next((p for p in gdb_dir.iterdir() if p.suffix == ".gdb"), None)
    if inner is None:
        return None
    try:
        gdf = gpd.read_file(inner, layer="fkb_bygning_omrade", columns=["bygningstype"])
    except Exception as e:
        print(f"  [bygning] skip {gdb_dir.name}: {e}")
        return None
    if gdf.empty:
        return None
    if gdf.crs is None or str(gdf.crs).lower() != TARGET_CRS.lower():
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf[["geometry"]]


def read_fkb_bygning(r_root: Path, counties: list[County]) -> gpd.GeoDataFrame:
    """Glob and read all per-kommune FKB Bygning 2018 building polygons for counties."""
    bygning_dir = (
        r_root / "Buildings" / "Norway_FKB_Buildings" / "Processed" / "FKB_Norway_buildings"
        / "Original" / "versjon20181231" / "FKB-Bygning FGDB-format"
    )
    if not bygning_dir.exists():
        raise FileNotFoundError(f"Missing FKB Bygning dir: {bygning_dir}")

    prefixes = tuple(prefix for c in counties for prefix in COUNTY_KOMM_PREFIXES[c.name])
    candidates = sorted(
        d for d in bygning_dir.iterdir()
        if d.is_dir() and d.name.startswith("Basisdata_") and any(
            d.name.startswith(f"Basisdata_{pref}") for pref in prefixes
        )
    )
    print(f"[bygning] {len(candidates)} per-kommune GDBs matching prefixes {prefixes}")

    n_workers = min(8, max(1, os.cpu_count() or 1))
    parts: list[gpd.GeoDataFrame] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for fut in as_completed({pool.submit(_read_one_bygning_gdb, d): d for d in candidates}):
            gdf = fut.result()
            if gdf is not None and not gdf.empty:
                parts.append(gdf)
    if not parts:
        raise RuntimeError("No FKB Bygning buildings read.")
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=TARGET_CRS)
    print(f"[bygning] read {len(combined):,} buildings from {len(parts)} kommunes in {time.time() - t0:.1f}s")
    return combined


def process_built_v2(
    dst,
    p_root: Path,
    r_root: Path,
    counties: list[County],
    boundary_25832: gpd.GeoDataFrame,
) -> None:
    """v2 built pass: buildings from FKB Bygning 2018, roads + grey from FKB Grønnstruktur.

    Buildings are sourced from per-kommune FKB Bygning 2018 GDBs (the dedicated
    building product), while road and grey-area features remain from the FKB
    Grønnstruktur GDB used by v1. FKB_bygg is excluded from the grey read to
    avoid double-burning building footprints from two sources.
    """
    # --- Buildings: FKB Bygning 2018 -----------------------------------------
    buildings = read_fkb_bygning(r_root, counties)
    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty]
    buildings = gpd.GeoDataFrame(
        {"class_code": np.full(len(buildings), 10, dtype=np.uint8)},
        geometry=buildings.geometry.values, crs=TARGET_CRS,
    )
    print(f"[built-v2] burning {len(buildings):,} buildings (FKB Bygning 2018) ...")
    burn_sindex(dst, buildings, label="bygning")
    del buildings

    # --- Roads + grey: FKB Grønnstruktur (exclude FKB_bygg) ------------------
    grey_gdb = p_root / "GIS" / "FKB" / "0000_25833_grønnstruktur_gdb.gdb"
    if not grey_gdb.exists():
        raise FileNotFoundError(f"Missing FKB grey gdb: {grey_gdb}")
    bnd25833 = boundary_25832.to_crs(GREY_CRS)
    bbox25833 = tuple(bnd25833.total_bounds)
    where = "klasse_navn IN ('FKB_vei', 'greyArea')"
    print(f"[built-v2] reading FKB Grønnstruktur roads + grey (bbox + SQL filter)")
    t0 = time.time()
    grey = gpd.read_file(grey_gdb, layer="grønnstruktur", bbox=bbox25833, where=where,
                         columns=["klasse_navn"])
    print(f"[built-v2] read {len(grey):,} road/grey polygons in {time.time() - t0:.1f}s")
    if grey.empty:
        return
    grey = grey[grey.geometry.notna() & ~grey.geometry.is_empty]
    grey = grey.to_crs(TARGET_CRS)
    grey = gpd.GeoDataFrame(
        {"class_code": np.full(len(grey), 10, dtype=np.uint8)},
        geometry=grey.geometry.values, crs=grey.crs,
    )
    print(f"[built-v2] burning {len(grey):,} road/grey polygons ...")
    burn_sindex(dst, grey, label="vei_grey")


def process_built(dst, root: Path, union_bounds_25832: tuple[float, float, float, float],
                  boundary_25832: gpd.GeoDataFrame) -> None:
    grey_gdb = root / "GIS" / "FKB" / "0000_25833_grønnstruktur_gdb.gdb"
    if not grey_gdb.exists():
        raise FileNotFoundError(f"Missing FKB grey gdb: {grey_gdb}")

    # Project union bbox into 25833 to filter the read.
    bnd25833 = boundary_25832.to_crs(GREY_CRS)
    bbox25833 = tuple(bnd25833.total_bounds)
    where = "klasse_navn IN ('FKB_bygg', 'FKB_vei', 'greyArea')"
    print(f"[built] reading FKB grey (bbox in 25833 + SQL filter)")
    t0 = time.time()
    grey = gpd.read_file(grey_gdb, layer="grønnstruktur", bbox=bbox25833, where=where,
                         columns=["klasse_navn"])
    print(f"[built] read {len(grey):,} polygons in {time.time() - t0:.1f}s")
    if grey.empty:
        print("[built] no built features in bbox; skipping")
        return

    # Filter before reprojecting to avoid transforming geometries we'd discard.
    # The bbox read already limits grey to the study-area extent; skip the
    # expensive per-polygon boundary clip (which also fails on invalid FKB geoms).
    grey = grey[grey.geometry.notna() & ~grey.geometry.is_empty]
    grey = grey.to_crs(TARGET_CRS)
    grey = gpd.GeoDataFrame(
        {"class_code": np.full(len(grey), 10, dtype=np.uint8)},
        geometry=grey.geometry.values, crs=grey.crs,
    )
    print(f"[built] {len(grey):,} built polygons; burning block-by-block (overwrite)")
    burn_sindex(dst, grey, label="built")


def main():
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ.setdefault("GDAL_CACHEMAX", "4096")  # MB; tiles stay hot across R-M-W passes

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None, help="Output GeoTIFF path.")
    p.add_argument("--resolution", type=float, default=RES)
    p.add_argument("--skip-built", action="store_true", help="Skip FKB built overwrite pass.")
    p.add_argument("--built-only", action="store_true",
                   help="Skip grunnkart; open existing output in r+ and burn built pass only.")
    p.add_argument("--v2", action="store_true",
                   help="v2 built pass: buildings from FKB Bygning 2018 (per-kommune GDBs on R:), "
                        "roads + grey from FKB Grønnstruktur (FKB_bygg excluded).")
    p.add_argument("--r-root", default=None, help="Root of R:/GeoSpatialData mount (auto-detected if omitted).")
    p.add_argument("--copy-from", default=None,
                   help="If set together with --built-only, copy this existing tif to --out before burning. "
                        "Useful for producing a v2 raster that re-uses the v1 grunnkart base.")
    p.add_argument("--only", nargs="+", choices=[c.name for c in COUNTIES], default=None,
                   help="Restrict to a subset of counties. NOTE: output grid is sized to the selected "
                        "counties only, so the result is not drop-in compatible with a full run.")
    args = p.parse_args()

    root = detect_data_root(args.data_root)
    out_path = Path(args.out) if args.out else root / "GIS" / "NIBIO" / "Version_2" / "rasterized_10m" / "grunnkart_nyvest_10m.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    boundaries_shp = root / "GIS" / "Boundaries" / "nyvest_fylker.shp"
    if not boundaries_shp.exists():
        raise FileNotFoundError(f"Missing boundary shapefile: {boundaries_shp}")
    boundary = load_boundary(boundaries_shp)
    print(f"Loaded {len(boundary)} county boundaries from {boundaries_shp.name}")

    counties = [c for c in COUNTIES if args.only is None or c.name in args.only]
    selected = boundary[boundary["navn"].isin([c.boundary_navn for c in counties])]
    if selected.empty:
        raise ValueError(f"No boundaries matched: {[c.boundary_navn for c in counties]}")

    transform, width, height = aligned_grid(tuple(selected.total_bounds), args.resolution)
    print(f"Output grid: {width:,} x {height:,} px ({width * height / 1e6:.0f} Mpx) at {args.resolution} m")
    print(f"Output path: {out_path}")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": 0,
        "tiled": True,
        "blockxsize": BLOCK,
        "blockysize": BLOCK,
        "compress": "zstd",
        "zstd_level": 3,
        "predictor": 2,
        "interleave": "band",
        "sparse_ok": True,
        "bigtiff": "IF_SAFER",
    }

    t_start = time.time()
    # If --copy-from is given alongside --built-only, seed the output with the
    # existing v1 raster so we can swap only the built layer.
    if args.copy_from:
        if not args.built_only:
            raise ValueError("--copy-from requires --built-only.")
        src_path = Path(args.copy_from)
        if not src_path.exists():
            raise FileNotFoundError(f"--copy-from path does not exist: {src_path}")
        if src_path.resolve() != out_path.resolve():
            import shutil
            print(f"Copying {src_path} -> {out_path}")
            shutil.copyfile(src_path, out_path)

    open_mode = "r+" if args.built_only else "w+"
    if args.built_only and not out_path.exists():
        raise FileNotFoundError(f"--built-only requires existing output at {out_path}")
    with rasterio.open(out_path, open_mode, **({} if args.built_only else profile)) as dst:
        if not args.built_only:
            # Rasterize counties in parallel (pyogrio reads and rasterio.features.rasterize
            # both release the GIL, so threads get real concurrency). Each future returns
            # (arr, src_transform, label); merge into dst serially as futures complete so
            # each subgrid is freed immediately rather than holding all three at once.
            n_workers = min(len(counties), os.cpu_count() or 1)
            print(f"Rasterizing {len(counties)} counties with {n_workers} threads in parallel ...")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_rasterize_county, root, c, args.resolution): c
                    for c in counties
                }
                for future in as_completed(futures):
                    arr, src_transform, label = future.result()
                    merge_into_dst(dst, arr, src_transform, label=label)
                    del arr
        if not args.skip_built:
            if args.v2:
                r_root = detect_r_root(args.r_root)
                print(f"v2 built pass — R: root: {r_root}")
                process_built_v2(dst, root, r_root, counties, selected)
            else:
                process_built(dst, root, tuple(selected.total_bounds), selected)
        print("Building overviews ...")
        dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.mode)
        dst.update_tags(ns="rio_overview", resampling="mode")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

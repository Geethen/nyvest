"""Fast batch inference on AlphaEarth GeoTIFF stacks (P-drive / GEE / source.coop).

Source-agnostic: `--in` is any rasterio-openable path — a local GeoTIFF, a VRT,
or a Cloud-Optimized GeoTIFF URL (source.coop / GCS / S3, via GDAL /vsicurl).
Designed for wall-to-wall county-scale runs at 10 m (order 1e9 pixels): the model
compute is minutes; the wall is I/O, so this OVERLAPS read -> GPU -> write with a
producer/consumer pipeline instead of the naive read-infer-write-repeat loop.

Pipeline
--------
  READER threads  --(raw block queue)-->  GPU thread  --(class block queue)-->  WRITER thread
  * N reader threads prefetch blocks (rasterio.read releases the GIL, so threads
    give real read parallelism, and COG range-reads overlap network latency).
  * one GPU thread keeps the A40 fed: standardize + mean-softmax ensemble entirely
    on-device (dnn_core.Ensemble.predict_classmap_gpu), only the small int16 class
    map comes back to host.
  * one writer thread serializes GeoTIFF writes (GDAL is not thread-safe on one
    dataset).

Input contract
--------------
  * 64 AlphaEarth bands in band order A00..A63 (band index 1..64) — the extraction
    contract (EXPECTED_BANDS), matching ee.Image.toBands / source.coop AE exports.
  * lidar-trained models: pass --lidar-raster (bands elevation,tri,tch on the SAME
    grid/window) or omit to fill with the saved training medians.

Output: single-band int16 GeoTIFF of RAW class codes (e.g. 12 = snow/ice); nodata
pixels -> --nodata-class (default 0). Optional uncertainty quantification via
--uq-out (opt-in), written as THREE typed rasters derived from the given stem
(uq.tif -> uq_pcal.tif / uq_setsize.tif / uq_inset.tif):
  * uq_pcal.tif    — C-band uint16, per-class calibrated probability scaled by
                     60000 (÷60000 -> [0,1]; temperature scaling or Venn-Abers,
                     whichever fit_calibration.py found better-calibrated)
  * uq_setsize.tif — 1-band uint8, LAC+Mondrian conformal set size (0..C)
  * uq_inset.tif   — C-band uint8, per-class 0/1 conformal-set membership
Typed + deflate/predictor=2 instead of one 2C+1 float32 stack: ~3x smaller on
disk and each file opens on its own (a 3-county run went ~31 GB -> ~10 GB).

Run:
  ~/myprojects/recover/.venv/bin/python DNN/predict_raster.py \
    --in ae_tile.tif --out classified.tif \
    [--model models/dnn_final.pt] [--calib models/dnn_final_calib.npz] \
    [--uq-out uq.tif] \
    [--lidar-raster lidar.tif] [--nodata-class 0] [--readers 4] [--block 2048]
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

# BEFORE rasterio imports GDAL. The GTiff driver decompresses single-threaded by
# default, which is ~4x off this box's ceiling on a 64-band deflate tile
# (bench_io_formats.py --thread-sweep). It is NOT a free 4x on top of --readers:
# reader threads and GDAL's own threads are substitutes for the same cores and
# saturate together. It is worth having because the reader pool is bounded by
# --readers while a single large window read is not, and because blocks skipped
# by the AOI pre-filter leave reader threads idle. Overridable from the
# environment for anyone who needs to pin cores.
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import rasterio                                              # noqa: E402
import torch                                                 # noqa: E402
from rasterio.features import geometry_mask
from rasterio.windows import Window, bounds as window_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_core as C  # noqa: E402
# The one definition of the AEF quantisation, shared with the writer side so the
# two can never drift into disagreeing about what an int8 tile means.
from prep_aef_tiles import _DEQUANT_LUT  # noqa: E402

N_AE = 64
AE_BANDS = [f"A{i:02d}" for i in range(N_AE)]
LIDAR_COLS = ["elevation", "tri", "tch"]
# UQ typed-output encoding (see the udst setup in main()).
PCAL_SCALE = 60000     # calibrated proba [0,1] -> uint16 [0, 60000]; leaves
                       # 65535 free as nodata and keeps ~1e-5 precision
PCAL_NODATA = 65535    # uint16 nodata for the scaled-proba raster
UINT8_NODATA = 255     # uint8 nodata for set_size / inset (real values are 0..C)


def _blocks(width, height, bx, by):
    for row in range(0, height, by):
        h = min(by, height - row)
        for col in range(0, width, bx):
            w = min(bx, width - col)
            yield Window(col, row, w, h)


def _assemble_features(ae, lid, feat_cols, lidar_med):
    """Stack AE (+lidar) bands into the model's feat_cols order -> [F, h, w].

    `ae` may be float32 (dequantised tiles) or int8 (quantised tiles, see
    `_Dequant`). The output dtype follows `ae`: for int8 the AE columns stay
    quantised all the way to the GPU and only the lidar columns are float, so
    this returns a structure the GPU thread finishes rather than a finished
    array. Keeping one function for both keeps `feat_cols` ordering in one place.
    """
    band_of = {b: i for i, b in enumerate(AE_BANDS)}
    h, w = ae.shape[1], ae.shape[2]
    if ae.dtype == np.int8:
        # AE columns stay int8; lidar columns (at most 3) are built as float32
        # and carried alongside. See _Dequant.to_gpu for the join.
        lid_f = None
        if any(c in LIDAR_COLS for c in feat_cols):
            lid_f = np.empty((sum(c in LIDAR_COLS for c in feat_cols), h, w),
                             dtype=np.float32)
            for j, col in enumerate(c for c in feat_cols if c in LIDAR_COLS):
                if lid is not None:
                    b = lid[LIDAR_COLS.index(col)]
                    lid_f[j] = np.where(np.isfinite(b), b, lidar_med.get(col, 0.0))
                else:
                    lid_f[j] = lidar_med.get(col, 0.0)
        return (ae, lid_f)
    feats = np.empty((len(feat_cols), h, w), dtype=np.float32)
    for fi, col in enumerate(feat_cols):
        if col in band_of:
            feats[fi] = ae[band_of[col]]
        elif col in LIDAR_COLS:
            if lid is not None:
                b = lid[LIDAR_COLS.index(col)]
                feats[fi] = np.where(np.isfinite(b), b, lidar_med.get(col, 0.0))
            else:
                feats[fi] = lidar_med.get(col, 0.0)
        else:
            raise ValueError(f"model feature {col!r} is neither an AE band nor lidar")
    return feats


class _Dequant:
    """Finishes an int8 AE block on the GPU instead of in the reader thread.

    Quantised tiles (`prep_aef_tiles.py --int8`) store the AEF int8 exactly as
    downloaded; the dequantisation is a 256-entry LUT. Doing that LUT in numpy in
    the reader costs more than it saves — it is a 268M-element gather per block,
    and it measured 0.60x the float32 tile, i.e. SLOWER than not quantising at
    all. On the GPU the same gather runs at ~300 M px/s, ~6x faster than the
    fastest read, so the read speedup (~4x) survives intact. As a side effect the
    PCIe transfer shrinks 4x too, since int8 crosses instead of float32.

    Only the columns the model actually asks for are gathered, in `feat_cols`
    order, so the result is identical to the float32 path by construction.
    """

    def __init__(self, feat_cols, device):
        band_of = {b: i for i, b in enumerate(AE_BANDS)}
        self.ae_src, self.ae_dst, self.lid_dst = [], [], []
        for fi, col in enumerate(feat_cols):
            if col in band_of:
                self.ae_src.append(band_of[col])
                self.ae_dst.append(fi)
            elif col in LIDAR_COLS:
                self.lid_dst.append(fi)
            else:
                raise ValueError(f"model feature {col!r} is neither AE nor lidar")
        self.n_feat = len(feat_cols)
        self.device = device
        # index 0 is raw -128 (nodata). The float32 path makes it NaN and the
        # validity mask drops those pixels before they ever reach here, so the
        # value is unobservable; 0.0 keeps a stray NaN from poisoning a softmax.
        lut = _DEQUANT_LUT.copy()
        lut[0] = 0.0
        self.lut = torch.as_tensor(lut, device=device)
        self.ae_src_t = torch.as_tensor(np.array(self.ae_src), device=device)
        self.ae_dst_t = torch.as_tensor(np.array(self.ae_dst), device=device)
        self.lid_dst_t = (torch.as_tensor(np.array(self.lid_dst), device=device)
                          if self.lid_dst else None)

    def to_gpu(self, ae_i8, lid_f, vidx):
        """(int8 [64,h,w], float32 [L,h,w] | None, valid indices) -> [nvalid, F]."""
        ae_v = ae_i8.reshape(ae_i8.shape[0], -1)[:, vidx]          # [64, nv] int8
        t = torch.from_numpy(np.ascontiguousarray(ae_v)).to(self.device)
        X = torch.empty((vidx.size, self.n_feat), device=self.device,
                        dtype=torch.float32)
        # (v + 128) as int16 then index: the LUT is 256 entries, so this is one
        # gather per element and nothing widens on the host.
        X[:, self.ae_dst_t] = self.lut[
            (t.index_select(0, self.ae_src_t).to(torch.int16) + 128).long()].T
        if self.lid_dst_t is not None:
            lv = lid_f.reshape(lid_f.shape[0], -1)[:, vidx]
            X[:, self.lid_dst_t] = torch.from_numpy(
                np.ascontiguousarray(lv)).to(self.device).T
        return X


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] /
                                           "models" / "dnn_final.pt"))
    ap.add_argument("--calib", default=str(Path(__file__).resolve().parents[1] /
                                           "models" / "dnn_final_calib.npz"))
    ap.add_argument("--uq-out", default=None,
                    help="optional UQ output stem; writes three typed rasters "
                         "<stem>_pcal.tif (uint16 calibrated proba x60000), "
                         "<stem>_setsize.tif (uint8), <stem>_inset.tif (uint8 "
                         "per-class 0/1 conformal membership)")
    ap.add_argument("--lidar-raster", default=None,
                    help="GeoTIFF with bands elevation,tri,tch on the same grid")
    ap.add_argument("--nodata-class", type=int, default=0)
    ap.add_argument("--block", type=int, default=2048, help="block size (px); read/infer unit")
    ap.add_argument("--readers", type=int, default=4, help="reader threads (I/O parallelism)")
    ap.add_argument("--scale", type=float, default=None,
                    help="divide AE bands by this if source packs scaled ints "
                         "(P-drive embeddings are ~1000x: use --scale 1000)")
    ap.add_argument("--mask-allzero", action="store_true",
                    help="treat all-band-zero pixels as nodata (VRT tile gaps / "
                         "untagged AE nodata); safe since real AE is unit-norm")
    ap.add_argument("--aoi", default=None,
                    help="optional AOI vector (any OGR format): predict ONLY "
                         "pixels inside these polygons, everything outside -> "
                         "nodata. Reprojected to the raster CRS; useful when the "
                         "input rectangle overflows the true study area (e.g. AEF "
                         "tiles overflow the 3-county boundary).")
    ap.add_argument("--gpu-chunk", type=int, default=0,
                    help="rows per GPU forward chunk (bounds activation memory); "
                         "0 = the model's own default (Ensemble.default_chunk, "
                         "which is smaller for a MoE — dense experts hold a "
                         "chunk x 40 x 64 activation)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ens = C.Ensemble.load(args.model)
    gpu_chunk = args.gpu_chunk or ens.default_chunk
    feat_cols = ens.feat_cols
    needs_lidar = any(c in feat_cols for c in LIDAR_COLS)
    lidar_med = ens.lidar_med or {}
    with_uq = args.uq_out is not None
    calib = C.Calibration.load(args.calib) if with_uq else None
    if calib is not None and calib.classes != ens.classes:
        raise ValueError(
            f"--model/--calib class mismatch: ens.classes={ens.classes} vs "
            f"calib.classes={calib.classes} — regenerate dnn_final_calib.npz "
            f"(fit_calibration.py) against the current model, they are indexed "
            f"positionally and a mismatch silently mislabels UQ bands.")
    # Classes alone are NOT enough to prove a calibration belongs to a model:
    # the MLP and the MoE share this exact class list, so a calibration fit on
    # one passed the check above while being applied to the other. Venn-Abers
    # breakpoints and LAC/Mondrian taus are fit to ONE model's score
    # distribution; using another's leaves the class map correct but silently
    # voids the 90% conformal coverage and the calibrated-probability band.
    if calib is not None and calib.arch is not None and calib.arch != ens.cfg.arch:
        raise ValueError(
            f"--model/--calib architecture mismatch: model arch={ens.cfg.arch!r} "
            f"but {args.calib} was fit on arch={calib.arch!r}. Refit:\n"
            f"  ARCH={ens.cfg.arch} N_EXPERTS={ens.cfg.n_experts} "
            f"TOP_K={ens.cfg.top_k} "
            f"EXPERT_HIDDEN={','.join(str(h) for h in ens.cfg.expert_hidden)} \\\n"
            f"  OUT=<calib>.npz  python DNN/fit_calibration.py")
    if calib is not None and calib.arch is None:
        print(f"  WARNING: {args.calib} predates architecture stamping — it "
              f"cannot be checked against this {ens.cfg.arch!r} model. If it was "
              f"fit on a different architecture the class map is still correct "
              f"but the UQ bands are not calibrated for this model.", flush=True)
    print(f"model: {len(ens.models)} nets, {len(feat_cols)} feats, classes={ens.classes}, "
          f"needs_lidar={needs_lidar}, uq={with_uq}"
          + (f" (calib_method={calib.calib_method})" if with_uq else ""), flush=True)

    src = rasterio.open(args.inp)
    if src.count < N_AE:
        raise ValueError(f"{args.inp} has {src.count} bands; expected >= {N_AE}")
    if needs_lidar and not args.lidar_raster:
        print(f"  no --lidar-raster: median-fill {LIDAR_COLS} = "
              f"{ {k: round(lidar_med.get(k, 0.0), 2) for k in LIDAR_COLS} }", flush=True)
    if needs_lidar and args.lidar_raster:
        # The reader reads the lidar raster with the SAME pixel Window as the AE
        # input and does no resampling, so a mismatched grid silently pairs each
        # AE pixel with lidar from a different location. Fail fast instead.
        with rasterio.open(args.lidar_raster) as _l:
            if ((_l.width, _l.height) != (src.width, src.height)
                    or _l.crs != src.crs
                    or not _l.transform.almost_equals(src.transform, precision=1e-6)):
                raise ValueError(
                    f"--lidar-raster grid does not match --in: "
                    f"lidar={_l.width}x{_l.height} {_l.crs} {_l.transform!r} vs "
                    f"ae={src.width}x{src.height} {src.crs} {src.transform!r}. "
                    f"predict_raster.py reads both on the same Window with NO "
                    f"resampling — reproject the lidar onto the AE grid first "
                    f"(see DATA_INFERENCE.md 'Lidar raster derivation').")
            if _l.count < len(LIDAR_COLS):
                raise ValueError(
                    f"--lidar-raster has {_l.count} bands; expected "
                    f">= {len(LIDAR_COLS)} ({LIDAR_COLS})")

    # force GTiff output: when --in is a .vrt the source profile's driver is
    # "VRT", which is read-only — writing pixels through it raises
    # "Writing through VRTSourcedRasterBand is not supported".
    # BIGTIFF=IF_SAFER: with deflate GDAL can't predict the compressed size, so
    # a county-scale write silently corrupts once it crosses the 4 GB classic-
    # TIFF limit mid-stream unless BigTIFF is forced.
    prof = src.profile.copy()
    prof.update(driver="GTiff", count=1, dtype="int16", nodata=args.nodata_class,
                compress="deflate", tiled=True, blockxsize=512, blockysize=512,
                bigtiff="IF_SAFER")
    dst = rasterio.open(args.out, "w", **prof)
    # UQ is written as THREE typed rasters instead of one 21-band float32 stack:
    # the three components have very different value domains and float32 wastes
    # ~3x the disk (a 3-county run was ~31 GB float32 vs ~10 GB split). Files are
    # derived from --uq-out's stem (uq.tif -> uq_pcal.tif / uq_setsize.tif /
    # uq_inset.tif). Encodings:
    #   * pcal_*  : calibrated proba, uint16 scaled x PCAL_SCALE (lossless to
    #               ~1e-5; max round-trip err 8e-6). nodata = 65535.
    #   * set_size: conformal set size (int 0..C), uint8. nodata = 255.
    #   * inset_* : 0/1 per-class conformal-set membership, uint8. nodata = 255.
    # deflate + predictor=2 (horizontal) — integer data with spatial structure
    # compresses far better than the old no-predictor float32.
    udst_pcal = udst_ss = udst_inset = None
    if with_uq:
        C_ = ens.n_classes
        stem = args.uq_out[:-4] if args.uq_out.lower().endswith(".tif") else args.uq_out
        pcal_path, ss_path, inset_path = (stem + "_pcal.tif",
                                          stem + "_setsize.tif",
                                          stem + "_inset.tif")
        base = dict(driver="GTiff", compress="deflate", predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, bigtiff="IF_SAFER",
                    crs=src.crs, transform=src.transform,
                    width=src.width, height=src.height)

        pp = dict(base, count=C_, dtype="uint16", nodata=PCAL_NODATA)
        udst_pcal = rasterio.open(pcal_path, "w", **pp)
        for j, c in enumerate(ens.classes):
            udst_pcal.set_band_description(j + 1, f"pcal_c{c}")

        sp = dict(base, count=1, dtype="uint8", nodata=UINT8_NODATA)
        udst_ss = rasterio.open(ss_path, "w", **sp)
        udst_ss.set_band_description(1, "set_size")

        ip = dict(base, count=C_, dtype="uint8", nodata=UINT8_NODATA)
        udst_inset = rasterio.open(inset_path, "w", **ip)
        for j, c in enumerate(ens.classes):
            udst_inset.set_band_description(j + 1, f"inset_c{c}")
        print(f"  uq -> {pcal_path} (uint16 x{PCAL_SCALE}), {ss_path} (uint8), "
              f"{inset_path} (uint8)", flush=True)

    ae_nodata = src.nodata

    # Quantised input (prep_aef_tiles.py --int8) is detected from the dtype, not
    # a flag: the two encodings are interchangeable inputs and a run should not
    # depend on the caller remembering which one a VRT points at.
    ae_int8 = src.dtypes[0] == "int8"
    dequant = _Dequant(feat_cols, C.DEVICE) if ae_int8 else None
    if ae_int8:
        if args.scale:
            raise SystemExit(
                "--scale with an int8 (quantised) input: --scale is for P-drive "
                "tiles that pack scaled floats. Quantised AEF tiles are "
                "dequantised by the LUT, and dividing the int8 codes first would "
                "corrupt every embedding.")
        print(f"  input is int8 (quantised AEF): LUT dequant on {C.DEVICE}",
              flush=True)

    # --aoi: load + reproject the study-area polygons to the raster CRS once.
    # geometry_mask is applied per-block in the reader (a full-grid mask would be
    # ~1 GB of bool at county scale), so we keep the shapely geoms here and rely
    # on each block's own transform. Bail early if the AOI misses the raster
    # entirely — that is almost always a CRS/extent mistake, not an empty map.
    aoi_geoms = None
    aoi_union = None
    if args.aoi:
        import geopandas as gpd
        from shapely.geometry import box as _box
        gdf = gpd.read_file(args.aoi).to_crs(src.crs)
        aoi_geoms = list(gdf.geometry.values)
        aoi_union = gdf.union_all()
        if not aoi_union.intersects(_box(*src.bounds)):
            raise ValueError(
                f"--aoi {args.aoi!r} does not intersect the input raster extent "
                f"(after reprojecting to {src.crs}); check the AOI/CRS.")
        print(f"  --aoi: {len(aoi_geoms)} polygon(s), masking to study area",
              flush=True)

    wins = list(_blocks(src.width, src.height, args.block, args.block))

    # --aoi block pre-filter: with the AOI union in hand, classify every window by
    # its bounding box BEFORE any pixels are read — a window whose bbox misses the
    # AOI entirely is skipped (no 64-band read/decompress, no GPU), a window whose
    # bbox is fully covered by the AOI needs no per-block geometry_mask (every
    # pixel is inside), and only genuinely straddling windows pay the per-block
    # rasterize. For an AOI that fills a fraction of the tile rectangle (AEF tiles
    # overflow the 3-county boundary by ~67%) this drops most of the I/O, which is
    # the measured wall (CIFS-bound, ~0.2 M px/s). Windows are still ALL written by
    # the writer (outside ones as pure-nodata blocks) so the len(wins) completeness
    # assertion and the UQ-raster nodata fill are unchanged.
    #   win_kind[i]: 0 = outside (skip read), 1 = fully inside (skip mask),
    #                2 = straddling (per-block mask).  None-AOI runs are all 2-ish
    #                (mask is simply never applied).
    win_kind = None
    if aoi_union is not None:
        from shapely import STRtree, box as _sbox, prepared
        win_boxes = [_sbox(*window_bounds(w, src.transform)) for w in wins]
        # Vectorized bbox-vs-AOI test via an STRtree over the window boxes: which
        # boxes intersect / are covered by the AOI union. covered_by ⊆ intersects.
        tree = STRtree(win_boxes)
        inter_idx = set(tree.query(aoi_union, predicate="intersects").tolist())
        # "covers" on the union tells us a window entirely inside the AOI; use a
        # prepared union so the (potentially many) contains tests are fast.
        prep = prepared.prep(aoi_union)
        win_kind = np.full(len(wins), 0, dtype=np.uint8)  # default: outside
        n_in = n_edge = 0
        for i in inter_idx:
            if prep.covers(win_boxes[i]):
                win_kind[i] = 1
                n_in += 1
            else:
                win_kind[i] = 2
                n_edge += 1
        n_out = len(wins) - n_in - n_edge
        print(f"  --aoi block pre-filter: {n_in} inside, {n_edge} straddling, "
              f"{n_out} outside skipped (of {len(wins)} blocks)", flush=True)

    total_valid = [0]

    # ---- pipeline queues ---------------------------------------------------
    read_q: "queue.Queue" = queue.Queue(maxsize=args.readers * 2)   # -> GPU
    write_q: "queue.Queue" = queue.Queue(maxsize=args.readers * 2)  # -> writer
    win_q: "queue.Queue" = queue.Queue()
    for i, w in enumerate(wins):
        # Carry the pre-filter verdict alongside each window (2 = "apply per-block
        # AOI mask" / no-AOI default). The reader uses it to skip the read of
        # outside blocks and the rasterize of fully-inside ones.
        win_q.put((w, 2 if win_kind is None else int(win_kind[i])))
    lid_path = args.lidar_raster if (needs_lidar and args.lidar_raster) else None

    # Any worker thread that raises records it here and sets `errored`. Without
    # this a dead reader would just stop consuming windows (the run finishes and
    # prints success, but its blocks are never written -> silent holes in the
    # output), and a dead GPU thread would deadlock the readers on a full queue.
    errored = threading.Event()
    err_box: list = []

    def _fail(exc):
        err_box.append(exc)
        errored.set()

    def _drain(q):
        """Empty `q` so no producer stays parked in a blocking put on it.

        Readers put into read_q with a blocking put, and read_q is bounded. If
        the GPU thread stops consuming while readers are parked in that put,
        nothing ever wakes them: the reader threads never reach their own
        `errored` check, main blocks forever in `for r in readers: r.join()`,
        and the run HANGS instead of raising "inference aborted". After a drain
        each reader completes at most one more put before its next `errored`
        check, and read_q holds `readers * 2` — so there is always room, both
        for those and for main's poison pill.
        """
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def reader():
        # per-thread dataset handles: decompression is the wall on compressed
        # 64-band tiles, and one shared handle serializes reads. Separate GDAL
        # handles let decompression run on multiple cores in parallel.
        rsrc = rasterio.open(args.inp)
        rlid = rasterio.open(lid_path) if lid_path else None
        try:
            while not errored.is_set():
                try:
                    win, kind = win_q.get_nowait()
                except queue.Empty:
                    return
                if kind == 0:
                    # Window entirely outside the AOI (bbox test in the pre-filter).
                    # Skip the 64-band read/decompress and feature build entirely;
                    # emit an all-invalid block so the GPU thread forwards it as a
                    # pure-nodata write. This is the I/O win — outside blocks never
                    # touch the (CIFS-bound) raster.
                    valid = np.zeros((int(win.height), int(win.width)), dtype=bool)
                    read_q.put((win, None, valid))
                    continue
                # Quantised tiles stay int8 here — see _Dequant for why the LUT
                # runs on the GPU. The masks below are all order-preserving
                # comparisons (== nodata, == 0), so they read the same on the
                # int8 codes as on the dequantised floats: the LUT is strictly
                # monotone and maps 0 -> 0.0 and -128 -> nodata.
                ae = rsrc.read(range(1, N_AE + 1), window=win)
                if not ae_int8:
                    ae = ae.astype(np.float32)
                # nodata / zero / finite tests must run on the RAW values, BEFORE
                # --scale divides the bands: comparing scaled data against the
                # source's unscaled nodata (e.g. -32768 -> -32.768) would never
                # match, so nodata areas would be classified as real land cover.
                # (isfinite / ==0 are scale-invariant but kept here for one pass.)
                valid = np.ones(ae.shape[1:], dtype=bool)
                if ae_nodata is not None:
                    valid &= ~np.any(ae == ae_nodata, axis=0)
                if not ae_int8:                 # integers are always finite
                    valid &= np.all(np.isfinite(ae), axis=0)
                if args.mask_allzero:
                    # VRT gaps between tiles (and untiled AE nodata) read as all
                    # bands == 0. A genuine AlphaEarth embedding is unit-norm, so
                    # an all-zero pixel is never real data — exclude it.
                    valid &= ~np.all(ae == 0, axis=0)
                if aoi_geoms is not None and kind != 1:
                    # Drop pixels outside the study-area polygons. Rasterize the
                    # AOI on THIS block's transform (invert=True -> True inside).
                    # kind==1 means the pre-filter proved this block's bbox is fully
                    # covered by the AOI, so every pixel is inside -> skip the
                    # per-block rasterize (a pure win, identical result).
                    win_transform = rsrc.window_transform(win)
                    inside = geometry_mask(
                        aoi_geoms, out_shape=valid.shape,
                        transform=win_transform, invert=True, all_touched=True)
                    valid &= inside
                if args.scale:
                    ae /= args.scale
                lid = None
                if rlid is not None:
                    lid = rlid.read(range(1, len(LIDAR_COLS) + 1), window=win).astype(np.float32)
                feats = _assemble_features(ae, lid, feat_cols, lidar_med)
                read_q.put((win, feats, valid))
        except BaseException as exc:  # noqa: BLE001 — surface, don't swallow
            _fail(exc)
        finally:
            rsrc.close()
            if rlid is not None:
                rlid.close()

    def gpu_worker():
        try:
            while True:
                if errored.is_set():
                    # Another thread failed (typically the writer — a full disk
                    # is the usual cause). Drain before leaving, or the readers
                    # parked on a full read_q are never released.
                    _drain(read_q)
                    return
                item = read_q.get()
                if item is None:
                    return
                win, feats, valid = item
                h, w = valid.shape
                out_cls = np.full((h, w), args.nodata_class, dtype=np.int16)
                # Always allocate the UQ blocks when UQ output is requested, even
                # for an all-nodata block (vidx.size == 0) — writer() always calls
                # write unconditionally then, so no block is silently left
                # unwritten. Prefill with each raster's nodata (valid pixels get
                # overwritten below via `vidx`).
                if with_uq:
                    C_ = ens.n_classes
                    u_pcal = np.full((C_, h, w), PCAL_NODATA, dtype=np.uint16)
                    u_ss = np.full((h, w), UINT8_NODATA, dtype=np.uint8)
                    u_inset = np.full((C_, h, w), UINT8_NODATA, dtype=np.uint8)
                    ublocks = (u_pcal, u_ss, u_inset)
                else:
                    ublocks = None
                vidx = np.flatnonzero(valid.ravel())
                if vidx.size:
                    if dequant is not None:
                        # int8 tile: gather + LUT on-device (4x less over PCIe)
                        X_t = dequant.to_gpu(feats[0], feats[1], vidx)
                    else:
                        Xv = feats.reshape(len(feat_cols), -1)[:, vidx].T  # [nvalid, F]
                        X_t = torch.as_tensor(np.ascontiguousarray(Xv), device=C.DEVICE)
                    if with_uq:
                        full = ens.predict_full_gpu(X_t, calib, gpu_chunk)
                        out_cls.ravel()[vidx] = full["pred_class"]
                        # scale proba -> uint16, clip so a rounded 1.0 can't hit
                        # the nodata sentinel (65535).
                        pc = np.clip(np.rint(full["proba_calibrated"] * PCAL_SCALE),
                                     0, PCAL_SCALE).astype(np.uint16)
                        u_pcal.reshape(C_, h * w)[:, vidx] = pc.T
                        u_ss.reshape(h * w)[vidx] = full["set_size"].astype(np.uint8)
                        u_inset.reshape(C_, h * w)[:, vidx] = full["included"].T.astype(np.uint8)
                    else:
                        cls = ens.predict_classmap_gpu(X_t, gpu_chunk).cpu().numpy()
                        out_cls.ravel()[vidx] = cls
                    total_valid[0] += vidx.size
                # timeout-loop the put so a dead writer (full write_q that never
                # drains) can't deadlock this thread — bail out on `errored`.
                while not errored.is_set():
                    try:
                        write_q.put((win, out_cls, ublocks), timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 — surface, don't swallow
            _fail(exc)
            # unblock any reader parked on a full read_q so join() can return
            _drain(read_q)

    def writer():
        done = [0]

        def _drain():
            while True:
                item = write_q.get()
                if item is None:
                    return
                win, out_cls, ublocks = item
                dst.write(out_cls, 1, window=win)
                if udst_pcal is not None and ublocks is not None:
                    u_pcal, u_ss, u_inset = ublocks
                    udst_pcal.write(u_pcal, window=win)
                    udst_ss.write(u_ss, 1, window=win)
                    udst_inset.write(u_inset, window=win)
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == len(wins):
                    el = time.perf_counter() - t0
                    print(f"  {done[0]}/{len(wins)} blocks  {total_valid[0]/1e6:.1f}M px  "
                          f"{total_valid[0]/max(el,1e-9)/1e6:.1f} M px/s", flush=True)

        try:
            _drain()
        except BaseException as exc:  # noqa: BLE001 — surface, don't swallow
            _fail(exc)
        finally:
            writer.blocks_written = done[0]

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(args.readers)]
    gpu_t = threading.Thread(target=gpu_worker, daemon=True)
    write_t = threading.Thread(target=writer, daemon=True)
    for r in readers:
        r.start()
    gpu_t.start()
    write_t.start()
    for r in readers:
        r.join()
    read_q.put(None)      # signal GPU (poison-pill; gpu_worker re-raises None-safe)
    gpu_t.join()
    write_q.put(None)     # signal writer
    write_t.join()

    src.close()
    dst.close()
    for u in (udst_pcal, udst_ss, udst_inset):
        if u is not None:
            u.close()

    # Fail loud: a worker exception, or fewer blocks written than tiled, means
    # the output raster has holes — never report success on a partial write.
    written = getattr(writer, "blocks_written", 0)
    if errored.is_set():
        raise RuntimeError(
            f"inference aborted: a worker thread failed "
            f"({written} / {len(wins)} blocks written before abort)") from err_box[0]
    if written != len(wins):
        raise RuntimeError(
            f"incomplete write: {written} / {len(wins)} blocks reached the "
            f"writer — output {args.out} has unwritten (nodata) holes")

    el = time.perf_counter() - t0
    print(f"done: {total_valid[0]:,} valid px -> {args.out}"
          f"{'  + '+args.uq_out if with_uq else ''}  "
          f"{el:.1f}s  ({total_valid[0]/max(el,1e-9)/1e6:.1f} M px/s)", flush=True)


if __name__ == "__main__":
    main()

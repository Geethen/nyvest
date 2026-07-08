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
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import geometry_mask
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_core as C  # noqa: E402

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
    """Stack AE (+lidar) bands into the model's feat_cols order -> [F, h, w] f32."""
    band_of = {b: i for i, b in enumerate(AE_BANDS)}
    h, w = ae.shape[1], ae.shape[2]
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
    ap.add_argument("--gpu-chunk", type=int, default=262144,
                    help="rows per GPU forward chunk (bounds activation memory)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ens = C.Ensemble.load(args.model)
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

    # --aoi: load + reproject the study-area polygons to the raster CRS once.
    # geometry_mask is applied per-block in the reader (a full-grid mask would be
    # ~1 GB of bool at county scale), so we keep the shapely geoms here and rely
    # on each block's own transform. Bail early if the AOI misses the raster
    # entirely — that is almost always a CRS/extent mistake, not an empty map.
    aoi_geoms = None
    if args.aoi:
        import geopandas as gpd
        from shapely.geometry import box as _box
        gdf = gpd.read_file(args.aoi).to_crs(src.crs)
        aoi_geoms = list(gdf.geometry.values)
        if not gdf.union_all().intersects(_box(*src.bounds)):
            raise ValueError(
                f"--aoi {args.aoi!r} does not intersect the input raster extent "
                f"(after reprojecting to {src.crs}); check the AOI/CRS.")
        print(f"  --aoi: {len(aoi_geoms)} polygon(s), masking to study area",
              flush=True)

    wins = list(_blocks(src.width, src.height, args.block, args.block))
    total_valid = [0]

    # ---- pipeline queues ---------------------------------------------------
    read_q: "queue.Queue" = queue.Queue(maxsize=args.readers * 2)   # -> GPU
    write_q: "queue.Queue" = queue.Queue(maxsize=args.readers * 2)  # -> writer
    win_q: "queue.Queue" = queue.Queue()
    for w in wins:
        win_q.put(w)
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

    def reader():
        # per-thread dataset handles: decompression is the wall on compressed
        # 64-band tiles, and one shared handle serializes reads. Separate GDAL
        # handles let decompression run on multiple cores in parallel.
        rsrc = rasterio.open(args.inp)
        rlid = rasterio.open(lid_path) if lid_path else None
        try:
            while not errored.is_set():
                try:
                    win = win_q.get_nowait()
                except queue.Empty:
                    return
                ae = rsrc.read(range(1, N_AE + 1), window=win).astype(np.float32)
                # nodata / zero / finite tests must run on the RAW values, BEFORE
                # --scale divides the bands: comparing scaled data against the
                # source's unscaled nodata (e.g. -32768 -> -32.768) would never
                # match, so nodata areas would be classified as real land cover.
                # (isfinite / ==0 are scale-invariant but kept here for one pass.)
                valid = np.ones(ae.shape[1:], dtype=bool)
                if ae_nodata is not None:
                    valid &= ~np.any(ae == ae_nodata, axis=0)
                valid &= np.all(np.isfinite(ae), axis=0)
                if args.mask_allzero:
                    # VRT gaps between tiles (and untiled AE nodata) read as all
                    # bands == 0. A genuine AlphaEarth embedding is unit-norm, so
                    # an all-zero pixel is never real data — exclude it.
                    valid &= ~np.all(ae == 0, axis=0)
                if aoi_geoms is not None:
                    # Drop pixels outside the study-area polygons. Rasterize the
                    # AOI on THIS block's transform (invert=True -> True inside).
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
                    Xv = feats.reshape(len(feat_cols), -1)[:, vidx].T   # [nvalid, F]
                    X_t = torch.as_tensor(np.ascontiguousarray(Xv), device=C.DEVICE)
                    if with_uq:
                        full = ens.predict_full_gpu(X_t, calib, args.gpu_chunk)
                        out_cls.ravel()[vidx] = full["pred_class"]
                        # scale proba -> uint16, clip so a rounded 1.0 can't hit
                        # the nodata sentinel (65535).
                        pc = np.clip(np.rint(full["proba_calibrated"] * PCAL_SCALE),
                                     0, PCAL_SCALE).astype(np.uint16)
                        u_pcal.reshape(C_, h * w)[:, vidx] = pc.T
                        u_ss.reshape(h * w)[vidx] = full["set_size"].astype(np.uint8)
                        u_inset.reshape(C_, h * w)[:, vidx] = full["included"].T.astype(np.uint8)
                    else:
                        cls = ens.predict_classmap_gpu(X_t, args.gpu_chunk).cpu().numpy()
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
            try:
                while True:
                    read_q.get_nowait()
            except queue.Empty:
                pass

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

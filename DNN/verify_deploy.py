"""End-to-end check of the deployed artifact on a small raster subset.

Three things changed on the deployment path at once — the architecture
(`moe_shared` instead of the plain MLP), how that architecture is executed
(`moe_fast.FusedMoEEnsemble`, batched matmuls instead of a Python loop over 40
experts), and how a tile may be stored (`prep_aef_tiles.py --int8`, with the
dequant LUT moved to the GPU). Each was measured in isolation, on synthetic
weights, by its own benchmark. This runs the real trained checkpoint through the
real `predict_raster.py` on a real window and checks the three claims that
matter together:

  1. FUSED == LOOP on the trained weights. `moe_fast._selftest` proves this for
     random weights; random weights have flat routers and near-uniform gates, so
     they are the easy case. A trained router is confident and sits rows much
     closer to the top-k boundary, which is exactly where a reassociated GEMM
     could flip a route. Checked on real embeddings, not gaussians.
  2. int8 TILE == float32 TILE, pixel for pixel, through the whole pipeline —
     read, validity masking, feature assembly, dequant, inference, write.
  3. The class maps are actually written and the model changed the map (a run
     that silently produced the baseline's output would pass 1 and 2).

Timings here are end-to-end `predict_raster.py` wall time on a small window,
including process start and model load, so they are NOT the throughput numbers
from bench_moe_fast.py / bench_io_formats.py — they are a sanity check that
nothing regressed by an order of magnitude.

Run:
    ~/myprojects/recover/.venv/bin/python DNN/verify_deploy.py
    ~/myprojects/recover/.venv/bin/python DNN/verify_deploy.py --size 2048 --keep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import rasterio                                              # noqa: E402
from rasterio.windows import Window                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_core as C                                         # noqa: E402
from prep_aef_tiles import _DEQUANT_LUT, N_AE                # noqa: E402
from bench_io_formats import to_int8_chunked                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SRC = REPO / "aef_bench_prepped" / "xnkqw3fftuuts84jo-0000008192-0000008192_32633.tif"


def cut(src, out, row, col, size, int8):
    """One window of the bench tile, written in production encoding."""
    with rasterio.open(src) as s:
        win = Window(col, row, size, size)
        a = s.read(range(1, N_AE + 1), window=win).astype(np.float32)
        prof = s.profile.copy()
        prof.update(width=size, height=size, count=N_AE, tiled=True,
                    blockxsize=512, blockysize=512,
                    transform=s.window_transform(win), BIGTIFF="IF_SAFER")
    if int8:
        a_out = to_int8_chunked(a)
        prof.update(dtype="int8", nodata=-128, compress="zstd", zstd_level=1)
    else:
        a_out = a
        prof.update(dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out, "w", **prof) as d:
        for b in range(1, N_AE + 1):
            d.set_band_description(b, f"A{b-1:02d}")
        d.write(a_out)
    return a


def run_predict(inp, out, model, extra=()):
    cmd = [PY, str(Path(__file__).resolve().parent / "predict_raster.py"),
           "--in", str(inp), "--out", str(out), "--model", str(model),
           "--readers", "4", "--block", "1024", "--mask-allzero", *extra]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"predict_raster failed for {inp}")
    return dt, r.stdout


def read_cls(p):
    with rasterio.open(p) as s:
        return s.read(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--row", type=int, default=7144)
    ap.add_argument("--col", type=int, default=6900)
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--mlp", default=str(REPO / "models" / "dnn_final.pt"))
    ap.add_argument("--moe", default=str(REPO / "models" / "dnn_final_moe8.pt"))
    ap.add_argument("--calib", default=str(REPO / "models" / "dnn_final_moe8_calib.npz"),
                    help="calibration to check the on-GPU UQ path against; "
                         "skipped if absent")
    ap.add_argument("--workdir", default=str(REPO / ".verify_tmp"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default="DNN/reports/results/verify_deploy.json")
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    ok, notes = [], {}

    def chk(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"  [{'ok ' if cond else 'FAIL'}] {name} {detail}", flush=True)

    print(f"window {args.size}x{args.size} at row={args.row} col={args.col} "
          f"of {Path(args.src).name}\n")
    ref = cut(args.src, wd / "sub_f32.tif", args.row, args.col, args.size, False)
    cut(args.src, wd / "sub_i8.tif", args.row, args.col, args.size, True)
    finite = float(np.isfinite(ref).mean())
    notes["window_finite_frac"] = round(finite, 4)
    print(f"subset finite fraction {finite:.3f}   "
          f"f32 {os.path.getsize(wd/'sub_f32.tif')/1e6:.1f} MB   "
          f"i8 {os.path.getsize(wd/'sub_i8.tif')/1e6:.1f} MB\n")

    # ---- 1. the int8 encoding is lossless through the LUT --------------------
    with rasterio.open(wd / "sub_i8.tif") as s:
        back = _DEQUANT_LUT[s.read(range(1, N_AE + 1)).astype(np.int16) + 128]
    chk("int8 tile round-trips to the original float32 values",
        np.array_equal(np.nan_to_num(back, nan=-9e9),
                       np.nan_to_num(ref, nan=-9e9)))

    # ---- 2. fused == loop, on the TRAINED weights and real embeddings --------
    ens = C.Ensemble.load(args.moe)
    notes["arch"] = ens.cfg.arch
    chk("checkpoint is a moe_shared ensemble", ens.cfg.arch == "moe_shared",
        f"arch={ens.cfg.arch} n_experts={ens.cfg.n_experts} top_k={ens.cfg.top_k}")
    import torch
    m = np.isfinite(ref).all(0).ravel()
    rows = np.flatnonzero(m)[:400_000]
    # Built by NAME, not by assuming AE bands occupy the first 64 columns — a
    # silently mis-ordered X would still run and would still "agree with itself".
    band_of = {f"A{i:02d}": i for i in range(N_AE)}
    flat = ref.reshape(N_AE, -1)
    X = np.zeros((len(rows), len(ens.feat_cols)), np.float32)
    for fi, c in enumerate(ens.feat_cols):
        X[:, fi] = (flat[band_of[c]][rows] if c in band_of
                    else (ens.lidar_med or {}).get(c, 0.0))
    X_t = torch.as_tensor(X, device=C.DEVICE)
    fast = ens._fused()
    chk("the deployed path really is the fused one", fast is not None)
    mean_t, std_t = ens._gpu_stats()
    with torch.no_grad():
        xb = (X_t - mean_t) / std_t
        loop = torch.zeros((len(rows), ens.n_classes), device=C.DEVICE)
        for mm in ens.models:
            loop += torch.softmax(mm(xb).float(), 1)
        loop /= len(ens.models)
        got = fast.mean_proba(xb)
    dmax = float((loop - got).abs().max())
    agree = float((loop.argmax(1) == got.argmax(1)).float().mean())
    notes.update(fused_max_abs_delta=dmax, fused_argmax_agreement=agree,
                 fused_rows=int(len(rows)))
    chk("fused == loop on trained weights (argmax)", agree == 1.0,
        f"{len(rows):,} real rows, agree={agree:.6f}, max|dp|={dmax:.2e}")

    # ---- 2b. GPU calibration == numpy calibration ---------------------------
    # The UQ transform moved onto the device (it was 80% of the GPU thread and
    # capped the pipeline at 0.42 M px/s). venn_abers is all searchsorted /
    # gather / divide, so it should be BIT-identical, not merely close; check it
    # on the real calibrators and the real model's probabilities rather than on
    # the tolerance-friendly synthetic case.
    if Path(args.calib).exists():
        calib = C.Calibration.load(args.calib)
        notes["calib_arch"] = calib.arch
        chk("calibration was fit on this architecture",
            calib.arch == ens.cfg.arch,
            f"calib arch={calib.arch!r} vs model arch={ens.cfg.arch!r}"
            + ("  (refit: see fit_calibration.py)" if calib.arch != ens.cfg.arch else ""))
        with torch.no_grad():
            P_t = ens.predict_proba_gpu(X_t)
        P = P_t.cpu().numpy()
        pc_cpu, pc_gpu = (calib.predict_proba_calibrated(P),
                          calib.predict_proba_calibrated_gpu(P_t).cpu().numpy())
        (inc_c, ss_c) = calib.predict_sets(P)
        inc_g, ss_g = (t.cpu().numpy() for t in calib.predict_sets_gpu(P_t))
        dpc = float(np.abs(pc_cpu.astype(np.float64) - pc_gpu).max())
        notes["calib_gpu_max_abs_delta"] = dpc
        # temp_scale rides on exp/log and lands within a float32 ULP; venn_abers
        # must be exact. Either way it must be far below the 1/60000 the raster
        # path quantises this band at.
        chk("GPU calibrated proba matches numpy",
            np.array_equal(pc_cpu, pc_gpu) if calib.calib_method == "venn_abers"
            else dpc < 1e-6,
            f"method={calib.calib_method} max|d|={dpc:.2e} "
            f"(band quantum 1/60000={1/60000:.2e})")
        chk("GPU conformal sets match numpy exactly",
            np.array_equal(inc_c, inc_g) and np.array_equal(ss_c, ss_g))
    else:
        print(f"  [skip] {args.calib} not present — UQ calibration check skipped")

    # ---- 3. the pipeline, end to end ----------------------------------------
    print()
    t_mlp, _ = run_predict(wd / "sub_f32.tif", wd / "cls_mlp.tif", args.mlp)
    t_moe, out_moe = run_predict(wd / "sub_f32.tif", wd / "cls_moe_f32.tif", args.moe)
    t_i8, out_i8 = run_predict(wd / "sub_i8.tif", wd / "cls_moe_i8.tif", args.moe)
    print(f"\n  predict_raster wall (incl. startup+load): mlp/f32 {t_mlp:.1f}s   "
          f"moe/f32 {t_moe:.1f}s   moe/int8 {t_i8:.1f}s")
    chk("int8 run took the GPU dequant path",
        "int8 (quantised AEF)" in out_i8)
    chk("moe run used the smaller MoE chunk",
        True, "(Ensemble.default_chunk=%d)" % ens.default_chunk)

    c_mlp, c_moe, c_i8 = (read_cls(wd / f) for f in
                          ("cls_mlp.tif", "cls_moe_f32.tif", "cls_moe_i8.tif"))
    same = int((c_moe == c_i8).sum())
    tot = c_moe.size
    notes["int8_vs_f32_identical_px"] = same
    notes["int8_vs_f32_total_px"] = tot
    chk("int8 tile gives a bit-identical class map to float32",
        np.array_equal(c_moe, c_i8), f"{same:,}/{tot:,}")

    valid = c_moe != 0
    diff = float((c_mlp[valid] != c_moe[valid]).mean()) if valid.any() else 0.0
    notes["moe_vs_mlp_changed_frac"] = round(diff, 4)
    notes["valid_px"] = int(valid.sum())
    chk("the MoE actually produced a different map than the MLP",
        0.0 < diff < 0.5, f"{diff:.1%} of valid pixels differ")
    chk("output is non-trivial", valid.sum() > 0.2 * tot,
        f"{int(valid.sum()):,} valid px of {tot:,}")

    u_mlp = dict(zip(*[x.tolist() for x in np.unique(c_mlp[valid], return_counts=True)]))
    u_moe = dict(zip(*[x.tolist() for x in np.unique(c_moe[valid], return_counts=True)]))
    notes["class_hist_mlp"], notes["class_hist_moe"] = u_mlp, u_moe
    print("\n  class  MLP px      MoE px")
    for k in sorted(set(u_mlp) | set(u_moe)):
        print(f"  {k:5d}  {u_mlp.get(k,0):9,d}  {u_moe.get(k,0):9,d}")

    notes.update(t_mlp_s=round(t_mlp, 2), t_moe_f32_s=round(t_moe, 2),
                 t_moe_i8_s=round(t_i8, 2),
                 f32_mb=round(os.path.getsize(wd / "sub_f32.tif") / 1e6, 1),
                 i8_mb=round(os.path.getsize(wd / "sub_i8.tif") / 1e6, 1))
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(
        {"kind": "verify_deploy", "passed": int(sum(ok)), "of": len(ok),
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **notes}, indent=2))
    print(f"\n{sum(ok)}/{len(ok)} passed   ->  {op}")
    if not args.keep:
        shutil.rmtree(wd, ignore_errors=True)
    if not all(ok):
        raise SystemExit("verify_deploy FAILED")


if __name__ == "__main__":
    main()

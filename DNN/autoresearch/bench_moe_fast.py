"""How much of the 5.0x inference penalty on `moe_shared__n_experts8_top_k2`
is recoverable, and by which change.

`diag_inference_cost.json` measured the arm as written and stopped there. This
script is the follow-up: an ablation ladder from that number down, so the
speedup is attributed rather than just claimed. Every rung runs the deployed
path — `dnn_core.Ensemble.predict_classmap_gpu` semantics, raw features already
on-GPU, standardize on-device, mean-softmax over 5 members, argmax to int16 —
and every rung is checked against the reference loop's class map for exact
agreement before it is timed, so a fast wrong answer cannot win.

The ladder:
  as-written        the arm exactly as `moe_layers.SharedExpertMoE` runs it.
  no-sync           drop the per-expert device->host `float(we.abs().sum())`.
  fused             `moe_fast.FusedMoEEnsemble`: 40 experts and 5 trunks folded
                    into batched matmuls. Same weights, same routing.
  fused+tf32        Ampere TF32 for the fp32 matmuls (10-bit mantissa).
  fused+fp16        fp16 weights and activations.
  fused+bf16        bf16, kept in as the control that says fp16 is the right
                    reduced precision here: same speed class, 8x the class-map
                    disagreement, because 8 mantissa bits is not enough for the
                    ensemble mean-softmax to keep its argmax.
  fused+compile     torch.compile at a fixed chunk shape, fp32 and fp16.

Reported against two references, because they answer different questions:
`baseline` (the deployed MLP) is the cost the arm has to justify, and
`local_off` (this arm with its experts switched off) is the floor fusing could
possibly reach.

The reduced-precision and compile rungs are NOT exact, so they are scored on
class-map agreement against the fp32 reference rather than asserted equal. Note
that agreement here is a PESSIMISTIC proxy: these are random weights with an
untrained (deliberately sharpened) router, so the ensemble mean-softmax margins
are far tighter than a trained model's, and near-tied argmaxes are exactly what
low precision flips.

The number every rung should be read against is the I/O ceiling, not each
other: README.md measures `predict_raster.py` end-to-end at 1.4 M px/s on the
8-core VDI with 6 reader threads, and the GPU idles most of that run. A rung
above that line has stopped being the bottleneck, and going faster still buys
nothing wall-to-wall.

Run:
    ~/myprojects/recover/.venv/bin/python bench_moe_fast.py
    ~/myprojects/recover/.venv/bin/python bench_moe_fast.py --chunk-sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ar_common as ac
import moe_layers as ML
from moe_fast import FusedMoEEnsemble

IN_DIM, N_CLASSES = 67, 10
N_EXPERTS, TOP_K = 8, 2
AOI_PX = 1.3e9
DECODE = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]


# ------------------------------------------------------------------ the models
def make_members(dev, n=5):
    """The arm's 5 ensemble members, shapes as trained (config in results/)."""
    ms = []
    for i in range(n):
        torch.manual_seed(i)
        m = ML.SharedExpertMoE(
            IN_DIM, N_CLASSES, hidden=ac.HIDDEN, dropout=ac.DROPOUT,
            n_experts=N_EXPERTS, top_k=TOP_K, expert_hidden=(64, 32),
            gate_src="content", zero_init=False)
        with torch.no_grad():          # an untrained router is flat and would
            m.router.weight.mul_(30.0)  # route every row to the same experts
        ms.append(m.to(dev).eval())
    return ms


def make_mlp(dev, n=5):
    h1, h2 = ac.HIDDEN
    ms = []
    for i in range(n):
        torch.manual_seed(i)
        ms.append(nn.Sequential(
            nn.Linear(IN_DIM, h1), nn.ReLU(), nn.Dropout(ac.DROPOUT),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(ac.DROPOUT),
            nn.Linear(h2, N_CLASSES)).to(dev).eval())
    return ms


class _NoSync(ML.SharedExpertMoE):
    """The reference loop minus the per-expert host sync and the eval census."""

    def forward(self, x):
        feat = x[:, :self.feat_dim]
        if self.local_off:
            return self.shared(feat)
        w, _ = self._weights(self.router(self.gate_in(x)))
        out = self.shared(feat)
        for e, expert in enumerate(self.experts):
            out = out + w[:, e:e + 1] * expert(feat)
        return out


def as_nosync(models):
    out = []
    for m in models:
        c = _NoSync.__new__(_NoSync)
        c.__dict__ = m.__dict__
        c.__class__ = _NoSync
        out.append(c)
    return out


# ------------------------------------------------------------------ the runners
def loop_runner(models, cast=None):
    """dnn_core.Ensemble.predict_classmap_gpu, verbatim."""
    @torch.no_grad()
    def run(X, mean_t, std_t, decode, chunk):
        n = X.shape[0]
        cls = torch.empty(n, device=X.device, dtype=torch.int16)
        for i in range(0, n, chunk):
            xb = (X[i:i + chunk] - mean_t) / std_t
            acc = torch.zeros((xb.shape[0], N_CLASSES), device=X.device)
            for m in models:
                acc += F.softmax(m(xb).float(), 1)
            cls[i:i + chunk] = decode[acc.argmax(1)]
        return cls
    return run


def fused_runner(proba, dtype=None):
    """`proba` is any (B, in) -> (B, C) mean-softmax: a module's bound method or
    a torch.compile'd wrapper of one."""
    @torch.no_grad()
    def run(X, mean_t, std_t, decode, chunk):
        n = X.shape[0]
        cls = torch.empty(n, device=X.device, dtype=torch.int16)
        for i in range(0, n, chunk):
            xb = (X[i:i + chunk] - mean_t) / std_t
            if dtype is not None:
                xb = xb.to(dtype)
            cls[i:i + chunk] = decode[proba(xb).argmax(1)]
        return cls
    return run


# ---------------------------------------------------------------- measurement
def timed(run, X, mean_t, std_t, decode, chunk, reps):
    for _ in range(2):
        run(X, mean_t, std_t, decode, chunk)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run(X, mean_t, std_t, decode, chunk)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return X.shape[0] / float(np.median(ts))


def compiled_runner(dev, dtype=None, maxauto=False):
    f = fresh(dev, dtype)
    return fused_runner(torch.compile(f.mean_proba, dynamic=False,
                                      mode="max-autotune" if maxauto else None), dtype)


def fresh(dev, dtype=None):
    """A new fused ensemble per rung — one warmed module reused across rungs
    would let an earlier rung's cuBLAS autotune decide a later rung's timing."""
    f = FusedMoEEnsemble.from_models(make_members(dev))
    return f if dtype is None else f.to(dtype)


def build_rungs(dev, compile_on):
    members = make_members(dev)
    fast = FusedMoEEnsemble.from_models(members)
    rungs = [
        ("baseline", "deployed MLP 256,128 (reference)",
         loop_runner(make_mlp(dev)), False, None),
        ("as_written", "moe8/2 as written",
         loop_runner(members), True, None),
        ("no_sync", "  + drop per-expert host sync / census",
         loop_runner(as_nosync(make_members(dev))), True, None),
        ("fused", "  + fuse 40 experts + 5 trunks (moe_fast)",
         fused_runner(fast.mean_proba), True, None),
        ("fused_tf32", "  + TF32 matmuls",
         fused_runner(fresh(dev).mean_proba), False, "tf32"),
        ("fused_fp16", "  + fp16 weights/activations",
         fused_runner(fresh(dev, torch.float16).mean_proba, torch.float16), False, None),
        ("fused_bf16", "  (control) bf16 instead of fp16",
         fused_runner(fresh(dev, torch.bfloat16).mean_proba, torch.bfloat16), False, None),
    ]
    if compile_on:
        rungs += [
            ("fused_compile", "  + torch.compile, fp32",
             compiled_runner(dev), False, "tf32"),
            ("fused_fp16_compile_ma", "  + fp16 + torch.compile max-autotune",
             compiled_runner(dev, torch.float16, maxauto=True), False, "tf32"),
        ]

    off = fresh(dev)
    off.local_off = True
    rungs.append(("local_off", "moe8/2 experts OFF (fusing's speed floor)",
                  fused_runner(off.mean_proba), False, None))
    return rungs, fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--rows", type=int, default=4_194_304)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--chunk-sweep", action="store_true",
                    help="throughput vs chunk for the fused path, with its memory")
    ap.add_argument("--out", default=str(ac.RESULTS_DIR / "moe_fast_bench.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the A40; this is a GPU deployment question")
    dev = torch.device("cuda")
    torch.manual_seed(123)
    X = torch.randn(args.rows, IN_DIM, device=dev)
    mean_t = torch.randn(IN_DIM, device=dev)
    std_t = torch.rand(IN_DIM, device=dev) + 0.5
    decode = torch.tensor(DECODE, device=dev, dtype=torch.int16)

    if args.chunk_sweep:
        fast = FusedMoEEnsemble.from_models(make_members(dev))
        run = fused_runner(fast.mean_proba)
        print(f"{'chunk':>9} {'M px/s':>9} {'act GB':>8}")
        for ch in (16384, 32768, 65536, 131072, 262144, 524288):
            torch.cuda.empty_cache()
            r = timed(run, X, mean_t, std_t, decode, ch, 3)
            print(f"{ch:>9,} {r/1e6:>9.3f} {fast.activation_bytes(ch)/1e9:>8.2f}")
        return

    print(f"device={torch.cuda.get_device_name(0)}  chunk={args.chunk:,}  "
          f"rows={args.rows:,}  ensemble=5  reps={args.reps}\n")

    rungs, fast = build_rungs(dev, not args.no_compile)
    io_ceiling = 1.4e6

    # exactness gate: every rung's class map, against the arm as written
    ref_cls = loop_runner(make_members(dev))(X[:524288], mean_t, std_t, decode, 65536)

    out_rows, base_rate, moe_rate = [], None, None
    for key, label, run, must_match, backend in rungs:
        torch.backends.cuda.matmul.allow_tf32 = (backend == "tf32")
        torch.backends.cudnn.allow_tf32 = (backend == "tf32")
        agree = None
        if key != "baseline":
            got = run(X[:524288], mean_t, std_t, decode, 65536)
            agree = float((got == ref_cls).float().mean())
            if must_match and agree != 1.0:
                raise SystemExit(f"{key}: class map differs from the reference "
                                 f"({agree:.6f} agree) — not a valid optimisation")
        torch.cuda.empty_cache()
        rate = timed(run, X, mean_t, std_t, decode, args.chunk, args.reps)
        if base_rate is None:
            base_rate = rate
        if key == "as_written":
            moe_rate = rate
        rec = {"name": key, "label": label.strip(),
               "px_per_s": round(rate),
               "aoi_minutes": round(AOI_PX / rate / 60, 2),
               "vs_baseline_time": round(base_rate / rate, 2),
               "speedup_vs_as_written": (None if moe_rate is None
                                         else round(rate / moe_rate, 2)),
               "classmap_agreement": agree, "exact": bool(must_match),
               "above_io_ceiling": bool(rate > io_ceiling)}
        out_rows.append(rec)
        sp = "" if rec["speedup_vs_as_written"] is None else \
            f"{rec['speedup_vs_as_written']:5.2f}x arm"
        ag = "exact" if agree == 1.0 else ("" if agree is None else f"agree={agree:.6f}")
        print(f"{label:44s} {rate/1e6:7.3f} M px/s  "
              f"{rec['aoi_minutes']:6.2f} min/1.3Bpx  "
              f"{rec['vs_baseline_time']:5.2f}x base  {sp:10s} "
              f"{'  ' if rate > io_ceiling else 'IO'} {ag}")

    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"\n  IO = below README.md's measured end-to-end raster ceiling "
          f"({io_ceiling/1e6:.1f} M px/s, 6 readers): still the bottleneck.")
    out = {
        "kind": "moe_fast_bench",
        "arm": "moe_shared__n_experts8_top_k2",
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "chunk": args.chunk, "rows": args.rows, "reps": args.reps,
        "ensemble": 5, "n_experts": N_EXPERTS, "top_k": TOP_K,
        "aoi_px": AOI_PX,
        "io_px_per_s": {"local ssd, 6 readers": 1.4e6, "cifs P-drive": 0.2e6},
        "fused_activation_bytes_at_chunk": fast.activation_bytes(args.chunk),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rungs": out_rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

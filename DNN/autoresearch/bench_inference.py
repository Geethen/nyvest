"""What each locality arm would cost to DEPLOY, measured on the deployment path.

The round scored mechanisms on macro-F1 and stopped there. Nothing here ships on
F1 alone: `predict_raster.py` runs the chosen net over ~1.3 billion pixels, so a
mechanism that buys +0.004 for 2x the forward pass has to be worth 2x the forward
pass. This script produces the other column of that decision.

What is measured
----------------
The hot path exactly as deployed — `dnn_core.Ensemble.predict_classmap_gpu`:
raw features already on-GPU, standardize on-device, mean-softmax over the 5-seed
ensemble, argmax to an int16 class map, in `--chunk` row blocks. No host
round-trip, because the raster pipeline does not do one either.

Three numbers per arm:

  params      what a checkpoint carries (x5 for the ensemble).
  MACs/row    ANALYTIC, and given twice. `dense` is what the code as written
              actually executes; `routed` is what an ideal sparse kernel would
              execute given the arm's measured routing. SharedExpertMoE loops
              over every expert and multiplies by a gate weight that is zero for
              the ones not selected, and MoDNet computes its extra block for all
              rows and gates the result — so top-k here buys accuracy, not FLOPs.
              The gap between the two columns is the work a gather/scatter
              implementation could remove, and it is the honest ceiling on "the
              MoE is cheap because it is sparse".
  M px/s      MEASURED, median of `--reps` timed passes after warmup.

Run:
    ~/myprojects/recover/.venv/bin/python bench_inference.py
    ~/myprojects/recover/.venv/bin/python bench_inference.py --cpu --reps 3
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ar_common as ac          # noqa: E402
import moe_layers as ML         # noqa: E402

IN_DIM, N_CLASSES = 67, 10      # 64 AlphaEarth bands + elevation/tri/tch
# Weights are random — timing depends on SHAPE, not on values — but the shapes
# are the deployed ones: models/dnn_final.pt carries cfg hidden=[256,128],
# dropout=0.3, n_ensemble=5, which is what the `baseline` arm builds.
# Wall-to-wall target from README.md's "Inference performance" section: three
# Norwegian counties at 10 m. Reported so the numbers here sit in the same units
# as the I/O measurements they have to be compared against.
AOI_PX = 1.3e9


# --------------------------------------------------------------- the arms
def _spec():
    """(key, label, builder, kwargs, measured routing) for each arm.

    `routed_frac` is what the trial's own fold_info recorded on HELD-OUT rows,
    not a design constant: MoDNet's capacity is 0.5 by construction during
    training but its inference-time auxiliary predictor chose 0.515 of rows.
    """
    def mlp(h1, h2):
        return lambda: nn.Sequential(
            nn.Linear(IN_DIM, h1), nn.ReLU(), nn.Dropout(ac.DROPOUT),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(ac.DROPOUT),
            nn.Linear(h2, N_CLASSES))

    def moe(n_experts, top_k, eh=(64, 32)):
        return lambda: ML.SharedExpertMoE(
            IN_DIM, N_CLASSES, hidden=ac.HIDDEN, dropout=ac.DROPOUT,
            n_experts=n_experts, top_k=top_k, expert_hidden=eh,
            gate_src="content")

    return [
        ("baseline", "deployed MLP 256,128", mlp(256, 128), {}),
        ("ctrl_capacity", "plain MLP 320,170 (matched capacity)", mlp(320, 170), {}),
        ("moe_shared", "shared + 4 experts, top-2", moe(4, 2), {}),
        ("moe_shared__n_experts8_top_k2", "shared + 8 experts, top-2", moe(8, 2), {}),
        ("moe_shared__n_experts16_top_k4", "shared + 16 experts, top-4", moe(16, 4), {}),
        ("mod_depth", "adaptive depth, capacity 0.5",
         lambda: ML.MoDNet(IN_DIM, N_CLASSES, hidden=ac.HIDDEN,
                           dropout=ac.DROPOUT, capacity=0.5), {}),
    ]


# ------------------------------------------------------------- analytic cost
def _lin_macs(seq_dims):
    return sum(a * b for a, b in zip(seq_dims[:-1], seq_dims[1:]))


def analytic(key):
    """MACs per row, as executed and as an ideal sparse kernel would.

    Counted by hand rather than by a profiler hook so the two columns can differ:
    a profiler sees the dense loop and would report it twice.
    """
    trunk = _lin_macs([IN_DIM, 256, 128, N_CLASSES])
    if key == "baseline":
        return trunk, trunk, 1.0
    if key == "ctrl_capacity":
        w = _lin_macs([IN_DIM, 320, 170, N_CLASSES])
        return w, w, 1.0
    if key.startswith("moe_shared"):
        n_e, k = (4, 2)
        if "n_experts8" in key:
            n_e, k = 8, 2
        elif "n_experts16" in key:
            n_e, k = 16, 4
        expert = _lin_macs([IN_DIM, 64, 32, N_CLASSES])
        router = IN_DIM * n_e
        return (trunk + n_e * expert + router,      # every expert, as written
                trunk + k * expert + router,        # only the selected ones
                k / n_e)
    if key == "mod_depth":
        block = _lin_macs([128, 128, 128])
        heads = 128 * 2                             # router + aux predictor
        frac = 0.515                                # measured te_deep_frac
        return (trunk + block + heads,
                trunk + round(block * frac) + heads,
                frac)
    raise KeyError(key)


# --------------------------------------------------------------- measurement
def bench(build, device, chunk, rows, reps, ensemble=5, local_off=False):
    """Median rows/s of the deployed classmap path over `rows` synthetic rows."""
    models = []
    for i in range(ensemble):
        torch.manual_seed(i)
        m = build().to(device).eval()
        if local_off:
            m.local_off = True
        models.append(m)
    X = torch.randn(rows, IN_DIM, device=device)
    mean_t = torch.zeros(IN_DIM, device=device)
    std_t = torch.ones(IN_DIM, device=device)
    decode = torch.arange(N_CLASSES, device=device, dtype=torch.int16)

    def one_pass():
        cls = torch.empty(rows, device=device, dtype=torch.int16)
        for i in range(0, rows, chunk):
            xb = (X[i:i + chunk] - mean_t) / std_t
            acc = torch.zeros((xb.shape[0], N_CLASSES), device=device)
            with torch.no_grad():
                for m in models:
                    acc += F.softmax(m(xb).float(), 1)
            cls[i:i + chunk] = decode[acc.argmax(1)]
        return cls

    for _ in range(2):                              # warmup: cuBLAS autotune
        one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        one_pass()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    del models, X
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows / float(np.median(ts))


def n_params(build):
    return sum(p.numel() for p in build().parameters())


def _selftest(verbose=True):
    """The two claims in this file that are asserted rather than measured.

    1. The analytic MAC counts. Every layer here is an nn.Linear, so a model's
       weight count and its MACs-per-row are THE SAME NUMBER: params minus the
       1-D (bias) parameters must equal the dense MAC count exactly. That turns
       a hand-count into a checkable identity — and a hand-count is what the
       dense-vs-sparse column pair depends on.
    2. `local_off` really is the deployed model. The claim "the routed capacity
       can be dropped at deployment without retraining" is only true if the
       switched-off net computes the shared expert and nothing else.

    Run:  python bench_inference.py --selftest
    """
    torch.manual_seed(0)
    x = torch.randn(256, IN_DIM)
    ok = []

    def chk(name, cond, detail=""):
        ok.append(bool(cond))
        if verbose:
            print(f"  [{'ok ' if cond else 'FAIL'}] {name} {detail}")

    for key, label, build, _ in _spec():
        m = build().eval()
        dense, routed, frac = analytic(key)
        weights = sum(p.numel() for p in m.parameters() if p.ndim > 1)
        chk(f"{key}: dense MACs == weight count", weights == dense,
            f"{weights:,} vs {dense:,}")
        chk(f"{key}: ideal sparse <= dense", routed <= dense,
            f"{routed:,} <= {dense:,}")
        if hasattr(m, "local_off"):
            m.local_off = True
            off = m(x)
            m.local_off = False
            ref = (m.shared(x) if hasattr(m, "shared") else
                   m.head(m.drop(F.relu(m.l2(m.drop(F.relu(m.l1(x))))))))
            chk(f"{key}: local_off == the deployed trunk",
                torch.allclose(off, ref, atol=1e-6),
                f"max|d|={float((off - ref).abs().max()):.2e}")
            chk(f"{key}: routed_frac is a fraction", 0.0 < frac <= 1.0, f"{frac}")

    print(f"\n{sum(ok)}/{len(ok)} passed")
    return all(ok), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=262144,
                    help="rows per forward chunk (predict_raster.py --gpu-chunk)")
    ap.add_argument("--rows", type=int, default=4_194_304, help="rows per timed pass")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ensemble", type=int, default=ac.N_ENSEMBLE)
    ap.add_argument("--cpu", action="store_true", help="also time the CPU path")
    ap.add_argument("--cpu-rows", type=int, default=524_288)
    ap.add_argument("--out", default=str(ac.RESULTS_DIR / "diag_inference_cost.json"))
    ap.add_argument("--selftest", action="store_true",
                    help="check the analytic counts and the local_off claim, then exit")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if _selftest()[0] else 1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor()
    print(f"device={dev} ({gpu_name})  chunk={args.chunk:,}  rows={args.rows:,}  "
          f"ensemble={args.ensemble}  reps={args.reps}\n")

    rows_out = []
    base_rate = base_macs = None
    for key, label, build, _ in _spec():
        p = n_params(build)
        dense, routed, frac = analytic(key)
        rate = bench(build, dev, args.chunk, args.rows, args.reps, args.ensemble)
        rec = {
            "name": key, "label": label,
            "params": p, "params_ensemble": p * args.ensemble,
            "macs_row_dense": dense, "macs_row_routed_ideal": routed,
            "routed_frac": frac,
            "px_per_s": round(rate),
            "aoi_minutes": round(AOI_PX / rate / 60, 2),
        }
        # The MoE arms can be scored with their local branch switched off, which
        # is the deployed MLP exactly — the same weights, one flag. Worth a
        # number: it is the fallback if the routed capacity ever has to be
        # dropped at deployment without retraining.
        if key.startswith("moe_shared") or key == "mod_depth":
            rec["px_per_s_local_off"] = round(
                bench(build, dev, args.chunk, args.rows, args.reps,
                      args.ensemble, local_off=True))
        if args.cpu:
            rec["cpu_px_per_s"] = round(
                bench(build, torch.device("cpu"), args.chunk, args.cpu_rows,
                      max(2, args.reps // 2), args.ensemble))
        if base_rate is None:
            base_rate, base_macs, base_p = rate, dense, p
        rec["rel_params"] = round(p / base_p, 2)
        rec["rel_macs"] = round(dense / base_macs, 2)
        rec["rel_time"] = round(base_rate / rate, 2)
        rows_out.append(rec)
        print(f"{key:32s} {p:8,d} par  {dense:8,d} MAC/row  "
              f"{rate/1e6:6.2f} M px/s  {rec['aoi_minutes']:6.2f} min/1.3Bpx  "
              f"({rec['rel_time']:.2f}x baseline time)")

    out = {
        "kind": "inference_cost",
        "device": gpu_name, "torch": torch.__version__,
        "chunk": args.chunk, "rows": args.rows, "reps": args.reps,
        "ensemble": args.ensemble, "in_dim": IN_DIM, "n_classes": N_CLASSES,
        "aoi_px": AOI_PX,
        # For context, not measured here: README.md's end-to-end raster figures.
        # The comparison that decides deployment is arm-vs-IO, not arm-vs-arm.
        "io_px_per_s": {"local ssd, 6 readers": 1.4e6, "cifs P-drive": 0.2e6},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": rows_out,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

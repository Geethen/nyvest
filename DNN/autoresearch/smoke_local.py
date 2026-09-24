"""Integration smoke test for the locality round.

moe_layers._selftest() checks the mechanisms in isolation; this checks they
survive contact with the harness — the gate-only column plumbing, ctx keys the
builders read, the auxiliary-loss path, and the two full fold overrides. Runs
the real data at 2 epochs / 1 member into a scratch results dir, so a shape or
key error surfaces in a minute instead of forty.

    python smoke_local.py [trial ...]
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MAX_EPOCHS", "2")
os.environ.setdefault("PATIENCE", "1")
os.environ.setdefault("N_ENSEMBLE", "1")
os.environ.setdefault("WANDB_MODE", "offline")
SCRATCH = Path(os.environ.setdefault(
    "AR_RESULTS_DIR", tempfile.mkdtemp(prefix="ar_smoke_")))

import torch                      # noqa: E402
import torch.nn as nn             # noqa: E402

import ar_common as ac            # noqa: E402
import moe_layers as M            # noqa: E402
import trials as T                # noqa: E402


def unit_pass():
    """Every builder gets a fake ctx and one forward/backward."""
    print("== builders on synthetic data ==")
    B, D, C = 256, 67, 10
    for name in T.QUEUE_LOCAL:
        t = T.TRIALS[name]
        if t.build_fn is None:
            print(f"  [skip] {name} (fold override — covered by the live pass)")
            continue
        n_extra = 2 if t.extra_gate == "geo" else 0
        ctx = {"trial": t, "lidar_cols": [64, 65, 66], "n_gate_extra": n_extra,
               "n_classes": C}
        x = torch.randn(B, D + n_extra)
        y = torch.randint(0, C, (B,))
        m = t.build_fn(D + n_extra, C, ctx)
        crit = nn.CrossEntropyLoss()
        loss = t.loss_fn(m, x, y, crit, ctx) if t.loss_fn else crit(m(x), y)
        loss.backward()
        gnorm = sum(float(p.grad.abs().sum()) for p in m.parameters()
                    if p.grad is not None)
        n_par = sum(p.numel() for p in m.parameters())
        assert gnorm > 0, f"{name}: no gradient reached any parameter"
        print(f"  [ok ] {name:14s} {n_par:>7,} params  loss={float(loss):.4f}")


def live_pass(names):
    print(f"\n== live 2-epoch pass (results -> {SCRATCH}) ==")
    base = Path(__file__).resolve().parent / "results" / "baseline.json"
    (SCRATCH / "_baseline.json").write_text(base.read_text())
    (SCRATCH / "baseline_gval.json").write_text(
        (base.parent / "baseline_gval.json").read_text())
    for name in names:
        rec = ac.run_trial(T.resolve(name))
        info = rec["fold_info"][0]
        keys = {k: v for k, v in info.items() if k != "f1_raw_argmax"}
        print(f"  [ok ] {name:14s} F1={rec['f1_mean']:.4f}  fold0 info={keys}")


if __name__ == "__main__":
    M._selftest(verbose=False)
    print("moe_layers self-test: passed")
    unit_pass()
    want = sys.argv[1:] or T.QUEUE_LOCAL
    live_pass(want)
    print("\nsmoke: all passed")

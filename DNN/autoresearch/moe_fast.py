"""Deployment-time rewrite of the `moe_shared` arms: same weights, ~20 kernels.

WHY

`moe_shared__n_experts8_top_k2` scored +0.004 macro-F1 and cost 5.0x the
baseline forward pass (`results/diag_inference_cost.json`), which is the reason
it did not ship. That 5.0x is not FLOPs — the arm is only 2.05x the baseline in
MACs/row, and switching its experts off (`local_off`) recovers baseline speed
exactly. The gap is LAUNCH COUNT. `SharedExpertMoE.forward` is written for
research legibility: it loops over experts in Python, and each expert is an
`nn.Sequential` of three Linears whose hidden dims are 64 and 32. Per ensemble
member that is 8 x (3 GEMM + 2 ReLU) plus 8 x (slice, mul, add) ~= 64 kernels
against the shared trunk's 5, and `dnn_core.Ensemble.predict_classmap_gpu` runs
five members, so a 262144-row chunk dispatches ~350 kernels that each do a few
microseconds of work on an A40. The GPU is idle between them.

Nothing about that is inherent to the mechanism. Every expert sees the SAME
rows (moe_layers.py rule 2: experts are computed densely), every ensemble member
sees the same rows, and all 40 experts have identical shape — so the whole thing
is three batched matmuls.

WHAT IS FUSED

  * 40 experts (5 members x 8) -> one (67, 40*64) GEMM, then two bmms over a
    40-wide batch. Ensemble members fuse with experts because neither dimension
    interacts with the other until the final weighted sum.
  * 5 shared trunks -> one (67, 5*256) GEMM + two bmms.
  * 5 routers -> one (67, 5*8) GEMM, with top-k taken batched over (B, M, E).
  * The gate-weighted sum stays on the (M, E, B, 10) logit tensor, which is
    small (C=10); folding it into the last bmm would need a permute of the
    (M*E, B, 32) activations and cost more than it saves.

WHAT IS DROPPED (deployment-only, and the reason this is a separate class)

  * The eval-time routing census (`route_counts`/`route_rows`). Telemetry the
    trial harness reads; a raster run does not.
  * The per-expert `float(we.abs().sum()) == 0.0` skip. It is a device->host
    sync per expert per member, and over a 262144-row chunk every expert is
    always used, so the branch never fires. (Measured: worth ~0% on its own —
    the launches, not the syncs, are the cost.)
  * `nn.Dropout`, which is identity in eval.

WHAT IS NOT CHANGED

The arithmetic. Same weights, same routing rule, same dense-expert semantics.
`_selftest` checks the fused logits against the reference `SharedExpertMoE` loop
and reports the argmax agreement rate; see its docstring for why the match is
"equal up to fp32 reassociation" rather than bitwise.

SPARSE DISPATCH, AND WHY IT IS NOT HERE

top-2-of-8 means 75% of the dense expert work is multiplied by a zero gate, so a
gather/scatter kernel would cut expert MACs 4x (the `macs_row_routed_ideal`
column in diag_inference_cost.json) and, since the fused path turns out to be
bandwidth-bound rather than launch-bound, would cut the dominant (B, 40*64)
activation by 4x as well. So it is a real optimisation, not a fake one — worth
maybe another 1.5-2x on the expert stack.

It is not here because of where the finish line is. `predict_raster.py` is I/O
bound: README.md measures the full pipeline at 1.4 M px/s on the 8-core VDI with
6 readers, decompression-limited, with the GPU idle most of the run. Fused fp32
already runs the arm ABOVE that line, so sparse dispatch would be making a
component faster that has stopped being the bottleneck. It would also cost the
two things this file is otherwise free of: a host sync per chunk (the group
sizes are data-dependent, and capacity padding without one either drops rows or
allocates more than the dense path it replaces) and a ragged-group loop that
walks back toward the launch-bound structure fusing just removed.

If the I/O ceiling ever moves — an uncompressed local source, or many more read
cores — this is the next thing to build, and `bench_moe_fast.py`'s stage timings
say to build it for the expert L1/L2 activations specifically.

Use:
    from moe_fast import FusedMoEEnsemble
    fast = FusedMoEEnsemble.from_models(ensemble.models)      # trained weights
    cls  = fast.predict_classmap_gpu(X_raw_t, mean_t, std_t, decode, chunk=131072)

Check:
    ~/myprojects/recover/.venv/bin/python moe_fast.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _linears(seq):
    return [m for m in seq if isinstance(m, nn.Linear)]


def _stack_mlp(seqs):
    """G identically-shaped `_mlp` Sequentials -> one fused-first-layer stack.

    Returns (W0, b0, Ws, bs) where W0 is (in, G*h0) so layer 0 is a single GEMM
    that reads x once, and Ws[i] is (G, h_i, h_{i+1}) for the batched rest.
    """
    lins = [_linears(s) for s in seqs]
    L = len(lins[0])
    if L < 2:
        raise ValueError("need at least one hidden layer to fuse")
    if any(len(l) != L for l in lins):
        raise ValueError("stacked MLPs differ in depth")
    W0 = torch.cat([l[0].weight.detach().t() for l in lins], dim=1).contiguous()
    b0 = torch.cat([l[0].bias.detach() for l in lins]).contiguous()
    Ws = [torch.stack([l[i].weight.detach().t() for l in lins]).contiguous()
          for i in range(1, L)]
    bs = [torch.stack([l[i].bias.detach() for l in lins]).unsqueeze(1).contiguous()
          for i in range(1, L)]
    return W0, b0, Ws, bs


def _run_stack(x, W0, b0, Ws, bs, G, h0):
    """(B, in) -> (G, B, out) through a stack fused by `_stack_mlp`.

    The `.view(B, G, h0).transpose(0, 1)` is free: it hands cuBLAS a strided
    batched operand (batch stride h0, lda G*h0) rather than a copy.
    """
    h = torch.addmm(b0, x, W0).relu_().view(-1, G, h0).transpose(0, 1)
    last = len(Ws) - 1
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = torch.baddbmm(b, h, W)
        if i < last:
            h = h.relu_()
    return h


class FusedMoEEnsemble(nn.Module):
    """An ensemble of `SharedExpertMoE` members as three fused stacks.

    Built from trained members, not trained itself — `from_models` copies
    weights out of the reference modules and never writes back.
    """

    def __init__(self, models):
        super().__init__()
        ref = models[0]
        for m in models:
            if (m.n_experts, m.top_k, m.feat_dim, m.balance) != (
                    ref.n_experts, ref.top_k, ref.feat_dim, ref.balance):
                raise ValueError("ensemble members must share MoE geometry")
        self.M, self.E, self.K = len(models), ref.n_experts, ref.top_k
        self.feat_dim = ref.feat_dim
        # echoice routing is un-renormalised at eval (moe_layers.SharedExpertMoE
        # docstring); every other balance mode renormalises over the top-k.
        self.renorm = ref.balance != "echoice"
        self.lossfree = ref.balance == "lossfree"
        self.local_off = False

        sW0, sb0, sWs, sbs = _stack_mlp([m.shared for m in models])
        self.sh0 = sW0.shape[1] // self.M
        self._reg("s", sW0, sb0, sWs, sbs)
        self.n_classes = sWs[-1].shape[-1]

        experts = [models[m].experts[e] for m in range(self.M) for e in range(self.E)]
        eW0, eb0, eWs, ebs = _stack_mlp(experts)
        self.eh0 = eW0.shape[1] // (self.M * self.E)
        self._reg("e", eW0, eb0, eWs, ebs)

        self.register_buffer("rW", torch.cat(
            [m.router.weight.detach().t() for m in models], dim=1).contiguous())
        self.register_buffer("rb", torch.cat(
            [m.router.bias.detach() for m in models]).contiguous())
        self.register_buffer("bias", torch.stack(
            [m.bias.detach() for m in models]).contiguous())      # (M, E)

        # Gate columns. "content" is the whole feature block, so the gate reads a
        # view; "terrain"/"geo" are a real gather and are kept as an index.
        cols = ref.gate_in.cols
        self.gate_all = bool(cols.numel() == self.feat_dim and
                             torch.equal(cols.cpu(), torch.arange(self.feat_dim)))
        self.register_buffer("gate_cols", cols.detach().clone())

    def _reg(self, tag, W0, b0, Ws, bs):
        self.register_buffer(f"{tag}W0", W0)
        self.register_buffer(f"{tag}b0", b0)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            self.register_buffer(f"{tag}W{i + 1}", W)
            self.register_buffer(f"{tag}b{i + 1}", b)
        setattr(self, f"{tag}_n", len(Ws))

    def _stack(self, tag):
        n = getattr(self, f"{tag}_n")
        return (getattr(self, f"{tag}W0"), getattr(self, f"{tag}b0"),
                [getattr(self, f"{tag}W{i + 1}") for i in range(n)],
                [getattr(self, f"{tag}b{i + 1}") for i in range(n)])

    @classmethod
    def from_models(cls, models):
        ref = models[0]
        dev = next(ref.parameters()).device
        return cls([m.eval() for m in models]).to(dev).eval()

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        """(B, in) -> (M, B, C) per-member logits, same values as the loop."""
        feat = x if x.shape[1] == self.feat_dim else x[:, :self.feat_dim]
        W0, b0, Ws, bs = self._stack("s")
        shared = _run_stack(feat, W0, b0, Ws, bs, self.M, self.sh0)   # (M,B,C)
        if self.local_off:
            return shared

        gx = feat if self.gate_all else x[:, self.gate_cols]
        s = torch.addmm(self.rb, gx, self.rW).view(-1, self.M, self.E)
        probs = s.softmax(2)
        sel = s + self.bias if self.lossfree else s
        top = sel.topk(self.K, dim=2).indices
        g = probs.gather(2, top)
        if self.renorm:
            g = g / g.sum(2, keepdim=True).clamp_min(1e-9)
        w = torch.zeros_like(probs).scatter_(2, top, g)               # (B,M,E)

        W0, b0, Ws, bs = self._stack("e")
        o = _run_stack(feat, W0, b0, Ws, bs, self.M * self.E, self.eh0)
        o = o.view(self.M, self.E, -1, self.n_classes)                # (M,E,B,C)
        return shared + (o * w.permute(1, 2, 0).unsqueeze(-1)).sum(1)

    def mean_proba(self, x):
        """Ensemble mean softmax, the quantity `predict_classmap_gpu` argmaxes."""
        return self(x).float().softmax(-1).mean(0)

    # --------------------------------------------------------------- deployment
    @torch.no_grad()
    def predict_classmap_gpu(self, X_raw_t, mean_t, std_t, decode, chunk=65536):
        """Drop-in for `dnn_core.Ensemble.predict_classmap_gpu`.

        `chunk` defaults to a QUARTER of dnn_core's 262144. Dense experts hold a
        (B, M*E, h0) activation — 4.0 GB of it at 262144 rows and 40 experts —
        and `bench_moe_fast.py --chunk-sweep` shows throughput flat from 65536
        upward, so the larger chunk buys nothing and costs 4x the memory. Below
        65536 the batched matmuls stop filling the device and it does cost speed.
        See `activation_bytes`.
        """
        n = X_raw_t.shape[0]
        cls = torch.empty(n, device=X_raw_t.device, dtype=torch.int16)
        for i in range(0, n, chunk):
            xb = (X_raw_t[i:i + chunk] - mean_t) / std_t
            cls[i:i + chunk] = decode[self.mean_proba(xb).argmax(1)]
        return cls

    def activation_bytes(self, chunk):
        """Peak expert-activation footprint, the constraint on `chunk`."""
        g = self.M * self.E
        return chunk * 4 * (g * self.eh0 + g * self.eW1.shape[-1])


# ------------------------------------------------------------------- pre-flight
def _selftest(verbose=True):
    """Fused == loop, on the arms that would actually be deployed.

    The match is "equal up to fp32 reassociation", not bitwise, and that is a
    property of the fusion rather than a slack tolerance: the reference computes
    each member's router as a (B,67)@(67,8) GEMM while the fused path computes
    one (B,67)@(67,40), so the accumulation order differs and scores land a few
    ULP apart. That matters in exactly one place — if two experts are tied at
    the top-k boundary the two paths can pick different ones — so the test
    reports route disagreements separately from logit error, and requires the
    ENSEMBLE ARGMAX (the only thing a class map records) to agree on every row.
    """
    import moe_layers as ML

    ok = []

    def chk(name, cond, detail=""):
        ok.append(bool(cond))
        if verbose:
            print(f"  [{'ok ' if cond else 'FAIL'}] {name} {detail}")

    D, C, B = 67, 10, 20_000
    cases = [
        ("n_experts8_top_k2 (the arm)", dict(n_experts=8, top_k=2, expert_hidden=(64, 32))),
        ("n_experts4_top_k2", dict(n_experts=4, top_k=2, expert_hidden=(64, 32))),
        ("n_experts16_top_k4", dict(n_experts=16, top_k=4, expert_hidden=(64, 32))),
        ("expert_hidden 128,64", dict(n_experts=8, top_k=2, expert_hidden=(128, 64))),
        ("lossfree balance", dict(n_experts=8, top_k=2, expert_hidden=(64, 32),
                                  balance="lossfree")),
        ("terrain gate", dict(n_experts=8, top_k=2, expert_hidden=(64, 32),
                              gate_src="terrain", gate_slice=[64, 65, 66])),
    ]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for label, kw in cases:
        torch.manual_seed(0)
        models = []
        for i in range(5):
            torch.manual_seed(i)
            m = ML.SharedExpertMoE(D, C, hidden=(256, 128), dropout=0.3,
                                   zero_init=False, **kw).to(dev).eval()
            with torch.no_grad():       # untrained experts are zero-init'd; a
                m.router.weight.mul_(30.0)   # flat router would hide route bugs
                for e in m.experts:
                    for p in e.parameters():
                        p.add_(torch.randn_like(p) * 0.2)
            models.append(m)
        x = torch.randn(B, D, device=dev)

        fast = FusedMoEEnsemble.from_models(models)
        with torch.no_grad():
            ref = torch.stack([m(x) for m in models])
            got = fast(x)
        d = float((ref - got).abs().max())
        # a route flip shows up as a large per-row logit delta, so count rows
        # rather than trusting a global max
        flips = int(((ref - got).abs().amax(-1) > 1e-3).any(0).sum())
        chk(f"{label}: logits match", d < 1e-3 or flips <= B // 10_000,
            f"max|d|={d:.2e}  route-flip rows={flips}/{B}")

        with torch.no_grad():
            pr = torch.stack([F.softmax(m(x).float(), 1) for m in models]).mean(0)
        agree = float((pr.argmax(1) == fast.mean_proba(x).argmax(1)).float().mean())
        chk(f"{label}: ensemble argmax agrees", agree == 1.0, f"{agree:.6f}")

        with torch.no_grad():
            fast.local_off = True
            off = fast(x)
            fast.local_off = False
            ref_off = torch.stack([m.shared(x[:, :m.feat_dim]) for m in models])
        chk(f"{label}: local_off == the shared trunks",
            torch.allclose(off, ref_off, atol=1e-4),
            f"max|d|={float((off - ref_off).abs().max()):.2e}")

    # the deployed entry point, end to end against dnn_core's own loop
    torch.manual_seed(0)
    models = []
    for i in range(5):
        torch.manual_seed(i)
        m = ML.SharedExpertMoE(D, C, hidden=(256, 128), dropout=0.3, n_experts=8,
                               top_k=2, expert_hidden=(64, 32), zero_init=False).to(dev).eval()
        with torch.no_grad():
            m.router.weight.mul_(30.0)
            for e in m.experts:
                for p in e.parameters():
                    p.add_(torch.randn_like(p) * 0.2)
        models.append(m)
    X = torch.randn(50_000, D, device=dev)
    mean_t = torch.randn(D, device=dev)
    std_t = torch.rand(D, device=dev) + 0.5
    decode = torch.tensor([2, 3, 4, 5, 6, 7, 8, 10, 11, 12], device=dev, dtype=torch.int16)
    fast = FusedMoEEnsemble.from_models(models)
    with torch.no_grad():
        ref_cls = torch.empty(50_000, device=dev, dtype=torch.int16)
        for i in range(0, 50_000, 8192):
            xb = (X[i:i + 8192] - mean_t) / std_t
            acc = torch.zeros((xb.shape[0], C), device=dev)
            for m in models:
                acc += F.softmax(m(xb).float(), 1)
            ref_cls[i:i + 8192] = decode[acc.argmax(1)]
    for ch in (8192, 16384, 50_000):
        got_cls = fast.predict_classmap_gpu(X, mean_t, std_t, decode, chunk=ch)
        chk(f"classmap matches dnn_core loop (chunk={ch})",
            bool((ref_cls == got_cls).all()),
            f"{float((ref_cls == got_cls).float().mean()):.6f} agree")

    print(f"\nmoe_fast self-test: {sum(ok)}/{len(ok)} passed")
    if not all(ok):
        raise SystemExit("moe_fast self-test FAILED")
    return True


if __name__ == "__main__":
    _selftest()

"""Locality mechanisms: MoE routing, low-rank local adapters, adaptive depth,
and context-conditioned modulation.

WHY THIS FILE EXISTS, given that nine MoE variants already lost here
(stage6/stage7 class- and spatial-MoE, Soft-MoE in DNN/research):

Every one of those gated on POSITION — KMeans over lon/lat, or lon/lat straight
into the gate. Under this evaluation that is self-defeating. Folds are
GroupKFold over cell_id, so a test fold IS a geographic region the model has
never seen; a position gate maps all of it to whichever expert owns the nearest
training region, and 84-95% of the fold lands on ONE expert. That expert saw a
quarter of the data, so a position-routed MoE is a strictly smaller training set
dressed up as specialisation.

An LLM MoE does not route on position. It routes on the token REPRESENTATION —
on content. Content routing survives a new region because a row in an unseen
region still resembles some rows in the training set: the partition is over the
feature manifold, not over the map. That is the untested version of "local
models beat global models", and it is what these modules implement, with the
position gate kept as an explicit control arm so the contrast is paired.

Two design rules run through everything here:

  1. START AT THE GLOBAL MODEL. Every mechanism is a global trunk of exactly the
     deployed shape plus a correction whose output projection is zero-initialised
     (routed experts, LoRA B, the MoD block, FiLM's beta). At step 0 the network
     computes the deployed model bit for bit, so a loss is never a failure to
     reach the baseline, and a win is local specialisation actually adding
     something. This is LoRA's argument and DeepSeekMoE's shared-expert
     isolation, and it is the specific thing the old spatial MoEs could not say.
  2. EXPERTS ARE COMPUTED DENSELY. With 4-16 tiny experts on a 4096-row batch the
     gather/scatter is pure overhead, so all experts run and the routing weight
     vector is sparse instead. Mathematically identical to sparse dispatch; this
     is a quality experiment, not a FLOPs one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- helpers
def _mlp(dims, dropout, out_dim):
    layers, d = [], dims[0]
    for h in dims[1:]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def _zero_last(seq):
    """Zero the output projection so the branch contributes nothing at init."""
    last = [m for m in seq if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class _LocalSwitch(nn.Module):
    """Base for every mechanism here: a global trunk plus a local correction that
    can be switched OFF at inference without changing a single weight.

    This is the measurement the round is really about. Comparing a locality arm
    to the baseline compares two different training runs, so a tie could always
    be two models that landed in different places and scored the same. Scoring
    ONE set of trained weights with its local branch on and then off is paired at
    the weight level: whatever the difference is, it is exactly what locality
    contributed, and nothing else moved.
    """

    local_off: bool = False

    def local_disabled(self):
        class _Ctx:
            def __init__(s, m): s.m = m
            def __enter__(s): s.m.local_off = True
            def __exit__(s, *a): s.m.local_off = False
        return _Ctx(self)


class _GateInput(nn.Module):
    """Which columns the router is allowed to see — the whole experiment.

    content : the row's own features (what an LLM MoE routes on)
    terrain : elevation / tri / tch only. Physical properties that mean the same
              thing in Vestland as in Rogaland, so a partition built on them is
              portable in a way a coordinate partition is not.
    geo     : lon/lat, appended as gate-only columns by the harness. The CONTROL
              — this is what stage7 did, and it is expected to collapse.
    """

    def __init__(self, kind, feat_dim, gate_slice=None, n_gate_extra=0):
        super().__init__()
        self.kind, self.feat_dim, self.n_gate_extra = kind, feat_dim, n_gate_extra
        if kind == "content":
            self.register_buffer("cols", torch.arange(feat_dim), persistent=False)
        elif kind == "terrain":
            if not gate_slice:
                raise ValueError("terrain gate needs gate_slice (lidar columns)")
            self.register_buffer("cols", torch.tensor(sorted(gate_slice)), persistent=False)
        elif kind == "geo":
            if n_gate_extra <= 0:
                raise ValueError("geo gate needs extra_gate='geo' on the Trial")
            self.register_buffer(
                "cols", torch.arange(feat_dim, feat_dim + n_gate_extra), persistent=False)
        else:
            raise ValueError(f"unknown gate_src {kind}")
        self.dim = int(self.cols.numel())

    def forward(self, x):
        return x[:, self.cols]


# ------------------------------------------------------------ shared-expert MoE
class SharedExpertMoE(_LocalSwitch):
    """DeepSeekMoE-style shared expert + fine-grained routed experts.

    logits = shared(x) + sum_{e in top-k} g_e * expert_e(x)

    The shared expert is the deployed 256,128 MLP and is ALWAYS on: whatever is
    common to all of Norway is learned once, in full capacity, instead of being
    re-learned inside every expert. The routed experts are deliberately small
    (fine-grained segmentation) so what they hold is a local correction, not a
    competing global model. This is the structural difference from stage6/stage7,
    where routing replaced the global model rather than adding to it.

    balance:
      none     plain top-k softmax routing.
      lossfree DeepSeek-V3's auxiliary-loss-free balancing: a per-expert bias is
               added to the SELECTION scores only (never to the combining
               weights, so the gradient signal is untouched) and nudged each step
               toward equal load. Balancing without an aux loss that competes
               with the task loss.
      aux      the Switch/GShard load-balance loss plus ST-MoE's router z-loss,
               written to self.aux_loss for the trial's loss_fn to add.
      echoice  expert-choice routing: each expert takes its top-C rows in the
               batch, so load is balanced BY CONSTRUCTION. Caveat, stated up
               front: this makes a row's routing depend on the rest of its batch,
               which cannot hold at inference, so eval falls back to per-row
               top-k with the same un-renormalised gate probabilities. That
               train/test routing mismatch is the known cost of expert choice,
               and it is why this is a separate arm rather than the default.
    """

    def __init__(self, in_dim, n_classes, hidden=(256, 128), dropout=0.3,
                 n_experts=4, top_k=2, expert_hidden=(64, 32), gate_src="content",
                 gate_slice=None, n_gate_extra=0, zero_init=True, balance="none",
                 bias_gamma=1e-3, aux_lam=0.01, z_lam=1e-3):
        super().__init__()
        self.feat_dim = in_dim - n_gate_extra
        self.n_experts, self.top_k = n_experts, min(top_k, n_experts)
        self.balance, self.bias_gamma = balance, bias_gamma
        self.aux_lam, self.z_lam = aux_lam, z_lam
        self.aux_loss = torch.zeros(())

        self.shared = _mlp((self.feat_dim, *hidden), dropout, n_classes)
        self.experts = nn.ModuleList([
            _mlp((self.feat_dim, *expert_hidden), dropout, n_classes)
            for _ in range(n_experts)])
        if zero_init:
            for e in self.experts:
                _zero_last(e)

        self.gate_in = _GateInput(gate_src, self.feat_dim, gate_slice, n_gate_extra)
        self.router = nn.Linear(self.gate_in.dim, n_experts)
        nn.init.normal_(self.router.weight, std=0.01)
        nn.init.zeros_(self.router.bias)

        # selection bias for loss-free balancing; NOT a parameter, updated by rule
        self.register_buffer("bias", torch.zeros(n_experts))
        # eval-time routing census — how much of a HELD-OUT region one expert eats
        self.register_buffer("route_counts", torch.zeros(n_experts))
        self.register_buffer("route_rows", torch.zeros(()))

    def _weights(self, scores):
        """(B,E) router scores -> (B,E) sparse combining weights."""
        probs = F.softmax(scores, dim=1)
        if self.balance == "echoice" and self.training:
            # each expert claims its top-C rows; weights are the gate probs,
            # left un-renormalised exactly as in the paper (and at eval below)
            cap = max(1, math.ceil(scores.shape[0] * self.top_k / self.n_experts))
            cap = min(cap, scores.shape[0])
            w = torch.zeros_like(probs)
            idx = probs.topk(cap, dim=0).indices                    # (C,E)
            w.scatter_(0, idx, probs.gather(0, idx))
            return w, probs
        sel = scores + self.bias if self.balance == "lossfree" else scores
        top = sel.topk(self.top_k, dim=1).indices
        if self.balance == "echoice":
            w = torch.zeros_like(probs).scatter_(1, top, probs.gather(1, top))
            return w, probs                                         # eval fallback
        # renormalise over the selected experts so the correction has unit gain
        gsel = probs.gather(1, top)
        gsel = gsel / gsel.sum(1, keepdim=True).clamp_min(1e-9)
        return torch.zeros_like(probs).scatter_(1, top, gsel), probs

    def _balance_updates(self, w, probs, scores):
        used = (w > 0).float()
        if self.training and self.balance == "lossfree":
            with torch.no_grad():
                load = used.mean(0)
                target = self.top_k / self.n_experts
                self.bias += self.bias_gamma * torch.sign(target - load)
        if self.balance == "aux":
            # Switch: E * sum_e (fraction of rows routed to e)(mean prob of e)
            aux = self.n_experts * (used.mean(0) * probs.mean(0)).sum()
            z = (torch.logsumexp(scores, dim=1) ** 2).mean()
            self.aux_loss = self.aux_lam * aux + self.z_lam * z
        if not self.training:
            with torch.no_grad():
                self.route_counts += torch.bincount(
                    w.argmax(1), minlength=self.n_experts).float()
                self.route_rows += w.shape[0]

    def forward(self, x):
        feat = x[:, :self.feat_dim]
        if self.local_off:
            return self.shared(feat)
        scores = self.router(self.gate_in(x))
        w, probs = self._weights(scores)
        self._balance_updates(w, probs, scores)
        out = self.shared(feat)
        for e, expert in enumerate(self.experts):
            we = w[:, e:e + 1]
            if not self.training and float(we.abs().sum()) == 0.0:
                continue                       # nothing routed here in this batch
            out = out + we * expert(feat)
        return out

    def routing_stats(self):
        n = float(self.route_rows)
        if n <= 0:
            return {}
        p = (self.route_counts / n).cpu().numpy()
        ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))
        return {"route_share": [round(float(v), 4) for v in p],
                "route_max_share": round(float(p.max()), 4),
                "route_entropy_norm": round(ent, 4)}


# ------------------------------------------------------ mixture of LoRA experts
class MoLoRALinear(nn.Module):
    """W x + b + sum_k g_k * (alpha/r) B_k A_k x, with B zero-initialised.

    The frozen-shape global weight is the whole model everyone shares; each
    expert owns only a rank-r correction to it. This is the cheapest possible
    formalisation of "one global model plus local adjustments", and the one the
    LLM world actually deploys for per-domain adaptation.
    """

    def __init__(self, in_dim, out_dim, n_experts=4, rank=8, alpha=16.0):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.A = nn.Parameter(torch.empty(n_experts, rank, in_dim))
        self.B = nn.Parameter(torch.zeros(n_experts, out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x, g):
        # (B,K,r) <- (B,in) x (K,r,in); then (B,K,out) contracted with gate g
        h = torch.einsum("bi,kri->bkr", x, self.A)
        d = torch.einsum("bkr,kor->bko", h, self.B)
        return self.base(x) + self.scale * torch.einsum("bko,bk->bo", d, g)


class MoLoRAMLP(_LocalSwitch):
    """The deployed trunk with every Linear carrying K routed low-rank experts.

    One router, read once from the gate columns, drives all layers — the local
    identity of a row is a property of the row, not of the depth."""

    def __init__(self, in_dim, n_classes, hidden=(256, 128), dropout=0.3,
                 n_experts=4, rank=8, alpha=16.0, top_k=0, gate_src="content",
                 gate_slice=None, n_gate_extra=0):
        super().__init__()
        self.feat_dim = in_dim - n_gate_extra
        self.n_experts, self.top_k = n_experts, top_k
        self.gate_in = _GateInput(gate_src, self.feat_dim, gate_slice, n_gate_extra)
        self.router = nn.Linear(self.gate_in.dim, n_experts)
        nn.init.normal_(self.router.weight, std=0.01)
        nn.init.zeros_(self.router.bias)
        dims = [self.feat_dim, *hidden]
        self.layers = nn.ModuleList([
            MoLoRALinear(dims[i], dims[i + 1], n_experts, rank, alpha)
            for i in range(len(dims) - 1)])
        self.head = MoLoRALinear(dims[-1], n_classes, n_experts, rank, alpha)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("route_counts", torch.zeros(n_experts))
        self.register_buffer("route_rows", torch.zeros(()))

    def forward(self, x):
        feat = x[:, :self.feat_dim]
        if self.local_off:
            g = torch.zeros(feat.shape[0], self.n_experts, device=feat.device)
            h = feat
            for lin in self.layers:
                h = self.drop(F.relu(lin(h, g)))
            return self.head(h, g)
        g = F.softmax(self.router(self.gate_in(x)), dim=1)
        if self.top_k and self.top_k < self.n_experts:
            top = g.topk(self.top_k, dim=1).indices
            gs = g.gather(1, top)
            g = torch.zeros_like(g).scatter_(1, top, gs / gs.sum(1, keepdim=True))
        if not self.training:
            with torch.no_grad():
                self.route_counts += torch.bincount(
                    g.argmax(1), minlength=self.n_experts).float()
                self.route_rows += g.shape[0]
        h = feat
        for lin in self.layers:
            h = self.drop(F.relu(lin(h, g)))
        return self.head(h, g)

    routing_stats = SharedExpertMoE.routing_stats


# ---------------------------------------------------------- mixture of depths
class MoDNet(_LocalSwitch):
    """Adaptive depth: only the hardest rows pay for the extra block.

    Raposo et al. 2024 route a capacity-limited fraction of tokens through a
    block and let the rest take the residual path. The land-cover analogue is
    concrete: a lake pixel is settled by band 3, a crop/grassland pixel is not,
    and spending the same depth on both is what a fixed net has to do. The extra
    block is zero-initialised, so depth is only ever added where it earns its
    place.

    Top-k within a batch is not available at inference, so the paper's auxiliary
    predictor is trained alongside (BCE against top-k membership) and takes over
    at eval. `router_agreement` reports how well it reproduces the training rule.
    """

    def __init__(self, in_dim, n_classes, hidden=(256, 128), dropout=0.3,
                 capacity=0.5, n_gate_extra=0):
        super().__init__()
        self.feat_dim = in_dim - n_gate_extra
        self.capacity = capacity
        h1, h2 = hidden
        self.l1 = nn.Linear(self.feat_dim, h1)
        self.l2 = nn.Linear(h1, h2)
        self.block = nn.Sequential(nn.Linear(h2, h2), nn.ReLU(), nn.Linear(h2, h2))
        _zero_last(self.block)
        self.head = nn.Linear(h2, n_classes)
        self.drop = nn.Dropout(dropout)
        self.router = nn.Linear(h2, 1)          # scores the row for extra depth
        self.aux_pred = nn.Linear(h2, 1)        # inference stand-in for top-k
        self.aux_loss = torch.zeros(())
        self.register_buffer("route_rows", torch.zeros(()))
        self.register_buffer("route_deep", torch.zeros(()))

    def forward(self, x):
        feat = x[:, :self.feat_dim]
        h = self.drop(F.relu(self.l1(feat)))
        h = self.drop(F.relu(self.l2(h)))
        if self.local_off:
            return self.head(h)
        s = self.router(h).squeeze(1)
        if self.training:
            k = max(1, int(h.shape[0] * self.capacity))
            idx = s.topk(k).indices
            sel = torch.zeros_like(s).scatter_(0, idx, 1.0)
            # the auxiliary predictor learns the SAME decision from the row alone
            self.aux_loss = F.binary_cross_entropy_with_logits(
                self.aux_pred(h.detach()).squeeze(1), sel)
        else:
            sel = (torch.sigmoid(self.aux_pred(h).squeeze(1)) > 0.5).float()
            with torch.no_grad():
                self.route_rows += h.shape[0]
                self.route_deep += sel.sum()
        gate = torch.sigmoid(s).unsqueeze(1) * sel.unsqueeze(1)
        h = h + gate * self.block(h)
        return self.head(h)

    def routing_stats(self):
        n = float(self.route_rows)
        return {} if n <= 0 else {"deep_frac": round(float(self.route_deep) / n, 4)}


# ------------------------------------------------- context-conditioned FiLM net
class FiLMHyperNet(_LocalSwitch):
    """A hypernetwork emits per-row (gamma, beta) that modulate every hidden layer.

    The continuous limit of the MoE idea: instead of choosing among N local
    models, generate one on the fly from the row's context. There is no
    partition to get wrong and no expert to starve, which is the failure mode of
    every hard-routed variant tried here. The hypernet's output layer is zero-
    initialised so it starts at gamma=1, beta=0 — the deployed model exactly.
    """

    def __init__(self, in_dim, n_classes, hidden=(256, 128), dropout=0.3,
                 ctx_src="terrain", gate_slice=None, n_gate_extra=0, hyper_hidden=64):
        super().__init__()
        self.feat_dim = in_dim - n_gate_extra
        self.hidden = hidden
        self.ctx_in = _GateInput(ctx_src, self.feat_dim, gate_slice, n_gate_extra)
        self.hyper = nn.Sequential(
            nn.Linear(self.ctx_in.dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, 2 * sum(hidden)))
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)
        dims = [self.feat_dim, *hidden]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                   for i in range(len(dims) - 1)])
        self.head = nn.Linear(dims[-1], n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        feat = x[:, :self.feat_dim]
        mod = (torch.zeros(feat.shape[0], 2 * sum(self.hidden), device=feat.device)
               if self.local_off else self.hyper(self.ctx_in(x)))
        h, off = feat, 0
        for lin, size in zip(self.lins, self.hidden):
            gamma = 1.0 + mod[:, off:off + size]
            beta = mod[:, off + size:off + 2 * size]
            off += 2 * size
            h = self.drop(F.relu(gamma * lin(h) + beta))
        return self.head(h)


import numpy as np  # noqa: E402  (only used by routing_stats)


# ------------------------------------------------------------------- pre-flight
def _selftest(verbose=True):
    """Assertions that would have caught every bug this harness has ever hit.

    The claims worth checking BEFORE spending GPU hours are structural, not
    numerical: does the model really start at the baseline, does the router
    really get gradient, does balancing really balance, and does the geo gate
    really see only lon/lat.
    """
    torch.manual_seed(0)
    B, D, C, EX = 512, 67, 10, 4
    x = torch.randn(B, D)
    xg = torch.cat([x, torch.randn(B, 2)], 1)   # with gate-only geo columns
    y = torch.randint(0, C, (B,))
    ok = []

    def chk(name, cond, detail=""):
        ok.append(bool(cond))
        if verbose:
            print(f"  [{'ok ' if cond else 'FAIL'}] {name} {detail}")

    # 1. zero-init MoE reproduces its own shared expert exactly
    m = SharedExpertMoE(D, C, n_experts=EX, zero_init=True).eval()
    chk("moe zero-init == shared expert",
        torch.allclose(m(x), m.shared(x), atol=1e-6),
        f"max|d|={float((m(x)-m.shared(x)).abs().max()):.2e}")

    # 2. router gradient is exactly zero at init (all experts agree) and becomes
    #    non-zero once one step has let them differentiate. If this ever failed
    #    silently the "starts at baseline" argument would be worthless.
    m.train()
    F.cross_entropy(m(x), y).backward()
    g0 = float(m.router.weight.grad.abs().max())
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    opt.zero_grad()
    F.cross_entropy(m(x), y).backward()
    g1 = float(m.router.weight.grad.abs().max())
    chk("router grad 0 at init, non-zero after steps", g0 < 1e-9 < g1,
        f"g0={g0:.2e} g1={g1:.2e}")

    # 3. loss-free balancing actually equalises load on a deliberately skewed
    #    router (the stage7 failure mode, reproduced in miniature)
    mb = SharedExpertMoE(D, C, n_experts=EX, balance="lossfree", bias_gamma=0.05)
    with torch.no_grad():
        mb.router.bias[0] = 5.0                 # force everything to expert 0
    mb.train()
    for _ in range(200):
        mb(x)
    mb.eval()
    mb.route_counts.zero_(); mb.route_rows.zero_()
    mb(x)
    share = mb.routing_stats()["route_max_share"]
    chk("loss-free balancing breaks a collapsed router", share < 0.9,
        f"max expert share {share:.2f} (was 1.00)")

    # 4. expert-choice load is balanced by construction
    me = SharedExpertMoE(D, C, n_experts=EX, balance="echoice").train()
    with torch.no_grad():
        me.router.bias[0] = 5.0
        w, _ = me._weights(me.router(me.gate_in(x)))
    load = (w > 0).float().mean(0)
    chk("expert-choice equalises load by construction",
        float(load.max() - load.min()) < 1e-6, f"loads={[round(float(v),3) for v in load]}")

    # 5. the geo gate must see lon/lat and NOTHING else — otherwise the control
    #    arm quietly becomes a content gate and the whole contrast is void
    mg = SharedExpertMoE(D + 2, C, n_experts=EX, gate_src="geo", n_gate_extra=2).eval()
    s_a = mg.router(mg.gate_in(xg))
    xg2 = xg.clone(); xg2[:, :D] = torch.randn(B, D)      # scramble features only
    chk("geo gate ignores feature columns",
        torch.allclose(s_a, mg.router(mg.gate_in(xg2)), atol=1e-6))
    xg3 = xg.clone(); xg3[:, D:] = torch.randn(B, 2)      # scramble lon/lat only
    chk("geo gate responds to lon/lat",
        not torch.allclose(s_a, mg.router(mg.gate_in(xg3)), atol=1e-4))

    # 6. terrain gate reads exactly the columns it was given
    mt = SharedExpertMoE(D, C, n_experts=EX, gate_src="terrain",
                         gate_slice=[64, 65, 66]).eval()
    chk("terrain gate dim == 3", mt.gate_in.dim == 3)

    # 7. MoLoRA starts as its own base linear stack
    ml = MoLoRAMLP(D, C, n_experts=EX, rank=8).eval()
    ref = ml.head.base(ml.drop(F.relu(ml.layers[1].base(
        ml.drop(F.relu(ml.layers[0].base(x)))))))
    chk("MoLoRA zero-init B == plain MLP", torch.allclose(ml(x), ref, atol=1e-6),
        f"max|d|={float((ml(x)-ref).abs().max()):.2e}")

    # 8. MoD's extra block contributes nothing at init
    md = MoDNet(D, C).eval()
    z0 = md(x)
    with torch.no_grad():
        for p in md.block.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    chk("MoD block is a no-op at init (and not after)",
        not torch.allclose(z0, md(x), atol=1e-6))

    # 9. FiLM starts at gamma=1, beta=0
    mf = FiLMHyperNet(D, C, ctx_src="terrain", gate_slice=[64, 65, 66]).eval()
    ref = mf.head(mf.drop(F.relu(mf.lins[1](mf.drop(F.relu(mf.lins[0](x)))))))
    chk("FiLM zero-init == plain MLP", torch.allclose(mf(x), ref, atol=1e-6),
        f"max|d|={float((mf(x)-ref).abs().max()):.2e}")

    # 10. the locality switch must be a real off switch: with an UNTRAINED but
    #     non-zero correction, on and off have to differ, and after training the
    #     switch must still reach the same trunk. A silently ineffective switch
    #     would report "locality contributed nothing" for every arm in the round.
    for name, mk in (("moe", lambda: SharedExpertMoE(D, C, n_experts=EX, zero_init=False)),
                     ("lora", lambda: MoLoRAMLP(D, C, n_experts=EX, rank=8)),
                     ("mod", lambda: MoDNet(D, C)),
                     ("film", lambda: FiLMHyperNet(D, C, ctx_src="terrain",
                                                   gate_slice=[64, 65, 66]))):
        mm = mk().eval()
        with torch.no_grad():                       # give the branch something to say
            for p in (list(mm.experts.parameters()) if hasattr(mm, "experts") else
                      [mm.B] if hasattr(mm, "B") else
                      list(getattr(mm, "block", getattr(mm, "hyper", mm)).parameters())):
                p.add_(torch.randn_like(p) * 0.3)
            if hasattr(mm, "layers"):
                for lin in list(mm.layers) + [mm.head]:
                    lin.B.add_(torch.randn_like(lin.B) * 0.3)
        on = mm(x)
        with mm.local_disabled():
            off = mm(x)
        chk(f"local switch is a real off switch ({name})",
            not torch.allclose(on, off, atol=1e-5) and not mm.local_off,
            f"mean|on-off|={float((on - off).abs().mean()):.3e}")

    if not all(ok):
        raise SystemExit("moe_layers self-test FAILED — fix before spending GPU time")
    if verbose:
        print(f"moe_layers self-test: {len(ok)}/{len(ok)} passed")
    return True


if __name__ == "__main__":
    _selftest()

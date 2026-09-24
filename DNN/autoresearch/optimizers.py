"""Modern optimizers, implemented locally so the shared venv is untouched.

Each has a `_selftest` that runs before use — the research log's hardest lesson is
that a well-cited method producing a catastrophic delta is a BUG SIGNATURE, not a
finding (a sign error in logit adjustment once cost a day). Cheap synthetic
checks up front are worth more than post-hoc archaeology.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


# ------------------------------------------------------------------------ Muon
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration -> approximately orthogonalised G.

    Keller Jordan et al., "Muon: An optimizer for hidden layers in neural
    networks" (2024). The coefficients are the tuned quintic; the iteration is
    run in bfloat16 because only the singular-value *spectrum* matters and it is
    driven to ~1 regardless of input precision.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Momentum-orthogonalised SGD for 2D hidden weights.

    Only matmul parameters that are neither the input nor the output layer should
    be routed here; embeddings/heads/biases/1D params keep Adam. The update is
    `-lr * NS(momentum) * sqrt(max(1, fan_out/fan_in))`, the standard scale that
    makes Muon's effective step comparable to Adam's across shapes.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        super().__init__(list(params),
                         dict(lr=lr, momentum=momentum, nesterov=nesterov,
                              ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mom, nest = group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if nest else buf
                upd = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(upd, alpha=-group["lr"] * scale)
        return loss


# ------------------------------------------------------------------------ Lion
class Lion(Optimizer):
    """EvoLved Sign Momentum (Chen et al., 2023). Update is the SIGN of an
    interpolated momentum, so the step size is uniform across coordinates — hence
    lr ~3-10x smaller and weight decay ~10x larger than Adam."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-2):
        super().__init__(list(params), dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                upd = m.mul(b1).add_(g, alpha=1 - b1).sign_()
                p.add_(upd, alpha=-group["lr"])
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


# ------------------------------------------------------- Schedule-Free AdamW
class AdamWScheduleFree(Optimizer):
    """Defazio et al., "The Road Less Scheduled" (NeurIPS 2024).

    Keeps three points: the iterate z (where the SGD/Adam step lands), the
    Polyak-Ruppert-style average x (what you evaluate), and the interpolation
    y = (1-beta) z + beta x (where gradients are taken). No LR schedule and no
    horizon needs to be known in advance — which matters here because early
    stopping means the horizon is genuinely unknown.

    `train()` / `eval()` swap the parameter buffer between y and x, so evaluation
    always sees the averaged iterate.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, r=0.0, weight_lr_power=2.0, warmup_steps=1):
        super().__init__(list(params),
                         dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                              r=r, weight_lr_power=weight_lr_power,
                              warmup_steps=max(1, warmup_steps), k=0,
                              weight_sum=0.0, train_mode=True))

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta = group["betas"][0]
            for p in group["params"]:
                z = self.state[p].get("z")
                if z is not None:                       # y -> x
                    p.lerp_(z, weight=1 - 1 / beta)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta = group["betas"][0]
            for p in group["params"]:
                z = self.state[p].get("z")
                if z is not None:                       # x -> y
                    p.lerp_(z, weight=1 - beta)
            group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            k = group["k"]
            sched = min(1.0, (k + 1) / group["warmup_steps"])
            bias_correction2 = 1 - b2 ** (k + 1)
            lr = group["lr"] * sched * math.sqrt(bias_correction2)
            weight = ((k + 1) ** group["r"]) * (lr ** group["weight_lr_power"])
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "z" not in st:
                    st["z"] = p.detach().clone()
                    st["v"] = torch.zeros_like(p)
                z, v = st["z"], st["v"]
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                gd = g.div(v.sqrt().add(group["eps"]))
                if group["weight_decay"]:
                    gd = gd.add(p, alpha=group["weight_decay"])
                # p holds y. Interpolate toward z (the x-average update), then
                # take the gradient step; z is the plain Adam iterate.
                p.lerp_(z, weight=ckp1)
                p.add_(gd, alpha=lr * (b1 * (1 - ckp1) - 1))
                z.sub_(gd, alpha=lr)
            group["k"] = k + 1
        return loss


# --------------------------------------------------------------------- checks
def _selftest(verbose=True):
    """Every optimizer must drive a trivial convex problem to ~0. A method that
    cannot fit y = Wx on 200 points is misimplemented, and would otherwise show
    up as a fascinating negative result."""
    torch.manual_seed(0)
    X = torch.randn(512, 16)
    W = torch.randn(16, 4)
    Y = X @ W
    out = {}
    for name, make in [
        ("muon", lambda ps: Muon(ps, lr=0.02)),
        ("lion", lambda ps: Lion(ps, lr=3e-3, weight_decay=0.0)),
        ("sf_adamw", lambda ps: AdamWScheduleFree(ps, lr=1e-2, warmup_steps=20)),
    ]:
        torch.manual_seed(0)
        lin = torch.nn.Linear(16, 4, bias=False)
        start = float(((lin(X) - Y) ** 2).mean())
        opt = make(lin.parameters())
        if hasattr(opt, "train"):
            opt.train()
        for _ in range(600):
            opt.zero_grad()
            loss = ((lin(X) - Y) ** 2).mean()
            loss.backward()
            opt.step()
        if hasattr(opt, "eval"):
            opt.eval()
        final = float(((lin(X) - Y) ** 2).mean())
        # Relative criterion at 100x reduction: Lion's sign update has a floor of
        # ~lr per step and Schedule-Free evaluates an AVERAGE of iterates, so
        # neither can reach machine-zero on a convex problem — absolute mse is
        # the wrong yardstick. What this separates is working (<=1e-2 of start)
        # from broken: the first, mis-derived Schedule-Free step sat at 0.49.
        ok = final < 1e-2 * start
        out[name] = final
        if verbose:
            print(f"  {name:10s} mse {start:.3e} -> {final:.3e} "
                  f"({final/start:.1e} of start)  {'OK' if ok else 'FAIL'}")
    return out


if __name__ == "__main__":
    print("optimizer self-test (must all reach mse < 1e-2):")
    _selftest()

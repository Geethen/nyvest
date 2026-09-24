"""Architecture pieces: modern activations, numerical feature embeddings,
BatchEnsemble/TabM, Chebyshev-KAN, and the hierarchical coarse->fine head.

Everything here keeps the deployed 256,128 trunk shape wherever possible, because
the scaling study already proved capacity past that point converts into
geographic memorisation rather than test accuracy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ activations
class Snake(nn.Module):
    """x + sin^2(ax)/a  (Ziyin et al., NeurIPS 2020). Periodic inductive bias —
    plausible for spectral embeddings where class boundaries are not monotone in
    any single dimension."""

    def __init__(self, dim, a=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((dim,), float(a)))

    def forward(self, x):
        a = self.a.clamp(min=1e-3)
        return x + torch.sin(a * x) ** 2 / a


class XIELU(nn.Module):
    """Trainable expanded IELU (Nvidia, 2024): quadratic-positive / ELU-negative
    with learned slopes. A strictly larger family than ReLU/GELU, so it can only
    be beaten by them through optimisation or overfitting."""

    def __init__(self, alpha_p=0.8, alpha_n=0.8, beta=0.5, eps=-1e-6):
        super().__init__()
        self.ap = nn.Parameter(torch.tensor(math.log(math.expm1(alpha_p))))
        self.an = nn.Parameter(torch.tensor(math.log(math.expm1(alpha_n - beta))))
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x):
        ap = F.softplus(self.ap)
        an = F.softplus(self.an) + self.beta
        pos = ap * x * x + self.beta * x
        neg = an * torch.expm1(torch.minimum(x, self.eps.expand_as(x))) - an * x + self.beta * x
        return torch.where(x > 0, pos, neg)


ACTIVATIONS = {
    "relu": lambda d: nn.ReLU(),
    "gelu": lambda d: nn.GELU(),
    "silu": lambda d: nn.SiLU(),
    "mish": lambda d: nn.Mish(),
    "prelu": lambda d: nn.PReLU(num_parameters=d),
    "snake": lambda d: Snake(d),
    "xielu": lambda d: XIELU(),
}


def mlp_with_act(in_dim, n_classes, hidden, dropout, act="relu"):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), ACTIVATIONS[act](h), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, n_classes)]
    return nn.Sequential(*layers)


# -------------------------------------------------- numerical feature embeddings
class PeriodicEmbeddings(nn.Module):
    """Per-feature periodic ("PLR") embeddings — Gorishniy et al., "On Embeddings
    for Numerical Features in Tabular Deep Learning" (NeurIPS 2022).

    Each scalar x_i -> [sin(2*pi*c_i*x_i), cos(2*pi*c_i*x_i)] with LEARNED
    frequencies c_i ~ N(0, sigma), then a per-feature linear + ReLU. The point is
    that an MLP's first layer can only cut a feature with hyperplanes; periodic
    features let it resolve non-monotone thresholds (e.g. an elevation band that
    is mire low down and scrub higher up) without extra depth.
    """

    def __init__(self, n_features, n_freq=12, d_emb=8, sigma=0.5, cols=None):
        super().__init__()
        self.cols = cols                     # None -> embed every feature
        self.n_sel = n_features if cols is None else len(cols)
        self.coef = nn.Parameter(torch.randn(self.n_sel, n_freq) * sigma)
        self.lin = nn.Parameter(torch.empty(self.n_sel, 2 * n_freq, d_emb))
        self.bias = nn.Parameter(torch.zeros(self.n_sel, d_emb))
        bound = 1 / math.sqrt(2 * n_freq)
        nn.init.uniform_(self.lin, -bound, bound)
        self.out_dim = self.n_sel * d_emb + (0 if cols is None
                                             else n_features - len(cols))

    def forward(self, x):
        sel = x if self.cols is None else x[:, self.cols]
        z = 2 * math.pi * sel.unsqueeze(-1) * self.coef            # [B, S, F]
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)        # [B, S, 2F]
        z = torch.einsum("bsf,sfd->bsd", z, self.lin) + self.bias  # [B, S, D]
        z = F.relu(z).flatten(1)
        if self.cols is None:
            return z
        keep = [i for i in range(x.shape[1]) if i not in set(self.cols)]
        return torch.cat([z, x[:, keep]], dim=1)


def numemb_mlp(in_dim, n_classes, hidden, dropout, cols=None, n_freq=12,
               d_emb=8, sigma=0.5):
    emb = PeriodicEmbeddings(in_dim, n_freq, d_emb, sigma, cols)
    trunk, d = [], emb.out_dim
    for h in hidden:
        trunk += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    trunk += [nn.Linear(d, n_classes)]
    return nn.Sequential(emb, *trunk)


# ------------------------------------------------------------------------ TabM
class BatchEnsembleLinear(nn.Module):
    """Shared weight matrix with per-member rank-1 input/output adapters
    (Wen et al., 2020). k "members" for ~1x the parameters of one."""

    def __init__(self, in_dim, out_dim, k, first=False):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # TabM-mini: only the FIRST layer gets randomly-signed adapters; deeper
        # ones start at 1 so members diverge from the input side, which is where
        # the ensemble's useful diversity comes from on tabular data.
        r_init = (torch.randint(0, 2, (k, in_dim)).float() * 2 - 1) if first \
            else torch.ones(k, in_dim)
        self.r = nn.Parameter(r_init)
        self.s = nn.Parameter(torch.ones(k, out_dim))
        self.bias = nn.Parameter(torch.zeros(k, out_dim))

    def forward(self, x):                    # x: [B, k, in]
        return ((x * self.r) @ self.weight) * self.s + self.bias


class TabM(nn.Module):
    """TabM (Gorishniy et al., ICLR 2025) — an implicit deep ensemble that shares
    almost all weights. Relevant here because 5-seed ensembling is the ONLY lever
    that ever moved this problem, yet 5->15 independent seeds bought nothing:
    TabM's members are trained JOINTLY, so their diversity is regularisation
    rather than independent-init variance."""

    returns_log_probs = True   # forward()[0] is log P, not logits

    def __init__(self, in_dim, n_classes, hidden, dropout, k=8):
        super().__init__()
        self.k = k
        dims, layers = [in_dim] + list(hidden), []
        for i in range(len(hidden)):
            layers.append(BatchEnsembleLinear(dims[i], dims[i + 1], k, first=(i == 0)))
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.head = BatchEnsembleLinear(hidden[-1], n_classes, k)

    def forward(self, x):
        z = x.unsqueeze(1).expand(-1, self.k, -1)
        for lay in self.layers:
            z = self.drop(F.relu(lay(z)))
        member_logits = self.head(z)                          # [B, k, C]
        # element 0 = the ensemble's log-mean-probability (what gets predicted);
        # element 1 = per-member logits (what the loss supervises).
        log_mean = torch.logsumexp(F.log_softmax(member_logits, dim=-1), dim=1) \
            - math.log(self.k)
        return log_mean, member_logits


# ------------------------------------------------------------------------- KAN
class ChebyKANLayer(nn.Module):
    """Kolmogorov-Arnold layer with Chebyshev polynomial edges (Liu et al., 2024,
    efficient variant). Learnable univariate functions on the EDGES instead of
    fixed activations on the nodes — a different function basis entirely, which
    is the kind of change the flat 'everything ties' result has not yet had."""

    def __init__(self, in_dim, out_dim, degree=5):
        super().__init__()
        self.degree = degree
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, degree + 1)
                                 / (in_dim * (degree + 1)) ** 0.5)

    def forward(self, x):
        x = torch.tanh(x)                                  # Chebyshev needs [-1,1]
        t = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            t.append(2 * x * t[-1] - t[-2])
        T = torch.stack(t, dim=-1)                          # [B, in, deg+1]
        return torch.einsum("bid,iod->bo", T, self.coef)


class KAN(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(64,), degree=5, dropout=0.3):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.layers = nn.ModuleList(
            [ChebyKANLayer(dims[i], dims[i + 1], degree) for i in range(len(hidden))])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden[-1], n_classes)

    def forward(self, x):
        for lay, nrm in zip(self.layers, self.norms):
            x = self.drop(nrm(lay(x)))
        return self.head(x)


# ------------------------------------------------------- hierarchical head
class HierHead(nn.Module):
    """Coarse->fine factorised classifier.

    P(fine c | x) = P(coarse g(c) | x) * P(c | g(c), x)

    The ontology experiment showed that collapsing {crop,grassland} and
    {scrub,sparse-veg} lifts macro-F1 by 0.044 — i.e. almost all remaining error
    lives INSIDE two superclasses. A flat softmax has to spend one shared set of
    logits on both the easy between-group problem and the hard within-group one.
    Factorising gives the within-group split its own parameters and its own
    gradient, trained only on the rows where the distinction exists.
    """

    returns_log_probs = True

    def __init__(self, in_dim, n_classes, hidden, dropout, groups):
        super().__init__()
        trunk, d = [], in_dim
        for h in hidden:
            trunk += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*trunk)
        self.groups = groups                       # list[list[int]] of class idx
        self.n_groups = len(groups)
        self.coarse = nn.Linear(d, self.n_groups)
        self.fine = nn.ModuleList(
            [nn.Linear(d, len(g)) if len(g) > 1 else nn.Identity() for g in groups])
        gmap = torch.zeros(n_classes, dtype=torch.long)
        for gi, g in enumerate(groups):
            for c in g:
                gmap[c] = gi
        self.register_buffer("gmap", gmap)
        self.n_classes = n_classes

    def forward(self, x):
        z = self.trunk(x)
        log_coarse = F.log_softmax(self.coarse(z), dim=1)
        out = torch.empty(x.shape[0], self.n_classes, device=x.device,
                          dtype=log_coarse.dtype)
        for gi, g in enumerate(self.groups):
            lc = log_coarse[:, gi:gi + 1]
            if len(g) == 1:
                out[:, g[0]] = lc[:, 0]
            else:
                lf = F.log_softmax(self.fine[gi](z), dim=1)
                out[:, g] = lc + lf
        return out                                  # LOG-probabilities

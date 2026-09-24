"""Batch 1 — LLM-research-derived architectures, adapted to 67 tabular features.

Each idea is included only if it has a MECHANISM for the two known bottlenecks
(spatial generalization; confusion-bound vegetation classes) rather than being a
transformer part transplanted for its own sake. Prior work already ruled out:
depth/residual (memorizes folds), feature self-attention (AlphaEarth bands are
already a transformer embedding), routed MoE x8 configs.

EXP env var picks the variant:

  swiglu     - SwiGLU FFN (Shazeer 2020; LLaMA/PaLM). Gated activation is the one
               architectural change that survived contact with every modern LLM.
               Tests whether the gate helps where plain GLU (0.7288) didn't, at
               matched param count.
  rmsnorm    - Pre-RMSNorm (Zhang & Sennrich; LLaMA). BatchNorm memorized folds
               (stage2); RMSNorm has NO batch statistics, so it can't leak the
               train fold's distribution into the model the way BN did. This is
               the specific reason to retry normalization.
  softmoe    - Soft-MoE (Puigcerver et al. 2023). Unlike stage6/7 hard/gated MoE,
               experts process WEIGHTED SLOT MIXTURES of the batch, so every
               expert sees every row -> cannot collapse onto one region, which is
               exactly how stage7's spatial MoE failed.
  deepnarrow - 4x64 deep-narrow at ~matched params to 256,128. Tests the scaling
               claim (depth>width per param) under spatial CV.
  gaussmix   - No LLM lineage: 5-member ensemble but each member gets a different
               dropout rate (0.15..0.45). Diversity-of-regularization ensembling;
               rides the one lever that has ever worked here (variance reduction).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as ec  # noqa: E402

EXP = os.environ.get("EXP", "swiglu")


class SwiGLUBlock(nn.Module):
    """SwiGLU FFN: (Swish(xW) * xV) W2 — LLaMA/PaLM's feed-forward.
    Hidden is scaled by 2/3 to keep params ~equal to a plain Linear+ReLU."""

    def __init__(self, d_in, d_hidden, dropout):
        super().__init__()
        h = int(d_hidden * 2 / 3)
        self.w = nn.Linear(d_in, h)
        self.v = nn.Linear(d_in, h)
        self.out_dim = h
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.silu(self.w(x)) * self.v(x))


class SwiGLUNet(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, dropout):
        super().__init__()
        blocks, d = [], in_dim
        for h in hidden:
            b = SwiGLUBlock(d, h, dropout)
            blocks.append(b)
            d = b.out_dim
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.head(self.blocks(x))


class RMSNormNet(nn.Module):
    """Pre-RMSNorm MLP. No batch statistics -> no train-fold distribution leak."""

    def __init__(self, in_dim, n_classes, hidden, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.RMSNorm(d), nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.RMSNorm(d), nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SoftMoE(nn.Module):
    """Soft-MoE (Puigcerver et al. 2023) over a batch of tokens-as-rows.

    Each expert owns `slots` slots. A slot is a softmax-weighted average over the
    WHOLE batch, so every expert always receives gradient from every region —
    structurally impossible to collapse the way stage7's hard router did. Outputs
    are scattered back with a second softmax (combine weights).
    """

    def __init__(self, in_dim, n_classes, n_experts, slots, hidden, dropout):
        super().__init__()
        self.n_experts, self.slots = n_experts, slots
        self.phi = nn.Parameter(torch.randn(in_dim, n_experts, slots) * 0.02)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden[0], hidden[1]), nn.ReLU())
            for _ in range(n_experts)])
        self.head = nn.Linear(hidden[1], n_classes)

    def forward(self, x):
        # x: [B, D]; logits over (expert, slot)
        lg = torch.einsum("bd,des->bes", x, self.phi)          # [B,E,S]
        d_w = F.softmax(lg.flatten(1), dim=0).view_as(lg)      # dispatch: over BATCH
        c_w = F.softmax(lg.flatten(1), dim=-1).view_as(lg)     # combine: over slots
        slots_in = torch.einsum("bd,bes->esd", x, d_w)         # [E,S,D]
        ys = torch.stack([self.experts[e](slots_in[e]) for e in range(self.n_experts)])
        y = torch.einsum("esh,bes->bh", ys, c_w)               # [B,H]
        return self.head(y)


def build_deepnarrow(in_dim, n_classes):
    # 4 x 96 ~= 256,128 params (~44k vs ~50k), depth-vs-width at matched budget
    layers, d = [], in_dim
    for _ in range(4):
        layers += [nn.Linear(d, 96), nn.ReLU(), nn.Dropout(ec.DROPOUT)]
        d = 96
    layers += [nn.Linear(d, n_classes)]
    return nn.Sequential(*layers)


def main():
    if EXP == "swiglu":
        ec.run_experiment(
            "swiglu", lambda i, c: SwiGLUNet(i, c, ec.HIDDEN, ec.DROPOUT),
            notes="SwiGLU FFN (Shazeer 2020, LLaMA/PaLM) at matched params",
            extra={"idea": "swiglu"})
    elif EXP == "rmsnorm":
        ec.run_experiment(
            "rmsnorm", lambda i, c: RMSNormNet(i, c, ec.HIDDEN, ec.DROPOUT),
            notes="Pre-RMSNorm MLP; batch-stat-free norm (BN memorized folds in stage2)",
            extra={"idea": "rmsnorm"})
    elif EXP == "softmoe":
        n_e = int(os.environ.get("N_EXPERTS", "4"))
        slots = int(os.environ.get("SLOTS", "8"))
        ec.run_experiment(
            f"softmoe_e{n_e}s{slots}",
            lambda i, c: SoftMoE(i, c, n_e, slots, ec.HIDDEN, ec.DROPOUT),
            notes="Soft-MoE (Puigcerver 2023): slot mixtures, every expert sees "
                  "every row -> cannot region-collapse like stage7",
            extra={"idea": "soft_moe", "n_experts": n_e, "slots": slots})
    elif EXP == "deepnarrow":
        ec.run_experiment(
            "deepnarrow", build_deepnarrow,
            notes="4x96 deep-narrow at matched params vs 256,128",
            extra={"idea": "depth_vs_width"})
    elif EXP == "gaussmix":
        rates = [0.15, 0.225, 0.3, 0.375, 0.45]
        import stage3_robust_mlp as s3

        class VarDrop(nn.Module):
            """Dropout rate varies by member via a mutable module-level counter."""
            def __init__(self, in_dim, n_classes):
                super().__init__()
                r = rates[VarDrop.i % len(rates)]
                VarDrop.i += 1
                self.net = s3.MLP(in_dim, n_classes, ec.HIDDEN, r)

            def forward(self, x):
                return self.net(x)
        VarDrop.i = 0
        ec.run_experiment(
            "hetero_dropout_ens", lambda i, c: VarDrop(i, c),
            notes=f"5-member ensemble w/ heterogeneous dropout {rates}; "
                  "diversity-of-regularization variance reduction",
            extra={"idea": "hetero_ensemble", "rates": rates})
    else:
        raise SystemExit(f"unknown EXP={EXP}")


if __name__ == "__main__":
    main()

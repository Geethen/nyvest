"""Round 3 of the autoresearch program: LOCALITY.

The question this round exists to settle: "local models are better than global
models for mapping local conditions." Nine MoE variants have already lost on
this dataset, so the round is only worth running if it tests something those
nine did not. It does, and the distinction is sharp:

    every previous variant routed on POSITION.  These route on CONTENT.

Under GroupKFold-on-cell_id a test fold is a region with no training labels at
all, so a coordinate gate has to extrapolate its partition into territory it has
never seen — and stage7 measured the consequence directly, 84-95% of a held-out
fold landing on a single expert. Routing on the row's own representation, or on
physical terrain that means the same thing everywhere, defines locality in a
space that a new region can actually land in. `moe_geo` keeps the position gate
as a control so that claim is tested rather than asserted.

Every mechanism here starts AT the deployed model (zero-initialised corrections,
see moe_layers) so a loss means local specialisation actively hurt, not that the
optimiser failed to find its way back to the baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ar_common as ac
import moe_layers as M
from ar_common import Trial


# --------------------------------------------------------------- build helpers
def _cfg(ctx, **defaults):
    """Trial.config overrides, so the loop can sweep an axis without edits here."""
    cfg = dict(defaults)
    cfg.update({k: v for k, v in ctx["trial"].config.items() if k in defaults})
    return cfg


def _moe_builder(**fixed):
    def build(in_dim, n_classes, ctx):
        c = _cfg(ctx, n_experts=4, top_k=2, expert_h1=64, expert_h2=32,
                 zero_init=1, bias_gamma=1e-3, aux_lam=0.01, z_lam=1e-3, **fixed)
        return M.SharedExpertMoE(
            in_dim, n_classes, hidden=ac.HIDDEN, dropout=ac.DROPOUT,
            n_experts=int(c["n_experts"]), top_k=int(c["top_k"]),
            expert_hidden=(int(c["expert_h1"]), int(c["expert_h2"])),
            gate_src=c["gate_src"], gate_slice=ctx["lidar_cols"],
            n_gate_extra=ctx.get("n_gate_extra", 0),
            zero_init=bool(int(c["zero_init"])), balance=c["balance"],
            bias_gamma=float(c["bias_gamma"]), aux_lam=float(c["aux_lam"]),
            z_lam=float(c["z_lam"]))
    return build


def build_molora(in_dim, n_classes, ctx):
    c = _cfg(ctx, n_experts=4, rank=8, alpha=16.0, top_k=0, gate_src="content")
    return M.MoLoRAMLP(
        in_dim, n_classes, hidden=ac.HIDDEN, dropout=ac.DROPOUT,
        n_experts=int(c["n_experts"]), rank=int(c["rank"]), alpha=float(c["alpha"]),
        top_k=int(c["top_k"]), gate_src=c["gate_src"], gate_slice=ctx["lidar_cols"],
        n_gate_extra=ctx.get("n_gate_extra", 0))


def build_capacity_ctrl(in_dim, n_classes, ctx):
    """A plain MLP widened to the MoE's parameter count, and nothing else.

    Without this the round cannot tell its two possible readings apart. The MoE
    lifts inner-val by about +0.014 while leaving the held-out region untouched,
    which reads as "locality is learnable but does not transfer" — except that
    the scaling study found ANY extra width lifts val and not test on this model,
    and the MoE carries 27k extra parameters. If a plain net with the same
    parameter count lifts val by the same amount, the routing contributed
    nothing even in-distribution and the honest statement is much stronger.
    """
    c = _cfg(ctx, h1=320, h2=170)
    return nn.Sequential(
        nn.Linear(in_dim, int(c["h1"])), nn.ReLU(), nn.Dropout(ac.DROPOUT),
        nn.Linear(int(c["h1"]), int(c["h2"])), nn.ReLU(), nn.Dropout(ac.DROPOUT),
        nn.Linear(int(c["h2"]), n_classes))


def build_mod(in_dim, n_classes, ctx):
    c = _cfg(ctx, capacity=0.5)
    return M.MoDNet(in_dim, n_classes, hidden=ac.HIDDEN, dropout=ac.DROPOUT,
                    capacity=float(c["capacity"]),
                    n_gate_extra=ctx.get("n_gate_extra", 0))


def build_film(in_dim, n_classes, ctx):
    c = _cfg(ctx, hyper_hidden=64, ctx_src="terrain")
    return M.FiLMHyperNet(in_dim, n_classes, hidden=ac.HIDDEN, dropout=ac.DROPOUT,
                          ctx_src=c["ctx_src"], gate_slice=ctx["lidar_cols"],
                          n_gate_extra=ctx.get("n_gate_extra", 0),
                          hyper_hidden=int(c["hyper_hidden"]))


def loss_with_aux(model, xb, yb, crit, ctx):
    """CE plus whatever auxiliary term the module accumulated this forward pass
    (router load-balance + z-loss for MoE, the depth predictor's BCE for MoD)."""
    return crit(model(xb), yb) + model.aux_loss


# ------------------------------------------------------------- routing census
def post_route_stats(P_te, P_val, y_val, ctx):
    """Report WHERE the router sent the held-out region, next to where it sent
    the training rows.

    This is the measurement that condemned the old spatial MoE, so it is worth
    taking on every arm rather than inferring collapse from a bad F1. Counters
    are zeroed and re-run on a clean pass because the training loop's early-
    stopping evaluations would otherwise mix inner-val rows into the census.
    """
    m = ctx.get("last_model")
    if m is None:
        return P_te, {}
    out = {}

    # What did locality actually contribute? Same weights, correction switched
    # off. A trial can tie the baseline either because the local branch learned
    # something that cancels out, or because it learned to stay silent — those
    # are different findings and this is what separates them. One ensemble
    # member, so it is a mechanism probe, not a headline score.
    #
    # Read it as RELIANCE, not as evidence for locality. The trunk was trained
    # with the correction present, so switching it off breaks a co-adapted pair
    # and the "off" number is NOT the baseline model. A large gap means the
    # network leans on its local branch; whether that leaning was worth anything
    # is still the paired delta against the baseline and nothing else.
    if hasattr(m, "local_disabled"):
        on = ac.predict_probs(m, ctx["Xte_t"], ctx["n_classes"]).argmax(1)
        with m.local_disabled():
            off_P = ac.predict_probs(m, ctx["Xte_t"], ctx["n_classes"])
        off = off_P.argmax(1)
        out["local_switched_rows"] = round(float((on != off).mean()), 4)
        out["f1_local_on_1seed"] = round(float(ac.du.macro_f1(
            ctx["y_te"], on, ctx["n_classes"])), 4)
        out["f1_local_off_1seed"] = round(float(ac.du.macro_f1(
            ctx["y_te"], off, ctx["n_classes"])), 4)
        print(f"    locality: changes {out['local_switched_rows']:.1%} of rows, "
              f"F1 {out['f1_local_off_1seed']:.4f} -> {out['f1_local_on_1seed']:.4f} "
              "(1 member, switch off -> on)")

    if not hasattr(m, "routing_stats"):
        return P_te, out

    def census(X):
        for buf in ("route_counts", "route_rows", "route_deep"):
            if hasattr(m, buf):
                getattr(m, buf).zero_()
        ac.predict_probs(m, X, ctx["n_classes"])
        return m.routing_stats()

    out.update({f"te_{k}": v for k, v in census(ctx["Xte_t"]).items()})
    out.update({f"tr_{k}": v for k, v in census(ctx["Xtr_t"]).items()})
    if "te_route_max_share" in out:
        print(f"    routing: held-out region max expert share "
              f"{out['te_route_max_share']:.2f} (train rows "
              f"{out.get('tr_route_max_share', float('nan')):.2f})")
    return P_te, out


# ------------------------------------------------ kNN-LM retrieval interpolation
def _inner(trial, name="_base"):
    """A Trial carrying the outer trial's architecture hooks into a fold
    override. Without this a `combo:moe_shared+knn_retrieval` would quietly
    train a plain MLP and report it as the combination."""
    return Trial(name=name, tier="", idea="", hypothesis="",
                 n_ensemble=trial.n_ensemble, weight_mode=trial.weight_mode,
                 label_smooth=trial.label_smooth, build_fn=trial.build_fn,
                 loss_fn=trial.loss_fn, opt_fn=trial.opt_fn,
                 after_fit=trial.after_fit)


def _encoder(model):
    """The trained net minus its output layer, for the retrieval datastore."""
    if isinstance(model, nn.Sequential):
        return nn.Sequential(*list(model.children())[:-1]).eval()
    if hasattr(model, "penultimate"):
        return model.penultimate
    raise SystemExit(
        f"knn_retrieval has no defined representation for {type(model).__name__}; "
        "give the module a `penultimate` method before combining it with retrieval")


def _knn_topk(R_q, R_db, k, q_chunk=2048, db_chunk=65536):
    """Chunked exact top-k by L2, merging partial results so neither the query
    nor the datastore has to fit in one distance matrix.

    Chunks are sized so one distance block is ~0.5 GB: the datastore is 440k
    rows and the queries 220k, and the loop runs three other trials on the same
    GPU, so the full matrix is four orders of magnitude too big to hold."""
    n_q = R_q.shape[0]
    D = torch.empty(n_q, k, device=R_q.device)
    I = torch.empty(n_q, k, dtype=torch.long, device=R_q.device)
    for i in range(0, n_q, q_chunk):
        q = R_q[i:i + q_chunk]
        bd = bi = None
        for j in range(0, R_db.shape[0], db_chunk):
            d = torch.cdist(q, R_db[j:j + db_chunk])
            kk = min(k, d.shape[1])
            dv, di = d.topk(kk, dim=1, largest=False)
            di = di + j
            if bd is None:
                bd, bi = dv, di
            else:
                cd = torch.cat([bd, dv], 1)
                ci = torch.cat([bi, di], 1)
                dv2, sel = cd.topk(k, dim=1, largest=False)
                bd, bi = dv2, ci.gather(1, sel)
        D[i:i + q_chunk], I[i:i + q_chunk] = bd, bi
    return D, I


def _knn_probs(D, I, y_db, n_classes, T):
    w = torch.softmax(-D / T, dim=1)
    lab = y_db[I]
    P = torch.zeros(D.shape[0], n_classes, device=D.device)
    return P.scatter_add_(1, lab, w)


def fold_knn(ctx):
    """kNN-LM: interpolate the parametric posterior with a retrieval
    distribution over TRAINING rows, in the network's own representation space.

    p(c) = lambda * p_model(c) + (1 - lambda) * p_kNN(c),
    p_kNN(c) ∝ sum_{i in top-k} exp(-d_i / T) * 1[y_i = c]

    This is the purest form of a local model that is still deployable: no
    partition, no experts, no coordinates — just "which training rows does this
    pixel actually resemble, and what were they". It is a genuinely different
    lever from the feat_proto control that tied earlier (class centroids only
    reformat what the model already computed) because the datastore carries
    individual rows, including the ones the parametric model averages away.

    lambda and T are fitted on the inner val, which for this trial is carved by
    whole cell_ids. That matters more here than anywhere else: with a random
    inner val, a val row's nearest neighbours include its own cell and retrieval
    looks spectacular, then evaporates on a held-out region. The grid contains
    lambda=1.0, so "retrieval is worthless" is inside the hypothesis space and
    the fit can decline to use it.
    """
    trial = ctx["trial"]
    c = _cfg(ctx, k=32, t_mult=1.0)
    n_classes, n_ens = ctx["n_classes"], trial.n_ensemble
    base = _inner(trial)

    P_te = np.zeros((ctx["Xte_t"].shape[0], n_classes), dtype=np.float64)
    P_val = np.zeros((ctx["Xval_t"].shape[0], n_classes), dtype=np.float64)
    reps, vf1s = {}, []
    for e in range(n_ens):
        model, vf1 = ac.train_member(base, ctx, ac.SEED + 100 * e)
        P_te += ac.predict_probs(model, ctx["Xte_t"], n_classes)
        P_val += ac.predict_probs(model, ctx["Xval_t"], n_classes)
        vf1s.append(vf1)
        if e == 0:
            enc = _encoder(model)
            with torch.no_grad():
                reps = {k: torch.cat([enc(X[i:i + 16384])
                                      for i in range(0, X.shape[0], 16384)])
                        for k, X in (("tr", ctx["Xtr_t"]), ("val", ctx["Xval_t"]),
                                     ("te", ctx["Xte_t"]))}
            del enc
        del model
        torch.cuda.empty_cache()
    P_te /= n_ens
    P_val /= n_ens

    k = int(c["k"])
    y_db = torch.tensor(ctx["ytr_np"], device=ac.DEVICE)
    Dv, Iv = _knn_topk(reps["val"], reps["tr"], k)
    Dt, It = _knn_topk(reps["te"], reps["tr"], k)
    scale = float(Dv[:, 0].median())
    Pv_t = torch.tensor(P_val, device=ac.DEVICE, dtype=torch.float32)
    y_val = ctx["yval_np"]

    best = (-1.0, 1.0, scale)
    for tm in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        T = max(scale * tm, 1e-6)
        Kv = _knn_probs(Dv, Iv, y_db, n_classes, T)
        for lam in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3):
            pred = (lam * Pv_t + (1 - lam) * Kv).argmax(1).cpu().numpy()
            f1 = ac.du.macro_f1(y_val, pred, n_classes)
            if f1 > best[0]:
                best = (f1, lam, T)
    _, lam, T = best
    Kt = _knn_probs(Dt, It, y_db, n_classes, T).cpu().numpy().astype(np.float64)
    knn_only = ac.du.macro_f1(ctx["y_te"], Kt.argmax(1), n_classes)
    P_mix = lam * P_te + (1 - lam) * Kt
    print(f"    kNN: lambda={lam:.2f} T={T:.3f} (val F1 {best[0]:.4f}); "
          f"retrieval alone on the test fold {knn_only:.4f}")
    return P_mix, float(np.mean(vf1s)), {
        "knn_lambda": lam, "knn_T": round(T, 4), "knn_val_f1": round(best[0], 4),
        "knn_only_test_f1": round(float(knn_only), 4),
        "f1_model_only": round(float(ac.du.macro_f1(
            ctx["y_te"], P_te.argmax(1), n_classes)), 4)}


# -------------------------------------------------- regional fine-tune + merge
def fold_soup(ctx):
    """Train the global model, fine-tune a copy per geographic region, then
    average the copies in WEIGHT space (Wortsman et al., Model Soups; the same
    operation the LLM world now calls model merging).

    This is the most literal reading of "local models are better": each expert
    genuinely is a local model, fitted on its own region. The merge is what makes
    it deployable — the souped weights are a single global network, so there is
    no router to extrapolate into an unseen region and none of the routing
    collapse that killed the earlier spatial MoE. If regional specialisation
    carries transferable information, averaging the specialists should retain
    some of it. If "local" only ever meant "fitted to that region's quirks", the
    merge washes out to something no better than the global model it started
    from, and possibly worse.

    This arm buys ft_epochs of optimisation that the baseline does not get, so
    `n_regions=1` is pre-registered as its matched control: identical extra
    fine-tuning, on all the rows at once, with nothing local about it. Any gain
    that survives THAT comparison is locality; any gain that does not was extra
    training.
    """
    from sklearn.cluster import KMeans
    trial = ctx["trial"]
    c = _cfg(ctx, n_regions=4, ft_epochs=20, ft_lr=3e-4)
    n_classes, n_ens = ctx["n_classes"], trial.n_ensemble
    base = _inner(trial)

    G = ctx["lonlat_tr"]
    Gz = (G - G.mean(0)) / (G.std(0) + 1e-9)
    lab = KMeans(int(c["n_regions"]), n_init=10, random_state=0).fit_predict(Gz)
    masks = [torch.tensor(np.flatnonzero(lab == r), device=ac.DEVICE)
             for r in range(int(c["n_regions"]))]
    print(f"    regions: {[int(m.numel()) for m in masks]} rows")

    crit = nn.CrossEntropyLoss(weight=ctx["w"], label_smoothing=trial.label_smooth)
    P_te = np.zeros((ctx["Xte_t"].shape[0], n_classes), dtype=np.float64)
    vf1s = []
    for e in range(n_ens):
        model, vf1 = ac.train_member(base, ctx, ac.SEED + 100 * e)
        vf1s.append(vf1)
        global_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        soup = {k: torch.zeros_like(v) for k, v in global_sd.items()}
        for r, idx in enumerate(masks):
            model.load_state_dict(global_sd)
            opt = torch.optim.Adam(model.parameters(), lr=float(c["ft_lr"]),
                                   weight_decay=ac.WEIGHT_DECAY)
            g = torch.Generator(device=ac.DEVICE).manual_seed(ac.SEED + 100 * e + r)
            model.train()
            for _ in range(int(c["ft_epochs"])):
                order = idx[torch.randperm(idx.numel(), device=ac.DEVICE, generator=g)]
                for i in range(0, order.numel(), ac.BATCH):
                    b = order[i:i + ac.BATCH]
                    opt.zero_grad()
                    crit(model(ctx["Xtr_t"][b]), ctx["ytr_t"][b]).backward()
                    opt.step()
            for k, v in model.state_dict().items():
                soup[k] += v.detach() / len(masks)
        model.load_state_dict(soup)
        P_te += ac.predict_probs(model, ctx["Xte_t"], n_classes)
        del model
        torch.cuda.empty_cache()
    return P_te / n_ens, float(np.mean(vf1s)), {"n_regions": int(c["n_regions"])}


def fold_region_oracle(ctx):
    """The ceiling on regional specialisation, and the number that decides
    whether any of this could ever have worked.

    Fine-tune one copy of the global model per KMeans region exactly as the soup
    does, then score the held-out fold three ways:

      best single region  the most useful specialist for this region, chosen
                          with hindsight — an upper bound on any router that
                          picks ONE expert for the whole fold.
      oracle per row      for each row, whichever model happens to be right — an
                          upper bound on ANY per-row routing scheme, however
                          clever, including ones nobody has invented.
      global              the same members with no fine-tuning at all.

    If the per-row oracle sits at or below the global model, routing is not
    under-engineered here, it is impossible, and every arm in this round was
    dead on arrival. If it sits far above, the null is about ROUTING rather than
    about locality, and the gap says how much a better router could be worth.
    This mirrors the ensemble-oracle bound that closed the MoE question the last
    time, and it uses test labels, so it is a bound and never a result.
    """
    from sklearn.cluster import KMeans
    trial = ctx["trial"]
    c = _cfg(ctx, n_regions=4, ft_epochs=20, ft_lr=3e-4)
    n_classes, n_ens = ctx["n_classes"], trial.n_ensemble
    base = _inner(trial)
    K = int(c["n_regions"])

    G = ctx["lonlat_tr"]
    Gz = (G - G.mean(0)) / (G.std(0) + 1e-9)
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(Gz)
    masks = [torch.tensor(np.flatnonzero(lab == r), device=ac.DEVICE)
             for r in range(K)]

    crit = nn.CrossEntropyLoss(weight=ctx["w"], label_smoothing=trial.label_smooth)
    P_glob = np.zeros((ctx["Xte_t"].shape[0], n_classes))
    P_reg = np.zeros((K, ctx["Xte_t"].shape[0], n_classes))
    vf1s = []
    for e in range(n_ens):
        model, vf1 = ac.train_member(base, ctx, ac.SEED + 100 * e)
        vf1s.append(vf1)
        gsd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        P_glob += ac.predict_probs(model, ctx["Xte_t"], n_classes) / n_ens
        for r, idx in enumerate(masks):
            model.load_state_dict(gsd)
            opt = torch.optim.Adam(model.parameters(), lr=float(c["ft_lr"]),
                                   weight_decay=ac.WEIGHT_DECAY)
            g = torch.Generator(device=ac.DEVICE).manual_seed(ac.SEED + 100 * e + r)
            model.train()
            for _ in range(int(c["ft_epochs"])):
                order = idx[torch.randperm(idx.numel(), device=ac.DEVICE, generator=g)]
                for i in range(0, order.numel(), ac.BATCH):
                    b = order[i:i + ac.BATCH]
                    opt.zero_grad()
                    crit(model(ctx["Xtr_t"][b]), ctx["ytr_t"][b]).backward()
                    opt.step()
            P_reg[r] += ac.predict_probs(model, ctx["Xte_t"], n_classes) / n_ens
        del model
        torch.cuda.empty_cache()

    y = ctx["y_te"]
    f1_glob = ac.du.macro_f1(y, P_glob.argmax(1), n_classes)
    per_region = [float(ac.du.macro_f1(y, P_reg[r].argmax(1), n_classes))
                  for r in range(K)]
    best_r = int(np.argmax(per_region))

    # per-row oracle: keep each region model's prediction where it is correct
    preds = np.stack([P_reg[r].argmax(1) for r in range(K)] + [P_glob.argmax(1)])
    correct = preds == y[None, :]
    pick = np.where(correct.any(0), correct.argmax(0), len(preds) - 1)
    oracle_pred = preds[pick, np.arange(len(y))]
    f1_oracle = ac.du.macro_f1(y, oracle_pred, n_classes)

    P_out = np.zeros_like(P_glob)
    P_out[np.arange(len(y)), oracle_pred] = 1.0
    print(f"    regions {['%.4f' % v for v in per_region]} | best single "
          f"{per_region[best_r]:.4f} | global {f1_glob:.4f} | per-row oracle "
          f"{f1_oracle:.4f}")
    return P_out, float(np.mean(vf1s)), {
        "f1_global_same_members": round(float(f1_glob), 4),
        "f1_per_region": [round(v, 4) for v in per_region],
        "f1_best_single_region": round(float(per_region[best_r]), 4),
        "f1_row_oracle": round(float(f1_oracle), 4),
        "oracle_headroom": round(float(f1_oracle - f1_glob), 4)}


# ------------------------------------------------------------------- registry
QUEUE_LOCAL = [
    "moe_shared", "moe_geo", "moe_terrain", "moe_lora", "knn_retrieval",
    "moe_lossfree", "mod_depth", "film_hyper", "moe_echoice", "moe_auxbal",
    "moe_soup",
]


def register(_add, _Trial):
    """Called at the bottom of trials.py so these land in the same TRIALS
    registry that run.py, resolve() and the loop already understand."""

    _add(_Trial(
        name="moe_shared", tier="locality",
        idea="always-on shared expert (the deployed MLP) + 4 fine-grained routed "
             "experts, top-2, router reads the row's own features",
        hypothesis="the nine MoE variants that lost here all REPLACED the global "
                   "model with a partition of it; shared-expert isolation ADDS "
                   "local capacity to a global model that is still trained on "
                   "every row, and the routed experts start at zero so the "
                   "network begins as the exact deployed model. If content-space "
                   "locality carries anything, this is the arm that shows it",
        provenance="Dai et al., DeepSeekMoE, ACL 2024 (shared-expert isolation + "
                   "fine-grained expert segmentation)",
        build_fn=_moe_builder(gate_src="content", balance="none"),
        post_fn=post_route_stats,
        config={"n_experts": 4, "top_k": 2, "gate_src": "content"}))

    _add(_Trial(
        name="moe_geo", tier="locality",
        idea="identical MoE, but the router may see ONLY lon/lat",
        hypothesis="the control that makes this round interpretable. This is what "
                   "stage7 did, and the prediction is that it collapses: a test "
                   "fold is a region the router has never seen, so a coordinate "
                   "partition must extrapolate and dumps the fold on one expert. "
                   "A large gap between this and the content gate is the evidence "
                   "that locality has to be defined in a transferable space; no "
                   "gap means the gate source was never the problem",
        provenance="stage7_spatial_moe.py, reproduced as a paired control",
        build_fn=_moe_builder(gate_src="geo", balance="none"),
        post_fn=post_route_stats, extra_gate="geo",
        config={"n_experts": 4, "top_k": 2, "gate_src": "geo"}))

    _add(_Trial(
        name="moe_terrain", tier="locality",
        idea="identical MoE, router reads only elevation / tri / tch",
        hypothesis="the explicitly untried idea left over from stage7. Terrain is "
                   "PHYSICAL: 900 m of elevation means the same thing in Vestland "
                   "as in Rogaland, whereas a coordinate does not. It is also the "
                   "feature family that demonstrably transfers here (real minus "
                   "median lidar is worth +0.009). A partition on terrain is a "
                   "partition an unseen region can land inside",
        build_fn=_moe_builder(gate_src="terrain", balance="none"),
        post_fn=post_route_stats,
        config={"n_experts": 4, "top_k": 2, "gate_src": "terrain"}))

    _add(_Trial(
        name="moe_lossfree", tier="locality",
        idea="content-gated MoE with DeepSeek-V3 auxiliary-loss-free load "
             "balancing (per-expert selection bias nudged toward equal load)",
        hypothesis="if the content gate still concentrates a held-out region on "
                   "one expert, the fix is to balance the router — but a Switch "
                   "aux loss buys balance by fighting the task loss. The bias "
                   "trick balances selection while leaving every gradient the "
                   "task loss produces untouched, so it separates 'routing was "
                   "unbalanced' from 'routing was uninformative'",
        provenance="Wang et al., Auxiliary-Loss-Free Load Balancing / DeepSeek-V3",
        build_fn=_moe_builder(gate_src="content", balance="lossfree"),
        post_fn=post_route_stats,
        config={"n_experts": 4, "top_k": 2, "bias_gamma": 1e-3, "gate_src": "content"}))

    _add(_Trial(
        name="moe_auxbal", tier="locality",
        idea="content-gated MoE with the Switch load-balance loss + ST-MoE router "
             "z-loss",
        hypothesis="the orthodox balancing recipe, run as the matched comparison "
                   "for the loss-free arm. Same balance target, different price: "
                   "this one perturbs the task gradient, and z-loss additionally "
                   "keeps router logits from drifting to the saturated regime "
                   "where top-k stops responding to the input",
        provenance="Fedus et al., Switch Transformer, JMLR 2022; Zoph et al., "
                   "ST-MoE, 2022 (router z-loss)",
        build_fn=_moe_builder(gate_src="content", balance="aux"),
        loss_fn=loss_with_aux, post_fn=post_route_stats,
        config={"n_experts": 4, "top_k": 2, "aux_lam": 0.01, "z_lam": 1e-3,
                "gate_src": "content"}))

    _add(_Trial(
        name="moe_echoice", tier="locality",
        idea="expert-choice routing during training (each expert takes its top-C "
             "rows), per-row top-k at inference",
        hypothesis="balance by construction rather than by penalty — no expert "
                   "can starve, so every expert sees a full quarter of the data. "
                   "The honest cost is that a row's training-time routing depends "
                   "on its batch, which cannot hold at inference; this arm "
                   "measures whether perfect balance is worth that mismatch",
        provenance="Zhou et al., Mixture-of-Experts with Expert Choice Routing, "
                   "NeurIPS 2022",
        build_fn=_moe_builder(gate_src="content", balance="echoice"),
        post_fn=post_route_stats,
        config={"n_experts": 4, "top_k": 2, "gate_src": "content"}))

    _add(_Trial(
        name="moe_lora", tier="locality",
        idea="one global trunk; every Linear carries 4 routed rank-8 LoRA "
             "adapters combined per row by a content router",
        hypothesis="the sharpest form of 'global model plus local correction'. "
                   "The shared weights stay global and full-rank; locality is "
                   "confined to a rank-8 subspace per expert, so an expert cannot "
                   "wander off into its own model the way a full-capacity expert "
                   "can. B is zero-initialised, so this is exactly the deployed "
                   "network plus a correction that has to earn every unit it moves",
        provenance="Hu et al., LoRA, ICLR 2022; Dou et al., LoRAMoE, ACL 2024",
        build_fn=build_molora, post_fn=post_route_stats,
        config={"n_experts": 4, "rank": 8, "alpha": 16.0, "gate_src": "content"}))

    _add(_Trial(
        name="mod_depth", tier="locality",
        idea="mixture-of-depths: a learned router sends the hardest 50% of rows "
             "through an extra residual block, the rest bypass it",
        hypothesis="locality in DEPTH rather than in parameters. A water pixel is "
                   "settled by the first layer and a crop/grassland pixel is not, "
                   "yet a fixed net spends identical compute on both. Giving the "
                   "contested rows more depth without giving the settled ones any "
                   "is the one form of extra capacity the scaling study did not "
                   "test — it found width converts into geographic memorisation, "
                   "and this adds capacity only where the decision is actually close",
        provenance="Raposo et al., Mixture-of-Depths, 2024",
        build_fn=build_mod, loss_fn=loss_with_aux, post_fn=post_route_stats,
        config={"capacity": 0.5}))

    _add(_Trial(
        name="film_hyper", tier="locality",
        idea="a hypernetwork reads the terrain context and emits per-row FiLM "
             "(gamma, beta) modulating every hidden layer",
        hypothesis="the continuous limit of this whole round: rather than pick "
                   "among N local models, generate one per row. No partition to "
                   "extrapolate, no expert to starve, and the failure mode of "
                   "every hard-routed arm is structurally absent. Zero-init means "
                   "gamma=1, beta=0 at step 0, so it starts as the deployed model "
                   "and any movement is modulation the data asked for",
        provenance="Perez et al., FiLM, AAAI 2018; conditional adapters / "
                   "hypernetwork conditioning in modern LLM PEFT",
        build_fn=build_film, post_fn=post_route_stats,
        config={"hyper_hidden": 64, "ctx_src": "terrain"}))

    _add(_Trial(
        name="knn_retrieval", tier="locality",
        idea="interpolate the ensemble posterior with a k=32 retrieval "
             "distribution over training rows in the net's representation space; "
             "lambda and T fitted on a spatially held-out inner val",
        hypothesis="a non-parametric local model — locality in FEATURE space with "
                   "no partition at all. The parametric net compresses 664k rows "
                   "into 52k weights and necessarily averages away the rare local "
                   "configurations; the datastore keeps them. This is the one arm "
                   "that can add information the network never encoded, which is "
                   "why it is scored against the group-val control and why "
                   "lambda=1 is in the grid",
        provenance="Khandelwal et al., Nearest Neighbor Language Models, ICLR 2020",
        fold_fn=fold_knn, val_mode="group",
        config={"k": 32}))

    _add(_Trial(
        name="moe_soup", tier="locality",
        idea="fine-tune one copy of the global model per KMeans region, then "
             "average the copies in weight space",
        hypothesis="the literal claim under test — build genuinely local models, "
                   "then merge them so the result is still deployable to regions "
                   "none of them owns. Merging removes the router, so this is the "
                   "one way to buy regional specialisation without paying the "
                   "extrapolation cost that sank the position-gated MoE",
        provenance="Wortsman et al., Model Soups, ICML 2022 (model merging)",
        fold_fn=fold_soup,
        config={"n_regions": 4, "ft_epochs": 20, "ft_lr": 3e-4}))

    _add(_Trial(
        name="ctrl_capacity", tier="locality",
        idea="plain MLP widened to 320,170 — the MoE's parameter count, with no "
             "router, no experts and no locality of any kind",
        hypothesis="the control that decides how to read this entire round. The "
                   "locality arms lift inner val and leave the held-out region "
                   "flat, which sounds like 'learned but not transferable' — but "
                   "the scaling study showed plain width does exactly that too, "
                   "and these arms carry 27k extra parameters. Whatever this "
                   "control's val gain is, the locality arms only own the part "
                   "above it",
        provenance="matched-capacity control, cf. scaling_grid.py",
        build_fn=build_capacity_ctrl, config={"h1": 320, "h2": 170}))

    _add(_Trial(
        name="bound_region_oracle", tier="locality",
        idea="ceiling on regional specialisation: best single regional model and "
             "a per-row oracle over all of them, both chosen with test labels",
        hypothesis="a null is only worth something next to its ceiling. If a "
                   "PERFECT router over genuinely local models cannot beat the "
                   "global one, no routing scheme in this round or any future one "
                   "could have; if it beats it by a lot, the failure is the router "
                   "rather than locality. This is the same instrument that closed "
                   "the MoE question last time, when the oracle over ensemble "
                   "members showed +0.063 of headroom no router could reach",
        provenance="oracle bound, cf. diag_ensemble.py's member oracle",
        fold_fn=fold_region_oracle, is_bound=True,
        config={"n_regions": 4, "ft_epochs": 20, "ft_lr": 3e-4}))

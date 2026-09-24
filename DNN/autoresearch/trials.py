"""The experiment registry.

Each entry changes exactly ONE mechanism against the deployed recipe. Ordering in
QUEUE is by (expected value for the weak classes) x (1 / cost), because the loop
should surface a real win early if one exists.

Design bias, from what has already been ruled out on this problem:
  - capacity, data volume, dropout/weight-decay, spatial regularisation, MoE
    routing, feature attention and ensemble diversity are ALL exhausted;
  - the only levers that ever moved macro-F1 were (a) better FEATURES (lidar) and
    (b) variance reduction (5-seed ensembling), and both are spent;
  - the ontology probe says 0.044 macro-F1 sits inside two superclass boundaries
    ({crop, grassland} and {scrub, sparse-veg}).
So the weight here goes on mechanisms that change WHERE the decision boundary
falls for the weak classes (margins, priors, hierarchy, plug-in decision rules)
and on representations the flat MLP cannot express (periodic embeddings, KAN),
rather than on yet another way to fit the same boundary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ar_common as ac
import layers as L
from ar_common import Trial
from optimizers import AdamWScheduleFree, Lion, Muon

# Encoded index of each raw class code, for the 10-class merged set.
CLASSES = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
E = {c: i for i, c in enumerate(CLASSES)}

# The two boundaries the ontology probe priced at 0.044 macro-F1, plus the two
# next-largest symmetric confusions from the confusion matrix.
CONFUSION_PAIRS = [(E[3], E[5]), (E[6], E[11])]
SUPERGROUPS = [[E[3], E[5]], [E[6], E[11]], [E[2]], [E[4]], [E[7]],
               [E[8]], [E[10]], [E[12]]]


# ---------------------------------------------------------------- loss helpers
def soft_nll(log_p, y, w, ls, n_classes):
    """Class-weighted NLL with label smoothing, for heads that emit log-probs."""
    nll = -log_p.gather(1, y[:, None]).squeeze(1)
    if ls > 0:
        nll = (1 - ls) * nll - (ls / n_classes) * log_p.sum(1)
    if w is not None:
        wt = w[y]
        return (nll * wt).sum() / wt.sum()
    return nll.mean()


def _priors(ctx, dev):
    p = ctx["counts"] / ctx["counts"].sum()
    return torch.tensor(p, dtype=torch.float32, device=dev)


def loss_balanced_softmax(model, xb, yb, crit, ctx):
    """Ren et al. 2020. ADD log-prior to the logits DURING TRAINING and predict
    with plain logits — the model must beat the prior to fire, so rare classes
    end up with a lower implicit threshold at test time."""
    if "logprior" not in ctx:
        ctx["logprior"] = _priors(ctx, xb.device).log()
    return F.cross_entropy(model(xb) + ctx["logprior"], yb,
                           label_smoothing=ctx["trial"].label_smooth)


def loss_ldam_drw(model, xb, yb, crit, ctx):
    """LDAM + deferred re-weighting (Cao et al. NeurIPS 2019). Per-class margin
    ~ n_c^{-1/4} subtracted from the TRUE class logit, so rare classes must be
    predicted with extra confidence to count as correct — which widens their
    decision region at test time. Re-weighting is deferred to epoch 40 so the
    representation is learned first (the paper's central trick)."""
    if "ldam_m" not in ctx:
        n = np.maximum(ctx["counts"], 1.0)
        m = 1.0 / np.power(n, 0.25)
        m = m / m.max() * ctx["trial"].config["max_margin"]
        ctx["ldam_m"] = torch.tensor(m, dtype=torch.float32, device=xb.device)
        eff = (1 - 0.9999 ** n) / (1 - 0.9999)
        cb = (1 / eff)
        ctx["cb_w"] = torch.tensor(cb / cb.mean(), dtype=torch.float32, device=xb.device)
    z = model(xb)
    z = z - ctx["ldam_m"][None, :] * F.one_hot(yb, z.shape[1]).float()
    w = ctx["cb_w"] if ctx.get("epoch", 0) >= ctx["trial"].config["drw_epoch"] else None
    return F.cross_entropy(z, yb, weight=w,
                           label_smoothing=ctx["trial"].label_smooth)


def loss_vs(model, xb, yb, crit, ctx):
    """Vector-Scaling loss (Kini et al. NeurIPS 2021): logits are both SCALED
    (Delta_c, multiplicative, changes the margin) and SHIFTED (iota_c, additive,
    changes the prior). It strictly generalises LDAM and logit adjustment, both of
    which have already been tried here in isolation — the claim being tested is
    that the two corrections are complementary."""
    if "vs_d" not in ctx:
        n = np.maximum(ctx["counts"], 1.0)
        g, t = ctx["trial"].config["gamma"], ctx["trial"].config["tau"]
        d = np.power(n / n.max(), g)
        i = t * np.log(n / n.sum())
        ctx["vs_d"] = torch.tensor(d, dtype=torch.float32, device=xb.device)
        ctx["vs_i"] = torch.tensor(i, dtype=torch.float32, device=xb.device)
    z = model(xb) * ctx["vs_d"] + ctx["vs_i"]
    return F.cross_entropy(z, yb, label_smoothing=ctx["trial"].label_smooth)


def loss_softf1(model, xb, yb, crit, ctx):
    """CE + a differentiable macro-F1 surrogate. Macro-F1 is what we are scored
    on but CE optimises log-likelihood; the surrogate makes each class's F1
    contribute equally to the gradient regardless of how many rows it has, which
    is a different correction from re-weighting (it is recall/precision balanced,
    not count balanced)."""
    z = model(xb)
    p = F.softmax(z, 1)
    Y = F.one_hot(yb, z.shape[1]).float()
    tp = (p * Y).sum(0)
    fp = (p * (1 - Y)).sum(0)
    fn = ((1 - p) * Y).sum(0)
    present = (Y.sum(0) > 0).float()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    soft = (f1 * present).sum() / present.sum().clamp(min=1)
    lam = ctx["trial"].config["lam"]
    return crit(z, yb) + lam * (1 - soft)


def loss_pair_margin(model, xb, yb, crit, ctx):
    """CE + an explicit margin ONLY across the two confusable pairs.

    Global margin losses (LDAM/VS) move every boundary. This one touches nothing
    except {crop, grassland} and {scrub, sparse-veg}: for a row whose true class
    is in a pair, the true logit must beat its PARTNER's logit by m, otherwise a
    hinge penalty applies. Rows of other classes contribute zero extra gradient,
    so by construction the strong classes cannot be traded away."""
    z = model(xb)
    base = crit(z, yb)
    m, lam = ctx["trial"].config["margin"], ctx["trial"].config["lam"]
    pen = z.new_zeros(())
    for a, b in CONFUSION_PAIRS:
        for u, v in ((a, b), (b, a)):
            sel = yb == u
            if sel.any():
                gap = z[sel, u] - z[sel, v]
                pen = pen + F.relu(m - gap).mean()
    return base + lam * pen / (2 * len(CONFUSION_PAIRS))


def loss_focal(model, xb, yb, crit, ctx):
    """alpha-balanced focal loss (Lin et al. ICCV 2017): down-weight rows the
    model already gets right so the gradient concentrates on the hard ones. The
    weak classes here are hard for a REASON (overlapping definitions), so this is
    a genuine two-sided test — focusing on hard examples helps if they are
    learnable and hurts if they are label noise."""
    z = model(xb)
    logp = F.log_softmax(z, 1)
    lp = logp.gather(1, yb[:, None]).squeeze(1)
    focal = (1 - lp.exp()).pow(ctx["trial"].config["gamma"]) * (-lp)
    if ctx["w"] is not None:
        wt = ctx["w"][yb]
        return (focal * wt).sum() / wt.sum()
    return focal.mean()


def fold_smote(ctx):
    """SMOTE (Chawla et al., JAIR 2002) restricted to the weak classes.

    Synthesises rows on the segment between a weak-class point and one of its
    k nearest SAME-CLASS neighbours. Included because it is the reference
    imbalance baseline and because it is a sharp test of a prior result: spatial
    jitter (blending within a geographic cell) monotonically HURT here. SMOTE
    blends within the class manifold in FEATURE space instead, so if the earlier
    negative was about geography rather than interpolation, this should behave
    differently; if interpolation itself destroys the signal, it will not.
    """
    trial = ctx["trial"]
    k, ratio = trial.config["k"], trial.config["ratio"]
    Xtr, ytr = ctx["Xtr_t"], ctx["ytr_t"]
    weak_enc = [E[c] for c in ac.WEAK_CLASSES]
    new_X, new_y = [], []
    for c in weak_enc:
        idx = torch.nonzero(ytr == c, as_tuple=True)[0]
        if len(idx) < k + 1:
            continue
        Xc = Xtr[idx]
        n_new = int(len(idx) * ratio)
        # k nearest same-class neighbours, chunked so the distance matrix never
        # materialises in full
        nbr = torch.empty(len(idx), k, dtype=torch.long, device=Xc.device)
        for i in range(0, len(idx), 2048):
            d = torch.cdist(Xc[i:i + 2048], Xc)
            nbr[i:i + 2048] = d.topk(k + 1, largest=False).indices[:, 1:]
        pick = torch.randint(0, len(idx), (n_new,), device=Xc.device)
        which = torch.randint(0, k, (n_new,), device=Xc.device)
        partner = nbr[pick, which]
        lam = torch.rand(n_new, 1, device=Xc.device)
        new_X.append(Xc[pick] + lam * (Xc[partner] - Xc[pick]))
        new_y.append(torch.full((n_new,), c, dtype=ytr.dtype, device=Xc.device))
    if new_X:
        ctx["Xtr_t"] = torch.cat([Xtr] + new_X)
        ctx["ytr_t"] = torch.cat([ytr] + new_y)
        y_np = ctx["ytr_t"].cpu().numpy()
        ctx["counts"] = np.bincount(y_np, minlength=ctx["n_classes"]).astype(np.float64)
        ctx["w"] = ac.class_weights(y_np, ctx["n_classes"], trial.weight_mode)
        print(f"    SMOTE: +{sum(len(x) for x in new_X):,} synthetic weak-class rows")
    base = Trial(name="_base", tier="", idea="", hypothesis="",
                 n_ensemble=trial.n_ensemble, weight_mode=trial.weight_mode)
    ctx["trial"] = base
    return ac.run_fold(base, ctx)


def loss_hier(model, xb, yb, crit, ctx):
    return soft_nll(model(xb), yb, ctx["w"], ctx["trial"].label_smooth, ctx["n_classes"])


def loss_hier_finewt(model, xb, yb, crit, ctx):
    """HXE-style level weighting (Bertinetto et al. CVPR 2020): up-weight the
    WITHIN-group term so the hard fine split gets more of the gradient than the
    already-easy coarse split."""
    mdl = model
    z = mdl.trunk(xb)
    log_c = F.log_softmax(mdl.coarse(z), 1)
    gy = mdl.gmap[yb]
    loss_c = F.nll_loss(log_c, gy, weight=None)
    wf = ctx["trial"].config["fine_weight"]
    loss_f = z.new_zeros(())
    for gi, g in enumerate(mdl.groups):
        if len(g) == 1:
            continue
        sel = gy == gi
        if not sel.any():
            continue
        lf = F.log_softmax(mdl.fine[gi](z[sel]), 1)
        local = torch.tensor([g.index(int(c)) for c in yb[sel].cpu()],
                             device=xb.device)
        loss_f = loss_f + F.nll_loss(lf, local)
    return loss_c + wf * loss_f


def loss_tabm(model, xb, yb, crit, ctx):
    """Supervise every TabM member independently (the paper's recipe); the
    log-mean-prob head is only used at predict time."""
    _, member_logits = model(xb)
    B, k, C = member_logits.shape
    return crit(member_logits.reshape(B * k, C), yb.repeat_interleave(k))


# --------------------------------------------------------------- post-fit hooks
def after_tau_norm(model, ctx):
    """tau-normalised classifier (Kang et al. ICLR 2020): divide each class's
    weight vector by ||w_c||^tau. Long-tail nets grow larger weight norms for
    frequent classes; shrinking that back is a one-line rebalance that needs no
    retraining. tau is chosen on the SPATIALLY held-out inner val."""
    lin = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    W0 = lin.weight.data.clone()
    norms = W0.norm(dim=1, keepdim=True).clamp(min=1e-8)
    best, best_tau = -1.0, 0.0
    for tau in np.arange(0.0, 1.01, 0.1):
        lin.weight.data = W0 / norms.pow(float(tau))
        with torch.no_grad():
            vp = ac.logits_of(model, ctx["Xval_t"]).argmax(1).cpu().numpy()
        import data_utils as du
        f1 = du.macro_f1(ctx["yval_np"], vp, ctx["n_classes"])
        if f1 > best:
            best, best_tau = f1, float(tau)
    lin.weight.data = W0 / norms.pow(best_tau)
    ctx.setdefault("tau_chosen", []).append(best_tau)
    return model


def after_crt(model, ctx):
    """Classifier re-training (cRT, Kang et al. ICLR 2020): freeze the learned
    representation, re-initialise the last layer, and retrain ONLY it under
    class-balanced sampling. Decoupling matters because instance-balanced data is
    the better teacher for features while class-balanced data is the better
    teacher for the decision boundary — the deployed recipe has to compromise
    between the two with a single sqrt-weighted loss."""
    lin = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    for p in model.parameters():
        p.requires_grad_(False)
    lin.reset_parameters()
    for p in lin.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(lin.parameters(), lr=ctx["trial"].config["crt_lr"])
    ytr = ctx["ytr_t"]
    counts = torch.tensor(ctx["counts"], dtype=torch.float32, device=ac.DEVICE)
    prob = (1.0 / counts)[ytr]
    prob = prob / prob.sum()
    n_steps = ctx["trial"].config["crt_steps"]
    model.train()
    for _ in range(n_steps):
        idx = torch.multinomial(prob, ac.BATCH, replacement=True)
        opt.zero_grad()
        loss = F.cross_entropy(ac.logits_of(model, ctx["Xtr_t"][idx]), ytr[idx])
        loss.backward()
        opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# --------------------------------------------------------------- optimizer defs
def opt_muon(model, ctx):
    """Muon on the hidden matmul weights, Adam on everything else (the
    prescribed split). Muon's update is spectrally normalised, which is a
    different IMPLICIT BIAS from Adam's per-coordinate scaling — worth a shot
    even though SAM and EMA both tied, since those two changed WHICH minimum is
    found, not the geometry of the path."""
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden_w = [m.weight for m in lins[:-1]]
    hidden_ids = {id(p) for p in hidden_w}
    rest = [p for p in model.parameters() if id(p) not in hidden_ids]
    return _MultiOpt([Muon(hidden_w, lr=ctx["trial"].config["muon_lr"],
                           weight_decay=ac.WEIGHT_DECAY),
                      torch.optim.Adam(rest, lr=ac.LR, weight_decay=ac.WEIGHT_DECAY)])


class _MultiOpt:
    def __init__(self, opts):
        self.opts = opts

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def opt_lion(model, ctx):
    return Lion(model.parameters(), lr=ctx["trial"].config["lion_lr"],
                weight_decay=ctx["trial"].config["lion_wd"])


def opt_sf(model, ctx):
    o = AdamWScheduleFree(model.parameters(), lr=ac.LR,
                          weight_decay=ac.WEIGHT_DECAY, warmup_steps=200)
    o.train()
    return o


# ------------------------------------------------------------- fold overrides
def fold_specialist(ctx):
    """Base ensemble, then a dedicated binary discriminator per confusable pair,
    consulted ONLY on rows the base model already believes are that pair.

    This is the zero-tradeoff version of hierarchy: a row whose base top-2 is not
    a registered pair is returned untouched, so no strong class can lose ground
    by construction. The specialist sees only the two classes' rows, so its
    entire capacity goes to the boundary that the flat model has to fit while
    also separating eight other classes.
    """
    import data_utils as du
    trial = ctx["trial"]
    n_classes = ctx["n_classes"]
    base = Trial(name="_base", tier="", idea="", hypothesis="",
                 n_ensemble=trial.n_ensemble, weight_mode=trial.weight_mode)
    P_te, vf1, info = ac.run_fold(base, ctx)

    top2 = np.argsort(-P_te, axis=1)[:, :2]
    pairset = {frozenset(p): p for p in CONFUSION_PAIRS}
    n_switch, n_touch = 0, 0
    for key, (a, b) in pairset.items():
        sel_tr = np.isin(ctx["ytr_np"], [a, b])
        sel_val = np.isin(ctx["yval_np"], [a, b])
        if sel_tr.sum() < 100:
            continue
        ytr_bin = (ctx["ytr_np"][sel_tr] == b).astype(np.int64)
        sub = dict(ctx)
        sub["Xtr_t"] = ctx["Xtr_t"][torch.tensor(np.flatnonzero(sel_tr), device=ac.DEVICE)]
        sub["ytr_t"] = torch.tensor(ytr_bin, device=ac.DEVICE)
        sub["Xval_t"] = ctx["Xval_t"][torch.tensor(np.flatnonzero(sel_val), device=ac.DEVICE)]
        sub["yval_np"] = (ctx["yval_np"][sel_val] == b).astype(np.int64)
        sub["n_classes"] = 2
        sub["ytr_np"] = ytr_bin
        sub["counts"] = np.bincount(ytr_bin, minlength=2).astype(np.float64)
        sub["w"] = ac.class_weights(ytr_bin, 2, trial.weight_mode)
        sub["trial"] = base

        rows = np.flatnonzero([frozenset(t) == key for t in top2])
        n_touch += len(rows)
        if len(rows) == 0:
            continue
        Pb = np.zeros((len(rows), 2), dtype=np.float64)
        Xrows = ctx["Xte_t"][torch.tensor(rows, device=ac.DEVICE)]
        for e in range(trial.n_ensemble):
            m, _ = ac.train_member(base, sub, ac.SEED + 100 * e)
            Pb += ac.predict_probs(m, Xrows, 2)
            del m
            torch.cuda.empty_cache()
        Pb /= trial.n_ensemble
        # Redistribute ONLY the mass the base model already assigned to the pair.
        before = P_te[rows].argmax(1)
        mass = P_te[rows, a] + P_te[rows, b]
        P_te[rows, a] = mass * Pb[:, 0]
        P_te[rows, b] = mass * Pb[:, 1]
        n_switch += int((P_te[rows].argmax(1) != before).sum())
    info.update({"rows_touched": n_touch, "rows_switched": n_switch,
                 "frac_touched": round(n_touch / len(P_te), 4)})
    print(f"    specialists: touched {n_touch:,} rows, switched {n_switch:,}")
    return P_te, vf1, info


def fold_selfdistill(ctx):
    """Born-again self-distillation: the 5-seed ensemble teaches a fresh net of
    identical shape. The teacher's soft targets carry inter-class similarity
    structure ("this pixel is 0.6 grassland / 0.3 crop") that hard labels throw
    away, and that structure is exactly what the weak classes lack. Costs 2x."""
    trial = ctx["trial"]
    n_classes = ctx["n_classes"]
    teacher = Trial(name="_teacher", tier="", idea="", hypothesis="",
                    n_ensemble=trial.n_ensemble, weight_mode=trial.weight_mode)
    P_te_t, vf1_t, _ = ac.run_fold(teacher, ctx)

    # teacher soft targets on the training rows
    Q = np.zeros((ctx["Xtr_t"].shape[0], n_classes), dtype=np.float32)
    for e in range(trial.n_ensemble):
        m, _ = ac.train_member(teacher, ctx, ac.SEED + 100 * e)
        Q += ac.predict_probs(m, ctx["Xtr_t"], n_classes)
        del m
        torch.cuda.empty_cache()
    Q /= trial.n_ensemble
    ctx["teacher_q"] = torch.tensor(Q, device=ac.DEVICE)

    T, alpha = trial.config["T"], trial.config["alpha"]

    def distill_loss(model, xb, yb, crit, c):
        z = model(xb)
        hard = crit(z, yb)
        q = c["teacher_q"][c["batch_idx"]]
        soft = F.kl_div(F.log_softmax(z / T, 1), q, reduction="batchmean") * T * T
        return alpha * hard + (1 - alpha) * soft

    student = Trial(name="_student", tier="", idea="", hypothesis="",
                    n_ensemble=trial.n_ensemble, weight_mode=trial.weight_mode,
                    loss_fn=distill_loss)
    ctx["trial"] = student
    P_te = _run_fold_indexed(student, ctx)
    ctx["trial"] = trial
    return P_te, vf1_t, {"teacher_f1_note": "teacher scored separately"}


def _run_fold_indexed(trial, ctx):
    """Like run_fold but exposes each batch's row indices in ctx['batch_idx'] so
    a loss can look up per-row teacher targets."""
    n_classes = ctx["n_classes"]
    P_te = np.zeros((ctx["Xte_t"].shape[0], n_classes), dtype=np.float64)
    for e in range(trial.n_ensemble):
        ac.set_seed(ac.SEED + 100 * e)
        model = ac.default_mlp(ctx["in_dim"], n_classes, ctx).to(ac.DEVICE)
        opt = ac.default_opt(model, ctx)
        crit = nn.CrossEntropyLoss(weight=ctx["w"], label_smoothing=trial.label_smooth)
        n_tr = ctx["Xtr_t"].shape[0]
        best, best_state, bad = -1.0, None, 0
        g = torch.Generator(device=ac.DEVICE).manual_seed(ac.SEED + 100 * e)
        import data_utils as du
        for epoch in range(ac.MAX_EPOCHS):
            model.train()
            order = torch.randperm(n_tr, device=ac.DEVICE, generator=g)
            for i in range(0, n_tr, ac.BATCH):
                b = order[i:i + ac.BATCH]
                ctx["batch_idx"] = b
                opt.zero_grad()
                loss = trial.loss_fn(model, ctx["Xtr_t"][b], ctx["ytr_t"][b], crit, ctx)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vp = ac.logits_of(model, ctx["Xval_t"]).argmax(1).cpu().numpy()
            f1 = du.macro_f1(ctx["yval_np"], vp, n_classes)
            if f1 > best + 1e-4:
                best, bad = f1, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= ac.PATIENCE:
                    break
        model.load_state_dict(best_state)
        P_te += ac.predict_probs(model, ctx["Xte_t"], n_classes)
        del model
        torch.cuda.empty_cache()
    return P_te / trial.n_ensemble


def fold_cbst(ctx):
    """Class-balanced self-training on the TARGET region (Zou et al. ECCV 2018).

    TRANSDUCTIVE: uses the test fold's FEATURES (never its labels), which is
    legitimate for this deployment — inference runs over whole rasters, so the
    target region's unlabelled pixels are always available. Pseudo-labels are
    selected per class at the same confidence QUANTILE, so rare classes get
    represented instead of being crowded out by the easy majority.
    """
    trial = ctx["trial"]
    n_classes = ctx["n_classes"]
    base = Trial(name="_base", tier="", idea="", hypothesis="",
                 n_ensemble=trial.n_ensemble, weight_mode=trial.weight_mode)
    P_te, vf1, info = ac.run_fold(base, ctx)

    q = trial.config["quantile"]
    conf, pl = P_te.max(1), P_te.argmax(1)
    keep = np.zeros(len(pl), dtype=bool)
    for c in range(n_classes):
        m = pl == c
        if m.sum() == 0:
            continue
        thr = np.quantile(conf[m], 1 - q)
        keep |= m & (conf >= thr)
    n_pl = int(keep.sum())
    Xaug = torch.cat([ctx["Xtr_t"], ctx["Xte_t"][torch.tensor(np.flatnonzero(keep),
                                                              device=ac.DEVICE)]])
    yaug = torch.cat([ctx["ytr_t"],
                      torch.tensor(pl[keep], device=ac.DEVICE, dtype=torch.long)])
    ctx2 = dict(ctx)
    ctx2["Xtr_t"], ctx2["ytr_t"] = Xaug, yaug
    yaug_np = yaug.cpu().numpy()
    ctx2["counts"] = np.bincount(yaug_np, minlength=n_classes).astype(np.float64)
    ctx2["w"] = ac.class_weights(yaug_np, n_classes, trial.weight_mode)
    ctx2["trial"] = base
    P2, vf2, _ = ac.run_fold(base, ctx2)
    info.update({"n_pseudo": n_pl, "pseudo_frac": round(n_pl / len(pl), 3)})
    print(f"    CBST: {n_pl:,} pseudo-labelled target rows ({n_pl/len(pl):.1%})")
    return P2, vf2, info


# ------------------------------------------------------------------ pre-flight
def sign_selftest(verbose=True):
    """Imbalance-correcting losses all shift the decision boundary TOWARD rare
    classes. A sign error flips that and looks like a fascinating negative
    result, so assert the direction on synthetic data before spending GPU hours.
    (This exact bug cost a day on logit adjustment.)"""
    torch.manual_seed(0)
    n_rare, n_common = 100, 5000
    Xc = torch.randn(n_common, 4)
    Xr = torch.randn(n_rare, 4) + 0.9
    X = torch.cat([Xc, Xr])
    y = torch.cat([torch.zeros(n_common), torch.ones(n_rare)]).long()
    counts = np.array([n_common, n_rare], dtype=np.float64)

    def fit(loss_name):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        crit = nn.CrossEntropyLoss()
        ctx = {"counts": counts, "n_classes": 2, "w": None,
               "trial": Trial(name="t", tier="", idea="", hypothesis="",
                              label_smooth=0.0,
                              config={"gamma": 0.2, "tau": 1.0, "lam": 1.0,
                                      "max_margin": 0.5, "drw_epoch": 0,
                                      "margin": 1.0}),
               "epoch": 99}
        fn = {"plain": None, "balanced_softmax": loss_balanced_softmax,
              "ldam_drw": loss_ldam_drw, "vs": loss_vs, "softf1": loss_softf1}[loss_name]
        for _ in range(300):
            opt.zero_grad()
            loss = crit(model(X), y) if fn is None else fn(model, X, y, crit, ctx)
            loss.backward()
            opt.step()
        pred = model(X).argmax(1)
        rare_recall = float((pred[y == 1] == 1).float().mean())
        return rare_recall

    base = fit("plain")
    out = {"plain": base}
    ok = True
    for name in ("balanced_softmax", "ldam_drw", "vs", "softf1"):
        r = fit(name)
        out[name] = r
        good = r >= base - 1e-6
        ok &= good
        if verbose:
            print(f"  {name:18s} rare-class recall {base:.3f} -> {r:.3f} "
                  f"{'OK' if good else 'FAIL (sign inverted?)'}")
    return ok, out


# ---------------------------------------------------------------------- trials
def _mlp_act(act):
    return lambda in_dim, n_classes, ctx: L.mlp_with_act(
        in_dim, n_classes, ac.HIDDEN, ac.DROPOUT, act)


def build_hier(in_dim, n_classes, ctx):
    return L.HierHead(in_dim, n_classes, ac.HIDDEN, ac.DROPOUT, SUPERGROUPS)


TRIALS = {}


def _add(t: Trial):
    TRIALS[t.name] = t
    return t


# --- references -------------------------------------------------------------
_add(Trial(
    name="baseline", tier="reference",
    idea="the deployed recipe, re-run inside this harness",
    hypothesis="establishes the paired per-fold and per-class reference; every "
               "other trial is scored against THIS run, not a number copied from "
               "another script",
    save_probs=True))

_add(Trial(
    name="baseline_gval", tier="reference",
    idea="baseline but the inner val is held out by whole cell_ids",
    hypothesis="a spatially held-out inner val is a transfer estimate, so "
               "anything fitted on it (priors, thresholds, tau) has a chance of "
               "surviving the test fold; this run is the control for every "
               "decision-rule trial and supplies their probabilities",
    val_mode="group", save_probs=True))

_add(Trial(
    name="ctrl_noweight", tier="reference",
    idea="plain CE, no sqrt class weights",
    hypothesis="control for the loss family below — those losses do their own "
               "imbalance correction, so they must be compared against no-weights, "
               "not against the sqrt-weighted deployed recipe",
    weight_mode="none"))

# --- imbalance / margin losses ---------------------------------------------
_add(Trial(
    name="balanced_softmax", tier="loss",
    idea="add log-prior to logits during training, predict plain",
    hypothesis="sqrt weights only partially correct the prior; the Bayes-optimal "
               "correction for a shifted label distribution is exactly this shift",
    provenance="Ren et al., Balanced Meta-Softmax, NeurIPS 2020",
    loss_fn=loss_balanced_softmax, weight_mode="none"))

_add(Trial(
    name="ldam_drw", tier="loss",
    idea="per-class margin ~ n_c^{-1/4} + deferred re-weighting at epoch 40",
    hypothesis="re-weighting alone rescales gradients but leaves the margin "
               "structure untouched; a rare class needs a WIDER decision region, "
               "not just a louder gradient",
    provenance="Cao et al., LDAM-DRW, NeurIPS 2019",
    loss_fn=loss_ldam_drw, weight_mode="none",
    config={"max_margin": 0.5, "drw_epoch": 40}))

_add(Trial(
    name="vs_loss", tier="loss",
    idea="multiplicative Delta_c AND additive iota_c on the logits",
    hypothesis="logit adjustment (additive) alone reached +0.0015 here — half the "
               "win threshold; VS adds the multiplicative term that LDAM supplies, "
               "and the two corrections are theoretically complementary",
    provenance="Kini et al., Label-Imbalanced Learning, NeurIPS 2021",
    loss_fn=loss_vs, weight_mode="none", config={"gamma": 0.2, "tau": 0.25}))

_add(Trial(
    name="softf1", tier="loss",
    idea="CE + differentiable macro-F1 surrogate",
    hypothesis="we are scored on macro-F1 but optimise likelihood; the surrogate "
               "makes every class's F1 contribute equally regardless of support",
    provenance="soft-F1 / Dice-style surrogate (classic)",
    loss_fn=loss_softf1, config={"lam": 1.0}))

_add(Trial(
    name="pair_margin", tier="loss",
    idea="hinge margin ONLY between crop/grassland and scrub/sparse-veg",
    hypothesis="the ontology probe located 0.044 macro-F1 inside these two "
               "boundaries; a penalty that is exactly zero for every other class "
               "cannot trade the strong classes away",
    loss_fn=loss_pair_margin, config={"margin": 1.0, "lam": 0.5}))

_add(Trial(
    name="focal", tier="loss",
    idea="alpha-balanced focal loss (gamma=2) in place of plain weighted CE",
    hypothesis="the reference imbalance loss, and a two-sided test: focusing the "
               "gradient on hard rows helps if the weak classes are hard-but-"
               "learnable and hurts if they are label noise, which is itself worth "
               "knowing",
    provenance="Lin et al., Focal Loss, ICCV 2017",
    loss_fn=loss_focal, config={"gamma": 2.0}))

_add(Trial(
    name="smote_weak", tier="loss",
    idea="SMOTE oversampling of the four weak classes (k=5, +50%)",
    hypothesis="the reference oversampling baseline; also a sharp test of the "
               "earlier spatial-jitter negative — if interpolation per se destroys "
               "signal it will lose the same way, if that result was about "
               "geography it will not",
    provenance="Chawla et al., SMOTE, JAIR 2002",
    fold_fn=fold_smote, config={"k": 5, "ratio": 0.5}))

# --- hierarchy --------------------------------------------------------------
_add(Trial(
    name="hier_c2f", tier="hierarchy",
    idea="factorised P(class) = P(supergroup) * P(class | supergroup)",
    hypothesis="a flat softmax spends one set of logits on both the easy "
               "between-group problem and the hard within-group one; factorising "
               "gives the within-group split its own parameters and a gradient "
               "computed only over the rows where the distinction exists",
    provenance="hierarchical softmax / coarse-to-fine classification",
    build_fn=build_hier, loss_fn=loss_hier))

_add(Trial(
    name="hier_finewt", tier="hierarchy",
    idea="same hierarchy, fine-level loss weighted 2x (HXE-style)",
    hypothesis="if factorising helps, the level weighting says WHERE the capacity "
               "should go; the coarse level is already at F1 0.85",
    provenance="Bertinetto et al., Making Better Mistakes, CVPR 2020",
    build_fn=build_hier, loss_fn=loss_hier_finewt, config={"fine_weight": 2.0}))

_add(Trial(
    name="hier_specialist", tier="hierarchy",
    idea="binary expert per confusable pair, consulted only on contested rows",
    hypothesis="the strongest form of the hierarchy claim with a structural "
               "guarantee: rows whose top-2 is not a registered pair are returned "
               "untouched, so strong classes cannot lose",
    fold_fn=fold_specialist))

# --- modern optimizers ------------------------------------------------------
_add(Trial(
    name="opt_muon", tier="optimizer",
    idea="Muon (orthogonalised momentum) on hidden weights, Adam elsewhere",
    hypothesis="SAM and EMA both tied, which ruled out flat-minima effects — but "
               "both search for a different minimum with the same geometry. Muon "
               "changes the metric of the step itself (spectral, not "
               "per-coordinate), so it explores a different solution family",
    provenance="Jordan et al., Muon, 2024",
    opt_fn=opt_muon, config={"muon_lr": 0.02}))

_add(Trial(
    name="opt_lion", tier="optimizer",
    idea="Lion — sign of interpolated momentum",
    hypothesis="uniform per-coordinate step size acts as an implicit "
               "regulariser; on a problem whose failure mode is memorising train "
               "regions, a coarser update may transfer better",
    provenance="Chen et al., Symbolic Discovery of Optimization Algorithms, 2023",
    opt_fn=opt_lion, config={"lion_lr": 1e-4, "lion_wd": 1e-3}))

_add(Trial(
    name="opt_schedulefree", tier="optimizer",
    idea="Schedule-Free AdamW (evaluate the averaged iterate)",
    hypothesis="early stopping means the training horizon is unknown, which is "
               "precisely the setting Schedule-Free targets; its iterate average "
               "is also a cheaper EMA than the one already tried",
    provenance="Defazio et al., The Road Less Scheduled, NeurIPS 2024",
    opt_fn=opt_sf))

# --- activations ------------------------------------------------------------
for _act, _why in [
    ("gelu", "smooth gating, the modern default"),
    ("mish", "self-regularised non-monotone; reported gains on small tabular nets"),
    ("prelu", "learned negative slope per unit — lets dead units recover"),
    ("snake", "periodic inductive bias for non-monotone spectral boundaries"),
    ("xielu", "trainable expanded IELU; strictly contains ReLU as a special case"),
]:
    _add(Trial(
        name=f"act_{_act}", tier="activation",
        idea=f"replace ReLU with {_act}",
        hypothesis=_why + " — the trunk shape and every other knob stay fixed, so "
                          "any delta is the nonlinearity alone",
        build_fn=_mlp_act(_act)))

# --- representation / architecture -----------------------------------------
_add(Trial(
    name="tabm", tier="architecture",
    idea="TabM: k=8 weight-shared submodels trained jointly (BatchEnsemble)",
    hypothesis="ensembling is the ONE lever that ever worked here, yet 5->15 "
               "independent seeds bought nothing — TabM's members share weights "
               "and are trained jointly, so their diversity is a regulariser "
               "rather than init variance, which is a different mechanism",
    provenance="Gorishniy et al., TabM, ICLR 2025",
    build_fn=lambda i, c, ctx: L.TabM(i, c, ac.HIDDEN, ac.DROPOUT, k=8),
    loss_fn=loss_tabm, config={"k": 8}))

_add(Trial(
    name="numemb_lidar", tier="architecture",
    idea="periodic (PLR) embeddings on the 3 lidar columns only",
    hypothesis="elevation/TRI/CHM have NON-MONOTONE class structure (mire low, "
               "scrub mid, bare high) that a first linear layer can only cut once "
               "per unit; periodic embeddings resolve bands directly. Restricted "
               "to lidar because those 3 columns are the physical, "
               "region-transferable features",
    provenance="Gorishniy et al., On Embeddings for Numerical Features, NeurIPS 2022",
    build_fn=lambda i, c, ctx: L.numemb_mlp(i, c, ac.HIDDEN, ac.DROPOUT,
                                            cols=ctx["lidar_cols"], sigma=0.5),
    config={"n_freq": 12, "d_emb": 8, "sigma": 0.5}))

_add(Trial(
    name="numemb_all", tier="architecture",
    idea="periodic (PLR) embeddings on all 67 features",
    hypothesis="same mechanism applied to the AlphaEarth bands too; the risk is "
               "that 64 already-learned embedding dims do not need re-embedding "
               "and the extra width just memorises regions",
    provenance="Gorishniy et al., NeurIPS 2022",
    build_fn=lambda i, c, ctx: L.numemb_mlp(i, c, ac.HIDDEN, ac.DROPOUT,
                                            cols=None, n_freq=8, d_emb=4, sigma=0.5),
    config={"n_freq": 8, "d_emb": 4, "sigma": 0.5}))

_add(Trial(
    name="kan", tier="architecture",
    idea="Chebyshev Kolmogorov-Arnold network (learnable edge functions)",
    hypothesis="every architecture tried so far is a fixed-nonlinearity MLP "
               "variant, and they all tie — KAN changes the function BASIS "
               "(learned univariate functions on edges), which is the one "
               "structural axis left untested",
    provenance="Liu et al., KAN, 2024 (Chebyshev variant)",
    build_fn=lambda i, c, ctx: L.KAN(i, c, hidden=(64,), degree=5,
                                     dropout=ac.DROPOUT),
    config={"hidden": [64], "degree": 5}))

# --- classifier surgery -----------------------------------------------------
_add(Trial(
    name="tau_norm", tier="classifier",
    idea="divide each class's weight vector by ||w_c||^tau, tau picked on spatial val",
    hypothesis="long-tail nets grow larger classifier norms for frequent classes; "
               "rebalancing the norms needs no retraining and cannot disturb the "
               "representation",
    provenance="Kang et al., Decoupling Representation and Classifier, ICLR 2020",
    after_fit=after_tau_norm, val_mode="group"))

_add(Trial(
    name="crt", tier="classifier",
    idea="freeze the trunk, re-init and retrain the last layer class-balanced",
    hypothesis="instance-balanced data teaches better FEATURES, class-balanced "
               "data teaches a better BOUNDARY; the deployed single sqrt-weighted "
               "loss has to compromise between the two, decoupling does not",
    provenance="Kang et al., ICLR 2020",
    after_fit=after_crt, config={"crt_lr": 1e-3, "crt_steps": 400}))

# --- knowledge transfer -----------------------------------------------------
_add(Trial(
    name="selfdistill", tier="transfer",
    idea="5-seed ensemble teaches a fresh identical net (T=2)",
    hypothesis="soft targets carry inter-class similarity that hard labels "
               "discard, and that structure is densest exactly where the weak "
               "classes overlap; also the standard way to keep an ensemble's "
               "benefit in one model",
    provenance="Hinton et al. 2015 / Furlanello et al., Born-Again Networks, 2018",
    fold_fn=fold_selfdistill, config={"T": 2.0, "alpha": 0.5}))

# --- transductive adaptation ------------------------------------------------
_add(Trial(
    name="tta_cbst", tier="transductive",
    idea="class-balanced self-training on the target fold's UNLABELLED features",
    hypothesis="the documented wall is distribution shift between geographic "
               "regions. Every previous attempt fought it with train-side "
               "regularisation, which cannot invent information about an unseen "
               "region — this uses the target region's own unlabelled pixels, "
               "which deployment always has, and balances the pseudo-label quota "
               "per class so rare classes are not crowded out",
    provenance="Zou et al., CBST, ECCV 2018",
    fold_fn=fold_cbst, transductive=True, config={"quantile": 0.3}))


# Ordering: cheap and directly aimed at the weak classes first, expensive or
# speculative last, so a real win shows up early.
QUEUE = [
    "baseline", "baseline_gval",
    "pair_margin", "hier_c2f", "hier_specialist",
    "vs_loss", "tau_norm", "crt",
    "balanced_softmax", "ldam_drw", "ctrl_noweight",
    "hier_finewt", "numemb_lidar", "tabm",
    "opt_muon", "opt_schedulefree", "opt_lion",
    "act_gelu", "act_mish", "act_prelu", "act_snake", "act_xielu",
    "softf1", "kan", "numemb_all",
    "focal", "smote_weak",
    "selfdistill", "tta_cbst",
]


# ------------------------------------------------------- round-2 trial algebra
FIELD_OVERRIDES = {"label_smooth": float, "n_ensemble": int,
                   "weight_mode": str, "val_mode": str}


def resolve(spec: str) -> Trial:
    """Turn a trial SPEC into a Trial. Three forms, so the loop can schedule
    follow-ups without anyone editing this file:

      `pair_margin`                    a registered trial
      `pair_margin@margin=2.0,lam=1.0` the same trial with config overrides
      `combo:hier_c2f+tau_norm`        two trials whose hooks do not collide,
                                       applied together (this is how a second
                                       round tests whether two independent wins
                                       are additive or redundant)
    """
    import dataclasses
    if spec in TRIALS:
        return TRIALS[spec]

    if spec.startswith("combo:"):
        parts = spec[len("combo:"):].split("+")
        base = dataclasses.replace(TRIALS[parts[0]])
        merged_cfg = dict(base.config)
        for p in parts[1:]:
            other = TRIALS[p]
            for hook in ("build_fn", "loss_fn", "opt_fn", "after_fit",
                         "post_fn", "fold_fn"):
                mine, theirs = getattr(base, hook), getattr(other, hook)
                if theirs is None:
                    continue
                if mine is not None:
                    raise SystemExit(
                        f"cannot combine {spec}: both set `{hook}`. Combining "
                        f"two mechanisms that occupy the same hook would test a "
                        f"third, unnamed mechanism.")
                setattr(base, hook, theirs)
            if other.val_mode != "random":
                base.val_mode = other.val_mode
            if other.weight_mode != "sqrt":
                base.weight_mode = other.weight_mode
            merged_cfg.update(other.config)
        base.name = spec.replace(":", "_").replace("+", "_")
        base.tier = "combo"
        base.idea = " + ".join(TRIALS[p].idea for p in parts)
        base.hypothesis = ("are these mechanisms additive or redundant? each won "
                           "or nearly won alone; if the gains come from the same "
                           "underlying error they will not stack")
        base.config = merged_cfg
        return base

    head, _, tail = spec.partition("@")
    t = dataclasses.replace(TRIALS[head])
    cfg = dict(t.config)
    for kv in tail.split(","):
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if k in FIELD_OVERRIDES:
            setattr(t, k, FIELD_OVERRIDES[k](v))
            continue
        try:
            cfg[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            cfg[k] = v
    t.config = cfg
    t.name = (spec.replace("@", "__").replace(",", "_").replace("=", "")
                  .replace(".", "p"))
    t.tier = t.tier + "-v2"
    t.hypothesis = f"follow-up sweep of `{head}`: " + t.hypothesis
    return t


# --- round 3: locality ------------------------------------------------------
# Registered from a separate module so this file stays the record of rounds 1-2,
# but into the SAME registry — `resolve`, `run.py` and the loop's sweep/combo
# algebra all work on the locality trials with no special-casing.
import trials_local  # noqa: E402

trials_local.register(_add, Trial)
QUEUE_LOCAL = trials_local.QUEUE_LOCAL

# --- classical references ---------------------------------------------------
# Not a round: the floor the whole leaderboard is measured above. Same registry,
# same rules, so they appear in the reference tier next to the baseline.
import trials_classic  # noqa: E402

trials_classic.register(_add, Trial)
QUEUE_CLASSIC = trials_classic.QUEUE_CLASSIC


if __name__ == "__main__":
    print("loss sign self-test (imbalance losses must RAISE rare-class recall):")
    ok, out = sign_selftest()
    print("PASS" if ok else "FAIL")
    print(f"registered {len(TRIALS)} trials, queue length {len(QUEUE)}")
    missing = [n for n in QUEUE if n not in TRIALS]
    assert not missing, missing

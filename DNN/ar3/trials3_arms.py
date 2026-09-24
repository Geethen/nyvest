"""ar3 trial arms: temporal consistency/decoding + calibration.

Imported by `trials3.py` (`trials3_arms.register(_add, Trial3)`). Every arm here
inherits the deployed moe8 build (none of them set `build_fn`); hyperparameters
travel in `Trial3.config` and are read with `_cfg` (copied from
`autoresearch/trials_local.py`, which reads `ctx["trial"].config`).

Temporal arms (tmp_*) exploit the fact that every FSCS point carries 9 year rows
(2017-2025) under one label, aligned via `meta["loc"]` (constant per location
across its year rows). See PLAN.md workstream W3.

Calibration arms (cal_*) vary the training loss / ensemble size only; the
temporal-decoding machinery does not apply to them.

A note on `A.base_ce`: the harness convention is "compute the CE term via
`A.base_ce(A.ac.logits_of(model, xb), yb, crit, ctx)`" so custom losses inherit
row/class-weight handling uniformly. `tmp_consist` and `cal_logitnorm` do this
literally (their CE term is a plain, or linearly-transformed, cross-entropy).
`cal_focal` cannot: the whole point of focal loss is per-example modulation by
(1-p_t)^gamma *before* reduction, and `crit`'s reduction mode is fixed by the
harness (mean when no row weights, none when row weights exist) rather than
switchable by a trial, so `base_ce`'s `crit(out, yb)` call cannot produce the
per-example tensor focal needs when row weights are absent. `cal_focal`
therefore builds its own per-example CE and mirrors `base_ce`'s row/class-weight
normalisation by hand (see its docstring). Flagged in the final report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import ar3_common as A

Trial3 = A.Trial3


def _cfg(ctx, **defaults):
    """Trial.config overrides, so the loop can sweep an axis without edits here."""
    cfg = dict(defaults)
    cfg.update({k: v for k, v in ctx["trial"].config.items() if k in defaults})
    return cfg


# ======================================================================
# tmp_consist — train-time KL consistency between same-loc, other-year rows
# ======================================================================
def _build_partner_idx(tr_meta: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """For each row, a random OTHER-year row index (into the same frame) of the
    same `loc`, or -1 if the loc has no other-year row in this training set.
    Grouped by loc (sorted) so the python loop runs over ~n_locs groups, not
    ~n_rows individual rows."""
    n = len(tr_meta)
    loc = tr_meta["loc"].values
    year = tr_meta["year"].values
    partner = np.full(n, -1, dtype=np.int64)
    order = np.argsort(loc, kind="stable")
    loc_sorted = loc[order]
    bounds = np.flatnonzero(np.r_[True, loc_sorted[1:] != loc_sorted[:-1], True])
    for gi in range(len(bounds) - 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e - s < 2:
            continue
        idxs = order[s:e]
        yrs = year[idxs]
        if len(np.unique(yrs)) < 2:
            continue
        for pos in range(len(idxs)):
            cand = idxs[yrs != yrs[pos]]
            if len(cand):
                partner[idxs[pos]] = cand[rng.integers(len(cand))]
    return partner


def prep_tmp_consist(ctx):
    partner = _build_partner_idx(ctx["tr_meta"], ctx["rng"])
    n_partnered = int((partner >= 0).sum())
    ctx["tmp_partner_idx"] = torch.tensor(partner, dtype=torch.long, device=A.DEVICE)
    print(f"    tmp_consist: {n_partnered:,}/{len(partner):,} training rows have "
          f"an other-year partner")


def loss_tmp_consist(model, xb, yb, crit, ctx):
    """base CE + lam * symmetric-KL(softmax(logits(xb)) || softmax(logits(x_partner)))
    over rows with a valid partner. Single-year inference, zero deploy cost."""
    lam = float(_cfg(ctx, lam=1.0)["lam"])
    logits = A.ac.logits_of(model, xb)
    ce = A.base_ce(logits, yb, crit, ctx)

    b = ctx["batch_idx"]
    partner = ctx["tmp_partner_idx"][b]
    valid = partner >= 0
    if valid.any():
        x_partner = ctx["Xtr_t"][partner[valid]]
        logits_partner = A.ac.logits_of(model, x_partner)
        logp = F.log_softmax(logits[valid], dim=1)
        logq = F.log_softmax(logits_partner, dim=1)
        p, q = logp.exp(), logq.exp()
        skl = ((p * (logp - logq)).sum(1) + (q * (logq - logp)).sum(1)).mean()
    else:
        skl = logits.sum() * 0.0
    return ce + lam * skl


# ======================================================================
# tmp_pair — append the partner-year embedding as extra input features
# ======================================================================
def _select_partner_year(others: np.ndarray, y: int) -> int:
    """Among `others` (years != y for this loc), the year whose |year - y| is
    closest to 6 (the 2018/2024 map gap); ties -> later year. Falls out of the
    generic rule for 2018<->2024 exactly (|2024-2018|=6 is the unique minimum
    whenever both are available)."""
    d = np.abs(others - y)
    key = np.abs(d - 6)
    best = key.min()
    cand = others[key == best]
    return int(cand.max())


def _partner_embeddings(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    n = len(meta)
    loc = meta["loc"].values
    year = meta["year"].values.astype(np.int64)
    emb = X[:, :64].astype(np.float32)
    out = emb.copy()  # default: own embedding (no other year available)
    order = np.argsort(loc, kind="stable")
    loc_sorted = loc[order]
    bounds = np.flatnonzero(np.r_[True, loc_sorted[1:] != loc_sorted[:-1], True])
    for gi in range(len(bounds) - 1):
        s, e = bounds[gi], bounds[gi + 1]
        idxs = order[s:e]
        if len(idxs) < 2:
            continue
        yrs = year[idxs]
        for pos in range(len(idxs)):
            others_mask = yrs != yrs[pos]
            others_yrs = yrs[others_mask]
            if len(others_yrs) == 0:
                continue
            py = _select_partner_year(others_yrs, yrs[pos])
            match = idxs[others_mask][others_yrs == py][0]
            out[idxs[pos]] = emb[match]
    return out


def feat_tmp_pair(X, meta, feat_cols):
    """input = [emb_y (+lidar), emb_partner_year] — 64 extra cols, original 67
    first. Deploy note: needs both 2018 and 2024 map-year VRTs (they exist)."""
    partner = _partner_embeddings(X, meta)
    return np.hstack([X, partner]).astype(np.float32)


# ======================================================================
# tmp_ctx / tmp_ctx_mean — per-loc across-year embedding summary stats
# ======================================================================
def _loc_embedding_stats(X: np.ndarray, meta: pd.DataFrame):
    """Per-loc mean and population std of the 64 embedding dims over ALL of
    that loc's available-year rows (own row included), broadcast back to every
    row of the loc."""
    loc = meta["loc"].values
    emb = pd.DataFrame(X[:, :64].astype(np.float64))
    emb["__loc"] = loc
    g = emb.groupby("__loc")
    mean = g.transform("mean").values.astype(np.float32)
    std = g.transform("std", ddof=0).fillna(0.0).values.astype(np.float32)
    return mean, std


def feat_tmp_ctx(X, meta, feat_cols):
    """input = [emb_y, mean_y', std_y'] over all 9 years — 128 extra cols.
    Deploy note: needs all 9 annual AEF rasters."""
    mean, std = _loc_embedding_stats(X, meta)
    return np.hstack([X, mean, std]).astype(np.float32)


def feat_tmp_ctx_mean(X, meta, feat_cols):
    """input = [emb_y, mean_y'] over all 9 years — 64 extra cols (mean only,
    ablation of tmp_ctx). Deploy note: needs all 9 annual AEF rasters."""
    mean, _ = _loc_embedding_stats(X, meta)
    return np.hstack([X, mean]).astype(np.float32)


# ======================================================================
# tmp_joint — post-hoc bi-temporal joint decoding (2018 x 2024 only)
# ======================================================================
def _sticky_T(nc: int, eps: float) -> np.ndarray:
    T = np.full((nc, nc), eps / (nc - 1), dtype=np.float64)
    np.fill_diagonal(T, 1.0 - eps)
    return T


def post_tmp_joint(P_te, ctx):
    """For test locs with both a 2018 and a 2024 row: joint P(a,b) ∝
    p18(a)*p24(b)*T(a,b), replace each of the two rows' probs by its
    normalised marginal. Other rows untouched. config eps=0.01."""
    eps = float(_cfg(ctx, eps=0.01)["eps"])
    nc = ctx["n_classes"]
    meta = ctx["te_meta"]
    loc = meta["loc"].values
    year = meta["year"].values.astype(int)

    idx18 = {l: i for i, (l, y) in enumerate(zip(loc, year)) if y == 2018}
    idx24 = {l: i for i, (l, y) in enumerate(zip(loc, year)) if y == 2024}
    common = sorted(set(idx18) & set(idx24))

    P = P_te.astype(np.float64).copy()
    T = _sticky_T(nc, eps)
    n_changed = 0
    for l in common:
        i18, i24 = idx18[l], idx24[l]
        p18, p24 = P[i18], P[i24]
        old18, old24 = int(p18.argmax()), int(p24.argmax())
        joint = np.outer(p18, p24) * T
        s = joint.sum()
        if s <= 0:
            continue
        joint /= s
        marg18 = joint.sum(1)
        marg24 = joint.sum(0)
        P[i18], P[i24] = marg18, marg24
        if int(marg18.argmax()) != old18 or int(marg24.argmax()) != old24:
            n_changed += 1
    frac = n_changed / len(common) if common else 0.0
    info = {"tmp_joint_eps": eps, "tmp_joint_n_pairs": len(common),
            "tmp_joint_frac_argmax_changed": frac}
    return P.astype(P_te.dtype), info


# ======================================================================
# tmp_hmm — full 2017-2025 chain, forward-backward smoothing
# ======================================================================
YEARS_ALL = np.arange(2017, 2026)  # 9 calendar slots; missing years => uninformative


def _forward_backward(obs_used: np.ndarray, Tm: np.ndarray) -> np.ndarray:
    """obs_used: [n_locs, T, nc] emission likelihoods (uniform where unobserved).
    Uniform prior. Per-step renormalisation (the standard *scaled* HMM trick):
    alpha_hat_t is exactly proportional to the true alpha_t at every t (the
    running normalising constant telescopes out of the recursion), and same for
    beta_hat_t, so alpha_hat_t * beta_hat_t, renormalised over states, IS the
    exact posterior — no separate un-scaled computation is needed."""
    n_locs, T, nc = obs_used.shape
    alpha = np.empty_like(obs_used)
    a0 = obs_used[:, 0] * (1.0 / nc)
    alpha[:, 0] = a0 / a0.sum(1, keepdims=True)
    for t in range(1, T):
        pred = alpha[:, t - 1] @ Tm
        a = pred * obs_used[:, t]
        s = a.sum(1, keepdims=True)
        s[s == 0] = 1.0
        alpha[:, t] = a / s

    beta = np.ones_like(obs_used)
    for t in range(T - 2, -1, -1):
        b = (obs_used[:, t + 1] * beta[:, t + 1]) @ Tm.T
        s = b.sum(1, keepdims=True)
        s[s == 0] = 1.0
        beta[:, t] = b / s

    post = alpha * beta
    post /= post.sum(2, keepdims=True)
    return post


def post_tmp_hmm(P_te, ctx):
    """Bi-temporal joint's generalisation over the full 2017-2025 chain per
    loc, sticky transition eps (config, default 0.005) per year-step, uniform
    prior, each year's P_te row as the emission likelihood. Replaces every
    row's probs with its posterior marginal. Vectorised over all locs at once
    (padded to the fixed 9-year calendar) so it runs in well under a minute on
    ~220k test rows."""
    eps = float(_cfg(ctx, eps=0.005)["eps"])
    nc = ctx["n_classes"]
    meta = ctx["te_meta"]
    loc = meta["loc"].values
    year = meta["year"].values.astype(int)
    n = len(loc)
    T = len(YEARS_ALL)

    uniq_locs, loc_inv = np.unique(loc, return_inverse=True)
    n_locs = len(uniq_locs)
    year_idx = year - YEARS_ALL[0]
    valid = (year_idx >= 0) & (year_idx < T)

    obs = np.zeros((n_locs, T, nc), dtype=np.float64)
    mask = np.zeros((n_locs, T), dtype=bool)
    row_of = np.full((n_locs, T), -1, dtype=np.int64)
    vrows = np.flatnonzero(valid)
    obs[loc_inv[vrows], year_idx[vrows]] = P_te[vrows]
    mask[loc_inv[vrows], year_idx[vrows]] = True
    row_of[loc_inv[vrows], year_idx[vrows]] = vrows

    obs_used = np.where(mask[..., None], obs, 1.0 / nc)
    Tm = _sticky_T(nc, eps)
    post = _forward_backward(obs_used, Tm)

    P_out = P_te.astype(np.float64).copy()
    for t in range(T):
        rows = row_of[:, t]
        m = rows >= 0
        P_out[rows[m]] = post[m, t]
    return P_out.astype(P_te.dtype), {"tmp_hmm_eps": eps, "tmp_hmm_n_locs": int(n_locs)}


# ======================================================================
# tmp_pool — bound: mean posterior over all years at a loc
# ======================================================================
def post_tmp_pool(P_te, ctx):
    """Replace every row's probs with the arithmetic mean of its loc's probs
    over all years present in the test fold. The ceiling for temporal
    denoising on stable points: it can only help where the point never
    actually changed, and on a genuinely changed pixel it averages the real
    2018 and 2024 signal together and destroys the change signal outright."""
    meta = ctx["te_meta"]
    loc = meta["loc"].values
    uniq, inv = np.unique(loc, return_inverse=True)
    nc = P_te.shape[1]
    sums = np.zeros((len(uniq), nc), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(sums, inv, P_te.astype(np.float64))
    np.add.at(counts, inv, 1.0)
    pooled = sums / counts[:, None]
    P_out = pooled[inv]
    return P_out.astype(P_te.dtype), {"tmp_pool_n_locs": int(len(uniq))}


# ======================================================================
# calibration arms
# ======================================================================
def loss_cal_focal(model, xb, yb, crit, ctx):
    """Focal loss (Mukhoti et al., NeurIPS 2020), gamma from config (default
    3.0), class weights as in base (ctx["w"], may be None). `crit`'s reduction
    is fixed by row-weight presence and is unusable here regardless (focal
    needs the PER-EXAMPLE ce before the modulating (1-p_t)^gamma factor, even
    when there are no row weights and `crit` would otherwise be reduction=
    'mean'), so this builds its own per-example CE and mirrors `base_ce`'s
    row/class-weight normalisation by hand rather than delegating to it."""
    gamma = float(_cfg(ctx, gamma=3.0)["gamma"])
    logits = A.ac.logits_of(model, xb)
    logp = F.log_softmax(logits, dim=1)
    logpt = logp.gather(1, yb.unsqueeze(1)).squeeze(1)
    pt = logpt.exp().clamp(max=1.0 - 1e-6)
    ce = F.nll_loss(logp, yb, weight=ctx["w"], reduction="none")
    focal = ((1.0 - pt) ** gamma) * ce

    row_w = ctx.get("wtr_t")
    cls_w = ctx["w"][yb] if ctx["w"] is not None else torch.ones_like(focal)
    if row_w is not None:
        b = ctx["batch_idx"]
        w = row_w[b] * cls_w
        return (focal * row_w[b]).sum() / w.sum()
    return focal.sum() / cls_w.sum()


def loss_cal_logitnorm(model, xb, yb, crit, ctx):
    """LogitNorm (Wei et al., ICML 2022): CE on logits / (tau * ||logits||_2).
    Inference is UNCHANGED (argmax is scale-invariant so predictions are
    identical; probabilities differ, which is the point, and post-hoc
    temperature scaling in metrics.py handles the scale). The CE term here IS
    a plain cross-entropy (just on rescaled logits), so it goes through
    `A.base_ce` directly."""
    tau = float(_cfg(ctx, tau=0.04)["tau"])
    logits = A.ac.logits_of(model, xb)
    norm = logits.norm(p=2, dim=1, keepdim=True).clamp_min(1e-7)
    logits_n = logits / (tau * norm)
    return A.base_ce(logits_n, yb, crit, ctx)


# ======================================================================
# registry
# ======================================================================
def register(_add, _Trial3):
    _add(_Trial3(
        name="tmp_consist", tier="temporal",
        idea="train-time symmetric-KL consistency between a row's prediction "
             "and a random OTHER-year row of the same location; single-year "
             "inference, zero deploy cost",
        hypothesis="the deployed model sees each year's row independently and "
                   "has no reason to agree with itself across years; a soft "
                   "consistency penalty at train time should lower the flip "
                   "rate (flip_1720, flip_1824) without changing what is fed "
                   "in at inference, so it can only cost accuracy where the "
                   "consistency prior is WRONG (real change)",
        prep_fn=prep_tmp_consist, loss_fn=loss_tmp_consist,
        config={"lam": 1.0}))

    _add(_Trial3(
        name="tmp_pair", tier="temporal",
        idea="input = [emb_y, emb_partner_year] (2018<->2024 exactly; else the "
             "available year closest to a 6-year gap, ties -> later year; own "
             "embedding if no other year exists) — 64 extra input columns",
        hypothesis="the network sees BOTH endpoints of the year gap it is "
                   "scored on and can learn what changed vs. what is sensor "
                   "noise directly, rather than inferring it post-hoc. Deploy "
                   "note: needs both map years, and the 2018/2024 VRTs "
                   "already exist so this is deployable today",
        feat_fn=feat_tmp_pair))

    _add(_Trial3(
        name="tmp_ctx", tier="temporal",
        idea="input = [emb_y, mean_y', std_y'] over all 9 available years at "
             "the location — 128 extra input columns",
        hypothesis="mean/std over the full annual record is a cheap summary "
                   "of what is temporally STABLE at a point (mean) and how "
                   "noisy/volatile it is (std); a network conditioned on both "
                   "may down-weight a single year's sensor noise. Deploy "
                   "note: needs all 9 annual AEF rasters, flagged as a real "
                   "operational cost vs. tmp_pair's two",
        feat_fn=feat_tmp_ctx))

    _add(_Trial3(
        name="tmp_ctx_mean", tier="temporal",
        idea="ablation of tmp_ctx: input = [emb_y, mean_y'] only (64 extra "
             "cols, no std)",
        hypothesis="separates what tmp_ctx's gain (if any) is actually buying "
                   "— the stable-location signal (mean) or the volatility "
                   "signal (std). Same 9-raster deploy cost as tmp_ctx",
        feat_fn=feat_tmp_ctx_mean))

    _add(_Trial3(
        name="tmp_joint", tier="temporal",
        idea="post-hoc bi-temporal joint decoding: for test locs with both a "
             "2018 and 2024 row, P(c18,c24) ∝ p18(c18)*p24(c24)*T(c18,c24), "
             "sticky T=(1-eps)*I + eps/(C-1)*(1-I); replace each row's probs "
             "with its normalised marginal",
        hypothesis="no fitting on stable points (where the optimum is "
                   "trivially eps->0, collapse to agreement); a small, "
                   "pre-registered eps nudges the joint toward temporal "
                   "agreement only as far as the prior says is plausible, "
                   "which should lower flip_1824 with a bounded accuracy "
                   "cost on real 2018-2024 change",
        post3_fn=post_tmp_joint, config={"eps": 0.01}))

    _add(_Trial3(
        name="tmp_hmm", tier="temporal",
        idea="tmp_joint's chain generalisation over the full 2017-2025 record "
             "per location: forward-backward smoothing with a sticky "
             "per-year-step transition (eps, default 0.005), uniform prior, "
             "each year's P_te row as the emission likelihood; replaces "
             "EVERY row's probs with its posterior marginal",
        hypothesis="uses 9 years of evidence instead of 2, so it should "
                   "denoise single-year sensor noise harder than tmp_joint "
                   "while the sticky-not-absorbing transition still lets a "
                   "sustained multi-year shift (real change) through, unlike "
                   "tmp_pool's flat average",
        post3_fn=post_tmp_hmm, config={"eps": 0.005}))

    _add(_Trial3(
        name="tmp_pool", tier="temporal",
        idea="post-hoc: replace every row's probs with the arithmetic mean of "
             "its location's probs over all years in the test fold",
        hypothesis="THE CEILING of temporal denoising on stable points, not "
                   "a candidate to adopt: it has no transition prior at all, "
                   "so it destroys real 2018-2024 change sensitivity by "
                   "construction (flip_1824 -> ~0 regardless of whether the "
                   "pixel actually changed). Bounds how much flip-rate "
                   "improvement tmp_joint/tmp_hmm are leaving on the table",
        post3_fn=post_tmp_pool, is_bound=True))

    _add(_Trial3(
        name="cal_ls0", tier="calibration",
        idea="label smoothing 0.0 (deployed default is 0.05)",
        hypothesis="label smoothing pulls predicted probabilities away from "
                   "0/1 uniformly, which trades away calibration for a "
                   "regularisation effect (Müller et al., NeurIPS 2019, "
                   "'When Does Label Smoothing Help?' shows it can hurt "
                   "calibration even when it helps accuracy); removing it "
                   "should move ECE/NLL, direction untested here",
        provenance="Müller et al., NeurIPS 2019",
        label_smooth=0.0))

    _add(_Trial3(
        name="cal_focal", tier="calibration",
        idea="focal loss (gamma=3) replacing CE, label_smooth=0",
        hypothesis="focal loss down-weights already-confident correct "
                   "examples and concentrates gradient on the hard/uncertain "
                   "ones, which Mukhoti et al. show implicitly regularises "
                   "predicted confidence and improves calibration relative "
                   "to plain CE without needing post-hoc temperature scaling",
        provenance="Mukhoti et al., 'Calibrating Deep Neural Networks using "
                   "Focal Loss', NeurIPS 2020",
        loss_fn=loss_cal_focal, label_smooth=0.0, config={"gamma": 3.0}))

    _add(_Trial3(
        name="cal_logitnorm", tier="calibration",
        idea="LogitNorm: CE computed on logits / (tau * ||logits||_2), "
             "tau=0.04; inference/argmax unchanged, only the training "
             "objective changes",
        hypothesis="Wei et al. show unconstrained logit NORM growth (not "
                   "direction) is what drives overconfidence under plain CE; "
                   "normalising it out at train time should improve raw ECE/"
                   "NLL directly, and since post-hoc temperature scaling "
                   "already exists downstream this measures what LogitNorm "
                   "buys ON TOP of TS, not instead of it",
        provenance="Wei et al., 'Mitigating Neural Network Overconfidence "
                   "with Logit Normalization', ICML 2022",
        loss_fn=loss_cal_logitnorm, config={"tau": 0.04}))

    _add(_Trial3(
        name="cal_ens5", tier="calibration",
        idea="n_ensemble=5 (matches the deployed model's ensemble size; "
             "ar3's own default is N_ENSEMBLE=3), everything else identical "
             "to baseline",
        hypothesis="ensembling is a free calibration lever (deep ensembles, "
                   "Lakshminarayanan et al. 2017); this measures how much of "
                   "the deployed model's calibration/confidence profile is "
                   "attributable to ensemble size alone, on the SAME ar3 "
                   "harness the other calibration arms are compared against",
        provenance="Lakshminarayanan et al., NeurIPS 2017 (deep ensembles)",
        n_ensemble=5))

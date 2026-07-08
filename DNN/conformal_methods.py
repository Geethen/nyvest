"""Conformal / uncertainty score functions for the DNN ensemble.

Self-contained implementations (no torchcp dependency) of the split-conformal
score functions and predictors compared in Investigation 3. All operate on
softmax probabilities and follow the standard split-conformal recipe:

    cal_scores = s(P_cal, y_cal)                 # nonconformity at the true label
    tau        = conformal_quantile(cal_scores)  # 1-alpha quantile (finite-sample)
    sets       = { c : s(P_test, c) <= tau }     # prediction set per test row

Score functions (higher score = more nonconforming):
  LAC     s = 1 - p[y]                       (Least Ambiguous set-valued Classifier)
  APS     s = cumulative sorted prob up to & incl. y (randomized)   -> conformal_utils
  RAPS    APS + lam * max(0, rank(y) - k_reg)   (regularized, shrinks sets)
  SAPS    ranked: s = u*p_max (top) else p_max + (rank-1..)*? ; sorted-APS variant
  Margin  s = p_max_other - p[y]            (difference to best competing class)

Predictors:
  split            one global tau                          (marginal coverage)
  classconditional one tau_c per TRUE class (Mondrian)     (per-class coverage)

Also: temperature_scale() fits a single T on the cal set (minimizes NLL) and
returns calibrated probs + ECE, for the point-uncertainty comparison.

Everything takes a fixed uniform-noise vector u for the randomized scores so a
run is reproducible and cal/test use independent draws.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return np.inf
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q, method="higher"))


# ---------- helpers on sorted probabilities ----------

def _sorted(probs):
    """order[i] = class indices sorted by DESC prob; sorted_p descending;
    rank[i,c] = 1-based rank of class c (1 = highest prob)."""
    n, C = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(1, C + 1)[None, :]
    return order, sorted_p, rank


# ---------- score functions: s(probs) -> full [n, C] score matrix ----------
# For each, the nonconformity of assigning class c to row i is score[i, c].

def score_lac(probs, u=None):
    return 1.0 - probs


def score_margin(probs, u=None):
    """p_best_other(c) - p[c]; smaller (more negative) = more conforming."""
    n, C = probs.shape
    s = np.empty_like(probs)
    for c in range(C):
        other = np.delete(probs, c, axis=1).max(1)
        s[:, c] = other - probs[:, c]
    return s


def score_aps(probs, u):
    """Randomized APS: cumulative prob mass of classes ranked >= c, with the
    class-c mass added fractionally by u. Matches conformal_utils.aps_*."""
    n, C = probs.shape
    order, sorted_p, rank = _sorted(probs)
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_before_at = np.take_along_axis(cs_before, rank - 1, axis=1)
    return cum_before_at + u[:, None] * probs


def score_raps(probs, u, k_reg=1, lam=0.1):
    """Regularized APS: APS + lam * max(0, rank - k_reg). Penalizes deep-rank
    classes so prediction sets shrink (Angelopoulos et al. 2021)."""
    n, C = probs.shape
    _, _, rank = _sorted(probs)
    aps = score_aps(probs, u)
    reg = lam * np.maximum(0, rank - k_reg)
    return aps + reg


def score_saps(probs, u, lam=0.1):
    """SAPS (Huang et al. 2024): keep only the top-1 probability; replace the
    ranked masses of the rest with a constant lam so the score depends mostly on
    rank, not on unreliable tail probabilities.
        rank 1 : u * p_max
        rank r>1: p_max + (r-2)*lam + u*lam
    """
    n, C = probs.shape
    order, sorted_p, rank = _sorted(probs)
    p_max = sorted_p[:, 0]
    s = np.empty_like(probs)
    # score in sorted order then scatter back
    s_sorted = np.empty_like(sorted_p)
    s_sorted[:, 0] = u * p_max
    if C > 1:
        r = np.arange(2, C + 1)[None, :]                      # ranks 2..C
        s_sorted[:, 1:] = p_max[:, None] + (r - 2) * lam + u[:, None] * lam
    s[np.arange(n)[:, None], order] = s_sorted
    return s


SCORES = {
    "LAC": score_lac,
    "Margin": score_margin,
    "APS": score_aps,
    "RAPS": score_raps,
    "SAPS": score_saps,
}
RANDOMIZED = {"APS", "RAPS", "SAPS"}


def calibrate_split(score_mat_cal, y_cal, alpha):
    """One global tau from the true-label nonconformity scores."""
    s_true = score_mat_cal[np.arange(len(y_cal)), y_cal]
    return conformal_quantile(s_true, alpha)


def calibrate_classcond(score_mat_cal, y_cal, alpha, n_classes):
    """Mondrian: one tau per TRUE class (class-conditional coverage)."""
    s_true = score_mat_cal[np.arange(len(y_cal)), y_cal]
    taus = np.full(n_classes, np.inf)
    for c in range(n_classes):
        m = y_cal == c
        if m.any():
            taus[c] = conformal_quantile(s_true[m], alpha)
    return taus


def sets_from_tau(score_mat_test, tau):
    """tau scalar (split) OR array[n_classes] (class-conditional, compared
    per predicted class column). Returns bool [n_test, n_classes]."""
    if np.isscalar(tau) or np.ndim(tau) == 0:
        return score_mat_test <= tau
    return score_mat_test <= tau[None, :]


def set_metrics(sets, y_test, n_classes):
    """Coverage (marginal + per-class), avg set size, singleton rate."""
    n = len(y_test)
    covered = sets[np.arange(n), y_test]
    sizes = sets.sum(1)
    per_class_cov = {}
    for c in range(n_classes):
        m = y_test == c
        per_class_cov[str(c)] = round(float(covered[m].mean()), 4) if m.any() else None
    return {
        "coverage": round(float(covered.mean()), 4),
        "avg_set_size": round(float(sizes.mean()), 3),
        "singleton_rate": round(float((sizes == 1).mean()), 4),
        "empty_rate": round(float((sizes == 0).mean()), 4),
        "per_class_coverage": per_class_cov,
    }


# ---------- point-uncertainty: temperature scaling + ECE ----------

def _softmax_from_logprobs(logp, T):
    z = logp / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def apply_venn_abers_ovr(calibrators, P):
    """Vectorized one-vs-rest Venn-Abers transform from stored per-class
    isotonic-regression breakpoints (mirrors venn_abers.venn_abers.calc_probs,
    numpy-only so no VennAbers objects are needed at inference time).

    `calibrators`: {class_idx: (p0, p1, c)} as fit by fit_venn_abers_ovr /
    DNN/fit_calibration.py. `P`: raw ensemble probabilities [n, C]. Returns
    renormalized calibrated probabilities [n, C] (rows sum to 1).
    """
    n, n_classes = P.shape
    p1_out = np.empty((n, n_classes), dtype=np.float64)
    for c in range(n_classes):
        p0, p1, cpts = calibrators[c]
        out = P[:, c]
        p0_at = p0[np.searchsorted(cpts, out, "right"), 1]
        p1_at = p1[np.searchsorted(cpts, out, "left"), 1]
        p1_out[:, c] = p1_at / (1.0 - p0_at + p1_at)
    total = p1_out.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return (p1_out / total).astype(np.float32)


def fit_venn_abers_ovr(P_cal, y_cal):
    """One binary VennAbers calibrator per class (one-vs-rest fit).

    Returns {class_idx: (p0, p1, c)} — the internal isotonic-regression
    breakpoint arrays `apply_venn_abers_ovr` needs (see
    venn_abers.venn_abers.calc_probs for the reference binary implementation
    these mirror).
    """
    import venn_abers as va
    n_classes = P_cal.shape[1]
    calibrators = {}
    for c in range(n_classes):
        p_bin = np.stack([1.0 - P_cal[:, c], P_cal[:, c]], axis=1)
        y_bin = (y_cal == c).astype(int)
        v = va.VennAbers()
        v.fit(p_bin, y_bin)
        calibrators[c] = (v.p0_.copy(), v.p1_.copy(), v.c_.copy())
    return calibrators


def temperature_scale(P_cal, y_cal, P_test):
    """Fit scalar T minimizing NLL on the cal set (grid + local refine). We treat
    log(P) as logits (P are already-averaged ensemble softmaxes). Returns
    (T, P_test_scaled, ece_before, ece_after)."""
    eps = 1e-12
    logp_cal = np.log(np.clip(P_cal, eps, 1.0))
    logp_test = np.log(np.clip(P_test, eps, 1.0))

    def nll(T):
        Q = _softmax_from_logprobs(logp_cal, T)
        return -np.log(np.clip(Q[np.arange(len(y_cal)), y_cal], eps, 1.0)).mean()

    Ts = np.linspace(0.5, 5.0, 46)
    best_T = min(Ts, key=nll)
    # local refine
    for _ in range(20):
        for step in (0.05, 0.01):
            for cand in (best_T - step, best_T + step):
                if 0.2 < cand < 8 and nll(cand) < nll(best_T):
                    best_T = cand
    P_test_scaled = _softmax_from_logprobs(logp_test, best_T)
    return best_T, P_test_scaled


def ece(P, y, n_bins=15):
    """Expected Calibration Error on max-prob confidence."""
    conf = P.max(1)
    pred = P.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def entropy(P):
    eps = 1e-12
    return -(P * np.log(np.clip(P, eps, 1.0))).sum(1)

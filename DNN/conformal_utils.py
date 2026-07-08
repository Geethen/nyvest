"""APS (Adaptive Prediction Sets) conformal helpers for DNN pseudo-labelling.

Direct port of the APS math in common_ground/scripts/llto_schemeB_allyears.py so
the gate behaves identically — the only difference is the probabilities come from
the DNN ensemble instead of CatBoost. Workflow:

  1. Get OOS calibration probs P_cc for held-out stable rows (true labels y_cc).
  2. cal_scores = APS score at the true label; tau = conformal quantile(alpha).
  3. For unstable rows, build the full APS score matrix and keep rows whose
     prediction set is a SINGLETON (exactly one class with score <= tau). Those
     get that class as a pseudo-label.

A singleton set at level alpha means the model is confident enough that the
1-alpha coverage set collapsed to one class — the standard CC-APS pseudo-label
criterion used by the reference pipeline.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q, method="higher"))


def _ranks_sorted(probs):
    n, C = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(1, C + 1)[None, :]
    return ranks, order, sorted_p


def aps_cal_scores(probs, y, u):
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = len(y)
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    s = cum_at + u[:, None] * probs
    return s[np.arange(n), y]


def aps_score_matrix(probs, u):
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = probs.shape[0]
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    return cum_at + u[:, None] * probs


def pseudo_label_singletons(P_unstable, P_cc, y_cc, alpha, seed):
    """Return (keep_mask, pseudo_labels, tau, pct_kept).

    P_cc/y_cc: OOS calibration probs + true labels (stable held-out).
    P_unstable: DNN probs for unstable rows. Singleton APS sets -> pseudo-label.
    """
    rng = np.random.default_rng(seed)
    u_cc = rng.uniform(size=len(y_cc))
    cal = aps_cal_scores(P_cc, y_cc, u_cc)
    tau = conformal_quantile(cal, alpha)
    u_u = rng.uniform(size=len(P_unstable))
    s_mat = aps_score_matrix(P_unstable, u_u)
    sets = s_mat <= tau
    singleton = sets.sum(axis=1) == 1
    pseudo = np.where(singleton, sets.argmax(axis=1), -1)
    return singleton, pseudo, float(tau), float(singleton.mean())

"""Metric suite for ar3: accuracy, calibration, confidence, temporal consistency.

Every function takes ensemble-averaged PROBABILITIES (not logits) because that is
what the deployed model emits and what `fit_calibration.py` calibrates. All of
them are per fold; the harness pairs them against the baseline fold by fold.

Direction convention (used by the scorecard): `HIGHER_BETTER[m]` is True when a
larger value is an improvement. Conformal COVERAGE is special. It is a
guarantee, not an objective, so it is scored as |coverage − (1−α)| (lower is
better) under the name `conf_cov_gap`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

EPS = 1e-12
ALPHA = 0.1

HIGHER_BETTER = {
    "f1": True, "oa": True, "nll": False, "brier": False, "ece": False,
    "nll_ts": False, "ece_ts": False, "conf_cov_gap": False, "conf_size": False,
    "conf_single": True, "conf_cov_gap_mondrian": False, "conf_size_mondrian": False,
    "flip_1720": False, "flip_1824": False, "flip_1824_conf": False,
    "hard_f1": True, "hard_oa": True, "hard_flip_1824": False,
    "chg_sens": True, "chg_fa": False,
}


# ------------------------------------------------------------------ accuracy
def macro_f1(y, pred, n_classes):
    return float(f1_score(y, pred, average="macro", labels=np.arange(n_classes),
                          zero_division=0))


def per_class_f1(y, pred, n_classes):
    return f1_score(y, pred, labels=np.arange(n_classes), average=None, zero_division=0)


# --------------------------------------------------------------- calibration
def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0)).mean())


def brier(P, y):
    """Multi-class Brier: mean over rows of sum_c (p_c − 1[y=c])²."""
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(((P - Y) ** 2).sum(1).mean())


def ece(P, y, n_bins=15):
    """Top-label ECE, equal-width confidence bins."""
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def temp_scale(P, T):
    z = np.log(np.clip(P, EPS, 1.0)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(P_cal, y_cal, grid=None):
    """1-D NLL minimisation over T by grid + golden refinement (no torch)."""
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 60)) if grid is None else grid
    losses = [nll(temp_scale(P_cal, t), y_cal) for t in grid]
    i = int(np.argmin(losses))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(30):
        c, d = b - g * (b - a), a + g * (b - a)
        if nll(temp_scale(P_cal, c), y_cal) < nll(temp_scale(P_cal, d), y_cal):
            b = d
        else:
            a = c
    return float((a + b) / 2)


# ---------------------------------------------------------------- conformal
def _qhat(scores, alpha):
    n = len(scores)
    if n == 0:
        return 1.0
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def conformal_lac(P_cal, y_cal, P_te, y_te, alpha=ALPHA, mondrian=False):
    """Split-conformal LAC (score 1 − p_y). Mondrian = one threshold per TRUE
    class (class-conditional coverage), applied to each candidate label c.
    Returns (coverage, mean set size, singleton rate, sets)."""
    n_classes = P_cal.shape[1]
    s_cal = 1.0 - P_cal[np.arange(len(y_cal)), y_cal]
    if mondrian:
        q = np.array([_qhat(s_cal[y_cal == c], alpha) for c in range(n_classes)])
        sets = (1.0 - P_te) <= q[None, :]
    else:
        sets = (1.0 - P_te) <= _qhat(s_cal, alpha)
    size = sets.sum(1)
    cov = sets[np.arange(len(y_te)), y_te].mean()
    return float(cov), float(size.mean()), float((size == 1).mean()), sets


# ----------------------------------------------------------------- temporal
def flip_rates(pred, meta, singleton=None):
    """Year-to-year prediction flips on locations whose label is constant.

    meta: DataFrame with lon, lat, year aligned to `pred`. FSCS points are
    CCDC-stable over (2017, 2020], so any flip between consecutive years in
    2017-2020 is an error by construction. 2018↔2024 is the deployment pair
    and can include real change, but only a small share of it on stable
    points. `flip_1824_conf` counts only pairs where BOTH years have a
    singleton conformal set, the analogue of the map's confident-change screen.
    """
    d = pd.DataFrame({"lon": meta["lon"].values, "lat": meta["lat"].values,
                      "year": meta["year"].values.astype(int), "p": pred})
    if singleton is not None:
        d["s"] = singleton.astype(bool)
    wide = d.pivot_table(index=["lon", "lat"], columns="year", values="p",
                         aggfunc="first")
    out = {}
    pairs = [(2017, 2018), (2018, 2019), (2019, 2020)]
    flips, n = 0, 0
    for a, b in pairs:
        if a in wide and b in wide:
            m = wide[a].notna() & wide[b].notna()
            flips += int((wide.loc[m, a] != wide.loc[m, b]).sum())
            n += int(m.sum())
    out["flip_1720"] = flips / n if n else np.nan
    if 2018 in wide and 2024 in wide:
        m = wide[2018].notna() & wide[2024].notna()
        out["flip_1824"] = float((wide.loc[m, 2018] != wide.loc[m, 2024]).mean())
        if singleton is not None:
            sw = d.pivot_table(index=["lon", "lat"], columns="year", values="s",
                               aggfunc="first")
            both = m & sw[2018].fillna(False).astype(bool) & sw[2024].fillna(False).astype(bool)
            out["flip_1824_conf"] = float(((wide[2018] != wide[2024]) & both)[m].mean())
    return out


# -------------------------------------------------------------------- suite
def fold_metrics(P_te, y_te, meta_te, P_cal, y_cal, n_classes, hard=None):
    """Everything for one fold. `hard` = optional dict(P, y, meta) for the
    consensus hard-area population (test cells only)."""
    pred = P_te.argmax(1)
    m = {"f1": macro_f1(y_te, pred, n_classes),
         "oa": float((pred == y_te).mean()),
         "nll": nll(P_te, y_te), "brier": brier(P_te, y_te), "ece": ece(P_te, y_te)}
    T = fit_temperature(P_cal, y_cal)
    Pt = temp_scale(P_te, T)
    m.update({"T": T, "nll_ts": nll(Pt, y_te), "ece_ts": ece(Pt, y_te)})
    cov, size, single, sets = conformal_lac(P_cal, y_cal, P_te, y_te)
    m.update({"conf_cov": cov, "conf_cov_gap": abs(cov - (1 - ALPHA)),
              "conf_size": size, "conf_single": single})
    covm, sizem, _, _ = conformal_lac(P_cal, y_cal, P_te, y_te, mondrian=True)
    m.update({"conf_cov_mondrian": covm,
              "conf_cov_gap_mondrian": abs(covm - (1 - ALPHA)),
              "conf_size_mondrian": sizem})
    m.update(flip_rates(pred, meta_te, singleton=sets.sum(1) == 1))
    pc = per_class_f1(y_te, pred, n_classes)
    if hard is not None and len(hard["y"]):
        hp = hard["P"].argmax(1)
        m["hard_f1"] = macro_f1(hard["y"], hp, n_classes)
        m["hard_oa"] = float((hp == hard["y"]).mean())
        m["hard_n"] = int(len(hard["y"]))
        if "pred_all" in hard:  # predictions on ALL hard rows (labelled or not)
            m["hard_flip_1824"] = flip_rates(hard["pred_all"], hard["meta_all"]).get(
                "flip_1824", np.nan)
            ma = hard["meta_all"]
            if "ext_change" in ma:
                d = pd.DataFrame({"pid": ma["pid"].values, "year": ma["year"].values,
                                  "p": hard["pred_all"]})
                w = d.pivot_table(index="pid", columns="year", values="p", aggfunc="first")
                flag = ma.drop_duplicates("pid").set_index("pid")
                if 2018 in w and 2024 in w:
                    both = w[2018].notna() & w[2024].notna()
                    flip = (w[2018] != w[2024])[both]
                    for key, col in (("chg_sens", "ext_change"), ("chg_fa", "ext_stable")):
                        sel = flag.loc[flip.index, col].fillna(False).astype(bool).values
                        m[key] = float(flip[sel].mean()) if sel.any() else np.nan
                        m[key + "_n"] = int(sel.sum())
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in m.items()}, pc

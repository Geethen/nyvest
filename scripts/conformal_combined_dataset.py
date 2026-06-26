"""Conformal prediction on a combined Megan + FSCS dataset harmonised to level-1.

Class harmonisation
-------------------
Both Megan (22-class fallbck codes) and FSCS Scheme B (10-class) are mapped to
Megan's level-1 taxonomy (11 classes). Coastal (1) and Marine (8) are dropped
because FSCS has no equivalent, leaving 9 shared classes:

  2=Cropland  3=Forest  4=Grassland  5=Heathland/scrub  6=Inland wetlands
  7=Lakes/water  9=Rivers/canals  10=Settlements  11=Sparsely vegetated

Feature alignment
-----------------
Megan: embdd_1..embd_64 → f00..f63  (LiDAR dropped)
FSCS:  A00..A63          → f00..f63

CP methods evaluated
--------------------
APS, RAPS (lam=0.01, k_reg=1), SAPS (lam=0.2), RANK, THR, Mondrian-RAPS

Usage
-----
    ~/myprojects/recover/.venv/bin/python scripts/conformal_combined_dataset.py
    ~/myprojects/recover/.venv/bin/python scripts/conformal_combined_dataset.py \\
        --device cpu --alphas 0.05,0.10,0.20
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FSCS_PARQUET = REPO_ROOT / "data" / "grunnkart_nyvest_fscs_alphaearth.parquet"
REPORTS_DIR = REPO_ROOT / "reports"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from benchmark_tabular import load_split, resolve_device  # noqa: E402

# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------

# Megan fallbck code → level-1 code
MEGAN_TO_LEVEL1: dict[int, int] = {
    2: 3,   # Broadleaved deciduous forest → Forest
    5: 3,   # Coniferous forests → Forest
    9: 3,   # Forest and woodlands → Forest
    16: 3,  # Mixed forests → Forest
    10: 4,  # Grassland → Grassland
    7: 2,   # Cropland → Cropland
    15: 6,  # Mires, bogs and fens → Inland wetlands
    19: 5,  # Scrub and heathland → Heathland and shrub
    21: 11, # Sparsely vegetated ecosystems → Sparsely vegetated
    13: 7,  # Lakes and ponds → Lakes and reservoirs
    18: 9,  # Rivers → Rivers and canals
    1: 7,   # Artificial reservoirs → Lakes and reservoirs
    3: 9,   # Canals, ditches and drains → Rivers and canals
    4: 1,   # Coastal beaches, dunes and wetlands → Coastal (DROPPED)
    14: 8,  # Marine ecosystems → Marine ecosystems (DROPPED)
    6: 10,  # Continuous settlement area → Settlements
    8: 10,  # Discontinuous settlement area → Settlements
    17: 10, # Other artificial areas → Settlements
    20: 10, # Settlements and other artificial areas → Settlements
    22: 10, # Urban greenspace → Settlements
    11: 11, # Ice sheets, glaciers and perennial snowfields → Sparsely vegetated
    12: 10, # Infrastructure → Settlements
}

# FSCS Scheme B class → level-1 code  (after merging 1→2 and 9→8)
FSCS_TO_LEVEL1: dict[int, int] = {
    2: 11,  # bare/rock/sand (merged 1+2) → Sparsely vegetated
    3: 2,   # cropland → Cropland
    4: 3,   # forest → Forest
    5: 4,   # grassland → Grassland
    6: 5,   # scrub → Heathland and shrub
    7: 6,   # wetland → Inland wetlands
    8: 7,   # water (merged 9+8) → Lakes and reservoirs
    10: 10, # settlement → Settlements
    11: 10, # infrastructure → Settlements
    12: 11, # snow/ice → Sparsely vegetated
}

# Shared level-1 codes and names (Coastal=1, Marine=8 dropped)
LEVEL1_CLASSES: dict[int, str] = {
    2: "Cropland",
    3: "Forest",
    4: "Grassland",
    5: "Heathland and shrub",
    6: "Inland wetlands",
    7: "Lakes and reservoirs",
    9: "Rivers and canals",
    10: "Settlements",
    11: "Sparsely vegetated",
}

# Ordered list of shared level-1 codes
SHARED_CODES = sorted(LEVEL1_CLASSES.keys())  # [2,3,4,5,6,7,9,10,11]
N_CLASSES = len(SHARED_CODES)

# Map level-1 code → contiguous 0-indexed label used by the model
LEVEL1_TO_IDX: dict[int, int] = {code: i for i, code in enumerate(SHARED_CODES)}
IDX_TO_NAME: dict[int, str] = {i: LEVEL1_CLASSES[code] for code, i in LEVEL1_TO_IDX.items()}

# Megan feature columns (AlphaEarth bands only, LiDAR dropped)
_MEGAN_AE_COLS = [f"embdd_{i}" for i in range(1, 10)] + [f"embd_{i}" for i in range(10, 65)]
# FSCS feature columns
_FSCS_AE_COLS = [f"A{i:02d}" for i in range(64)]
# Canonical feature names
CANONICAL_COLS = [f"f{i:02d}" for i in range(64)]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _apply_megan_mapping(y_raw: pd.Series) -> pd.Series:
    """Map Megan fallbck codes → level-1 codes; NaN for unmapped (Coastal/Marine)."""
    return y_raw.map(MEGAN_TO_LEVEL1)


def _apply_fscs_merge_and_map(y_raw: pd.Series) -> pd.Series:
    """Apply FSCS Scheme B merges then map to level-1."""
    y = y_raw.copy()
    y[y == 1] = 2   # bare → merged bare/rock/sand
    y[y == 9] = 8   # marine → merged water
    return y.map(FSCS_TO_LEVEL1)


def _rename_features(X: pd.DataFrame, src_cols: list[str]) -> pd.DataFrame:
    """Select src_cols from X and rename to canonical f00..f63."""
    missing = [c for c in src_cols if c not in X.columns]
    if missing:
        raise KeyError(f"Feature columns not found in dataframe: {missing[:5]} ...")
    return X[src_cols].rename(columns=dict(zip(src_cols, CANONICAL_COLS))).astype(np.float32)


def load_megan_splits() -> tuple[
    pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray
]:
    """Load Megan train/val/test, apply level-1 mapping, drop Coastal & Marine rows."""
    splits = {}
    for name in ("train", "val", "test"):
        X_raw, y_raw = load_split(name)
        y_l1 = _apply_megan_mapping(y_raw)
        # Drop rows whose level-1 code is not in the shared set (Coastal=1, Marine=8)
        mask = y_l1.isin(SHARED_CODES)
        X_raw = X_raw.loc[mask].reset_index(drop=True)
        y_l1 = y_l1[mask].reset_index(drop=True)
        # Rename features
        ae_cols = [c for c in _MEGAN_AE_COLS if c in X_raw.columns]
        if len(ae_cols) != 64:
            raise RuntimeError(
                f"Expected 64 AlphaEarth columns in Megan {name} split, "
                f"found {len(ae_cols)}: {ae_cols}"
            )
        X = _rename_features(X_raw, ae_cols)
        y_idx = y_l1.map(LEVEL1_TO_IDX).to_numpy(dtype=np.int32)
        splits[name] = (X, y_idx)
    return (
        splits["train"][0], splits["train"][1],
        splits["val"][0],   splits["val"][1],
        splits["test"][0],  splits["test"][1],
    )


def load_fscs_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load FSCS parquet, apply class merges + level-1 mapping."""
    import duckdb
    df = duckdb.sql(f"SELECT * FROM '{FSCS_PARQUET}'").df()
    y_raw = df["class"].astype(int)
    y_l1 = _apply_fscs_merge_and_map(y_raw)
    mask = y_l1.notna()
    df = df.loc[mask].reset_index(drop=True)
    y_l1 = y_l1[mask].reset_index(drop=True)
    fscs_cols = [c for c in _FSCS_AE_COLS if c in df.columns]
    if len(fscs_cols) != 64:
        raise RuntimeError(
            f"Expected 64 AlphaEarth columns in FSCS parquet, found {len(fscs_cols)}"
        )
    X = _rename_features(df[fscs_cols], fscs_cols)
    y_idx = y_l1.map(LEVEL1_TO_IDX).to_numpy(dtype=np.int32)
    return X, y_idx


def fscs_stratified_splits(
    X: pd.DataFrame, y: np.ndarray, seed: int = 0
) -> tuple[
    pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray
]:
    """70/15/15 stratified random split of FSCS data."""
    sss_tv = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    idx_train, idx_tmp = next(sss_tv.split(X, y))
    X_train = X.iloc[idx_train].reset_index(drop=True)
    y_train = y[idx_train]
    X_tmp = X.iloc[idx_tmp].reset_index(drop=True)
    y_tmp = y[idx_tmp]

    sss_vt = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    idx_val, idx_test = next(sss_vt.split(X_tmp, y_tmp))
    X_val = X_tmp.iloc[idx_val].reset_index(drop=True)
    y_val = y_tmp[idx_val]
    X_test = X_tmp.iloc[idx_test].reset_index(drop=True)
    y_test = y_tmp[idx_test]
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_combined_splits(
    megan_splits: tuple,
    fscs_splits: tuple,
) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray,   # X_train, y_train, src_train
    pd.DataFrame, np.ndarray, np.ndarray,   # X_val,   y_val,   src_val
    pd.DataFrame, np.ndarray, np.ndarray,   # X_test,  y_test,  src_test
]:
    """Concatenate Megan and FSCS splits; src=0 for Megan, src=1 for FSCS."""
    (
        mX_tr, my_tr, mX_va, my_va, mX_te, my_te,
        fX_tr, fy_tr, fX_va, fy_va, fX_te, fy_te,
    ) = (*megan_splits, *fscs_splits)

    def concat(mX, my, fX, fy):
        X = pd.concat([mX, fX], ignore_index=True)
        y = np.concatenate([my, fy]).astype(np.int32)
        src = np.array([0] * len(mX) + [1] * len(fX), dtype=np.int8)
        return X, y, src

    X_tr, y_tr, s_tr = concat(mX_tr, my_tr, fX_tr, fy_tr)
    X_va, y_va, s_va = concat(mX_va, my_va, fX_va, fy_va)
    X_te, y_te, s_te = concat(mX_te, my_te, fX_te, fy_te)
    return X_tr, y_tr, s_tr, X_va, y_va, s_va, X_te, y_te, s_te


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, n_estimators: int = 500,
                  device: str = "cpu", seed: int = 0):
    from xgboost import XGBClassifier
    print(f"Training XGBoost (n_estimators={n_estimators}, device={device}) "
          f"on {len(X_train):,} rows, {N_CLASSES} classes ...")
    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        device=device,
        random_state=seed,
        eval_metric="mlogloss",
        num_class=N_CLASSES,
    )
    model.fit(X_train, y_train)
    print(f"  trained in {time.perf_counter() - t0:.1f}s")
    return model


def get_probabilities(model, X) -> np.ndarray:
    probs = model.predict_proba(X)
    # Ensure shape (n, N_CLASSES) — XGBoost may omit absent classes
    if probs.shape[1] < N_CLASSES:
        full = np.zeros((len(X), N_CLASSES), dtype=np.float64)
        for i, cls in enumerate(model.classes_):
            full[:, cls] = probs[:, i]
        probs = full
    return probs.astype(np.float64)


# ---------------------------------------------------------------------------
# Conformal score functions (all implemented from scratch)
# ---------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """ceil((n+1)*(1-alpha))/n -quantile of cal scores, capped at 1.0."""
    n = len(scores)
    q_level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def _ranks_and_sorted(probs: np.ndarray):
    """Return (ranks, order, sorted_probs).

    ranks[i,k]      : 1-indexed rank of class k for sample i (rank 1 = max prob)
    order[i,:]      : class indices in descending-prob order
    sorted_probs[i,:]: probs sorted descending
    """
    n, C = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    ranks = np.empty_like(order)
    row = np.arange(n)[:, None]
    ranks[row, order] = np.arange(1, C + 1)[None, :]
    return ranks, order, sorted_probs


# --- THR ---

def thr_cal_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """THR calibration score = p_{y_i}(x_i)  (score at true label)."""
    return probs[np.arange(len(y)), y]


def thr_prediction_sets(probs: np.ndarray, tau: float) -> np.ndarray:
    """THR prediction set: {k : p_k >= tau}."""
    return probs >= tau


# --- APS (and RAPS with lam_reg > 0) ---

def _aps_all_scores(probs: np.ndarray, u: np.ndarray,
                    lam_reg: float = 0.0, k_reg: int = 1) -> np.ndarray:
    """APS / RAPS score matrix (n, C).

    V(p, k) = sum_{j=1}^{r-1} p_(j)  +  u * p_(r)  [+  lam * max(0, r - k_reg)]
    """
    ranks, order, sorted_probs = _ranks_and_sorted(probs)
    n, C = probs.shape
    cumsum_sorted = np.cumsum(sorted_probs, axis=1)          # through rank r (inclusive)
    cumsum_before = np.concatenate(
        [np.zeros((n, 1)), cumsum_sorted[:, :-1]], axis=1
    )                                                         # through rank r-1
    cum_at_class = np.take_along_axis(cumsum_before, ranks - 1, axis=1)
    scores = cum_at_class + u[:, None] * probs
    if lam_reg > 0:
        scores = scores + lam_reg * np.maximum(0, ranks - k_reg)
    return scores


def aps_cal_scores(probs: np.ndarray, y: np.ndarray, u: np.ndarray,
                   lam_reg: float = 0.0, k_reg: int = 1) -> np.ndarray:
    all_s = _aps_all_scores(probs, u, lam_reg=lam_reg, k_reg=k_reg)
    return all_s[np.arange(len(y)), y]


def aps_prediction_sets(probs: np.ndarray, u: np.ndarray, tau: float,
                        lam_reg: float = 0.0, k_reg: int = 1) -> np.ndarray:
    all_s = _aps_all_scores(probs, u, lam_reg=lam_reg, k_reg=k_reg)
    return all_s <= tau


# --- SAPS ---

def _saps_all_scores(probs: np.ndarray, u: np.ndarray, lam: float) -> np.ndarray:
    """SAPS score matrix (n, C).

    V(p, k) = p_(1) * u                if rank == 1
            = p_(1) + (rank-2+u) * lam  if rank >= 2
    """
    ranks, order, sorted_probs = _ranks_and_sorted(probs)
    n = probs.shape[0]
    top1 = sorted_probs[:, 0:1]   # (n, 1)
    scores = np.where(
        ranks == 1,
        top1 * u[:, None],
        top1 + (ranks - 2 + u[:, None]) * lam,
    )
    return scores


def saps_cal_scores(probs: np.ndarray, y: np.ndarray, u: np.ndarray,
                    lam: float) -> np.ndarray:
    all_s = _saps_all_scores(probs, u, lam)
    return all_s[np.arange(len(y)), y]


def saps_prediction_sets(probs: np.ndarray, u: np.ndarray, tau: float,
                         lam: float) -> np.ndarray:
    all_s = _saps_all_scores(probs, u, lam)
    return all_s <= tau


# --- RANK (Liu et al. 2025, Algorithm 1) ---

def rank_prediction_sets(
    probs_cal: np.ndarray, y_cal: np.ndarray,
    probs_test: np.ndarray, alpha: float,
) -> np.ndarray:
    """Two-stage RANK conformal sets."""
    n, C = probs_cal.shape
    ranks_cal, _, _ = _ranks_and_sorted(probs_cal)
    r_true = ranks_cal[np.arange(n), y_cal]

    kth = int(max(1, min(int(np.floor((n + 1) * alpha)), n)))
    r_sorted_desc = np.sort(r_true)[::-1]
    r_star = int(r_sorted_desc[kth - 1])

    count_below = int(np.sum(r_true <= r_star - 1))
    count_at = int(np.sum(r_true == r_star))
    if count_at == 0:
        use_rstar_threshold = -np.inf
    else:
        p_prop = float(np.clip(
            (n - int(np.floor((n + 1) * alpha)) - count_below) / count_at, 0.0, 1.0
        ))
        sorted_probs_cal = np.sort(probs_cal, axis=1)[:, ::-1]
        rstar_probs = sorted_probs_cal[:, r_star - 1]
        k_pick = int(np.ceil(n * p_prop))
        if k_pick <= 0:
            use_rstar_threshold = np.inf
        elif k_pick > len(rstar_probs):
            use_rstar_threshold = -np.inf
        else:
            use_rstar_threshold = float(np.sort(rstar_probs)[::-1][k_pick - 1])

    nt = probs_test.shape[0]
    ranks_test, _, sorted_test = _ranks_and_sorted(probs_test)
    rstar_prob_test = sorted_test[:, r_star - 1] if r_star >= 1 else np.zeros(nt)
    include_rstar = rstar_prob_test >= use_rstar_threshold
    size_per_sample = np.where(include_rstar, r_star, max(r_star - 1, 0))
    sets = ranks_test <= size_per_sample[:, None]
    return sets


# --- Mondrian RAPS ---

def mondrian_raps_prediction_sets(
    probs_cal: np.ndarray, y_cal: np.ndarray,
    probs_test: np.ndarray, alpha: float,
    lam_reg: float = 0.01, k_reg: int = 1,
    min_cal_class: int = 5,
    u_cal: np.ndarray | None = None,
    u_test: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mondrian RAPS: stratify calibration by argmax(probs), one tau per class.

    Falls back to the global tau for classes with < min_cal_class cal samples.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if u_cal is None:
        u_cal = rng.uniform(size=len(y_cal))
    if u_test is None:
        u_test = rng.uniform(size=len(probs_test))

    C = probs_cal.shape[1]
    pred_class_cal = np.argmax(probs_cal, axis=1)
    pred_class_test = np.argmax(probs_test, axis=1)

    # Global fallback tau
    cal_scores_global = aps_cal_scores(probs_cal, y_cal, u_cal, lam_reg=lam_reg, k_reg=k_reg)
    tau_global = conformal_quantile(cal_scores_global, alpha)

    # Per-class taus
    tau_per_class = np.full(C, tau_global)
    for c in range(C):
        mask = pred_class_cal == c
        if mask.sum() >= min_cal_class:
            scores_c = aps_cal_scores(
                probs_cal[mask], y_cal[mask], u_cal[mask],
                lam_reg=lam_reg, k_reg=k_reg,
            )
            tau_per_class[c] = conformal_quantile(scores_c, alpha)

    all_test_scores = _aps_all_scores(probs_test, u_test, lam_reg=lam_reg, k_reg=k_reg)
    tau_test = tau_per_class[pred_class_test]  # (n_test,)
    sets = all_test_scores <= tau_test[:, None]
    return sets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def set_metrics(sets: np.ndarray, y: np.ndarray) -> dict:
    """Compute coverage, set-size, singleton/empty fraction."""
    set_sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "pct_singleton": float((set_sizes == 1).mean()),
        "pct_empty": float((set_sizes == 0).mean()),
    }


def per_class_coverage(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Per-class empirical coverage (fraction of test samples whose set contains y)."""
    covered = sets[np.arange(len(y)), y]
    result = {}
    for idx, name in IDX_TO_NAME.items():
        mask = y == idx
        if mask.sum() > 0:
            result[name] = float(covered[mask].mean())
        else:
            result[name] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all_methods(
    probs_cal: np.ndarray, y_cal: np.ndarray,
    probs_test: np.ndarray, y_test: np.ndarray,
    src_test: np.ndarray,
    alphas: list[float],
    seed: int = 0,
    saps_lam: float = 0.2,
    raps_lam: float = 0.01,
    raps_kreg: int = 1,
) -> dict:
    rng = np.random.default_rng(seed)
    u_cal = rng.uniform(size=len(y_cal))
    u_test = rng.uniform(size=len(y_test))

    megan_mask = src_test == 0
    fscs_mask = src_test == 1

    results = {}

    for alpha in alphas:
        print(f"\n--- alpha={alpha}  (target coverage {1-alpha:.2f}) ---")

        # ---- THR ----
        key = f"THR_alpha{alpha}"
        cal_s = thr_cal_scores(probs_cal, y_cal)
        tau = conformal_quantile(cal_s, alpha)
        sets = thr_prediction_sets(probs_test, tau)
        m = set_metrics(sets, y_test)
        m["alpha"] = alpha
        m["per_class_coverage"] = per_class_coverage(sets, y_test)
        m["megan_coverage"] = float(sets[megan_mask][np.arange(megan_mask.sum()), y_test[megan_mask]].mean()) if megan_mask.any() else float("nan")
        m["fscs_coverage"] = float(sets[fscs_mask][np.arange(fscs_mask.sum()), y_test[fscs_mask]].mean()) if fscs_mask.any() else float("nan")
        m["megan_avg_size"] = float(sets[megan_mask].sum(axis=1).mean()) if megan_mask.any() else float("nan")
        m["fscs_avg_size"] = float(sets[fscs_mask].sum(axis=1).mean()) if fscs_mask.any() else float("nan")
        results[key] = m
        print(f"  THR   cov={m['coverage']:.3f}  avg_size={m['avg_set_size']:.2f}  "
              f"singleton={m['pct_singleton']:.2f}  empty={m['pct_empty']:.3f}")

        # ---- APS ----
        key = f"APS_alpha{alpha}"
        cal_s = aps_cal_scores(probs_cal, y_cal, u_cal)
        tau = conformal_quantile(cal_s, alpha)
        sets = aps_prediction_sets(probs_test, u_test, tau)
        m = set_metrics(sets, y_test)
        m["alpha"] = alpha
        m["per_class_coverage"] = per_class_coverage(sets, y_test)
        m["megan_coverage"] = float(sets[megan_mask][np.arange(megan_mask.sum()), y_test[megan_mask]].mean()) if megan_mask.any() else float("nan")
        m["fscs_coverage"] = float(sets[fscs_mask][np.arange(fscs_mask.sum()), y_test[fscs_mask]].mean()) if fscs_mask.any() else float("nan")
        m["megan_avg_size"] = float(sets[megan_mask].sum(axis=1).mean()) if megan_mask.any() else float("nan")
        m["fscs_avg_size"] = float(sets[fscs_mask].sum(axis=1).mean()) if fscs_mask.any() else float("nan")
        results[key] = m
        print(f"  APS   cov={m['coverage']:.3f}  avg_size={m['avg_set_size']:.2f}  "
              f"singleton={m['pct_singleton']:.2f}  empty={m['pct_empty']:.3f}")

        # ---- RAPS ----
        key = f"RAPS_alpha{alpha}"
        cal_s = aps_cal_scores(probs_cal, y_cal, u_cal, lam_reg=raps_lam, k_reg=raps_kreg)
        tau = conformal_quantile(cal_s, alpha)
        sets = aps_prediction_sets(probs_test, u_test, tau, lam_reg=raps_lam, k_reg=raps_kreg)
        m = set_metrics(sets, y_test)
        m["alpha"] = alpha
        m["per_class_coverage"] = per_class_coverage(sets, y_test)
        m["megan_coverage"] = float(sets[megan_mask][np.arange(megan_mask.sum()), y_test[megan_mask]].mean()) if megan_mask.any() else float("nan")
        m["fscs_coverage"] = float(sets[fscs_mask][np.arange(fscs_mask.sum()), y_test[fscs_mask]].mean()) if fscs_mask.any() else float("nan")
        m["megan_avg_size"] = float(sets[megan_mask].sum(axis=1).mean()) if megan_mask.any() else float("nan")
        m["fscs_avg_size"] = float(sets[fscs_mask].sum(axis=1).mean()) if fscs_mask.any() else float("nan")
        results[key] = m
        print(f"  RAPS  cov={m['coverage']:.3f}  avg_size={m['avg_set_size']:.2f}  "
              f"singleton={m['pct_singleton']:.2f}  empty={m['pct_empty']:.3f}  "
              f"lam={raps_lam}  k_reg={raps_kreg}")

        # ---- SAPS ----
        key = f"SAPS_alpha{alpha}"
        cal_s = saps_cal_scores(probs_cal, y_cal, u_cal, saps_lam)
        tau = conformal_quantile(cal_s, alpha)
        sets = saps_prediction_sets(probs_test, u_test, tau, saps_lam)
        m = set_metrics(sets, y_test)
        m["alpha"] = alpha
        m["per_class_coverage"] = per_class_coverage(sets, y_test)
        m["megan_coverage"] = float(sets[megan_mask][np.arange(megan_mask.sum()), y_test[megan_mask]].mean()) if megan_mask.any() else float("nan")
        m["fscs_coverage"] = float(sets[fscs_mask][np.arange(fscs_mask.sum()), y_test[fscs_mask]].mean()) if fscs_mask.any() else float("nan")
        m["megan_avg_size"] = float(sets[megan_mask].sum(axis=1).mean()) if megan_mask.any() else float("nan")
        m["fscs_avg_size"] = float(sets[fscs_mask].sum(axis=1).mean()) if fscs_mask.any() else float("nan")
        results[key] = m
        print(f"  SAPS  cov={m['coverage']:.3f}  avg_size={m['avg_set_size']:.2f}  "
              f"singleton={m['pct_singleton']:.2f}  empty={m['pct_empty']:.3f}  "
              f"lam={saps_lam}")

        # ---- RANK ----
        key = f"RANK_alpha{alpha}"
        sets = rank_prediction_sets(probs_cal, y_cal, probs_test, alpha)
        m = set_metrics(sets, y_test)
        m["alpha"] = alpha
        m["per_class_coverage"] = per_class_coverage(sets, y_test)
        m["megan_coverage"] = float(sets[megan_mask][np.arange(megan_mask.sum()), y_test[megan_mask]].mean()) if megan_mask.any() else float("nan")
        m["fscs_coverage"] = float(sets[fscs_mask][np.arange(fscs_mask.sum()), y_test[fscs_mask]].mean()) if fscs_mask.any() else float("nan")
        m["megan_avg_size"] = float(sets[megan_mask].sum(axis=1).mean()) if megan_mask.any() else float("nan")
        m["fscs_avg_size"] = float(sets[fscs_mask].sum(axis=1).mean()) if fscs_mask.any() else float("nan")
        results[key] = m
        print(f"  RANK  cov={m['coverage']:.3f}  avg_size={m['avg_set_size']:.2f}  "
              f"singleton={m['pct_singleton']:.2f}  empty={m['pct_empty']:.3f}")

        # ---- Mondrian-RAPS ----
        key = f"MondRaps_alpha{alpha}"
        rng2 = np.random.default_rng(seed)
        u_cal2 = rng2.uniform(size=len(y_cal))
        u_test2 = rng2.uniform(size=len(y_test))
        sets = mondrian_raps_prediction_sets(
            probs_cal, y_cal, probs_test, alpha,
            lam_reg=raps_lam, k_reg=raps_kreg,
            u_cal=u_cal2, u_test=u_test2,
        )
        m = set_metrics(sets, y_test)
        m["alpha"] = alpha
        m["per_class_coverage"] = per_class_coverage(sets, y_test)
        m["megan_coverage"] = float(sets[megan_mask][np.arange(megan_mask.sum()), y_test[megan_mask]].mean()) if megan_mask.any() else float("nan")
        m["fscs_coverage"] = float(sets[fscs_mask][np.arange(fscs_mask.sum()), y_test[fscs_mask]].mean()) if fscs_mask.any() else float("nan")
        m["megan_avg_size"] = float(sets[megan_mask].sum(axis=1).mean()) if megan_mask.any() else float("nan")
        m["fscs_avg_size"] = float(sets[fscs_mask].sum(axis=1).mean()) if fscs_mask.any() else float("nan")
        results[key] = m
        print(f"  MondRaps cov={m['coverage']:.3f}  avg_size={m['avg_set_size']:.2f}  "
              f"singleton={m['pct_singleton']:.2f}  empty={m['pct_empty']:.3f}  "
              f"lam={raps_lam}  k_reg={raps_kreg}")

    return results


# ---------------------------------------------------------------------------
# Dataset summary helpers
# ---------------------------------------------------------------------------

def class_counts_table(y: np.ndarray, label: str) -> dict[str, int]:
    counts = {}
    for idx, name in IDX_TO_NAME.items():
        counts[name] = int((y == idx).sum())
    return counts


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _cov_colour(cov: float, target: float) -> str:
    """Background colour for a coverage cell: green if near/above target, red if far below."""
    if math.isnan(cov):
        return "#cccccc"
    deficit = target - cov
    if deficit <= 0.01:
        return "#c8e6c9"   # green
    elif deficit <= 0.05:
        return "#fff9c4"   # yellow
    else:
        return "#ffcdd2"   # red


def build_html_report(
    results: dict,
    class_names: dict,
    n_train: int, n_val: int, n_test: int,
    megan_counts: dict, fscs_counts: dict, combined_counts: dict,
    megan_n_test: int, fscs_n_test: int,
    alphas: list[float],
) -> str:
    methods = ["THR", "APS", "RAPS", "SAPS", "RANK", "MondRaps"]

    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>Conformal Prediction — Combined Megan + FSCS</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;font-size:13px;}",
        "h1{color:#333;}",
        "h2{color:#555;border-bottom:1px solid #ccc;padding-bottom:4px;}",
        "h3{color:#666;}",
        "table{border-collapse:collapse;margin-bottom:20px;}",
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right;}",
        "th{background:#e8e8e8;text-align:center;}",
        ".left{text-align:left;}",
        ".section{margin-top:30px;}",
        "</style></head><body>",
        "<h1>Conformal Prediction: Combined Megan + FSCS (Level-1 Taxonomy)</h1>",
    ]

    # 1. Dataset summary
    html_parts.append("<div class='section'><h2>1. Dataset Summary</h2>")
    html_parts.append(
        f"<p>Combined dataset: <b>{n_train:,}</b> train / "
        f"<b>{n_val:,}</b> val / <b>{n_test:,}</b> test rows. "
        f"Megan test: {megan_n_test:,} &nbsp; FSCS test: {fscs_n_test:,}. "
        f"Shared classes: {len(class_names)} (Coastal and Marine excluded).</p>"
    )

    html_parts.append("<table>")
    all_names = list(class_names.values())
    html_parts.append(
        "<tr><th class='left'>Class</th><th>Megan (train+val+test)</th>"
        "<th>FSCS (train+val+test)</th><th>Combined (train+val+test)</th></tr>"
    )
    for name in all_names:
        mc = megan_counts.get(name, 0)
        fc = fscs_counts.get(name, 0)
        cc = combined_counts.get(name, 0)
        html_parts.append(
            f"<tr><td class='left'>{name}</td><td>{mc:,}</td><td>{fc:,}</td><td>{cc:,}</td></tr>"
        )
    html_parts.append("</table></div>")

    # 2. Per-method results tables at each alpha
    html_parts.append("<div class='section'><h2>2. CP Method Results by Alpha</h2>")
    for alpha in alphas:
        target_cov = 1 - alpha
        html_parts.append(f"<h3>Alpha = {alpha}  (target coverage ≥ {target_cov:.2f})</h3>")
        html_parts.append("<table>")
        html_parts.append(
            "<tr><th class='left'>Method</th><th>Coverage</th><th>Avg Set Size</th>"
            "<th>Median Set Size</th><th>% Singleton</th><th>% Empty</th>"
            "<th>Megan Cov</th><th>FSCS Cov</th>"
            "<th>Megan Avg Size</th><th>FSCS Avg Size</th></tr>"
        )
        for method in methods:
            key = f"{method}_alpha{alpha}"
            if key not in results:
                continue
            r = results[key]
            cov_col = _cov_colour(r["coverage"], target_cov)
            mc_col = _cov_colour(r.get("megan_coverage", float("nan")), target_cov)
            fc_col = _cov_colour(r.get("fscs_coverage", float("nan")), target_cov)

            def _fmt(v, decimals=3):
                return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.{decimals}f}"

            html_parts.append(
                f"<tr>"
                f"<td class='left'>{method}</td>"
                f"<td style='background:{cov_col}'>{_fmt(r['coverage'])}</td>"
                f"<td>{_fmt(r['avg_set_size'], 2)}</td>"
                f"<td>{_fmt(r['median_set_size'], 1)}</td>"
                f"<td>{_fmt(r['pct_singleton'])}</td>"
                f"<td>{_fmt(r['pct_empty'])}</td>"
                f"<td style='background:{mc_col}'>{_fmt(r.get('megan_coverage', float('nan')))}</td>"
                f"<td style='background:{fc_col}'>{_fmt(r.get('fscs_coverage', float('nan')))}</td>"
                f"<td>{_fmt(r.get('megan_avg_size', float('nan')), 2)}</td>"
                f"<td>{_fmt(r.get('fscs_avg_size', float('nan')), 2)}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")
    html_parts.append("</div>")

    # 3. Per-class coverage heatmap
    html_parts.append("<div class='section'><h2>3. Per-Class Coverage Heatmap</h2>")
    for alpha in alphas:
        target_cov = 1 - alpha
        html_parts.append(f"<h3>Alpha = {alpha}</h3>")
        html_parts.append("<table>")
        header_cells = "".join(f"<th>{n}</th>" for n in all_names)
        html_parts.append(f"<tr><th class='left'>Method</th>{header_cells}</tr>")
        for method in methods:
            key = f"{method}_alpha{alpha}"
            if key not in results:
                continue
            r = results[key]
            pc = r.get("per_class_coverage", {})
            cells = ""
            for name in all_names:
                cov = pc.get(name, float("nan"))
                bg = _cov_colour(cov, target_cov)
                cells += (
                    f"<td style='background:{bg}'>"
                    + (f"{cov:.2f}" if not math.isnan(cov) else "n/a")
                    + "</td>"
                )
            html_parts.append(f"<tr><td class='left'>{method}</td>{cells}</tr>")
        html_parts.append("</table>")
    html_parts.append("</div>")

    # 4. Megan vs FSCS breakdown
    html_parts.append("<div class='section'><h2>4. Megan vs FSCS Coverage Breakdown</h2>")
    for alpha in alphas:
        target_cov = 1 - alpha
        html_parts.append(f"<h3>Alpha = {alpha}</h3>")
        html_parts.append("<table>")
        html_parts.append(
            "<tr><th class='left'>Method</th>"
            "<th>Megan Coverage</th><th>FSCS Coverage</th>"
            "<th>Megan Avg Size</th><th>FSCS Avg Size</th></tr>"
        )
        for method in methods:
            key = f"{method}_alpha{alpha}"
            if key not in results:
                continue
            r = results[key]
            mc = r.get("megan_coverage", float("nan"))
            fc = r.get("fscs_coverage", float("nan"))
            mc_col = _cov_colour(mc, target_cov)
            fc_col = _cov_colour(fc, target_cov)

            def _fmt(v, decimals=3):
                return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.{decimals}f}"

            html_parts.append(
                f"<tr><td class='left'>{method}</td>"
                f"<td style='background:{mc_col}'>{_fmt(mc)}</td>"
                f"<td style='background:{fc_col}'>{_fmt(fc)}</td>"
                f"<td>{_fmt(r.get('megan_avg_size', float('nan')), 2)}</td>"
                f"<td>{_fmt(r.get('fscs_avg_size', float('nan')), 2)}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")
    html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CP on combined Megan + FSCS dataset (harmonised to level-1)."
    )
    parser.add_argument("--alphas", default="0.05,0.10,0.20",
                        help="Comma-separated miscoverage levels (default: 0.05,0.10,0.20)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--saps-lam", type=float, default=0.2)
    parser.add_argument("--raps-lam", type=float, default=0.01)
    parser.add_argument("--raps-kreg", type=int, default=1)
    args = parser.parse_args()

    device = resolve_device(args.device)
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading Megan splits ...")
    mX_tr, my_tr, mX_va, my_va, mX_te, my_te = load_megan_splits()
    print(f"  Megan train={len(mX_tr):,}  val={len(mX_va):,}  test={len(mX_te):,}")

    print("Loading FSCS parquet ...")
    fX_all, fy_all = load_fscs_data()
    print(f"  FSCS total rows after mapping: {len(fX_all):,}")
    fX_tr, fy_tr, fX_va, fy_va, fX_te, fy_te = fscs_stratified_splits(
        fX_all, fy_all, seed=args.seed
    )
    print(f"  FSCS train={len(fX_tr):,}  val={len(fX_va):,}  test={len(fX_te):,}")
    del fX_all, fy_all
    gc.collect()

    (
        X_train, y_train, src_train,
        X_val, y_val, src_val,
        X_test, y_test, src_test,
    ) = build_combined_splits(
        (mX_tr, my_tr, mX_va, my_va, mX_te, my_te),
        (fX_tr, fy_tr, fX_va, fy_va, fX_te, fy_te),
    )
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    megan_n_test = int((src_test == 0).sum())
    fscs_n_test = int((src_test == 1).sum())
    print(
        f"\nCombined: train={n_train:,}  val={n_val:,}  test={n_test:,}  "
        f"(Megan test={megan_n_test:,}  FSCS test={fscs_n_test:,})"
    )

    # ------------------------------------------------------------------
    # 2. Dataset summary counts (across all splits per source)
    # ------------------------------------------------------------------
    megan_all_y = np.concatenate([my_tr, my_va, my_te])
    fscs_all_y = np.concatenate([fy_tr, fy_va, fy_te])
    combined_all_y = np.concatenate([y_train, y_val, y_test])

    megan_counts = class_counts_table(megan_all_y, "Megan")
    fscs_counts = class_counts_table(fscs_all_y, "FSCS")
    combined_counts = class_counts_table(combined_all_y, "Combined")

    print("\nClass distribution (train split):")
    for idx, name in IDX_TO_NAME.items():
        n = int((y_train == idx).sum())
        print(f"  {name}: {n:,}")

    # ------------------------------------------------------------------
    # 3. Train base model
    # ------------------------------------------------------------------
    model = train_xgboost(
        X_train, y_train,
        n_estimators=args.n_estimators,
        device=device,
        seed=args.seed,
    )

    print("Computing probabilities on val and test sets ...")
    probs_cal = get_probabilities(model, X_val)
    probs_test = get_probabilities(model, X_test)
    del model
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Run CP methods
    # ------------------------------------------------------------------
    print("\nRunning conformal prediction methods ...")
    results = evaluate_all_methods(
        probs_cal, y_val,
        probs_test, y_test, src_test,
        alphas=alphas,
        seed=args.seed,
        saps_lam=args.saps_lam,
        raps_lam=args.raps_lam,
        raps_kreg=args.raps_kreg,
    )

    # ------------------------------------------------------------------
    # 5. Save JSON results
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(exist_ok=True)
    class_names_out = {str(code): name for code, name in LEVEL1_CLASSES.items()
                       if code in SHARED_CODES}

    output = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_classes": N_CLASSES,
        "class_names": class_names_out,
        "sources": {
            "megan": {"n_test": megan_n_test},
            "fscs": {"n_test": fscs_n_test},
        },
        "results": results,
    }

    json_path = REPORTS_DIR / "conformal_combined_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"\nJSON results saved to: {json_path}")

    # ------------------------------------------------------------------
    # 6. Save HTML report
    # ------------------------------------------------------------------
    html = build_html_report(
        results=results,
        class_names=class_names_out,
        n_train=n_train, n_val=n_val, n_test=n_test,
        megan_counts=megan_counts,
        fscs_counts=fscs_counts,
        combined_counts=combined_counts,
        megan_n_test=megan_n_test,
        fscs_n_test=fscs_n_test,
        alphas=alphas,
    )
    html_path = REPORTS_DIR / "conformal_combined_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved to:  {html_path}")


if __name__ == "__main__":
    main()

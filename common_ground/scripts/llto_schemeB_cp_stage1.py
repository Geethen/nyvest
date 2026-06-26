"""Scheme B: CP-method comparison for Stage-1 pseudo-label filtering.

Tests whether different conformal prediction (CP) methods produce better
pseudo-label quality and higher downstream Stage-2 F1 than the existing RAPS
singleton filter in the Scheme B LLTO pipeline.

Pipeline per fold:
  stable_train → 60% cb_train / 20% cp_cal / 20% stage2_stable_extra
  Stage-1: CatBoost-500 CPU on cb_train
  CP calibration on cp_cal; filter unstable with each CP method (singleton sets)
  Stage-2: TabICL n_est=16 kv_cache=True on biased_cls5 25k support
  Score on stable_test

CP methods (all α=0.10):
  no_filter, thr, aps, raps_01, raps_02, raps_05, saps, rank, mondrian_raps

Output:
  common_ground/reports/research/schemeB_cp_stage1_results.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import resolve_device  # noqa: E402

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

STABLE_PARQUET   = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET    = "class"
MERGE_MAP = {1: 2, 9: 8}
N_FOLDS   = 3
SEED      = 0
ALPHA     = 0.10
TOTAL_SUP = 25_000
REF_BEST  = 0.6927   # cb_biased_cls5 no CP filter
WEAK_ORIG = [2, 5, 7]
MIN_CAL_MONDRIAN = 5   # min cal samples per class for Mondrian RAPS


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(
        subset=FEATURE_COLS + [TARGET, "cell_id", "lon", "lat"]
    ).reset_index(drop=True)


def merge_classes(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype(int)
    for src, tgt in MERGE_MAP.items():
        y[y == src] = tgt
    return y


def clear_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Biased subsample
# ---------------------------------------------------------------------------

def biased_subsample(X: np.ndarray, y: np.ndarray, n_total: int,
                     forced_counts: dict[int, int],
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_total rows forcing exact counts for specified encoded classes."""
    parts_X, parts_y = [], []
    forced_mask = np.zeros(len(y), dtype=bool)

    for enc_c, n_force in forced_counts.items():
        idx_c  = np.flatnonzero(y == enc_c)
        take   = min(n_force, len(idx_c))
        chosen = rng.choice(idx_c, size=take, replace=False)
        parts_X.append(X[chosen])
        parts_y.append(y[chosen])
        forced_mask[chosen] = True

    n_forced    = sum(len(p) for p in parts_y)
    n_remainder = max(0, n_total - n_forced)
    pool_idx    = np.flatnonzero(~forced_mask)
    take_rem    = min(n_remainder, len(pool_idx))
    if take_rem > 0:
        chosen_rem = rng.choice(pool_idx, size=take_rem, replace=False)
        parts_X.append(X[chosen_rem])
        parts_y.append(y[chosen_rem])

    X_out = np.concatenate(parts_X)
    y_out = np.concatenate(parts_y)
    perm  = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def make_catboost_cpu(iterations: int):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU", loss_function="MultiClass",
    )


def classification_scores(y_true, y_pred, n_classes):
    f1m  = f1_score(y_true, y_pred, average="macro",
                    labels=np.arange(n_classes), zero_division=0)
    bal  = balanced_accuracy_score(y_true, y_pred)
    f1pc = f1_score(y_true, y_pred, labels=np.arange(n_classes),
                    average=None, zero_division=0)
    return float(f1m), float(bal), [float(v) for v in f1pc]


# ---------------------------------------------------------------------------
# Conformal prediction primitives
# ---------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return ceil((n+1)(1-alpha))/n -quantile of cal scores (capped at 1.0)."""
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def _ranks_and_sorted(probs: np.ndarray):
    """Return (ranks, order, sorted_probs).

    ranks[i, k]       = 1-indexed rank of class k for sample i (rank 1 = largest)
    order[i, :]       = class indices in descending-prob order
    sorted_probs[i,:] = probs sorted descending along axis 1
    """
    n, C  = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    ranks = np.empty_like(order)
    row   = np.arange(n)[:, None]
    ranks[row, order] = np.arange(1, C + 1)[None, :]
    return ranks, order, sorted_probs


# --- THR ---

def _thr_scores_all(probs: np.ndarray) -> np.ndarray:
    """Per-class THR score = p_k(x).  prediction set = {k: p_k >= tau}."""
    return probs


def _thr_cal_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return probs[np.arange(len(y)), y]


def apply_thr(probs_cal, y_cal, probs_test, alpha):
    cal_scores = _thr_cal_scores(probs_cal, y_cal)
    # THR: prediction set includes k where p_k >= tau; so score = p_k,
    # and we want score >= tau  ←→  use 1 - p_k as 'non-conformity' and
    # flip. Easier: tau = conformal_quantile but we want the *lower* quantile
    # because higher p means more conforming.
    # By convention: cal_score = p_{y_i}(x_i); tau = lower conformal quantile.
    # Prediction set = {k : p_k(x) >= tau}.
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    # For THR (coverage from below): we want the (1-alpha) quantile of scores
    # interpreted as "must be at least this large" → use the lower quantile,
    # i.e. alpha-quantile of 1-p, which equals 1 - (1-alpha)-quantile of p.
    # Equivalently: tau = np.quantile(cal_scores, alpha, method="lower")
    # but using the conformal finite-sample correction:
    # tau = floor((n+1)*alpha)/n quantile  → use "lower" on that level.
    q_lower = np.floor((n + 1) * alpha) / n
    q_lower = max(q_lower, 0.0)
    tau = float(np.quantile(cal_scores, q_lower, method="lower"))
    pred_sets = probs_test >= tau

    # Cal coverage check
    cal_sets_cov = (probs_cal >= tau)
    cal_cov = float(cal_sets_cov[np.arange(len(y_cal)), y_cal].mean())
    cal_avg_size = float(cal_sets_cov.sum(axis=1).mean())

    return pred_sets, tau, cal_cov, cal_avg_size


# --- APS / RAPS ---

def _aps_all_scores(probs: np.ndarray, u: np.ndarray,
                    lam_reg: float = 0.0, k_reg: int = 1) -> np.ndarray:
    """(n, C) APS/RAPS scores for every (sample, class). lam_reg=0 → APS."""
    ranks, order, sorted_probs = _ranks_and_sorted(probs)
    n, C = probs.shape
    cumsum_sorted = np.cumsum(sorted_probs, axis=1)
    cumsum_before = np.concatenate(
        [np.zeros((n, 1), dtype=sorted_probs.dtype), cumsum_sorted[:, :-1]], axis=1
    )
    # Place cumsum_before back at each class's rank position.
    cum_at_class = np.take_along_axis(cumsum_before, ranks - 1, axis=1)
    scores = cum_at_class + u[:, None] * probs
    if lam_reg > 0:
        scores = scores + lam_reg * np.maximum(0, ranks - k_reg)
    return scores


def _aps_cal_scores_v(probs: np.ndarray, y: np.ndarray, u: np.ndarray,
                      lam_reg: float = 0.0, k_reg: int = 1) -> np.ndarray:
    all_s = _aps_all_scores(probs, u, lam_reg=lam_reg, k_reg=k_reg)
    return all_s[np.arange(len(y)), y]


def apply_aps_like(probs_cal, y_cal, probs_test, u_cal, u_test, alpha,
                   lam_reg=0.0, k_reg=1):
    cal_scores = _aps_cal_scores_v(probs_cal, y_cal, u_cal, lam_reg=lam_reg, k_reg=k_reg)
    tau = conformal_quantile(cal_scores, alpha)
    test_all = _aps_all_scores(probs_test, u_test, lam_reg=lam_reg, k_reg=k_reg)
    pred_sets = test_all <= tau

    cal_all = _aps_all_scores(probs_cal, u_cal, lam_reg=lam_reg, k_reg=k_reg)
    cal_sets_cov = cal_all <= tau
    cal_cov = float(cal_sets_cov[np.arange(len(y_cal)), y_cal].mean())
    cal_avg_size = float(cal_sets_cov.sum(axis=1).mean())

    return pred_sets, tau, cal_cov, cal_avg_size


# --- SAPS ---

def _saps_all_scores(probs: np.ndarray, u: np.ndarray, lam: float) -> np.ndarray:
    """(n, C) SAPS scores (Huang et al. 2024)."""
    ranks, order, sorted_probs = _ranks_and_sorted(probs)
    top1 = probs[np.arange(len(probs))[:, None], order[:, :1]]  # (n,1)
    scores = np.where(
        ranks == 1,
        top1 * u[:, None],
        top1 + (ranks - 2 + u[:, None]) * lam,
    )
    return scores


def _saps_cal_scores_v(probs: np.ndarray, y: np.ndarray, u: np.ndarray,
                       lam: float) -> np.ndarray:
    all_s = _saps_all_scores(probs, u, lam)
    return all_s[np.arange(len(y)), y]


def apply_saps(probs_cal, y_cal, probs_test, u_cal, u_test, alpha, lam):
    cal_scores = _saps_cal_scores_v(probs_cal, y_cal, u_cal, lam)
    tau = conformal_quantile(cal_scores, alpha)
    test_all = _saps_all_scores(probs_test, u_test, lam)
    pred_sets = test_all <= tau

    cal_all = _saps_all_scores(probs_cal, u_cal, lam)
    cal_sets_cov = cal_all <= tau
    cal_cov = float(cal_sets_cov[np.arange(len(y_cal)), y_cal].mean())
    cal_avg_size = float(cal_sets_cov.sum(axis=1).mean())

    return pred_sets, tau, cal_cov, cal_avg_size


# --- RANK (Liu et al. 2025, Algorithm 1) ---

def apply_rank(probs_cal, y_cal, probs_test, alpha):
    n, C = probs_cal.shape
    ranks_cal, _, _ = _ranks_and_sorted(probs_cal)
    r_true = ranks_cal[np.arange(n), y_cal]  # rank of true class per cal sample

    # r*_α = floor((n+1)*alpha)-th largest rank in cal
    kth = int(np.floor((n + 1) * alpha))
    kth = max(1, min(kth, n))
    r_sorted_desc = np.sort(r_true)[::-1]
    r_star = int(r_sorted_desc[kth - 1])

    count_below = int(np.sum(r_true <= r_star - 1))
    count_at    = int(np.sum(r_true == r_star))

    if count_at == 0:
        use_rstar_threshold = -np.inf
    else:
        p_prop = (n - int(np.floor((n + 1) * alpha)) - count_below) / count_at
        p_prop = float(np.clip(p_prop, 0.0, 1.0))
        sorted_probs_cal = np.sort(probs_cal, axis=1)[:, ::-1]
        rstar_probs = sorted_probs_cal[:, r_star - 1]
        k_pick = int(np.ceil(n * p_prop))
        if k_pick <= 0:
            use_rstar_threshold = np.inf
        elif k_pick > len(rstar_probs):
            use_rstar_threshold = -np.inf
        else:
            use_rstar_threshold = float(np.sort(rstar_probs)[::-1][k_pick - 1])

    nt, _ = probs_test.shape
    ranks_test, _, sorted_test = _ranks_and_sorted(probs_test)
    rstar_prob_test = sorted_test[:, r_star - 1] if r_star >= 1 else np.zeros(nt)
    include_rstar   = rstar_prob_test >= use_rstar_threshold
    size_per_sample = np.where(include_rstar, r_star, max(r_star - 1, 0))
    pred_sets = ranks_test <= size_per_sample[:, None]

    # Cal coverage estimate
    include_rstar_cal = (np.sort(probs_cal, axis=1)[:, ::-1][:, r_star - 1]
                         >= use_rstar_threshold)
    size_cal  = np.where(include_rstar_cal, r_star, max(r_star - 1, 0))
    sets_cal  = ranks_cal <= size_cal[:, None]
    cal_cov   = float(sets_cal[np.arange(n), y_cal].mean())
    cal_avg_size = float(sets_cal.sum(axis=1).mean())

    return pred_sets, float("nan"), cal_cov, cal_avg_size


# --- Mondrian RAPS ---

def apply_mondrian_raps(probs_cal, y_cal, probs_test, u_cal, u_test, alpha,
                        n_classes, lam_reg=0.01, k_reg=1):
    """Mondrian RAPS: stratify calibration by argmax class, one tau per class."""
    pred_class_cal  = probs_cal.argmax(axis=1)
    pred_class_test = probs_test.argmax(axis=1)

    # Compute RAPS scores for all cal samples first
    cal_all_scores = _aps_all_scores(probs_cal, u_cal, lam_reg=lam_reg, k_reg=k_reg)
    cal_raps_scores_true = cal_all_scores[np.arange(len(y_cal)), y_cal]

    # Global fallback tau
    tau_global = conformal_quantile(cal_raps_scores_true, alpha)

    # Per-class tau
    taus = {}
    for c in range(n_classes):
        mask = pred_class_cal == c
        if mask.sum() >= MIN_CAL_MONDRIAN:
            taus[c] = conformal_quantile(cal_raps_scores_true[mask], alpha)
        else:
            taus[c] = tau_global

    # Build prediction sets for test
    test_all_scores = _aps_all_scores(probs_test, u_test, lam_reg=lam_reg, k_reg=k_reg)
    pred_sets = np.zeros((len(probs_test), n_classes), dtype=bool)
    for c in range(n_classes):
        mask = pred_class_test == c
        if mask.sum() > 0:
            pred_sets[mask] = test_all_scores[mask] <= taus[c]

    # Cal coverage
    cal_sets_cov = np.zeros((len(probs_cal), n_classes), dtype=bool)
    for c in range(n_classes):
        mask = pred_class_cal == c
        if mask.sum() > 0:
            cal_sets_cov[mask] = cal_all_scores[mask] <= taus[c]
    cal_cov      = float(cal_sets_cov[np.arange(len(y_cal)), y_cal].mean())
    cal_avg_size = float(cal_sets_cov.sum(axis=1).mean())

    return pred_sets, float("nan"), cal_cov, cal_avg_size


# ---------------------------------------------------------------------------
# Apply a CP method: returns (pred_sets_test, tau, cal_cov, cal_avg_size)
# ---------------------------------------------------------------------------

def run_cp_method(name: str, probs_cal, y_cal, probs_test, rng, n_classes):
    u_cal  = rng.uniform(size=len(y_cal)).astype(np.float32)
    u_test = rng.uniform(size=len(probs_test)).astype(np.float32)

    if name == "thr":
        return apply_thr(probs_cal, y_cal, probs_test, ALPHA)
    elif name == "aps":
        return apply_aps_like(probs_cal, y_cal, probs_test, u_cal, u_test,
                              ALPHA, lam_reg=0.0, k_reg=1)
    elif name == "raps_01":
        return apply_aps_like(probs_cal, y_cal, probs_test, u_cal, u_test,
                              ALPHA, lam_reg=0.01, k_reg=1)
    elif name == "raps_02":
        return apply_aps_like(probs_cal, y_cal, probs_test, u_cal, u_test,
                              ALPHA, lam_reg=0.02, k_reg=2)
    elif name == "raps_05":
        return apply_aps_like(probs_cal, y_cal, probs_test, u_cal, u_test,
                              ALPHA, lam_reg=0.05, k_reg=1)
    elif name == "saps":
        return apply_saps(probs_cal, y_cal, probs_test, u_cal, u_test,
                          ALPHA, lam=0.2)
    elif name == "rank":
        return apply_rank(probs_cal, y_cal, probs_test, ALPHA)
    elif name == "mondrian_raps":
        return apply_mondrian_raps(probs_cal, y_cal, probs_test, u_cal, u_test,
                                   ALPHA, n_classes=n_classes,
                                   lam_reg=0.01, k_reg=1)
    else:
        raise ValueError(f"Unknown CP method: {name}")


# ---------------------------------------------------------------------------
# Stage-1 quality metrics
# ---------------------------------------------------------------------------

def stage1_quality(pred_sets_test, y_u_m_enc, probs_test, le,
                   weak_enc_set, method_name):
    """Compute Stage-1 quality metrics for CP-filtered unstable rows.

    Parameters
    ----------
    pred_sets_test : (n_unstable, C) bool array of prediction sets
    y_u_m_enc     : (n_unstable,) encoded ground-truth labels for unstable
    probs_test    : (n_unstable, C) softmax probs (for pseudo-label = argmax)
    le            : fitted LabelEncoder
    weak_enc_set  : set of encoded indices for weak classes (2, 5, 7)
    method_name   : str, for no_filter special case

    Returns
    -------
    dict with n_kept, pct_kept, pseudo_acc, pseudo_f1_weak, kept_mask
    """
    n_total = len(pred_sets_test)

    if method_name == "no_filter":
        kept_mask  = np.ones(n_total, dtype=bool)
        pseudo_lbl = probs_test.argmax(axis=1)
    else:
        set_sizes  = pred_sets_test.sum(axis=1)
        kept_mask  = (set_sizes == 1)
        # singleton pseudo-label: the one class in the set
        pseudo_lbl = np.zeros(n_total, dtype=int)
        for i in np.flatnonzero(kept_mask):
            pseudo_lbl[i] = int(np.flatnonzero(pred_sets_test[i])[0])

    n_kept   = int(kept_mask.sum())
    pct_kept = float(n_kept / n_total * 100) if n_total > 0 else 0.0

    # Pseudo accuracy: fraction of kept rows where pseudo == grunnkart label
    if n_kept > 0:
        pseudo_acc = float((pseudo_lbl[kept_mask] == y_u_m_enc[kept_mask]).mean())
    else:
        pseudo_acc = float("nan")

    # Pseudo F1 for weak classes among kept rows only
    if n_kept > 0:
        y_true_kept   = y_u_m_enc[kept_mask]
        y_pseudo_kept = pseudo_lbl[kept_mask]
        n_classes_le  = len(le.classes_)
        f1_weak = f1_score(
            y_true_kept, y_pseudo_kept,
            labels=sorted(weak_enc_set),
            average="macro", zero_division=0
        )
        pseudo_f1_weak = float(f1_weak)
    else:
        pseudo_f1_weak = float("nan")

    return {
        "n_kept": n_kept,
        "pct_kept": round(pct_kept, 2),
        "pseudo_acc": round(pseudo_acc, 4) if not np.isnan(pseudo_acc) else float("nan"),
        "pseudo_f1_weak": round(pseudo_f1_weak, 4) if not np.isnan(pseudo_f1_weak) else float("nan"),
        "kept_mask": kept_mask,
        "pseudo_lbl": pseudo_lbl,
    }


# ---------------------------------------------------------------------------
# TabICL batch predict
# ---------------------------------------------------------------------------

def tabicl_predict_proba_chunked(clf, X, chunk_size=10_000):
    parts = []
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        parts.append(clf.predict_proba(X[start:end]).astype(np.float64))
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def _record(results, descs, name, desc, fold, f1m, bal, pc,
            n_kept, pct_kept, pseudo_acc, pseudo_f1_weak,
            cal_coverage, cal_avg_size, n_classes):
    if name not in results:
        results[name] = []
        descs[name]   = desc
    results[name].append({
        "fold":        fold,
        "f1_macro":    round(f1m, 4),
        "bal_acc":     round(bal, 4),
        "f1_per_class": [round(v, 4) for v in pc],
        "n_kept":       n_kept,
        "pct_kept":     round(pct_kept, 2),
        "pseudo_acc":   round(pseudo_acc, 4) if not np.isnan(pseudo_acc) else float("nan"),
        "pseudo_f1_weak": round(pseudo_f1_weak, 4) if not np.isnan(pseudo_f1_weak) else float("nan"),
        "cal_coverage": round(cal_coverage, 4) if not np.isnan(cal_coverage) else float("nan"),
        "cal_avg_size": round(cal_avg_size, 4) if not np.isnan(cal_avg_size) else float("nan"),
    })


def _record_fail(results, descs, name, desc, fold, n_classes,
                 n_kept, pct_kept, pseudo_acc, pseudo_f1_weak,
                 cal_coverage, cal_avg_size):
    _record(results, descs, name, desc, fold,
            float("nan"), float("nan"), [float("nan")] * n_classes,
            n_kept, pct_kept, pseudo_acc, pseudo_f1_weak,
            cal_coverage, cal_avg_size, n_classes)


# ---------------------------------------------------------------------------
# CP method registry
# ---------------------------------------------------------------------------

CP_METHODS = [
    ("no_filter",     "All unstable pseudo-labels (argmax), no CP filter"),
    ("thr",           "Threshold CP: keep k where p_k >= tau (alpha=0.10)"),
    ("aps",           "APS: cumsum score, singleton filter (alpha=0.10)"),
    ("raps_01",       "RAPS lam=0.01 k_reg=1, singleton filter (alpha=0.10)"),
    ("raps_02",       "RAPS lam=0.02 k_reg=2, singleton filter (alpha=0.10)"),
    ("raps_05",       "RAPS lam=0.05 k_reg=1, singleton filter (alpha=0.10)"),
    ("saps",          "SAPS lam=0.2, singleton filter (alpha=0.10)"),
    ("rank",          "RANK two-stage (Liu et al. 2025), singleton filter (alpha=0.10)"),
    ("mondrian_raps", "Mondrian RAPS lam=0.01 k_reg=1, per-class tau, singleton filter (alpha=0.10)"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  scheme=B  alpha={ALPHA}  "
          f"n_methods={len(CP_METHODS)}")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    y_stable_m   = merge_classes(stable[TARGET].values)
    y_unstable_m = merge_classes(unstable[TARGET].values)
    merged_classes = sorted(set(y_stable_m.tolist()) | set(y_unstable_m.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded   = le.classes_.tolist()
    print(f"classes: {decoded}  n={n_classes}")

    # Encoded indices for forced/weak classes
    enc5 = int(le.transform([5])[0])
    weak_enc_set = set()
    for c in WEAK_ORIG:
        if c in le.classes_:
            weak_enc_set.add(int(le.transform([c])[0]))
    print(f"enc5={enc5}  weak_enc={sorted(weak_enc_set)}")

    # Spatial LLTO folds on stable data
    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    results: dict[str, list] = {}
    descs:   dict[str, str]  = {}
    t_global = time.perf_counter()

    for k in range(N_FOLDS):
        s_test  = (stable_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_s_tr_full = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr_full = le.transform(y_stable_m[s_train])
        X_s_te = stable.loc[s_test, FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(y_stable_m[s_test])
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)
        y_u_tr_m = le.transform(y_unstable_m[u_train])  # ground-truth (for pseudo_acc)

        print(f"\n=== fold {k}  n_stable_train_full={len(y_s_tr_full):,}  "
              f"n_unstable_train={len(X_u_tr):,}  n_test={len(y_s_te):,} ===")

        # --- Stage-1 split: 60% cb_train / 20% cp_cal / 20% stage2_stable_extra ---
        idx_all = np.arange(len(X_s_tr_full))
        idx_cb, idx_cp_and_s2 = train_test_split(
            idx_all, test_size=0.40,
            stratify=y_s_tr_full, random_state=SEED)
        idx_cp, idx_s2 = train_test_split(
            idx_cp_and_s2, test_size=0.50,
            stratify=y_s_tr_full[idx_cp_and_s2], random_state=SEED)

        X_cb  = X_s_tr_full[idx_cb]
        y_cb  = y_s_tr_full[idx_cb]
        X_cp  = X_s_tr_full[idx_cp]
        y_cp  = y_s_tr_full[idx_cp]
        X_s2s = X_s_tr_full[idx_s2]   # stage2_stable_extra
        y_s2s = y_s_tr_full[idx_s2]

        print(f"  cb_train={len(y_cb):,}  cp_cal={len(y_cp):,}  "
              f"stage2_stable_extra={len(y_s2s):,}")

        # --- Train CatBoost-500 on cb_train ---
        t_cb = time.perf_counter()
        stage1 = make_catboost_cpu(500)
        stage1.fit(X_cb, y_cb)
        t_cb = time.perf_counter() - t_cb
        print(f"  CatBoost trained in {t_cb:.1f}s")

        # Get probabilities for cp_cal and unstable_train
        probs_cp = stage1.predict_proba(X_cp).astype(np.float32)
        probs_u  = stage1.predict_proba(X_u_tr).astype(np.float32)
        del stage1

        rng = np.random.default_rng(SEED + k)

        # --- Loop over CP methods ---
        for method_name, method_desc in CP_METHODS:
            clear_cuda()
            print(f"\n  [{method_name}] fold {k} ...")

            # Calibrate and build prediction sets
            if method_name == "no_filter":
                # No CP: keep all, pseudo = argmax
                pred_sets_u = np.zeros((len(probs_u), n_classes), dtype=bool)
                for i, am in enumerate(probs_u.argmax(axis=1)):
                    pred_sets_u[i, am] = True
                cal_coverage = float("nan")
                cal_avg_size = float("nan")
                # fake tau is unused
            else:
                pred_sets_u, tau_val, cal_coverage, cal_avg_size = run_cp_method(
                    method_name, probs_cp, y_cp, probs_u, rng, n_classes
                )

            # Stage-1 quality metrics
            s1q = stage1_quality(
                pred_sets_u, y_u_tr_m, probs_u, le, weak_enc_set, method_name
            )
            n_kept       = s1q["n_kept"]
            pct_kept     = s1q["pct_kept"]
            pseudo_acc   = s1q["pseudo_acc"]
            pseudo_f1_w  = s1q["pseudo_f1_weak"]
            kept_mask    = s1q["kept_mask"]
            pseudo_lbl   = s1q["pseudo_lbl"]

            print(f"    n_kept={n_kept} ({pct_kept:.1f}%)  pseudo_acc={pseudo_acc}  "
                  f"cal_cov={cal_coverage}  cal_avg_size={cal_avg_size}")

            # --- Build Stage-2 support: stage2_stable_extra + kept_unstable ---
            X_kept = X_u_tr[kept_mask]
            y_kept = pseudo_lbl[kept_mask]

            X_aug = np.concatenate([X_s2s, X_kept])
            y_aug = np.concatenate([y_s2s, y_kept])

            # Biased cls5 subsample to 25k
            X_sup, y_sup = biased_subsample(
                X_aug, y_aug, TOTAL_SUP, {enc5: 4_000}, rng
            )
            print(f"    support: n={len(y_sup):,}")

            # --- Stage-2: TabICL ---
            from tabicl import TabICLClassifier
            try:
                clf = TabICLClassifier(
                    n_estimators=16, kv_cache=True,
                    random_state=SEED, verbose=False, device="cuda")
                t_fit = time.perf_counter()
                clf.fit(X_sup, y_sup)
                t_fit = time.perf_counter() - t_fit
                t_pred = time.perf_counter()
                probs_te = tabicl_predict_proba_chunked(clf, X_s_te, chunk_size=10_000)
                t_pred = time.perf_counter() - t_pred
                pred = probs_te.argmax(axis=1)
                f1m, bal, pc = classification_scores(y_s_te, pred, n_classes)
                del clf
                print(f"    F1={f1m:.4f}  bal={bal:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s")

                _record(results, descs, method_name, method_desc, k,
                        f1m, bal, pc, n_kept, pct_kept, pseudo_acc,
                        pseudo_f1_w, cal_coverage, cal_avg_size, n_classes)

            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {str(exc)[:120]}")
                _record_fail(results, descs, method_name, method_desc, k,
                             n_classes, n_kept, pct_kept, pseudo_acc,
                             pseudo_f1_w, cal_coverage, cal_avg_size)

            clear_cuda()

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── Summarise ────────────────────────────────────────────────────────────
    summary = {}
    for name, folds in results.items():
        valid_f1  = [f["f1_macro"]  for f in folds if not np.isnan(f["f1_macro"])]
        valid_bal = [f["bal_acc"]   for f in folds if not np.isnan(f["bal_acc"])]
        valid_pcs = [f["f1_per_class"] for f in folds
                     if not any(np.isnan(v) for v in f["f1_per_class"])]
        pc_mean   = (np.array(valid_pcs).mean(axis=0).tolist()
                     if valid_pcs else [float("nan")] * n_classes)

        def safe_mean(vals):
            return float(np.mean(vals)) if vals else float("nan")
        def safe_std(vals):
            return float(np.std(vals))  if vals else float("nan")
        def safe_avg(key):
            vs = [f[key] for f in folds if not np.isnan(f[key])]
            return float(np.mean(vs)) if vs else float("nan")

        f1_mean = safe_mean(valid_f1)

        summary[name] = {
            "description": descs[name],
            "f1_mean":   round(f1_mean, 4),
            "f1_std":    round(safe_std(valid_f1), 4),
            "bal_mean":  round(safe_mean(valid_bal), 4),
            "f1_per_class": {
                str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)
            },
            "stage1_mean": {
                "n_kept":       round(safe_avg("n_kept"), 1),
                "pct_kept":     round(safe_avg("pct_kept"), 2),
                "pseudo_acc":   round(safe_avg("pseudo_acc"), 4),
                "pseudo_f1_weak": round(safe_avg("pseudo_f1_weak"), 4),
                "cal_coverage": round(safe_avg("cal_coverage"), 4),
                "cal_avg_size": round(safe_avg("cal_avg_size"), 4),
            },
            "per_fold": folds,
        }

    out_json = OUT_DIR / "schemeB_cp_stage1_results.json"
    with open(out_json, "w") as fh:
        json.dump({
            "scheme":     "B",
            "n_folds":    N_FOLDS,
            "alpha":      ALPHA,
            "references": {"cb_biased_cls5_no_filter": REF_BEST},
            "results":    summary,
        }, fh, indent=2)
    print(f"\nSaved → {out_json}")

    # ── Leaderboard ──────────────────────────────────────────────────────────
    print(f"\n{'Method':<22}  {'F1':>8}  {'std':>6}  {'Δ ref':>8}")
    print("-" * 52)
    for name, s in sorted(
        summary.items(),
        key=lambda x: -(x[1]["f1_mean"] if not np.isnan(x[1]["f1_mean"]) else -1)
    ):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<22}  {'NaN':>8}  {'NaN':>6}  [FAIL]")
        else:
            delta = s["f1_mean"] - REF_BEST
            flag  = " ★" if delta > 0.003 else ""
            print(f"{name:<22}  {s['f1_mean']:>8.4f}  "
                  f"{s['f1_std']:>6.4f}  {delta:>+8.4f}{flag}")

    # ── Stage-1 quality table ─────────────────────────────────────────────────
    print(f"\nStage-1 quality (means across folds):")
    print(f"{'Method':<22}  {'n_kept':>8}  {'pct_kept%':>10}  "
          f"{'pseudo_acc':>11}  {'cal_cov':>9}  {'cal_size':>9}")
    print("-" * 77)
    for name, s in summary.items():
        s1 = s["stage1_mean"]
        nk  = s1["n_kept"]
        pct = s1["pct_kept"]
        acc = s1["pseudo_acc"]
        cov = s1["cal_coverage"]
        sz  = s1["cal_avg_size"]
        nk_s  = f"{nk:>8.0f}"  if not np.isnan(nk)  else f"{'nan':>8}"
        pct_s = f"{pct:>10.2f}" if not np.isnan(pct) else f"{'nan':>10}"
        acc_s = f"{acc:>11.4f}" if not np.isnan(acc) else f"{'nan':>11}"
        cov_s = f"{cov:>9.4f}"  if not np.isnan(cov) else f"{'nan':>9}"
        sz_s  = f"{sz:>9.4f}"   if not np.isnan(sz)  else f"{'nan':>9}"
        print(f"{name:<22}  {nk_s}  {pct_s}  {acc_s}  {cov_s}  {sz_s}")

    # ── Per-class F1 for weak classes ─────────────────────────────────────────
    print(f"\nPer-class F1 for weak classes (2=bare, 5=grassland, 7=wetland):")
    print(f"{'Method':<22}  " + "  ".join(f"cls{c:>2}" for c in WEAK_ORIG))
    print("-" * 52)
    for name, s in summary.items():
        pc = s["f1_per_class"]
        vals = "  ".join(f"{pc.get(str(c), float('nan')):>6.4f}" for c in WEAK_ORIG)
        print(f"{name:<22}  {vals}")


if __name__ == "__main__":
    run()

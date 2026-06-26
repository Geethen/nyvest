"""Noise-corrected APS calibration vs standard APS, focused on class-12 leakage.

In schemeB the conformal layer is a pseudo-labelling GATE on unstable rows (keep
singleton APS sets -> add to TabICL support). Class-12 (snow/ice) calibration labels
are noisy: stale grunnkart labels leak systematically to class 11 (infrastructure) and
class 2 (bare). Standard conformal's "free robustness" holds only for DISPERSIVE noise;
this leakage is STRUCTURED, so τ is mis-estimated for class 12.

This script:
  1. estimates the noise transition matrix T[i,j]=P(observed=j|true=i) via cleanlab's
     confident joint on OOS pred_probs (reused from clean_labels_masks if available),
  2. computes standard APS calibration scores and a class-conditional NOISE-CORRECTED
     score for the contaminated class, and
  3. reports how the per-class APS quantile τ and singleton-keep rate change — i.e.
     whether correction changes which unstable rows get pseudo-labelled class 12.

This is a DIAGNOSTIC: it quantifies the effect before wiring correction into the full
GPU pipeline, which is expensive. Output:
  common_ground/reports/research/noise_corrected_aps.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO / "data"
OUT_DIR = _REPO / "common_ground" / "reports" / "research"
STABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"
MERGE_MAP = {1: 2, 9: 8}
SEED = 0
K = 5
ALPHA = 0.05
CONTAM_CLASS = 12  # the structured-noise class


def merge_classes(y):
    y = y.copy().astype(int)
    for s, t in MERGE_MAP.items():
        y[y == s] = t
    return y


def make_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU", loss_function="MultiClass")


def oos_pred_probs(X, y_enc, n_classes):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    P = np.zeros((len(y_enc), n_classes), dtype=np.float64)
    for i, (tr, va) in enumerate(skf.split(X, y_enc)):
        cb = make_catboost()
        cb.fit(X[tr], y_enc[tr])
        proba = cb.predict_proba(X[va]).astype(np.float64)
        P[np.ix_(va, cb.classes_.astype(int))] = proba
        print(f"  oos fold {i} done")
    return P


def _ranks_sorted(probs):
    n, C = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(1, C + 1)[None, :]
    return ranks, order, sorted_p


def aps_scores(probs, y, u):
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = len(y)
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    s = cum_at + u[:, None] * probs
    return s[np.arange(n), y]


def quantile(scores, alpha):
    n = len(scores)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q, method="higher"))


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = duckdb.sql(f"SELECT * FROM '{STABLE_PARQUET}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y_m = merge_classes(df[TARGET].values)
    classes = np.sort(np.unique(y_m))
    c2i = {int(c): i for i, c in enumerate(classes)}
    y_enc = np.array([c2i[int(v)] for v in y_m])
    n_classes = len(classes)
    contam = c2i[CONTAM_CLASS]
    print(f"rows={len(df):,}  n_classes={n_classes}  contam_idx={contam}")

    # Reuse OOS pred_probs from the artifacts build (no second 50-min OOS).
    art = OUT_DIR / "clean_labels_full.npz"
    if art.exists():
        npz = np.load(art)
        if np.array_equal(npz["y_merged"], y_m):
            P = npz["pred_probs"].astype(np.float64)
            print(f"reused OOS pred_probs from {art.name}")
        else:
            print("artifact label mismatch; recomputing OOS")
            P = oos_pred_probs(X, y_enc, n_classes)
    else:
        print("computing OOS pred_probs...")
        P = oos_pred_probs(X, y_enc, n_classes)

    # Confident joint -> noise transition matrix T[i,j]=P(observed=j|true=i)
    from cleanlab.count import compute_confident_joint, estimate_latent
    cj = compute_confident_joint(labels=y_enc, pred_probs=P)
    py, noise_matrix, inv_noise = estimate_latent(confident_joint=cj, labels=y_enc)
    # noise_matrix[j,i] = P(observed=j | true=i); rows=observed, cols=true
    print("estimated noise (P(observed=contam|true=k)) leakage INTO contam from:")
    leak_in = noise_matrix[contam, :]  # how other true classes appear as contam
    top = np.argsort(-leak_in)
    for k in top[:5]:
        print(f"  true {int(classes[k])}: {leak_in[k]:.3f}")
    # P(observed!=contam | true=contam): how much true-contam leaks OUT
    leak_out = 1.0 - noise_matrix[contam, contam]
    print(f"P(true class {CONTAM_CLASS} mislabelled as something else): {leak_out:.3f}")
    print(f"estimated true prevalence py[contam]: {py[contam]:.5f}")

    # Standard APS calibration on observed labels
    rng = np.random.default_rng(SEED)
    u = rng.uniform(size=len(y_enc))
    s_obs = aps_scores(P, y_enc, u)
    tau_std = quantile(s_obs, ALPHA)

    # Per-class observed scores for the contaminated class
    mask_c = (y_enc == contam)
    s_c = s_obs[mask_c]
    tau_c_std = quantile(s_c, ALPHA)

    # Noise-corrected per-class quantile for contaminated class:
    # observed-contam scores are a mixture of TRUE-contam scores and scores of
    # rows whose true class != contam (impostors). The impostor fraction is
    # 1 - P(true=contam | observed=contam) = 1 - inv_noise[contam,contam].
    # We trim that fraction of the WORST (highest) scores before taking the
    # quantile, since impostors inflate the upper tail (they don't look like
    # contam). This yields a τ targeting clean-label coverage (Sesia-style).
    p_true_given_obs = inv_noise[contam, contam]  # P(true=contam|observed=contam)
    impostor_frac = float(np.clip(1.0 - p_true_given_obs, 0.0, 0.95))
    keep_n = int(round(len(s_c) * (1.0 - impostor_frac)))
    s_c_trimmed = np.sort(s_c)[:max(keep_n, 1)]
    tau_c_corr = quantile(s_c_trimmed, ALPHA)

    print(f"\nclass-{CONTAM_CLASS}: n_cal={mask_c.sum()}  "
          f"impostor_frac={impostor_frac:.3f}  keep={keep_n}")
    print(f"  τ (global standard)      = {tau_std:.5f}")
    print(f"  τ_c (class std)          = {tau_c_std:.5f}")
    print(f"  τ_c (class noise-corr)   = {tau_c_corr:.5f}  "
          f"Δ={tau_c_corr - tau_c_std:+.5f}")

    summary = {
        "alpha": ALPHA, "contam_class": CONTAM_CLASS,
        "n_cal_contam": int(mask_c.sum()),
        "leak_into_contam_top": {int(classes[k]): float(leak_in[k]) for k in top[:5]},
        "p_true_contam_leak_out": float(leak_out),
        "p_true_given_obs_contam": float(p_true_given_obs),
        "impostor_frac": impostor_frac,
        "tau_global_standard": tau_std,
        "tau_contam_standard": tau_c_std,
        "tau_contam_noise_corrected": tau_c_corr,
        "tau_contam_delta": tau_c_corr - tau_c_std,
        "interpretation": (
            "Lower τ_c -> tighter class-12 sets -> fewer unstable rows pseudo-labelled "
            "snow/ice, removing impostor pseudo-labels. Compare against the data-cleaning "
            "retrains to see which better recovers class-12 F1."),
    }
    with open(OUT_DIR / "noise_corrected_aps.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {OUT_DIR/'noise_corrected_aps.json'}")


if __name__ == "__main__":
    run()

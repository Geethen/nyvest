"""SAPS conformal prediction on NYVEST land-cover probabilities.

Fits a base classifier (XGBoost by default — the benchmark leader), calibrates
SAPS (Sorted Adaptive Prediction Sets) on the val split, then evaluates marginal
coverage and prediction-set size on test.

SAPS — Huang et al., "Sorted Adaptive Prediction Sets", ICML 2024.
Reference impl (unused here due to a torchsort ABI clash with torch 2.5+cu124):
https://github.com/ml-stat-Sustech/TorchCP/blob/master/torchcp/classification/score/saps.py

Score for sample x with softmax probs p and class k at sorted-rank O_k
(O_k = 1 is the top class):

    V(p, k) = p_(1) * u                       if O_k == 1
            = p_(1) + (O_k - 2 + u) * lam     if O_k >= 2

where u is Uniform(0,1) (or u=1 in --deterministic mode). The conformal
threshold is the ceil((n_cal+1)(1-alpha))/n_cal -quantile of cal scores; the
prediction set at test is {k : V(p, k) <= tau}.

Usage:
    ~/myprojects/recover/.venv/bin/python scripts/conformal_saps.py
    ~/myprojects/recover/.venv/bin/python scripts/conformal_saps.py \
        --alpha 0.05,0.1,0.2 --lam 0.05,0.2,0.5
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_tabular import get_data_dir, load_split, resolve_device


def saps_scores(probs: np.ndarray, classes: np.ndarray, u: np.ndarray, lam: float) -> np.ndarray:
    """Vectorised SAPS scores V(p, k) for each (sample, class) pair.

    probs   : (n, C) softmax probabilities
    classes : (C,)   candidate class indices (0..C-1)
    u       : (n,)   per-sample uniform draws (or all ones for deterministic)
    lam     : scalar penalty weight
    returns : (n, C) scores — row i, col k = V(p_i, k)
    """
    n, C = probs.shape
    # rank each class per-sample: rank 1 = highest prob
    order = np.argsort(-probs, axis=1)  # (n, C) — class idx in descending-prob order
    ranks = np.empty_like(order)
    row = np.arange(n)[:, None]
    ranks[row, order] = np.arange(1, C + 1)[None, :]  # (n, C): rank of class k
    top1 = probs[row, order[:, :1]]  # (n, 1) — top-1 prob

    scores = np.where(
        ranks == 1,
        top1 * u[:, None],
        top1 + (ranks - 2 + u[:, None]) * lam,
    )
    return scores


def saps_cal_scores(probs: np.ndarray, y: np.ndarray, u: np.ndarray, lam: float) -> np.ndarray:
    """Calibration scores V(p_i, y_i) — just the score at the true label."""
    C = probs.shape[1]
    all_scores = saps_scores(probs, np.arange(C), u, lam)
    return all_scores[np.arange(len(y)), y]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal threshold: ceil((n+1)(1-alpha))/n -quantile of cal scores."""
    n = len(scores)
    # np.quantile uses the (k-1)/(n-1) convention; we want the k-th order stat where
    # k = ceil((n+1)(1-alpha)). Use 'higher' interpolation on the (k-1)/n position.
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def evaluate_sets(probs: np.ndarray, y: np.ndarray, tau: float, u: np.ndarray, lam: float) -> dict:
    C = probs.shape[1]
    scores = saps_scores(probs, np.arange(C), u, lam)
    sets = scores <= tau  # (n, C) bool mask
    set_sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    empty = set_sizes == 0
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "max_set_size": int(set_sizes.max()),
        "pct_singleton": float((set_sizes == 1).mean()),
        "pct_empty": float(empty.mean()),
    }


def fit_base_model(X_train, y_train, X_val, X_test, n_estimators: int, n_jobs: int,
                   device: str, seed: int, cache_path: Path, refit: bool) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not refit:
        print(f"Loading cached probabilities from {cache_path}")
        z = np.load(cache_path)
        return z["val_probs"], z["test_probs"]

    from xgboost import XGBClassifier
    print(f"Training XGBoost (n_estimators={n_estimators}, device={device}) on {len(X_train)} rows ...")
    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        random_state=seed,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    print(f"  trained in {time.perf_counter() - t0:.1f} s")
    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)
    cache_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache_path, val_probs=val_probs, test_probs=test_probs)
    print(f"  cached probabilities to {cache_path}")
    return val_probs, test_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default="0.05,0.1,0.2",
                        help="Miscoverage level(s); comma-separated")
    parser.add_argument("--lam", default="0.05,0.2,0.5",
                        help="SAPS penalty weight(s); comma-separated")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use u=1 instead of Uniform(0,1). Gives conservative coverage.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--refit", action="store_true",
                        help="Retrain the base model even if cached probs exist.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    alphas = [float(a) for a in args.alpha.split(",") if a.strip()]
    lams = [float(l) for l in args.lam.split(",") if l.strip()]

    data_dir = get_data_dir()
    print(f"Loading data from {data_dir} ...")
    X_train, y_train_raw = load_split("train")
    X_val, y_val_raw = load_split("val")
    X_test, y_test_raw = load_split("test")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(
        pd.concat([y_train_raw, y_val_raw, y_test_raw], ignore_index=True)
    )
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    n_classes = len(le.classes_)
    print(f"  train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}  classes: {n_classes}")

    cache_path = Path(__file__).parent.parent / "reports" / "saps_cache.npz"
    val_probs, test_probs = fit_base_model(
        X_train, y_train, X_val, X_test,
        n_estimators=args.n_estimators, n_jobs=args.n_jobs, device=device,
        seed=args.seed, cache_path=cache_path, refit=args.refit,
    )

    rng = np.random.default_rng(args.seed)
    if args.deterministic:
        u_cal = np.ones(len(y_val))
        u_test = np.ones(len(y_test))
    else:
        u_cal = rng.uniform(size=len(y_val))
        u_test = rng.uniform(size=len(y_test))

    rows = []
    for alpha in alphas:
        for lam in lams:
            cal_scores = saps_cal_scores(val_probs, y_val, u_cal, lam)
            tau = conformal_quantile(cal_scores, alpha)
            metrics = evaluate_sets(test_probs, y_test, tau, u_test, lam)
            row = {
                "alpha": alpha,
                "target_coverage": round(1 - alpha, 3),
                "lam": lam,
                "tau": round(tau, 4),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            }
            rows.append(row)
            print(
                f"alpha={alpha:.2f}  lam={lam:.2f}  tau={tau:.3f}  "
                f"cov={metrics['coverage']:.3f}  avg_size={metrics['avg_set_size']:.2f}  "
                f"singleton={metrics['pct_singleton']:.2f}  empty={metrics['pct_empty']:.3f}"
            )

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "reports" / "conformal_saps_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

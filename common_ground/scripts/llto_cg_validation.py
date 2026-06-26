"""LLTO validation for Common Ground (CG and CG+RAPS-singleton).

This is the redux LLTO setup discussed on 2026-05-19.

Spatial folds
  KMeans(k=n_folds) on (lon, lat) of the stable parquet. Each stable
  point is assigned a fold; unstable points are assigned the *same*
  fold via nearest-cluster-centroid lookup, so a fold = a spatial
  region across both parquets.

Per fold k
  train_stable    = stable rows where fold != k         (year=2020)
  train_unstable  = unstable rows where fold != k       (years 2017..2025)
  test_stable     = stable rows where fold == k         (year=2020)

  Stage 1: CatBoost on train_stable.
  Pseudo-labels: Stage 1 -> predict each (point, year) in train_unstable.
  Optional filter: RAPS singleton sets at alpha (calibrated on a
    stratified 20% slice held out of train_stable -- the slice is
    rebuilt per fold so it never overlaps the test fold).
  Stage 2: CatBoost on train_stable + pseudo-labelled train_unstable.
           Two variants saved per fold:
             cg_llto            -- no filter
             cg_llto_raps       -- RAPS singleton filter only

  Test: Stage 2 predicts test_stable; metrics vs grunnkart class.

What's deferred
  The "naive" arm of this redux compares against a Stage 1 model
  applied to OOF stable points at all years 2017..2025, but stable
  parquet currently has year=2020 only. The companion extraction
  script scripts/extraction/sample_feature_space_stable_allyears.py
  produces that data; once it lands, run llto_naive_validation.py.

Usage
  ~/myprojects/recover/.venv/bin/python \\
      common_ground/scripts/llto_cg_validation.py
  ~/myprojects/recover/.venv/bin/python \\
      common_ground/scripts/llto_cg_validation.py --n_folds 5 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import resolve_device  # noqa: E402

DATA_DIR = _REPO / "data"
OUT_REPORTS = _REPO / "common_ground" / "reports" / "llto_cg"
OUT_CM = OUT_REPORTS / "confusion"

STABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"

RAPS_LAM = 0.01
RAPS_KREG = 1
DEFAULT_ALPHA = 0.10


# ── Data ────────────────────────────────────────────────────────────
def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    df = df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id", "cluster",
                                          "lon", "lat"])
    return df.reset_index(drop=True)


# ── Model ───────────────────────────────────────────────────────────
def make_catboost(iterations, seed, n_jobs, device):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations, random_seed=seed, verbose=False,
        allow_writing_files=False, thread_count=n_jobs,
        task_type="GPU" if device == "cuda" else "CPU",
        loss_function="MultiClass",
    )


# ── RAPS (lifted from scripts/conformal_compare.py) ────────────────
def _ranks_and_sorted(probs: np.ndarray):
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    ranks = np.empty_like(order)
    row = np.arange(len(probs))[:, None]
    ranks[row, order] = np.arange(1, probs.shape[1] + 1)[None, :]
    return ranks, order, sorted_probs


def raps_scores(probs, u, lam=RAPS_LAM, k_reg=RAPS_KREG):
    ranks, _, sorted_probs = _ranks_and_sorted(probs)
    n, _ = probs.shape
    cumsum_sorted = np.cumsum(sorted_probs, axis=1)
    cumsum_before_rank = np.concatenate(
        [np.zeros((n, 1), dtype=sorted_probs.dtype), cumsum_sorted[:, :-1]],
        axis=1)
    cum_at_class = np.take_along_axis(cumsum_before_rank, ranks - 1, axis=1)
    scores = cum_at_class + u[:, None] * probs
    if lam > 0:
        scores = scores + lam * np.maximum(0, ranks - k_reg)
    return scores


def raps_calibrate(probs_cal, y_cal, alpha, rng):
    n = len(y_cal)
    u = rng.uniform(0, 1, size=n).astype(np.float32)
    all_s = raps_scores(probs_cal, u)
    cal_s = all_s[np.arange(n), y_cal]
    q_level = float(np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0))
    return float(np.quantile(cal_s, q_level, method="higher"))


def raps_singleton_mask(probs, tau, rng):
    u = rng.uniform(0, 1, size=len(probs)).astype(np.float32)
    sets_ = raps_scores(probs, u) <= tau
    return (sets_.sum(axis=1) == 1), sets_


# ── Plotting ────────────────────────────────────────────────────────
def plot_cm(cm: np.ndarray, classes: list, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    M = cm.astype(float)
    row_sums = M.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(row_sums > 0, M / row_sums, 0.0)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class (grunnkart)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            color = "white" if M[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center", color=color, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ── Pipeline ────────────────────────────────────────────────────────
def run(args):
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_CM.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"device={device}  iterations={args.iterations}  "
          f"n_folds={args.n_folds}  alpha={args.alpha}  "
          f"cal_frac={args.cal_frac}  seed={args.seed}")

    print(f"\nLoading stable    : {STABLE_PARQUET.name}")
    stable = load_parquet(STABLE_PARQUET)
    print(f"  {len(stable):,} rows  cells={stable['cell_id'].nunique()}  "
          f"years={sorted(stable['year'].unique().tolist())}")

    print(f"Loading unstable  : {UNSTABLE_PARQUET.name}")
    unstable = load_parquet(UNSTABLE_PARQUET)
    print(f"  {len(unstable):,} rows  cells={unstable['cell_id'].nunique()}  "
          f"years={sorted(unstable['year'].unique().tolist())}")

    classes = sorted(set(stable[TARGET].astype(int).unique()) |
                     set(int(c) for c in unstable[TARGET].unique()))
    le = LabelEncoder().fit(np.array(classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    print(f"  classes (decoded): {decoded}")

    # ---- Build folds: KMeans on stable (lon, lat); assign unstable
    #      to the same clusters by transform. ------------------------
    km = KMeans(n_clusters=args.n_folds, random_state=args.seed, n_init=10)
    stable_fold = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())
    print("\n  LLTO folds (stable | unstable):")
    for k in range(args.n_folds):
        print(f"    fold {k}: stable={int((stable_fold==k).sum()):,}  "
              f"unstable={int((unstable_fold==k).sum()):,}")

    # ---- Per-fold loop ----
    rng = np.random.default_rng(args.seed)
    rows = []
    per_class_rows = []
    cm_accum = {s: np.zeros((n_classes, n_classes), dtype=np.int64)
                for s in ("cg_llto", "cg_llto_raps")}

    for k in range(args.n_folds):
        t_fold = time.perf_counter()
        s_test_mask = (stable_fold == k)
        s_train_mask = ~s_test_mask
        u_train_mask = (unstable_fold != k)

        print(f"\n=== fold {k}  "
              f"stable_train={int(s_train_mask.sum()):,}  "
              f"stable_test={int(s_test_mask.sum()):,}  "
              f"unstable_train={int(u_train_mask.sum()):,} ===")
        if s_test_mask.sum() == 0:
            print("  skipped (empty test fold)")
            continue

        X_s_train = stable.loc[s_train_mask, FEATURE_COLS].reset_index(drop=True)
        y_s_train = le.transform(stable.loc[s_train_mask, TARGET]
                                 .astype(int).values)
        X_s_test = stable.loc[s_test_mask, FEATURE_COLS].reset_index(drop=True)
        y_s_test = le.transform(stable.loc[s_test_mask, TARGET]
                                .astype(int).values)
        X_u_train = unstable.loc[u_train_mask, FEATURE_COLS].reset_index(drop=True)

        # ---- Conformal-cal slice carved from this fold's stable_train ----
        idx_tr, idx_cal = train_test_split(
            np.arange(len(X_s_train)), test_size=args.cal_frac,
            stratify=y_s_train, random_state=args.seed)
        X_s_t = X_s_train.iloc[idx_tr].reset_index(drop=True)
        y_s_t = y_s_train[idx_tr]
        X_s_c = X_s_train.iloc[idx_cal].reset_index(drop=True)
        y_s_c = y_s_train[idx_cal]
        print(f"  stage1_train={len(X_s_t):,}  "
              f"raps_cal={len(X_s_c):,}")

        # ---- Stage 1 ----
        stage1 = make_catboost(args.iterations, args.seed,
                               args.n_jobs, device)
        t0 = time.perf_counter()
        stage1.fit(X_s_t, y_s_t)
        print(f"  stage1 fit in {time.perf_counter()-t0:.1f}s")

        # ---- RAPS calibration ----
        probs_cal = stage1.predict_proba(X_s_c)
        tau = raps_calibrate(probs_cal, y_s_c, args.alpha, rng)
        sing_cal, _ = raps_singleton_mask(probs_cal, tau, rng)
        cov_cal = float((probs_cal.argmax(axis=1) == y_s_c).mean())
        print(f"  RAPS tau={tau:.4f}  "
              f"cal_singleton={sing_cal.mean()*100:.1f}%  "
              f"top1_acc_cal={cov_cal:.3f}")

        # ---- Pseudo-labels on unstable_train (all years) ----
        probs_u = stage1.predict_proba(X_u_train)
        pseudo_u = probs_u.argmax(axis=1)
        sing_mask_u, _ = raps_singleton_mask(probs_u, tau, rng)
        kept_pct = float(sing_mask_u.mean() * 100)
        print(f"  pseudo-labels (unstable_train): n={len(pseudo_u):,}  "
              f"singletons={int(sing_mask_u.sum()):,} "
              f"({kept_pct:.1f}%)")

        # ---- Build Stage 2 train sets ----
        # Stage 1 was fit on X_s_t, but for Stage 2 we use ALL of
        # X_s_train (the conformal cal slice is okay to include here
        # because the conformal guarantee is already locked in -- we
        # only need disjointness with the *test fold*, not within the
        # training fold).
        for setup, mask in (("cg_llto", np.ones(len(X_u_train), dtype=bool)),
                            ("cg_llto_raps", sing_mask_u)):
            X_u_kept = X_u_train.iloc[mask].reset_index(drop=True)
            y_u_kept = pseudo_u[mask]
            X_aug = pd.concat([X_s_train, X_u_kept], ignore_index=True)
            y_aug = np.concatenate([y_s_train, y_u_kept])
            stage2 = make_catboost(args.iterations, args.seed,
                                   args.n_jobs, device)
            t0 = time.perf_counter()
            stage2.fit(X_aug, y_aug)
            fit_s = time.perf_counter() - t0
            y_pred = np.asarray(stage2.predict(X_s_test)).ravel().astype(int)
            f1 = f1_score(y_s_test, y_pred, average="macro",
                          labels=np.arange(n_classes), zero_division=0)
            bal = balanced_accuracy_score(y_s_test, y_pred)
            cm = confusion_matrix(y_s_test, y_pred,
                                  labels=np.arange(n_classes))
            cm_accum[setup] += cm
            print(f"    [{setup}] fit={fit_s:.1f}s  "
                  f"f1={f1:.4f}  bal={bal:.4f}  "
                  f"n_train={len(X_aug):,}  n_test={len(X_s_test):,}")
            rows.append({"fold": k, "setup": setup,
                         "n_stage1_train": int(len(X_s_t)),
                         "n_stage2_train": int(len(X_aug)),
                         "n_pseudo": int(mask.sum()),
                         "n_test": int(len(X_s_test)),
                         "raps_tau": round(float(tau), 4),
                         "raps_kept_pct": round(kept_pct, 1),
                         "fit_s": round(fit_s, 2),
                         "f1_macro": round(float(f1), 4),
                         "balanced_accuracy": round(float(bal), 4)})
            # Per-class F1
            f1_pc = f1_score(y_s_test, y_pred,
                             labels=np.arange(n_classes),
                             average=None, zero_division=0)
            for c_enc, score in enumerate(f1_pc):
                per_class_rows.append({"fold": k, "setup": setup,
                                       "class": int(decoded[c_enc]),
                                       "f1": round(float(score), 4),
                                       "support": int(
                                           (y_s_test == c_enc).sum())})
            # Per-fold CM
            cm_df = pd.DataFrame(cm, index=decoded, columns=decoded)
            cm_df.index.name = "true_class"
            cm_df.to_csv(OUT_CM / f"cm_{setup}_fold{k}.csv")
            plot_cm(cm, decoded,
                    f"{setup}  fold={k}   macro-F1={f1:.3f} "
                    f"(n={len(y_s_test)})",
                    OUT_CM / f"cm_{setup}_fold{k}.png")
        print(f"  fold {k} total: {time.perf_counter()-t_fold:.1f}s")

    # ---- Aggregate CMs across folds ----
    for setup, cm in cm_accum.items():
        cm_df = pd.DataFrame(cm, index=decoded, columns=decoded)
        cm_df.index.name = "true_class"
        cm_df.to_csv(OUT_CM / f"cm_{setup}_all_folds.csv")
        title = (f"{setup}  ALL folds aggregated "
                 f"(n={cm.sum()}, row-normalised)")
        plot_cm(cm, decoded, title,
                OUT_CM / f"cm_{setup}_all_folds.png")
    print(f"\nSaved aggregated CMs to {OUT_CM}")

    # ---- Persist scores ----
    pf = pd.DataFrame(rows)
    pf_path = OUT_REPORTS / "llto_cg_per_fold.csv"
    pf.to_csv(pf_path, index=False)

    summary = (pf.groupby("setup")
                  .agg(f1_mean=("f1_macro", "mean"),
                       f1_std=("f1_macro", "std"),
                       bal_acc_mean=("balanced_accuracy", "mean"),
                       bal_acc_std=("balanced_accuracy", "std"),
                       raps_kept_pct_mean=("raps_kept_pct", "mean"),
                       n_folds=("f1_macro", "count"))
                  .reset_index())
    sum_path = OUT_REPORTS / "llto_cg_summary.csv"
    summary.to_csv(sum_path, index=False)
    print("\n--- LLTO CG summary (macro-F1 mean ± std across folds) ---")
    print(summary.to_string(index=False))

    pc_df = pd.DataFrame(per_class_rows)
    pc_path = OUT_REPORTS / "llto_cg_per_class_f1.csv"
    pc_df.to_csv(pc_path, index=False)

    info = {
        "config": vars(args) | {"device": device,
                                "raps_lam": RAPS_LAM,
                                "raps_kreg": RAPS_KREG},
        "classes": decoded,
        "per_fold": str(pf_path),
        "summary": str(sum_path),
        "per_class": str(pc_path),
        "confusion_dir": str(OUT_CM),
    }
    with open(OUT_REPORTS / "llto_cg_run.json", "w") as f:
        json.dump(info, f, indent=2, default=str)


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--cal_frac", type=float, default=0.20)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

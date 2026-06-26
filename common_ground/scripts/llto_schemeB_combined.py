"""Scheme B LLTO with combined FSCS+Megan stable dataset.

Stacks the three best findings:
  1. Combined stable parquet (FSCS + Megan, 159k rows)
  2. Biased cls5 support draw (class 5 forced to 4 000 rows in 25k support)
  3. Cross-conformal APS Stage-1 filter (K=5 inner folds, α=0.10)

Pipeline per outer fold:
  - Spatial folds (KMeans k=3) derived from FSCS rows only (Megan has no lon/lat).
  - Megan rows are always in stable_train (never test).
  - Stage-1: CatBoost-500 CPU cross-conformal on FSCS stable_train only
    (Megan lacks ground-truth coordinates needed for LLTO integrity;
     we calibrate CP on FSCS rows, then apply filter to unstable).
  - Stage-2: TabICL n16 kvcache on biased_cls5 25k support drawn from
    (FSCS stable_train + Megan train-split rows + kept pseudo-labelled unstable).

Conditions:
  fscs_only_ccaps      FSCS stable only + cc_aps filter (reference from round cc)
  combined_nofilter    Combined stable + no CP filter
  combined_ccaps       Combined stable + cc_aps filter  ← primary hypothesis

Output:
  common_ground/reports/research/schemeB_combined_results.json
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import resolve_device  # noqa: E402

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

FSCS_PARQUET     = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
COMBINED_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth_combined.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET    = "class"
MERGE_MAP = {1: 2, 9: 8}   # Scheme B
N_FOLDS   = 3
K_INNER   = 5
SEED      = 0
ALPHA     = 0.10
TOTAL_SUP = 25_000
CLS5_FORCE = 4_000
REF_CCAPS = 0.6961   # cc_aps FSCS-only from cross-conformal run
REF_BEST  = 0.6927   # cb_biased_cls5 no-filter
WEAK_ORIG = [2, 5, 7]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet(path: Path, require_lonlat: bool = True) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    required = FEATURE_COLS + [TARGET]
    if require_lonlat:
        required += ["lon", "lat"]
    return df.dropna(subset=required).reset_index(drop=True)


def merge_classes(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype(int)
    for src, tgt in MERGE_MAP.items():
        y[y == src] = tgt
    return y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clear_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def biased_subsample(X, y, n_total, enc5, rng):
    forced_mask = np.zeros(len(y), dtype=bool)
    idx5 = np.flatnonzero(y == enc5)
    take5 = min(CLS5_FORCE, len(idx5))
    ch5 = rng.choice(idx5, size=take5, replace=False)
    forced_mask[ch5] = True
    pool = np.flatnonzero(~forced_mask)
    take_rem = min(n_total - take5, len(pool))
    ch_rem = rng.choice(pool, size=take_rem, replace=False)
    idx = np.concatenate([ch5, ch_rem])
    perm = rng.permutation(len(idx))
    return X[idx[perm]], y[idx[perm]]


def random_subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def score_metrics(y_true, y_pred, n_classes):
    f1m  = f1_score(y_true, y_pred, average="macro",
                    labels=np.arange(n_classes), zero_division=0)
    bal  = balanced_accuracy_score(y_true, y_pred)
    f1pc = f1_score(y_true, y_pred, labels=np.arange(n_classes),
                    average=None, zero_division=0)
    return float(f1m), float(bal), [float(v) for v in f1pc]


def predict_batched(clf, X, batch=10_000):
    parts = []
    for i in range(0, len(X), batch):
        parts.append(clf.predict_proba(X[i:i+batch]).astype(np.float64))
        clear_cuda()
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def make_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU", loss_function="MultiClass",
    )


# ---------------------------------------------------------------------------
# CP helpers (APS cross-conformal)
# ---------------------------------------------------------------------------

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


def aps_prediction_sets(probs, u, tau):
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = probs.shape[0]
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    s = cum_at + u[:, None] * probs
    return s <= tau


def cross_conformal_aps(X_tr, y_tr, K, seed):
    """K-fold cross-conformal APS calibration. Returns pooled cal scores."""
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    pooled = np.empty(len(y_tr), dtype=np.float64)
    u_all  = np.random.default_rng(seed).uniform(size=len(y_tr))
    for _, (idx_t, idx_v) in enumerate(skf.split(X_tr, y_tr)):
        cb = make_catboost()
        cb.fit(X_tr[idx_t], y_tr[idx_t])
        p_v = cb.predict_proba(X_tr[idx_v]).astype(np.float64)
        pooled[idx_v] = aps_cal_scores(p_v, y_tr[idx_v], u_all[idx_v])
    return pooled


def stage1_quality(sets, pseudo_argmax, y_true, n_classes, weak_enc):
    sizes = sets.sum(axis=1)
    sing  = sizes == 1
    n_kept = int(sing.sum())
    if n_kept == 0:
        return {"n_kept": 0, "pct_kept": 0.0,
                "pseudo_acc": float("nan"), "pseudo_f1_weak": float("nan")}
    kept_idx  = np.flatnonzero(sing)
    pseudo_kp = sets[kept_idx].argmax(axis=1)
    true_kp   = y_true[kept_idx]
    acc = float((pseudo_kp == true_kp).mean())
    f1w = float(f1_score(true_kp, pseudo_kp, labels=weak_enc,
                         average="macro", zero_division=0))
    return {"n_kept": n_kept,
            "pct_kept": round(100.0 * n_kept / len(sing), 2),
            "pseudo_acc": round(acc, 4),
            "pseudo_f1_weak": round(f1w, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  K_inner={K_INNER}  "
          f"alpha={ALPHA}  experiment=combined")

    # Load datasets
    # FSCS: has lon/lat for spatial folding
    fscs = load_parquet(FSCS_PARQUET, require_lonlat=True)
    # Combined: Megan rows have NaN lon/lat — load without requiring them
    combined_full = load_parquet(COMBINED_PARQUET, require_lonlat=False)
    unstable = load_parquet(UNSTABLE_PARQUET, require_lonlat=True)

    # Separate Megan rows (no lon/lat) from FSCS rows in combined parquet
    megan_rows = combined_full[combined_full["source"] == "megan"].reset_index(drop=True)
    # Verify FSCS rows in combined match standalone FSCS parquet
    fscs_in_combined = combined_full[combined_full["source"] == "fscs"].reset_index(drop=True)
    print(f"FSCS rows: {len(fscs):,}  "
          f"Megan rows in combined: {len(megan_rows):,}")

    # Apply Scheme B merge to all datasets
    y_fscs_m    = merge_classes(fscs[TARGET].values)
    y_megan_m   = merge_classes(megan_rows[TARGET].values)
    y_unstable_m = merge_classes(unstable[TARGET].values)

    merged_classes = sorted(
        set(y_fscs_m.tolist()) |
        set(y_megan_m.tolist()) |
        set(y_unstable_m.tolist())
    )
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded   = le.classes_.tolist()
    print(f"Scheme B classes: {decoded}  n={n_classes}")

    enc5 = int(le.transform([5])[0])
    weak_enc = [int(le.transform([c])[0]) for c in WEAK_ORIG if c in le.classes_]

    # Spatial folds from FSCS only
    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    fscs_fold     = km.fit_predict(fscs[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    # Feature arrays
    X_megan = megan_rows[FEATURE_COLS].values.astype(np.float32)
    y_megan = le.transform(y_megan_m)

    results: dict[str, list] = {}
    descs:   dict[str, str]  = {}
    t_global = time.perf_counter()

    conditions = [
        ("fscs_only_ccaps",
         "FSCS stable only + cross-conformal APS filter (cc reference)"),
        ("combined_nofilter",
         "Combined FSCS+Megan stable + no CP filter"),
        ("combined_ccaps",
         "Combined FSCS+Megan stable + cross-conformal APS filter"),
    ]

    for k in range(N_FOLDS):
        s_test  = (fscs_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        # FSCS train/test splits
        X_fscs_tr = fscs.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_fscs_tr = le.transform(y_fscs_m[s_train])
        X_fscs_te = fscs.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_fscs_te = le.transform(y_fscs_m[s_test])

        # Unstable
        X_u = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)
        y_u = le.transform(y_unstable_m[u_train])

        print(f"\n=== fold {k}  fscs_train={len(y_fscs_tr):,}  "
              f"megan={len(y_megan):,}  unstable={len(X_u):,}  "
              f"test={len(y_fscs_te):,} ===")

        rng = np.random.default_rng(SEED + k)

        # Cross-conformal calibration on FSCS train (used by cc conditions)
        print(f"  Running {K_INNER}-fold CC-APS on FSCS stable_train...")
        t_cc = time.perf_counter()
        pooled_scores = cross_conformal_aps(X_fscs_tr, y_fscs_tr, K_INNER, SEED + k)
        tau_cc = conformal_quantile(pooled_scores, ALPHA)
        print(f"  CC done in {time.perf_counter()-t_cc:.1f}s  tau={tau_cc:.4f}")

        # Final Stage-1 CatBoost on FSCS stable_train → pseudo-labels for unstable
        cb_final = make_catboost()
        cb_final.fit(X_fscs_tr, y_fscs_tr)
        probs_u = cb_final.predict_proba(X_u).astype(np.float64)
        pseudo_argmax = probs_u.argmax(axis=1)
        u_rng_u = rng.uniform(size=len(X_u))

        # CP sets for unstable
        sets_cc = aps_prediction_sets(probs_u, u_rng_u, tau_cc)
        q1_cc = stage1_quality(sets_cc, pseudo_argmax, y_u, n_classes, weak_enc)
        q1_nf = {"n_kept": len(X_u), "pct_kept": 100.0,
                  "pseudo_acc": round(float((pseudo_argmax == y_u).mean()), 4),
                  "pseudo_f1_weak": float("nan")}

        sizes_cc = sets_cc.sum(axis=1)
        kept_cc  = sizes_cc == 1
        X_u_cc   = X_u[kept_cc]
        y_u_cc   = sets_cc[kept_cc].argmax(axis=1)

        for cond_name, cond_desc in conditions:
            clear_cuda()

            use_megan  = "combined" in cond_name
            use_cc     = "ccaps" in cond_name

            # Build stable pool
            if use_megan:
                X_stable = np.concatenate([X_fscs_tr, X_megan])
                y_stable = np.concatenate([y_fscs_tr, y_megan])
            else:
                X_stable = X_fscs_tr
                y_stable = y_fscs_tr

            # Pseudo-labelled unstable
            if use_cc:
                X_u_use = X_u_cc
                y_u_use = y_u_cc
                q1 = q1_cc
            else:
                X_u_use = X_u
                y_u_use = pseudo_argmax
                q1 = q1_nf

            X_aug = np.concatenate([X_stable, X_u_use])
            y_aug = np.concatenate([y_stable, y_u_use])

            X_sup, y_sup = biased_subsample(X_aug, y_aug, TOTAL_SUP, enc5, rng)

            sup_cnt = np.bincount(y_sup, minlength=n_classes)
            print(f"  [{cond_name:<22}] n_aug={len(y_aug):,}  "
                  f"n_sup={len(y_sup):,}  "
                  f"cls5={sup_cnt[enc5]}({100*sup_cnt[enc5]/len(y_sup):.1f}%)  "
                  f"pseudo_kept={q1['n_kept']}({q1['pct_kept']:.1f}%)")

            from tabicl import TabICLClassifier
            try:
                stage2 = TabICLClassifier(
                    n_estimators=16, kv_cache=True,
                    random_state=SEED, verbose=False, device="cuda")
                t_fit = time.perf_counter()
                stage2.fit(X_sup, y_sup)
                t_fit = time.perf_counter() - t_fit
                t_pred = time.perf_counter()
                probs_te = predict_batched(stage2, X_fscs_te)
                t_pred = time.perf_counter() - t_pred
                pred_te = probs_te.argmax(axis=1)
                f1m, bal, pc = score_metrics(y_fscs_te, pred_te, n_classes)
                del stage2
                print(f"  [{cond_name:<22}] F1={f1m:.4f}  bal={bal:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s")
                _record(results, descs, cond_name, cond_desc,
                        k, f1m, bal, pc, t_fit, t_pred, len(y_sup), q1)
            except Exception as e:
                print(f"  [{cond_name:<22}] FAILED: {type(e).__name__}: {str(e)[:100]}")
                _record_fail(results, descs, cond_name, cond_desc,
                             k, n_classes, len(X_aug))
            clear_cuda()

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── summarise ────────────────────────────────────────────────────
    summary = {}
    for name, folds in results.items():
        f1s  = [f["f1_macro"] for f in folds if not np.isnan(f["f1_macro"])]
        bals = [f["bal_acc"]  for f in folds if not np.isnan(f["bal_acc"])]
        valid_pcs = [f["f1_per_class"] for f in folds
                     if not any(np.isnan(v) for v in f["f1_per_class"])]
        pc_mean = (np.array(valid_pcs).mean(axis=0).tolist()
                   if valid_pcs else [float("nan")] * n_classes)
        q1_keys = ["n_kept", "pct_kept", "pseudo_acc", "pseudo_f1_weak"]
        q1_mean = {k2: round(float(np.nanmean([f.get(k2, float("nan"))
                                                for f in folds])), 4)
                   for k2 in q1_keys}
        f1_mean = float(np.mean(f1s)) if f1s else float("nan")
        summary[name] = {
            "description": descs[name],
            "f1_mean":  round(f1_mean, 4),
            "f1_std":   round(float(np.std(f1s)), 4) if f1s else float("nan"),
            "bal_mean": round(float(np.mean(bals)), 4) if bals else float("nan"),
            "f1_per_class": {str(decoded[i]): round(v, 4)
                             for i, v in enumerate(pc_mean)},
            "stage1_mean": q1_mean,
            "per_fold": folds,
        }

    out_json = OUT_DIR / "schemeB_combined_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": "B", "merged_classes": decoded, "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1), "n_folds": N_FOLDS,
                   "references": {"cc_aps_fscs_only": REF_CCAPS,
                                  "cb_biased_cls5_no_filter": REF_BEST},
                   "results": summary}, f, indent=2)
    print(f"\nSaved → {out_json}")

    # ── leaderboard ──────────────────────────────────────────────────
    weak_classes = [2, 5, 7]
    print(f"\n{'Condition':<24}  {'F1':>8}  {'std':>6}  "
          f"{'Δ cc_aps':>9}  {'Δ best':>8}")
    print("-" * 62)
    for name, s in sorted(summary.items(),
                           key=lambda x: -(x[1]["f1_mean"]
                                           if x[1]["f1_mean"] == x[1]["f1_mean"] else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<24}  [FAIL]")
        else:
            d1 = s["f1_mean"] - REF_CCAPS
            d2 = s["f1_mean"] - REF_BEST
            flag = " ★" if s["f1_mean"] > REF_CCAPS else ""
            print(f"{name:<24}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  "
                  f"{d1:>+9.4f}  {d2:>+8.4f}{flag}")

    print(f"\nPer-class F1 (cls 2=bare, 5=grassland, 7=wetland, 12=snow):")
    show_cls = [2, 5, 7, 12]
    print(f"{'Condition':<24}  " + "  ".join(f"cls{c:>2}" for c in show_cls))
    print("-" * 58)
    for name, s in summary.items():
        pc = s["f1_per_class"]
        vals = "  ".join(f"{pc.get(str(c), float('nan')):>6.4f}"
                         for c in show_cls)
        print(f"{name:<24}  {vals}")


def _record(results, descs, name, desc, fold, f1, bal, pc,
            fit_s, pred_s, n_sup, q1):
    if name not in results:
        results[name] = []
        descs[name] = desc
    results[name].append({
        "fold": fold, "f1_macro": round(f1, 4), "bal_acc": round(bal, 4),
        "fit_s": round(fit_s, 2), "pred_s": round(pred_s, 2),
        "n_support": n_sup, "f1_per_class": [round(v, 4) for v in pc],
        **q1,
    })


def _record_fail(results, descs, name, desc, fold, n_classes, n_sup):
    _record(results, descs, name, desc, fold,
            float("nan"), float("nan"), [float("nan")] * n_classes,
            0.0, 0.0, n_sup,
            {"n_kept": 0, "pct_kept": 0.0,
             "pseudo_acc": float("nan"), "pseudo_f1_weak": float("nan")})


if __name__ == "__main__":
    run()

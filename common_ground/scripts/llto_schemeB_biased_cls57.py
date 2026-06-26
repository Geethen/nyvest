"""Scheme B: biased support draw for classes 5 and 7.

Round 4a showed biasing cls5 to ~17% is the best single lever (+0.0046).
Class 7 (wetland, F1=0.62) is also data-helpful and hasn't been biased yet.
This script tests forcing both cls5 and cls7 to higher support fractions,
combined with the best pipeline (cc_aps filter, CatBoost Stage-1).

Conditions (all: cc_aps α=0.10 filter):
  cls5_only        Force cls5→4000, rest random             (current best config re-run)
  cls57_v1         Force cls5→4000, cls7→2500, rest random  (cls7 at ~10%)
  cls57_v2         Force cls5→4000, cls7→3500, rest random  (cls7 at ~14%)
  cls7_only        Force cls7→3500, rest random             (cls7 in isolation)

Reference: cc_aps α=0.10 + biased_cls5 = F1 0.6961

Output:
  common_ground/reports/research/schemeB_biased_cls57_results.json
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

STABLE_PARQUET   = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET    = "class"
MERGE_MAP = {1: 2, 9: 8}
N_FOLDS   = 3
K_INNER   = 5
SEED      = 0
ALPHA     = 0.10
TOTAL_SUP = 25_000
REF_CCAPS = 0.6961
REF_BEST  = 0.6927
WEAK_ORIG = [2, 5, 7]


def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "lon", "lat"]).reset_index(drop=True)


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


def multi_biased_subsample(X, y, n_total, forced_counts: dict, rng):
    """Draw n_total rows forcing exact counts for multiple classes."""
    forced_mask = np.zeros(len(y), dtype=bool)
    parts_idx = []
    for enc_c, n_force in forced_counts.items():
        idx_c = np.flatnonzero(y == enc_c)
        take  = min(n_force, len(idx_c))
        chosen = rng.choice(idx_c, size=take, replace=False)
        parts_idx.append(chosen)
        forced_mask[chosen] = True
    n_forced = sum(len(p) for p in parts_idx)
    pool = np.flatnonzero(~forced_mask)
    take_rem = min(n_total - n_forced, len(pool))
    if take_rem > 0:
        parts_idx.append(rng.choice(pool, size=take_rem, replace=False))
    idx = np.concatenate(parts_idx)
    perm = rng.permutation(len(idx))
    return X[idx[perm]], y[idx[perm]]


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


def make_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU", loss_function="MultiClass",
    )


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
    return (cum_at + u[:, None] * probs)[np.arange(n), y]


def aps_sets(probs, u, tau):
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = probs.shape[0]
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    return (cum_at + u[:, None] * probs) <= tau


def cross_conformal_aps(X_tr, y_tr, K, seed):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    pooled = np.empty(len(y_tr), dtype=np.float64)
    u_all  = np.random.default_rng(seed).uniform(size=len(y_tr))
    for _, (idx_t, idx_v) in enumerate(skf.split(X_tr, y_tr)):
        cb = make_catboost()
        cb.fit(X_tr[idx_t], y_tr[idx_t])
        p_v = cb.predict_proba(X_tr[idx_v]).astype(np.float64)
        pooled[idx_v] = aps_cal_scores(p_v, y_tr[idx_v], u_all[idx_v])
    return pooled


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  K_inner={K_INNER}  "
          f"alpha={ALPHA}  experiment=biased_cls57")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    y_stable_m   = merge_classes(stable[TARGET].values)
    y_unstable_m = merge_classes(unstable[TARGET].values)
    merged_classes = sorted(set(y_stable_m.tolist()) | set(y_unstable_m.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded   = le.classes_.tolist()
    print(f"classes: {decoded}  n={n_classes}")

    enc5 = int(le.transform([5])[0])
    enc7 = int(le.transform([7])[0])

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    # (name, description, forced_counts dict)
    conditions = [
        ("cls5_only",
         "Force cls5→4000 (current best re-run)",
         {enc5: 4_000}),
        ("cls57_v1",
         "Force cls5→4000, cls7→2500 (~10%)",
         {enc5: 4_000, enc7: 2_500}),
        ("cls57_v2",
         "Force cls5→4000, cls7→3500 (~14%)",
         {enc5: 4_000, enc7: 3_500}),
        ("cls7_only",
         "Force cls7→3500 (~14%), cls5 random",
         {enc7: 3_500}),
    ]

    results: dict[str, list] = {}
    descs:   dict[str, str]  = {}
    t_global = time.perf_counter()

    for k in range(N_FOLDS):
        s_test  = (stable_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_s_tr = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr = le.transform(y_stable_m[s_train])
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(y_stable_m[s_test])
        X_u    = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)
        y_u    = le.transform(y_unstable_m[u_train])

        print(f"\n=== fold {k}  n_stable_train={len(y_s_tr):,}  "
              f"n_unstable={len(X_u):,}  n_test={len(y_s_te):,} ===")

        # Cross-conformal APS — shared across conditions
        print(f"  Running {K_INNER}-fold CC-APS...")
        t0 = time.perf_counter()
        pooled = cross_conformal_aps(X_s_tr, y_s_tr, K_INNER, SEED + k)
        tau = conformal_quantile(pooled, ALPHA)
        print(f"  CC done {time.perf_counter()-t0:.1f}s  tau={tau:.4f}")

        cb_final = make_catboost()
        cb_final.fit(X_s_tr, y_s_tr)
        probs_u = cb_final.predict_proba(X_u).astype(np.float64)

        rng = np.random.default_rng(SEED + k)
        u_test_u = rng.uniform(size=len(X_u))
        sets_u   = aps_sets(probs_u, u_test_u, tau)
        sing_u   = (sets_u.sum(axis=1) == 1)
        X_u_kept = X_u[sing_u]
        y_u_kept = sets_u[sing_u].argmax(axis=1)
        pseudo_acc = float((y_u_kept == y_u[sing_u]).mean()) if sing_u.sum() > 0 else float("nan")
        print(f"  CC filter: {sing_u.sum()} kept ({100*sing_u.mean():.1f}%)  "
              f"pseudo_acc={pseudo_acc:.4f}")

        X_aug = np.concatenate([X_s_tr, X_u_kept])
        y_aug = np.concatenate([y_s_tr, y_u_kept])
        aug_cnt = np.bincount(y_aug, minlength=n_classes)
        print(f"  aug: cls5_avail={aug_cnt[enc5]}  cls7_avail={aug_cnt[enc7]}")

        for cond_name, cond_desc, forced in conditions:
            clear_cuda()
            X_sup, y_sup = multi_biased_subsample(X_aug, y_aug, TOTAL_SUP, forced, rng)
            sup_cnt = np.bincount(y_sup, minlength=n_classes)
            print(f"  [{cond_name:<12}] n_sup={len(y_sup):,}  "
                  f"cls5={sup_cnt[enc5]}({100*sup_cnt[enc5]/len(y_sup):.1f}%)  "
                  f"cls7={sup_cnt[enc7]}({100*sup_cnt[enc7]/len(y_sup):.1f}%)")

            from tabicl import TabICLClassifier
            try:
                stage2 = TabICLClassifier(
                    n_estimators=16, kv_cache=True,
                    random_state=SEED, verbose=False, device="cuda")
                t_fit = time.perf_counter()
                stage2.fit(X_sup, y_sup)
                t_fit = time.perf_counter() - t_fit
                t_pred = time.perf_counter()
                probs_te = predict_batched(stage2, X_s_te)
                t_pred = time.perf_counter() - t_pred
                pred_te = probs_te.argmax(axis=1)
                f1m, bal, pc = score_metrics(y_s_te, pred_te, n_classes)
                del stage2
                print(f"  [{cond_name:<12}] F1={f1m:.4f}  bal={bal:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s")
                _record(results, descs, cond_name, cond_desc,
                        k, f1m, bal, pc, t_fit, t_pred, len(y_sup),
                        int(sing_u.sum()), round(pseudo_acc, 4))
            except Exception as e:
                print(f"  [{cond_name:<12}] FAILED: {type(e).__name__}: {str(e)[:100]}")
                _record_fail(results, descs, cond_name, cond_desc, k, n_classes, len(X_aug))
            clear_cuda()

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    summary = {}
    for name, folds in results.items():
        f1s  = [f["f1_macro"] for f in folds if not np.isnan(f["f1_macro"])]
        bals = [f["bal_acc"]  for f in folds if not np.isnan(f["bal_acc"])]
        valid_pcs = [f["f1_per_class"] for f in folds
                     if not any(np.isnan(v) for v in f["f1_per_class"])]
        pc_mean = (np.array(valid_pcs).mean(axis=0).tolist()
                   if valid_pcs else [float("nan")] * n_classes)
        f1_mean = float(np.mean(f1s)) if f1s else float("nan")
        summary[name] = {
            "description": descs[name],
            "f1_mean":  round(f1_mean, 4),
            "f1_std":   round(float(np.std(f1s)), 4) if f1s else float("nan"),
            "bal_mean": round(float(np.mean(bals)), 4) if bals else float("nan"),
            "f1_per_class": {str(decoded[i]): round(v, 4)
                             for i, v in enumerate(pc_mean)},
            "per_fold": folds,
        }

    out_json = OUT_DIR / "schemeB_biased_cls57_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": "B", "merged_classes": decoded, "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1), "n_folds": N_FOLDS,
                   "references": {"cc_aps_a10_cls5only": REF_CCAPS,
                                  "cb_biased_cls5_no_filter": REF_BEST},
                   "results": summary}, f, indent=2)
    print(f"\nSaved → {out_json}")

    show_cls = [2, 5, 7]
    print(f"\n{'Condition':<14}  {'F1':>8}  {'std':>6}  {'Δ ref':>8}")
    print("-" * 42)
    for name, s in sorted(summary.items(),
                           key=lambda x: -(x[1]["f1_mean"]
                                           if x[1]["f1_mean"] == x[1]["f1_mean"] else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<14}  [FAIL]")
        else:
            d = s["f1_mean"] - REF_CCAPS
            flag = " ★" if d > 0 else ""
            print(f"{name:<14}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  "
                  f"{d:>+8.4f}{flag}")

    print(f"\nPer-class F1 (cls 2, 5, 7):")
    print(f"{'Condition':<14}  " + "  ".join(f"cls{c:>2}" for c in show_cls))
    print("-" * 40)
    for name, s in summary.items():
        pc = s["f1_per_class"]
        vals = "  ".join(f"{pc.get(str(c), float('nan')):>6.4f}" for c in show_cls)
        print(f"{name:<14}  {vals}")


def _record(results, descs, name, desc, fold, f1, bal, pc,
            fit_s, pred_s, n_sup, n_kept, pseudo_acc):
    if name not in results:
        results[name] = []
        descs[name] = desc
    results[name].append({
        "fold": fold, "f1_macro": round(f1, 4), "bal_acc": round(bal, 4),
        "fit_s": round(fit_s, 2), "pred_s": round(pred_s, 2),
        "n_support": n_sup, "n_pseudo_kept": n_kept, "pseudo_acc": pseudo_acc,
        "f1_per_class": [round(v, 4) for v in pc],
    })


def _record_fail(results, descs, name, desc, fold, n_classes, n_sup):
    _record(results, descs, name, desc, fold,
            float("nan"), float("nan"), [float("nan")] * n_classes,
            0.0, 0.0, n_sup, 0, float("nan"))


if __name__ == "__main__":
    run()

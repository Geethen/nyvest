"""Scheme B round-4a: biased support draw for minority classes.

Tests whether forcing minority classes 2 (bare) and 5 (grassland) to a higher
share of the 25k support set improves macro-F1 and per-class recall, without
oversampling — just a capped stratified-without-replacement draw.

Conditions (all: TabICL n_est=16, kv_cache=True, random Stage-1 CatBoost-500 CPU):
  tabicl_random_25k    Pure random 25k draw (current best, baseline re-run)
  tabicl_biased_cls5   Force class 5 to 4 000 rows (~16 %), rest random from other classes
  tabicl_biased_cls2   Force class 2 to 750 rows (~3 %), rest random from other classes
  tabicl_biased_cls25  Both: class 5 → 4 000, class 2 → 750, rest random from others

Class distribution in stable (approx):
  class 2 (bare, merged 1+2): ~952 rows  → natural share in 25k ≈ 0.15%
  class 5 (grassland):        ~4739 rows → natural share in 25k ≈ 5.6%

Output:
  common_ground/reports/research/schemeB_round4_biased_support_results.json
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
TOTAL_SUP = 25_000
REF_BEST  = 0.6881


def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id", "lon", "lat"]).reset_index(drop=True)


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
            torch.cuda.empty_cache()
    except Exception:
        pass


def biased_subsample(X: np.ndarray, y: np.ndarray, n_total: int,
                     forced_counts: dict[int, int], rng: np.random.Generator):
    """Draw n_total rows, forcing exact counts for specified encoded classes.

    forced_counts: {encoded_class_idx: n_rows_to_force}
    Draws forced_counts[c] rows for each forced class (or all available if fewer),
    then fills the remainder from the non-forced pool without replacement.
    Returns shuffled (X, y) of length <= n_total.
    """
    parts_X, parts_y = [], []
    forced_mask = np.zeros(len(y), dtype=bool)

    for enc_c, n_force in forced_counts.items():
        idx_c = np.flatnonzero(y == enc_c)
        take  = min(n_force, len(idx_c))
        chosen = rng.choice(idx_c, size=take, replace=False)
        parts_X.append(X[chosen])
        parts_y.append(y[chosen])
        forced_mask[chosen] = True

    n_forced = sum(len(p) for p in parts_y)
    n_remainder = max(0, n_total - n_forced)

    # fill remainder from non-forced rows
    pool_idx = np.flatnonzero(~forced_mask)
    take_rem = min(n_remainder, len(pool_idx))
    if take_rem > 0:
        chosen_rem = rng.choice(pool_idx, size=take_rem, replace=False)
        parts_X.append(X[chosen_rem])
        parts_y.append(y[chosen_rem])

    X_out = np.concatenate(parts_X)
    y_out = np.concatenate(parts_y)
    perm  = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def random_subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def score(y_true, y_pred, n_classes):
    f1m  = f1_score(y_true, y_pred, average="macro",
                    labels=np.arange(n_classes), zero_division=0)
    bal  = balanced_accuracy_score(y_true, y_pred)
    f1pc = f1_score(y_true, y_pred, labels=np.arange(n_classes),
                    average=None, zero_division=0)
    return float(f1m), float(bal), [float(v) for v in f1pc]


def make_catboost_cpu(iterations):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
        task_type="CPU", loss_function="MultiClass",
    )


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  scheme=B  round=4a (biased support)")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    y_stable_m   = merge_classes(stable[TARGET].values)
    y_unstable_m = merge_classes(unstable[TARGET].values)
    merged_classes = sorted(set(y_stable_m.tolist()) | set(y_unstable_m.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    print(f"classes: {decoded}  n={n_classes}")

    # encoded indices for bias targets
    enc2 = int(le.transform([2])[0])
    enc5 = int(le.transform([5])[0])
    print(f"class 2 encoded={enc2},  class 5 encoded={enc5}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    results: dict[str, list] = {}
    descs:   dict[str, str]  = {}
    t_global = time.perf_counter()

    conditions = [
        ("tabicl_random_25k",
         "Pure random 25k support (current best pipeline, baseline re-run)",
         {}),
        ("tabicl_biased_cls5",
         "Force class 5 to 4 000 rows in 25k support (~16 %)",
         {enc5: 4_000}),
        ("tabicl_biased_cls2",
         "Force class 2 to 750 rows in 25k support (~3 %)",
         {enc2: 750}),
        ("tabicl_biased_cls25",
         "Force class 5→4 000 and class 2→750 in 25k support",
         {enc5: 4_000, enc2: 750}),
    ]

    for k in range(N_FOLDS):
        s_test  = (stable_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_s_tr = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr = le.transform(y_stable_m[s_train])
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(y_stable_m[s_test])
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)

        print(f"\n=== fold {k}  n_stable_train={len(y_s_tr):,}  "
              f"n_unstable_train={len(X_u_tr):,}  n_test={len(y_s_te):,} ===")

        # Stage 1: shared CatBoost-500 CPU across all conditions
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost_cpu(500)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        pseudo_u = stage1.predict_proba(X_u_tr).astype(np.float32).argmax(axis=1)
        del stage1

        X_aug = np.concatenate([X_s_tr, X_u_tr])
        y_aug = np.concatenate([y_s_tr, pseudo_u])

        # log actual class counts available in aug
        counts = np.bincount(y_aug, minlength=n_classes)
        print(f"  aug class counts: cls2={counts[enc2]}  cls5={counts[enc5]}  total={len(y_aug):,}")

        rng = np.random.default_rng(SEED + k)

        for cond_name, cond_desc, forced in conditions:
            clear_cuda()

            if forced:
                X_sup, y_sup = biased_subsample(X_aug, y_aug, TOTAL_SUP, forced, rng)
            else:
                X_sup, y_sup = random_subsample(X_aug, y_aug, TOTAL_SUP, rng)

            # log support class shares for bias targets
            sup_counts = np.bincount(y_sup, minlength=n_classes)
            bias_info = f"cls2={sup_counts[enc2]} ({100*sup_counts[enc2]/len(y_sup):.1f}%)  " \
                        f"cls5={sup_counts[enc5]} ({100*sup_counts[enc5]/len(y_sup):.1f}%)"

            from tabicl import TabICLClassifier
            try:
                clf = TabICLClassifier(
                    n_estimators=16, kv_cache=True,
                    random_state=SEED, verbose=False, device="cuda")
                t_fit = time.perf_counter()
                clf.fit(X_sup, y_sup)
                t_fit = time.perf_counter() - t_fit
                t_pred = time.perf_counter()
                probs = clf.predict_proba(X_s_te).astype(np.float64)
                t_pred = time.perf_counter() - t_pred
                pred = probs.argmax(axis=1)
                f1m, bal, pc = score(y_s_te, pred, n_classes)
                del clf
                print(f"  [{cond_name:<24}] F1={f1m:.4f}  bal={bal:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s  "
                      f"n_sup={len(y_sup):,}  {bias_info}")
                _record(results, descs, cond_name, cond_desc,
                        k, f1m, bal, pc, t_fit, t_pred, len(y_sup))
            except Exception as e:
                print(f"  [{cond_name:<24}] FAILED: {type(e).__name__}: {str(e)[:120]}")
                _record_fail(results, descs, cond_name, cond_desc, k, n_classes, len(X_sup))

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
        pc_mean = np.array(valid_pcs).mean(axis=0).tolist() if valid_pcs else [float("nan")] * n_classes
        f1_mean = float(np.mean(f1s)) if f1s else float("nan")
        summary[name] = {
            "description": descs[name],
            "f1_mean":  round(f1_mean, 4),
            "f1_std":   round(float(np.std(f1s)), 4) if f1s else float("nan"),
            "bal_mean": round(float(np.mean(bals)), 4) if bals else float("nan"),
            "bal_std":  round(float(np.std(bals)), 4) if bals else float("nan"),
            "f1_per_class": {str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)},
            "per_fold": folds,
        }

    out_json = OUT_DIR / "schemeB_round4_biased_support_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": "B", "merged_classes": decoded, "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1), "n_folds": N_FOLDS,
                   "references": {"tabicl_25k_n16_kvon": REF_BEST},
                   "results": summary}, f, indent=2)
    print(f"\nSaved → {out_json}")

    # ── leaderboard ──────────────────────────────────────────────────
    weak_classes = [2, 5, 7]
    print(f"\n{'Condition':<28}  {'F1':>8}  {'std':>6}  {'Δ ref':>8}")
    print("-" * 57)
    for name, s in sorted(summary.items(), key=lambda x: -(x[1]["f1_mean"] if not np.isnan(x[1]["f1_mean"]) else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<28}  {'NaN':>8}  {'NaN':>6}  [FAIL]")
        else:
            d = s["f1_mean"] - REF_BEST
            flag = " ★" if d > 0.003 else ""
            print(f"{name:<28}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  {d:>+8.4f}{flag}")

    print(f"\nPer-class F1 for weak classes (2=bare, 5=grassland, 7=wetland):")
    print(f"{'Condition':<28}  " + "  ".join(f"cls{c:>2}" for c in weak_classes))
    print("-" * 57)
    for name, s in summary.items():
        pc = s["f1_per_class"]
        vals = "  ".join(f"{pc.get(str(c), float('nan')):>6.4f}" for c in weak_classes)
        print(f"{name:<28}  {vals}")


def _record(results, descs, name, desc, fold, f1, bal, pc, fit_s, pred_s, n_sup):
    if name not in results:
        results[name] = []
        descs[name] = desc
    results[name].append({"fold": fold, "f1_macro": round(f1, 4),
                           "bal_acc": round(bal, 4), "fit_s": round(fit_s, 2),
                           "pred_s": round(pred_s, 2), "n_support": n_sup,
                           "f1_per_class": [round(v, 4) for v in pc]})


def _record_fail(results, descs, name, desc, fold, n_classes, n_sup):
    _record(results, descs, name, desc, fold,
            float("nan"), float("nan"), [float("nan")] * n_classes,
            0.0, 0.0, n_sup)


if __name__ == "__main__":
    run()

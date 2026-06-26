"""Scheme B round-4b: confidence-gated pseudo-labels.

Tests whether filtering low-confidence Stage-1 pseudo-labels for the three
weakest classes (2=bare, 5=grassland, 7=wetland) improves Stage-2 TabICL.

Conditions (all: TabICL n_est=16, kv_cache=True, 25k random support):
  tabicl_nogate     All pseudo-labels used (current best pipeline, baseline re-run)
  tabicl_gate_weak  Drop pseudo-labels for classes 2/5/7 with top-1 prob < 0.6
  tabicl_gate_all   Drop ALL pseudo-labels with top-1 prob < 0.6

Background: CatBoost Stage-1 pseudo-label quality (from schemeB_diagnostics.json):
  class 2: mean_conf=0.458, low_conf_pct=70.5%
  class 5: mean_conf=0.547, low_conf_pct=43.9%
  class 7: mean_conf=0.520, low_conf_pct=49.2%

Output:
  common_ground/reports/research/schemeB_round4_conf_gate_results.json
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

FEATURE_COLS    = [f"A{i:02d}" for i in range(64)]
TARGET          = "class"
MERGE_MAP       = {1: 2, 9: 8}
WEAK_ORIG       = {2, 5, 7}   # original (pre-encode) class labels
CONF_THRESHOLD  = 0.6
SUPPORT_N       = 25_000
N_FOLDS         = 3
SEED            = 0
REF_BEST        = 0.6881


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
    print(f"device={device}  n_folds={N_FOLDS}  scheme=B  round=4b (confidence gate)")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    y_stable_m   = merge_classes(stable[TARGET].values)
    y_unstable_m = merge_classes(unstable[TARGET].values)
    merged_classes = sorted(set(y_stable_m.tolist()) | set(y_unstable_m.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    print(f"classes: {decoded}  n={n_classes}")

    # encoded indices for the weak classes
    weak_enc = set()
    for c in WEAK_ORIG:
        if c in le.classes_:
            weak_enc.add(int(le.transform([c])[0]))
    print(f"weak class encoded indices: {sorted(weak_enc)}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    results: dict[str, list] = {}
    descs:   dict[str, str]  = {}
    t_global = time.perf_counter()

    conditions = [
        ("tabicl_nogate",
         "No confidence gate — all pseudo-labels used (baseline re-run)"),
        ("tabicl_gate_weak",
         f"Gate: drop cls 2/5/7 pseudo-labels with top-1 prob < {CONF_THRESHOLD}"),
        ("tabicl_gate_all",
         f"Gate: drop ALL pseudo-labels with top-1 prob < {CONF_THRESHOLD}"),
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

        # Stage 1: CatBoost CPU — shared across all three conditions
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost_cpu(500)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        proba_u = stage1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_u = proba_u.argmax(axis=1)
        top1_conf_u = proba_u.max(axis=1)
        del stage1

        # log Stage-1 pseudo quality per weak class
        for enc_c in sorted(weak_enc):
            mask_c = (pseudo_u == enc_c)
            if mask_c.sum() > 0:
                low = (top1_conf_u[mask_c] < CONF_THRESHOLD).mean()
                print(f"    cls{decoded[enc_c]} pseudo: n={mask_c.sum()}  "
                      f"low_conf={100*low:.1f}%")

        rng = np.random.default_rng(SEED + k)

        for cond_name, cond_desc in conditions:
            clear_cuda()

            # Build gate mask for unstable pseudo-labels
            if cond_name == "tabicl_nogate":
                keep_u = np.ones(len(X_u_tr), dtype=bool)
            elif cond_name == "tabicl_gate_weak":
                is_weak = np.array([p in weak_enc for p in pseudo_u])
                keep_u = ~is_weak | (top1_conf_u >= CONF_THRESHOLD)
            else:  # gate_all
                keep_u = (top1_conf_u >= CONF_THRESHOLD)

            n_kept   = int(keep_u.sum())
            n_dropped = int((~keep_u).sum())

            # log survival rate per weak class
            for enc_c in sorted(weak_enc):
                mask_c = (pseudo_u == enc_c)
                survived = int((mask_c & keep_u).sum())
                print(f"    [{cond_name}] cls{decoded[enc_c]}: "
                      f"kept {survived}/{mask_c.sum()}")

            X_aug = np.concatenate([X_s_tr, X_u_tr[keep_u]])
            y_aug = np.concatenate([y_s_tr, pseudo_u[keep_u]])
            X_sup, y_sup = random_subsample(X_aug, y_aug, SUPPORT_N, rng)

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
                print(f"  [{cond_name:<20}] F1={f1m:.4f}  bal={bal:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s  "
                      f"n_sup={len(y_sup):,}  kept={n_kept}  dropped={n_dropped}")
                _record(results, descs, cond_name, cond_desc,
                        k, f1m, bal, pc, t_fit, t_pred, len(y_sup),
                        n_pseudo_kept=n_kept, n_pseudo_dropped=n_dropped)
            except Exception as e:
                print(f"  [{cond_name:<20}] FAILED: {type(e).__name__}: {str(e)[:120]}")
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

    out_json = OUT_DIR / "schemeB_round4_conf_gate_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": "B", "merged_classes": decoded, "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1), "n_folds": N_FOLDS,
                   "references": {"tabicl_25k_n16_kvon": REF_BEST},
                   "results": summary}, f, indent=2)
    print(f"\nSaved → {out_json}")

    # ── leaderboard ──────────────────────────────────────────────────
    weak_classes = [2, 5, 7]
    print(f"\n{'Condition':<24}  {'F1':>8}  {'std':>6}  {'Δ ref':>8}")
    print("-" * 52)
    for name, s in sorted(summary.items(), key=lambda x: -(x[1]["f1_mean"] if not np.isnan(x[1]["f1_mean"]) else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<24}  {'NaN':>8}  {'NaN':>6}  [FAIL]")
        else:
            d = s["f1_mean"] - REF_BEST
            flag = " ★" if d > 0.003 else ""
            print(f"{name:<24}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  {d:>+8.4f}{flag}")

    print(f"\nPer-class F1 for weak classes (2=bare, 5=grassland, 7=wetland):")
    print(f"{'Condition':<24}  " + "  ".join(f"cls{c:>2}" for c in weak_classes))
    print("-" * 52)
    for name, s in summary.items():
        pc = s["f1_per_class"]
        vals = "  ".join(f"{pc.get(str(c), float('nan')):>6.4f}" for c in weak_classes)
        print(f"{name:<24}  {vals}")


def _record(results, descs, name, desc, fold, f1, bal, pc, fit_s, pred_s, n_sup,
            n_pseudo_kept=0, n_pseudo_dropped=0):
    if name not in results:
        results[name] = []
        descs[name] = desc
    results[name].append({"fold": fold, "f1_macro": round(f1, 4),
                           "bal_acc": round(bal, 4), "fit_s": round(fit_s, 2),
                           "pred_s": round(pred_s, 2), "n_support": n_sup,
                           "f1_per_class": [round(v, 4) for v in pc],
                           "n_pseudo_kept": n_pseudo_kept,
                           "n_pseudo_dropped": n_pseudo_dropped})


def _record_fail(results, descs, name, desc, fold, n_classes, n_sup):
    _record(results, descs, name, desc, fold,
            float("nan"), float("nan"), [float("nan")] * n_classes,
            0.0, 0.0, n_sup)


if __name__ == "__main__":
    run()

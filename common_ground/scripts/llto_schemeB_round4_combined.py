"""Scheme B round-4d: biased_cls5 support draw + TabICL n8 Stage-1 combined.

Tests whether the two best individual improvements from round 4 stack:
  - biased_cls5: force class 5 to 4 000 rows in the 25k support (~16%)
  - tabicl_stage1_n8: use TabICL n8 (kv_cache=False) as Stage-1 pseudo-labeller

Conditions (all: Stage-2 = TabICL n16, kv_cache=True):
  cb_random          CatBoost Stage-1 + random 25k support     (current best re-run)
  cb_biased_cls5     CatBoost Stage-1 + biased_cls5 25k        (best individual, 0.6927)
  tabicl_random      TabICL n8 Stage-1 + random 25k support    (round-4c best re-run)
  tabicl_biased_cls5 TabICL n8 Stage-1 + biased_cls5 25k       (the combined hypothesis)

Output:
  common_ground/reports/research/schemeB_round4_combined_results.json
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
CLS5_FORCE = 4_000
REF_BEST   = 0.6927   # tabicl_biased_cls5 from round 4a
REF_PREV   = 0.6881   # tabicl_25k_n16_kvon original best


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
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def biased_subsample(X, y, n_total, forced_counts, rng):
    """Draw n_total rows, forcing exact counts for specified encoded classes."""
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
    n_rem = max(0, n_total - n_forced)
    pool = np.flatnonzero(~forced_mask)
    take_rem = min(n_rem, len(pool))
    if take_rem > 0:
        chosen_rem = rng.choice(pool, size=take_rem, replace=False)
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


def make_tabicl_stage1(n_estimators):
    from tabicl import TabICLClassifier
    return TabICLClassifier(
        n_estimators=n_estimators, kv_cache=False,
        random_state=SEED, verbose=False, device="cuda",
    )


def make_tabicl_stage2():
    from tabicl import TabICLClassifier
    return TabICLClassifier(
        n_estimators=16, kv_cache=True,
        random_state=SEED, verbose=False, device="cuda",
    )


def predict_batched(clf, X, batch_size=10_000):
    """predict_proba in batches to avoid OOM on large test sets."""
    parts = []
    for i in range(0, len(X), batch_size):
        parts.append(clf.predict_proba(X[i:i+batch_size]).astype(np.float64))
        clear_cuda()
    return np.concatenate(parts, axis=0)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  scheme=B  round=4d (combined)")

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
    enc2 = int(le.transform([2])[0])
    print(f"class 2 encoded={enc2},  class 5 encoded={enc5}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    results: dict[str, list] = {}
    descs:   dict[str, str]  = {}
    t_global = time.perf_counter()

    # (name, description, use_tabicl_stage1, biased)
    conditions = [
        ("cb_random",
         "CatBoost-500 Stage-1 + random 25k support (current best re-run)",
         False, False),
        ("cb_biased_cls5",
         f"CatBoost-500 Stage-1 + biased_cls5 25k support (cls5→{CLS5_FORCE:,} rows)",
         False, True),
        ("tabicl_n8_random",
         "TabICL n8 nokv Stage-1 + random 25k support",
         True, False),
        ("tabicl_n8_biased_cls5",
         f"TabICL n8 nokv Stage-1 + biased_cls5 25k support (cls5→{CLS5_FORCE:,} rows) [combined]",
         True, True),
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

        rng = np.random.default_rng(SEED + k)

        # Pre-compute both Stage-1 pseudo-labels once per fold
        clear_cuda()
        # CatBoost Stage-1
        idx_tr, _ = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        cb1 = make_catboost_cpu(500)
        cb1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])
        pseudo_cb = cb1.predict_proba(X_u_tr).astype(np.float32).argmax(axis=1)
        del cb1

        # TabICL n8 Stage-1
        clear_cuda()
        ti1 = make_tabicl_stage1(8)
        t0 = time.perf_counter()
        ti1.fit(X_s_tr, y_s_tr)
        t_s1 = time.perf_counter() - t0
        top1_conf = ti1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_ti = top1_conf.argmax(axis=1)
        mean_conf = top1_conf.max(axis=1).mean()
        print(f"  TabICL n8 Stage-1: fit={t_s1:.1f}s  mean_conf={mean_conf:.3f}  "
              f"low_conf(<0.6)={100*(top1_conf.max(axis=1)<0.6).mean():.1f}%")
        del ti1, top1_conf
        clear_cuda()

        # log pseudo-label distributions
        for label, pseudo in [("CatBoost", pseudo_cb), ("TabICL-n8", pseudo_ti)]:
            cnt = np.bincount(pseudo, minlength=n_classes)
            print(f"  {label} pseudo: cls2={cnt[enc2]}  cls5={cnt[enc5]}  "
                  f"cls7={cnt[le.transform([7])[0]]}  total={len(pseudo):,}")

        for cond_name, cond_desc, use_tabicl, biased in conditions:
            clear_cuda()

            pseudo_u = pseudo_ti if use_tabicl else pseudo_cb
            X_aug = np.concatenate([X_s_tr, X_u_tr])
            y_aug = np.concatenate([y_s_tr, pseudo_u])

            if biased:
                X_sup, y_sup = biased_subsample(
                    X_aug, y_aug, TOTAL_SUP, {enc5: CLS5_FORCE}, rng)
            else:
                X_sup, y_sup = random_subsample(X_aug, y_aug, TOTAL_SUP, rng)

            sup_cnt = np.bincount(y_sup, minlength=n_classes)
            bias_info = (f"cls5={sup_cnt[enc5]} ({100*sup_cnt[enc5]/len(y_sup):.1f}%)  "
                         f"cls2={sup_cnt[enc2]} ({100*sup_cnt[enc2]/len(y_sup):.1f}%)")

            try:
                stage2 = make_tabicl_stage2()
                t_fit = time.perf_counter()
                stage2.fit(X_sup, y_sup)
                t_fit = time.perf_counter() - t_fit
                t_pred = time.perf_counter()
                probs = predict_batched(stage2, X_s_te)
                t_pred = time.perf_counter() - t_pred
                pred = probs.argmax(axis=1)
                f1m, bal, pc = score(y_s_te, pred, n_classes)
                del stage2
                print(f"  [{cond_name:<26}] F1={f1m:.4f}  bal={bal:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s  "
                      f"n_sup={len(y_sup):,}  {bias_info}")
                _record(results, descs, cond_name, cond_desc,
                        k, f1m, bal, pc, t_fit, t_pred, len(y_sup))
            except Exception as e:
                print(f"  [{cond_name:<26}] FAILED: {type(e).__name__}: {str(e)[:120]}")
                _record_fail(results, descs, cond_name, cond_desc, k, n_classes, len(X_aug))

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

    out_json = OUT_DIR / "schemeB_round4_combined_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": "B", "merged_classes": decoded, "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1), "n_folds": N_FOLDS,
                   "references": {"tabicl_biased_cls5": REF_BEST,
                                  "tabicl_25k_n16_kvon": REF_PREV},
                   "results": summary}, f, indent=2)
    print(f"\nSaved → {out_json}")

    # ── leaderboard ──────────────────────────────────────────────────
    weak_classes = [2, 5, 7]
    print(f"\n{'Condition':<28}  {'F1':>8}  {'std':>6}  {'Δ 0.6927':>9}  {'Δ 0.6881':>9}")
    print("-" * 68)
    for name, s in sorted(summary.items(),
                           key=lambda x: -(x[1]["f1_mean"] if x[1]["f1_mean"] == x[1]["f1_mean"] else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<28}  {'NaN':>8}  {'NaN':>6}  [FAIL]")
        else:
            d1 = s["f1_mean"] - REF_BEST
            d2 = s["f1_mean"] - REF_PREV
            flag = " ★" if d2 > 0.003 else ""
            print(f"{name:<28}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  "
                  f"{d1:>+9.4f}  {d2:>+9.4f}{flag}")

    print(f"\nPer-class F1 (cls 2=bare, 5=grassland, 7=wetland):")
    print(f"{'Condition':<28}  " + "  ".join(f"cls{c:>2}" for c in weak_classes))
    print("-" * 56)
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

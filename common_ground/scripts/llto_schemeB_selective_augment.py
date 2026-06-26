"""Scheme B: class-selective Megan augmentation.

Adding all Megan data hurt macro-F1 (-0.026) due to domain shift, but
class 5 (grassland) clearly benefited (+0.037-0.045). This script tests
adding Megan rows selectively — only for classes where Megan data is
expected to help.

Conditions (all: cc_aps filter + biased_cls5 25k support):
  fscs_only          FSCS stable only (cc_aps reference = 0.6961)
  aug_cls5           FSCS + Megan class 5 (grassland) only
  aug_cls57          FSCS + Megan classes 5 + 7 (grassland + wetland)
  aug_cls5_capped    FSCS + Megan class 5 capped at 2000 rows (half of available)

Megan rows available: cls5=4998, cls7=4949 (from combined parquet).

Output:
  common_ground/reports/research/schemeB_selective_augment_results.json
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
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import resolve_device  # noqa: E402

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

FSCS_PARQUET     = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
COMBINED_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth_combined.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET    = "class"
MERGE_MAP = {1: 2, 9: 8}
N_FOLDS   = 3
K_INNER   = 5
SEED      = 0
ALPHA     = 0.10
TOTAL_SUP = 25_000
CLS5_FORCE = 4_000
REF_CCAPS  = 0.6961
REF_BEST   = 0.6927
WEAK_ORIG  = [2, 5, 7]


def load_parquet(path: Path, require_lonlat: bool = True) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    req = FEATURE_COLS + [TARGET]
    if require_lonlat:
        req += ["lon", "lat"]
    return df.dropna(subset=req).reset_index(drop=True)


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
          f"alpha={ALPHA}  experiment=selective_augment")

    fscs     = load_parquet(FSCS_PARQUET, require_lonlat=True)
    unstable = load_parquet(UNSTABLE_PARQUET, require_lonlat=True)

    # Load only the Megan rows we want from combined parquet
    combined = load_parquet(COMBINED_PARQUET, require_lonlat=False)
    megan_all = combined[combined["source"] == "megan"].reset_index(drop=True)
    print(f"FSCS: {len(fscs):,}  Megan (all): {len(megan_all):,}  "
          f"Unstable: {len(unstable):,}")

    y_fscs_m    = merge_classes(fscs[TARGET].values)
    y_megan_m   = merge_classes(megan_all[TARGET].values)
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
    enc7 = int(le.transform([7])[0])
    weak_enc = [int(le.transform([c])[0]) for c in WEAK_ORIG if c in le.classes_]

    # Pre-encode all Megan rows
    X_megan_all = megan_all[FEATURE_COLS].values.astype(np.float32)
    y_megan_all = le.transform(y_megan_m)

    # Megan subsets by class (encoded)
    megan_cls5_mask = (y_megan_all == enc5)
    megan_cls7_mask = (y_megan_all == enc7)
    X_m5 = X_megan_all[megan_cls5_mask]
    y_m5 = y_megan_all[megan_cls5_mask]
    X_m7 = X_megan_all[megan_cls7_mask]
    y_m7 = y_megan_all[megan_cls7_mask]
    print(f"Megan cls5: {len(y_m5):,}  cls7: {len(y_m7):,}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    fscs_fold     = km.fit_predict(fscs[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    results: dict[str, list] = {}
    descs:   dict[str, str]  = {}
    t_global = time.perf_counter()

    # (name, description, megan_X_list, megan_y_list)
    conditions = [
        ("fscs_only",
         "FSCS stable only, cc_aps filter (reference)",
         [], []),
        ("aug_cls5",
         f"FSCS + Megan cls5 ({len(y_m5):,} rows), cc_aps filter",
         [X_m5], [y_m5]),
        ("aug_cls57",
         f"FSCS + Megan cls5+cls7 ({len(y_m5)+len(y_m7):,} rows), cc_aps filter",
         [X_m5, X_m7], [y_m5, y_m7]),
        ("aug_cls5_capped",
         "FSCS + Megan cls5 capped at 2000 rows, cc_aps filter",
         [X_m5[:2000]], [y_m5[:2000]]),
    ]

    for k in range(N_FOLDS):
        s_test  = (fscs_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_fscs_tr = fscs.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_fscs_tr = le.transform(y_fscs_m[s_train])
        X_fscs_te = fscs.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_fscs_te = le.transform(y_fscs_m[s_test])
        X_u       = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)
        y_u       = le.transform(y_unstable_m[u_train])

        print(f"\n=== fold {k}  fscs_train={len(y_fscs_tr):,}  "
              f"unstable={len(X_u):,}  test={len(y_fscs_te):,} ===")

        # Cross-conformal APS on FSCS train (shared across all conditions)
        print(f"  Running {K_INNER}-fold CC-APS...")
        t0 = time.perf_counter()
        pooled = cross_conformal_aps(X_fscs_tr, y_fscs_tr, K_INNER, SEED + k)
        tau = conformal_quantile(pooled, ALPHA)
        print(f"  CC done {time.perf_counter()-t0:.1f}s  tau={tau:.4f}")

        # Final CatBoost on FSCS train → pseudo-labels + CP sets for unstable
        cb_final = make_catboost()
        cb_final.fit(X_fscs_tr, y_fscs_tr)
        probs_u = cb_final.predict_proba(X_u).astype(np.float64)

        rng = np.random.default_rng(SEED + k)
        u_test_u = rng.uniform(size=len(X_u))
        sets_u   = aps_prediction_sets(probs_u, u_test_u, tau)
        sing_u   = (sets_u.sum(axis=1) == 1)
        X_u_kept = X_u[sing_u]
        y_u_kept = sets_u[sing_u].argmax(axis=1)
        pseudo_acc = float((y_u_kept == y_u[sing_u]).mean()) if sing_u.sum() > 0 else float("nan")
        print(f"  CC filter: {sing_u.sum()} kept ({100*sing_u.mean():.1f}%)  "
              f"pseudo_acc={pseudo_acc:.4f}")

        for cond_name, cond_desc, mx_list, my_list in conditions:
            clear_cuda()

            # Build augmented stable pool: FSCS + selected Megan classes
            parts_X = [X_fscs_tr] + mx_list + [X_u_kept]
            parts_y = [y_fscs_tr] + my_list + [y_u_kept]
            X_aug = np.concatenate(parts_X)
            y_aug = np.concatenate(parts_y)

            X_sup, y_sup = biased_subsample(X_aug, y_aug, TOTAL_SUP, enc5, rng)

            sup_cnt = np.bincount(y_sup, minlength=n_classes)
            megan_n = sum(len(m) for m in mx_list)
            print(f"  [{cond_name:<20}] n_aug={len(y_aug):,}  n_sup={len(y_sup):,}  "
                  f"megan_added={megan_n:,}  "
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
                probs_te = predict_batched(stage2, X_fscs_te)
                t_pred = time.perf_counter() - t_pred
                pred_te = probs_te.argmax(axis=1)
                f1m, bal, pc = score_metrics(y_fscs_te, pred_te, n_classes)
                del stage2
                print(f"  [{cond_name:<20}] F1={f1m:.4f}  bal={bal:.4f}  "
                      f"fit={t_fit:.1f}s  pred={t_pred:.1f}s")
                _record(results, descs, cond_name, cond_desc,
                        k, f1m, bal, pc, t_fit, t_pred, len(y_sup),
                        megan_n, int(sing_u.sum()), round(pseudo_acc, 4))
            except Exception as e:
                print(f"  [{cond_name:<20}] FAILED: {type(e).__name__}: {str(e)[:100]}")
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

    out_json = OUT_DIR / "schemeB_selective_augment_results.json"
    with open(out_json, "w") as f:
        json.dump({"scheme": "B", "merged_classes": decoded, "n_classes": n_classes,
                   "elapsed_s": round(elapsed, 1), "n_folds": N_FOLDS,
                   "references": {"cc_aps_fscs_only": REF_CCAPS,
                                  "cb_biased_cls5_no_filter": REF_BEST},
                   "results": summary}, f, indent=2)
    print(f"\nSaved → {out_json}")

    show_cls = [2, 5, 7, 12]
    print(f"\n{'Condition':<22}  {'F1':>8}  {'std':>6}  {'Δ cc_aps':>9}  {'Δ best':>8}")
    print("-" * 60)
    for name, s in sorted(summary.items(),
                           key=lambda x: -(x[1]["f1_mean"]
                                           if x[1]["f1_mean"] == x[1]["f1_mean"] else -1)):
        if np.isnan(s["f1_mean"]):
            print(f"{name:<22}  [FAIL]")
        else:
            d1 = s["f1_mean"] - REF_CCAPS
            d2 = s["f1_mean"] - REF_BEST
            flag = " ★" if s["f1_mean"] > REF_CCAPS else ""
            print(f"{name:<22}  {s['f1_mean']:>8.4f}  {s['f1_std']:>6.4f}  "
                  f"{d1:>+9.4f}  {d2:>+8.4f}{flag}")

    print(f"\nPer-class F1 (cls 2=bare, 5=grassland, 7=wetland, 12=snow):")
    print(f"{'Condition':<22}  " + "  ".join(f"cls{c:>2}" for c in show_cls))
    print("-" * 56)
    for name, s in summary.items():
        pc = s["f1_per_class"]
        vals = "  ".join(f"{pc.get(str(c), float('nan')):>6.4f}"
                         for c in show_cls)
        print(f"{name:<22}  {vals}")


def _record(results, descs, name, desc, fold, f1, bal, pc,
            fit_s, pred_s, n_sup, megan_added, n_kept, pseudo_acc):
    if name not in results:
        results[name] = []
        descs[name] = desc
    results[name].append({
        "fold": fold, "f1_macro": round(f1, 4), "bal_acc": round(bal, 4),
        "fit_s": round(fit_s, 2), "pred_s": round(pred_s, 2),
        "n_support": n_sup, "megan_added": megan_added,
        "n_pseudo_kept": n_kept, "pseudo_acc": pseudo_acc,
        "f1_per_class": [round(v, 4) for v in pc],
    })


def _record_fail(results, descs, name, desc, fold, n_classes, n_sup):
    _record(results, descs, name, desc, fold,
            float("nan"), float("nan"), [float("nan")] * n_classes,
            0.0, 0.0, n_sup, 0, 0, float("nan"))


if __name__ == "__main__":
    run()

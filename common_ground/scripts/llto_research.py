"""Auto-research: LLTO experiment grid for Common Ground (nyvest).

Tests multiple ideas against the baseline (cg_llto) on 3 spatial folds.
Target runtime < 10 minutes total on the A40 GPU.

Ideas tested
  baseline     -- CatBoost 500 iter, no weighting (reproduces cg_llto)
  class_weight -- CatBoost 500 iter + auto class weights (inverse freq)
  high_iter    -- CatBoost 1000 iter, no weighting
  conf_weight  -- pseudo-labels weighted by top-1 prob (soft trust)
  lgbm         -- LightGBM 500 iter, no weighting, GPU
  lgbm_cw      -- LightGBM 500 iter + class weights

Output
  common_ground/reports/research/research_results.json
  common_ground/reports/research/research_report.html
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
from benchmark_tabular import resolve_device  # noqa: E402

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

STABLE_PARQUET   = DATA_DIR / "grunnkart_nyvest_fscs_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"

BASELINE_F1  = 0.6197   # cg_llto mean macro-F1
BASELINE_PC  = {1: 0.0382, 2: 0.5479, 3: 0.7338, 4: 0.6913,
                5: 0.3635, 6: 0.6257, 7: 0.5839, 8: 0.7908,
                9: 0.9065, 10: 0.7561, 11: 0.7304, 12: 0.6679}

N_FOLDS = 3
SEED    = 0
N_JOBS  = 4


# ── data ────────────────────────────────────────────────────────────
def load_parquet(path: Path) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{path}'").df()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df.dropna(subset=FEATURE_COLS + [TARGET, "cell_id", "lon", "lat"]).reset_index(drop=True)


# ── class weights ────────────────────────────────────────────────────
def class_weights_from_y(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()   # normalised so mean weight = 1


# ── models ───────────────────────────────────────────────────────────
def make_catboost(iterations: int, device: str, class_weights=None, **kw):
    from catboost import CatBoostClassifier
    kwargs = dict(
        iterations=iterations, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=N_JOBS,
        task_type="GPU" if device == "cuda" else "CPU",
        loss_function="MultiClass",
    )
    if class_weights is not None:
        kwargs["class_weights"] = class_weights.tolist()
    kwargs.update(kw)
    return CatBoostClassifier(**kwargs)


def make_lgbm(device: str, class_weights=None):
    import lightgbm as lgb
    kwargs = dict(
        n_estimators=500, random_state=SEED, verbose=-1,
        n_jobs=N_JOBS, objective="multiclass",
        device="gpu" if device == "cuda" else "cpu",
    )
    if class_weights is not None:
        kwargs["class_weight"] = {i: float(w) for i, w in enumerate(class_weights)}
    return lgb.LGBMClassifier(**kwargs)


# ── per-fold evaluation ──────────────────────────────────────────────
def eval_model(model, X_train, y_train, X_test, y_test, n_classes,
               sample_weight=None):
    t0 = time.perf_counter()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    y_pred = np.asarray(model.predict(X_test)).ravel().astype(int)
    f1_macro = f1_score(y_test, y_pred, average="macro",
                        labels=np.arange(n_classes), zero_division=0)
    bal = balanced_accuracy_score(y_test, y_pred)
    f1_pc = f1_score(y_test, y_pred, labels=np.arange(n_classes),
                     average=None, zero_division=0)
    return {"f1_macro": round(float(f1_macro), 4),
            "bal_acc":  round(float(bal), 4),
            "fit_s":    round(fit_s, 2),
            "f1_per_class": [round(float(v), 4) for v in f1_pc]}


# ── experiment definitions ───────────────────────────────────────────
# Each experiment returns (model, sample_weight_or_None) given fold data.
# sample_weight applies to the Stage-2 training set (stable + pseudo).

def get_experiments(device, n_classes, y_s_train, probs_u, pseudo_u, sing_mask_u):
    """Build list of (name, description, model, sample_weight)."""
    cw = class_weights_from_y(y_s_train, n_classes)

    # confidence of each pseudo-label = top-1 prob
    conf_u = probs_u.max(axis=1)   # shape (n_unstable,)

    # For the augmented stage-2 set:
    #   stable rows get weight 1.0, unstable rows get weight = top-1 prob
    def aug_weights(mask, conf):
        n_stable = len(y_s_train)
        n_pseudo = int(mask.sum())
        w_stable = np.ones(n_stable, dtype=np.float32)
        w_pseudo = conf[mask].astype(np.float32)
        return np.concatenate([w_stable, w_pseudo])

    experiments = [
        # (name, description, model_factory_fn, sample_weight_for_stage2)
        ("baseline",
         "CatBoost 500 iter, no class weights (reproduces cg_llto)",
         lambda: make_catboost(500, device),
         None),

        ("class_weight",
         "CatBoost 500 iter + inverse-frequency class weights on Stage 2 training set",
         lambda: make_catboost(500, device, class_weights=cw),
         None),

        ("high_iter",
         "CatBoost 1 000 iter, no class weights (more boosting rounds)",
         lambda: make_catboost(1000, device),
         None),

        ("conf_weight",
         "CatBoost 500 iter + pseudo-label sample weights = top-1 prob (stable rows weight=1)",
         lambda: make_catboost(500, device),
         aug_weights(np.ones(len(pseudo_u), dtype=bool), conf_u)),

        ("class_weight_conf",
         "CatBoost 500 iter + class weights AND pseudo-label confidence sample weights",
         lambda: make_catboost(500, device, class_weights=cw),
         aug_weights(np.ones(len(pseudo_u), dtype=bool), conf_u)),

        ("lgbm",
         "LightGBM 500 estimators, no class weights",
         lambda: make_lgbm(device),
         None),

        ("lgbm_cw",
         "LightGBM 500 estimators + inverse-frequency class weights",
         lambda: make_lgbm(device, class_weights=cw),
         None),
    ]
    return experiments


# ── main loop ────────────────────────────────────────────────────────
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device={device}  n_folds={N_FOLDS}  seed={SEED}")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)

    classes = sorted(set(stable[TARGET].astype(int).unique()) |
                     set(unstable[TARGET].astype(int).unique()))
    le = LabelEncoder().fit(np.array(classes))
    n_classes = len(le.classes_)
    decoded = le.classes_.tolist()
    print(f"classes: {decoded}  n={n_classes}")

    km = KMeans(n_clusters=N_FOLDS, random_state=SEED, n_init=10)
    stable_fold   = km.fit_predict(stable[["lon", "lat"]].to_numpy())
    unstable_fold = km.predict(unstable[["lon", "lat"]].to_numpy())

    rng = np.random.default_rng(SEED)

    # results[exp_name] = list of per-fold dicts
    results: dict[str, list] = {}
    exp_meta: dict[str, str] = {}
    t_global = time.perf_counter()

    for k in range(N_FOLDS):
        s_test  = (stable_fold == k)
        s_train = ~s_test
        u_train = (unstable_fold != k)

        X_s_tr = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr = le.transform(stable.loc[s_train, TARGET].astype(int).values)
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(stable.loc[s_test,  TARGET].astype(int).values)
        X_u_tr = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)

        # Stage 1: always plain CatBoost on 80% of stable_train
        idx_tr, idx_cal = train_test_split(
            np.arange(len(X_s_tr)), test_size=0.20,
            stratify=y_s_tr, random_state=SEED)
        stage1 = make_catboost(500, device)
        stage1.fit(X_s_tr[idx_tr], y_s_tr[idx_tr])

        probs_u  = stage1.predict_proba(X_u_tr).astype(np.float32)
        pseudo_u = probs_u.argmax(axis=1)
        # simple top-1-prob singleton filter matching RAPS kept% ~30%
        conf_u   = probs_u.max(axis=1)
        sing_mask_u = conf_u >= np.percentile(conf_u, 70)   # top-30%

        exps = get_experiments(device, n_classes, y_s_tr,
                               probs_u, pseudo_u, sing_mask_u)

        print(f"\n=== fold {k}  n_train_stable={s_train.sum():,}  "
              f"n_test={s_test.sum():,}  n_unstable={u_train.sum():,} ===")

        for name, desc, model_fn, sw in exps:
            if name not in results:
                results[name] = []
                exp_meta[name] = desc

            # Build Stage-2 dataset: full stable_train + all pseudo
            X_aug = np.concatenate([X_s_tr, X_u_tr], axis=0)
            y_aug = np.concatenate([y_s_tr, pseudo_u], axis=0)

            # sample_weight: replicate stable weight=1, then pseudo weight
            if sw is not None:
                assert len(sw) == len(y_aug), \
                    f"{name}: sw len {len(sw)} != aug len {len(y_aug)}"

            model = model_fn()
            metrics = eval_model(model, X_aug, y_aug,
                                 X_s_te, y_s_te, n_classes,
                                 sample_weight=sw)
            metrics["fold"] = k
            results[name].append(metrics)
            print(f"  [{name}] F1={metrics['f1_macro']:.4f}  "
                  f"bal={metrics['bal_acc']:.4f}  fit={metrics['fit_s']:.1f}s")

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # ── summarise ────────────────────────────────────────────────────
    summary = {}
    for name, folds in results.items():
        f1s  = [f["f1_macro"] for f in folds]
        bals = [f["bal_acc"]  for f in folds]
        # per-class mean across folds
        pc_mat = np.array([f["f1_per_class"] for f in folds])
        pc_mean = pc_mat.mean(axis=0).tolist()
        summary[name] = {
            "description": exp_meta[name],
            "f1_mean":  round(float(np.mean(f1s)), 4),
            "f1_std":   round(float(np.std(f1s)),  4),
            "bal_mean": round(float(np.mean(bals)), 4),
            "bal_std":  round(float(np.std(bals)),  4),
            "delta_vs_baseline": round(float(np.mean(f1s)) - BASELINE_F1, 4),
            "f1_per_class": {str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)},
            "per_fold": folds,
        }

    out_json = OUT_DIR / "research_results.json"
    with open(out_json, "w") as f:
        json.dump({"elapsed_s": round(elapsed, 1),
                   "n_folds": N_FOLDS,
                   "device": device,
                   "classes": decoded,
                   "baseline_f1": BASELINE_F1,
                   "baseline_per_class": BASELINE_PC,
                   "results": summary}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── print quick table ────────────────────────────────────────────
    print(f"\n{'Experiment':<22}  {'F1 mean':>8}  {'±std':>6}  {'Δ baseline':>10}")
    print("-" * 52)
    for name, s in sorted(summary.items(), key=lambda x: -x[1]["f1_mean"]):
        flag = " ★" if s["delta_vs_baseline"] > 0.003 else ""
        print(f"{name:<22}  {s['f1_mean']:>8.4f}  "
              f"{s['f1_std']:>6.4f}  {s['delta_vs_baseline']:>+10.4f}{flag}")

    return out_json


if __name__ == "__main__":
    run()

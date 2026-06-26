"""Scheme B CC-APS with stable-allyears support.

Same pipeline as llto_schemeB_alpha_sweep.py (CC-APS α=0.05, class merging
1→2 and 9→8, biased cls5 support, TabICL stage-2) but uses the extended
stable-allyears parquet (663k rows, years 2017-2025) as the stable source.

Key design choices vs the original:
- CC calibration uses full long-format data per fold (more points → better τ)
- Stage-2 support is deduplicated to one row per unique (lon, lat) location
  (keeping the most recent year) before the biased subsample, so TabICL sees
  spatially diverse support rather than 9× pseudo-replicates of the same pixels
- TabICL prediction is chunked at 1k rows to bound GPU memory
- Support size 25k (deduped pool ~79k unique locations, same scale as the
  original 2020-only experiment). Runs on GPU (torch 2.5.1+cu121 on the A40);
  on a CPU fallback 25k OOMs (exit 137) since TabICL attention is O(n^2) in
  sequence length — drop to ~5k if forced onto CPU.

Output:
  common_ground/reports/research/schemeB_allyears_results.json
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
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]

DATA_DIR = _REPO / "data"
OUT_DIR  = _REPO / "common_ground" / "reports" / "research"

STABLE_PARQUET   = DATA_DIR / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"
UNSTABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_unstable_alphaearth.parquet"

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET    = "class"
MERGE_MAP = {1: 2, 9: 8}
N_FOLDS   = 3
K_INNER   = 5
SEED      = 0
import os
TOTAL_SUP = int(os.environ.get("TOTAL_SUP", "25000"))
                     # 25k default; deduped pool ~79k unique locations.
                     # Runs on GPU (torch 2.5.1+cu121 on the A40); on CPU this
                     # OOMs (exit 137) because TabICL attention is O(n^2) in
                     # sequence length — keep on cuda, or drop to ~5k for CPU.
CLS5_FORCE = 4_000   # same as original
# Classes exempt from the lon/lat dedup keep all their multi-year rows. EMPTY by
# design: this was tried for class 12 and made it far WORSE (0.665 -> 0.320),
# because class 12 is spectrally near-inseparable from the 62x-larger class 11
# (centroid dist 0.484 vs class-12 radius 0.446, ratio 1.09). Adding class-12
# support rows in that overlap zone just amplifies 11<->12 contradiction in
# TabICL's context, collapsing class-12 recall. The clean no-force config
# (DEDUP_EXEMPT empty, no cls12 forcing) is the winner at F1=0.6964. Keep empty
# unless a real 11-vs-12 discriminator is added. See memory: schemeB-class12.
DEDUP_EXEMPT_CLASSES: set = set()

# --- Stale snow/ice (class 12) label cleaning ---------------------------------
# Grunnkart labels are static but drawn over a long time frame; the AlphaEarth
# embeddings span 2017-2025, a period of fast glacier/snow retreat in the AOI
# (Vestland holds 46% of Norway's glaciers; Folgefonna/Hardangerjøkulen shrinking).
# So some class-12 pixels were snow/ice when grunnkart was made but are bare rock
# / meltwater in the embedding years — and those melted pixels are spectrally
# infrastructure-like (95.6% of stray cls12 rows fall nearest class 11). We flag a
# class-12 ROW as suspect if it is nearer to ANY other class centroid than to the
# class-12 centroid, then clean per unique location:
#   - suspect in ALL years   -> stale/deglaciated, drop the location entirely
#   - suspect in SOME years   -> drop only the suspect rows (partial melt)
#   - never suspect           -> keep
# Centroids are computed on the full stable set (unsupervised geometric prior on
# label quality; only REMOVES rows, never relabels using fold test info).
CLEAN_TARGET_CLASS = 12
# Cleaning is applied TRAIN-ONLY inside the CV loop; the test fold always keeps its
# original labels (so F1 is measured on a fixed, untouched test set — cleaning the
# test set leaks and inflates F1, as cleanlab_all v1 showed).
# CLEAN_MODE selects WHICH rows + WHICH treatment:
#   "off"               - no cleaning
#   "centroid"          - geometric heuristic, class 12 only; REMOVE (147 rows)
#   "cleanlab_c12_rm"   - cleanlab issues, class 12 only; REMOVE
#   "cleanlab_all_rm"   - cleanlab issues, all classes; REMOVE (~13% of TRAIN)
#   "cleanlab_all_fix"  - cleanlab issues, all classes; CORRECT to suggested label
#   "cleanlab_c12_fix"  - cleanlab issues, class 12 only; CORRECT to suggested label
# cleanlab_* modes reuse clean_labels_full.npz (built by clean_labels_artifacts.py;
# OOS pred_probs ~50 min). Masks/labels are keyed to full-parquet load order.
#   ── per-fold leak-free variants (cleanlab flagging done train-only PER FOLD,
#      so test rows never influence their own fold's cleaning) ──
#   "perfold_all_rm"  / "perfold_all_fix"  - all-class issues, remove / correct
# perfold_* modes read clean_labels_perfold.npz (built by clean_labels_perfold.py).
import os
CLEAN_MODE = os.environ.get("CLEAN_MODE", "centroid")
CLEAN_ARTIFACTS_NPZ = OUT_DIR / "clean_labels_full.npz"
CLEAN_PERFOLD_NPZ = OUT_DIR / "clean_labels_perfold.npz"
# Optional suffix to disambiguate output JSON across cleaning-mode runs.
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
# Stage-2 in-context model: "tabicl" (default) or "tabpfn" (TabPFNV3 / TabPFN-2.5).
STAGE2_MODEL = os.environ.get("STAGE2_MODEL", "tabicl")
# Conformal gate variant:
#   "standard" - single global APS τ (original behaviour)
#   "nc"       - noise-corrected class-12 gate: class-12 calibration labels carry
#                structured noise (stale snow/ice → infrastructure), which inflates
#                the class-12 APS quantile and lets impostor rows be pseudo-labelled
#                snow/ice. Per fold (train-only, leak-free) we estimate the impostor
#                fraction among observed-cls12 calibration rows via cleanlab's
#                confident joint on the inner-CV OOS probs, trim that fraction of
#                the worst (highest) cls12 scores, and re-take the quantile. The
#                tightened τ_c12 applies ONLY to class-12 set membership; all other
#                classes keep the global τ. See noise_corrected_aps.py (diagnostic:
#                τ_c12 0.972 → 0.853 on the full set).
GATE_MODE = os.environ.get("GATE_MODE", "standard")
# Extra (non-embedding) feature columns joined by exact (lon, lat):
#   ""      - 64 AlphaEarth bands only (original behaviour)
#   "dem"   - + elevation, slope from data/dem_features.parquet (Copernicus
#             GLO30, 30 m). Near-no-op (see memory dem-features-negative).
#   "lidar" - + elevation, tri, tch from data/lidar_features.parquet (3 m local
#             lidar; scripts/extraction/extract_lidar_features.py). The lean
#             3-feature set chosen by the CatBoost probe
#             (scripts/feature_probe_lidar.py): elevation+tri+tch beat the full
#             6-feature lidar set (aspect_sin/cos ≈0 importance, slope marginal).
#             Probe gain over embed-only: macro-F1 0.6964 -> 0.7043. Gains
#             concentrate in vegetation confusions (cls7 +0.023, cls5 +0.013,
#             cls3 +0.012); cls12 unchanged/slightly down.
EXTRA_FEATURES = os.environ.get("EXTRA_FEATURES", "")
DEM_PARQUET = DATA_DIR / "dem_features.parquet"
LIDAR_PARQUET = DATA_DIR / "lidar_features.parquet"
LIDAR_COLS = ["elevation", "tri", "tch"]

TABICL_PREDICT_CHUNK = 1_000  # chunk size for TabICL predict to bound GPU mem
ALPHA     = 0.05     # best from alpha sweep
WEAK_ORIG = [2, 5, 7]


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


def rss_gb() -> float:
    """Resident set size of this process in GiB (for memory-stability logging)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def load_parquet(path: Path) -> pd.DataFrame:
    # Cast features to FLOAT in SQL and select only needed columns: avoids
    # materialising the full float64 frame + a second astype copy.
    cols = duckdb.sql(f"DESCRIBE SELECT * FROM '{path}'").df()["column_name"]
    meta = [c for c in (TARGET, "cell_id", "lon", "lat", "year")
            if c in set(cols)]
    sel = ", ".join([f'CAST("{c}" AS FLOAT) AS "{c}"' for c in FEATURE_COLS]
                    + [f'"{c}"' for c in meta])
    df = duckdb.sql(f"SELECT {sel} FROM '{path}'").df()
    return df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)


def dedup_to_latest_year(df: pd.DataFrame, exempt_classes=None,
                         class_col: str = "_y_merged") -> pd.DataFrame:
    """Keep one row per unique (lon, lat) — the most recent year available —
    EXCEPT for `exempt_classes`, whose multi-year rows are kept in full.

    Dedup gives TabICL spatially diverse, non-replicated support, but for rare
    classes (e.g. class 12, ~146 unique locations) it strips the inter-year
    variety that helps generalization. Exempting those classes keeps all their
    yearly rows while still deduping abundant classes. `class_col` holds the
    MERGED label (so exemption keys off post-merge classes).
    """
    if "year" not in df.columns or "lon" not in df.columns:
        return df
    exempt_classes = set(exempt_classes or ())
    if exempt_classes and class_col in df.columns:
        is_exempt = df[class_col].isin(exempt_classes)
        kept_exempt = df[is_exempt]
        deduped = (df[~is_exempt].sort_values("year", ascending=False)
                                 .drop_duplicates(subset=["lon", "lat"]))
        return pd.concat([deduped, kept_exempt]).reset_index(drop=True)
    return (df.sort_values("year", ascending=False)
              .drop_duplicates(subset=["lon", "lat"])
              .reset_index(drop=True))


def merge_classes(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype(int)
    for src, tgt in MERGE_MAP.items():
        y[y == src] = tgt
    return y


def clean_stale_class(df: pd.DataFrame, y_merged: np.ndarray,
                      target_class: int) -> np.ndarray:
    """Return a boolean keep-mask that drops stale `target_class` rows.

    A target-class row is 'suspect' if it is nearer (Euclidean, embedding space)
    to ANY other class centroid than to its own class centroid — i.e. it does not
    look like its label. Per unique (lon, lat):
      - suspect in ALL years  -> drop the whole location (stale/mislabelled)
      - suspect in SOME years -> drop only the suspect rows (partial/transient)
      - never suspect          -> keep all rows
    Non-target rows are always kept. Centroids use the full df (unsupervised).
    """
    X = df[FEATURE_COLS].values.astype(np.float64)
    classes = np.unique(y_merged)
    cents = {int(c): X[y_merged == c].mean(0) for c in classes}
    if target_class not in cents:
        return np.ones(len(df), dtype=bool)

    keep = np.ones(len(df), dtype=bool)
    tgt_idx = np.flatnonzero(y_merged == target_class)
    Xt = X[tgt_idx]
    d_own = np.linalg.norm(Xt - cents[target_class], axis=1)
    others = [c for c in cents if c != target_class]
    OC = np.stack([cents[c] for c in others])
    d_min_other = np.linalg.norm(Xt[:, None, :] - OC[None, :, :], axis=2).min(1)
    suspect = d_min_other < d_own  # row looks more like some other class

    # Aggregate per location: drop whole location only if suspect in every year.
    sub = df.iloc[tgt_idx][["lon", "lat"]].copy()
    sub["_suspect"] = suspect
    sub["_pos"] = tgt_idx
    grp = sub.groupby(["lon", "lat"])
    loc_frac = grp["_suspect"].transform("mean")
    # all-years suspect (frac==1): drop every row at that location
    # some-years suspect (0<frac<1): drop only the suspect rows
    drop = ((loc_frac == 1.0) | ((loc_frac < 1.0) & sub["_suspect"].values))
    keep[sub["_pos"].values[drop.values]] = False
    return keep


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


def aps_score_matrix(probs, u):
    """Full APS score matrix s[i, c]; set membership is s <= τ."""
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = probs.shape[0]
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    return cum_at + u[:, None] * probs


def cross_conformal_aps(X_tr, y_tr, K, seed, n_classes):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    pooled = np.empty(len(y_tr), dtype=np.float64)
    P_pooled = np.zeros((len(y_tr), n_classes), dtype=np.float64)
    u_all  = np.random.default_rng(seed).uniform(size=len(y_tr))
    for _, (idx_t, idx_v) in enumerate(skf.split(X_tr, y_tr)):
        cb = make_catboost()
        cb.fit(X_tr[idx_t], y_tr[idx_t])
        p_v = cb.predict_proba(X_tr[idx_v]).astype(np.float64)
        P_pooled[np.ix_(idx_v, cb.classes_.astype(int))] = p_v
        pooled[idx_v] = aps_cal_scores(P_pooled[idx_v], y_tr[idx_v], u_all[idx_v])
    return pooled, P_pooled


def nc_class_tau(pooled, P_cc, y_cc, cls_enc, alpha, tau_global):
    """Noise-corrected per-class APS quantile for a structurally noisy class.

    Observed-cls calibration scores mix true-cls rows with impostors (rows whose
    true class differs); impostors inflate the upper tail. Estimate the impostor
    fraction from the confident joint on the inner-CV OOS probs (all train-only),
    trim that fraction of the highest scores, and re-take the quantile. Capped at
    the global τ so the correction can only tighten, never loosen, the gate.
    Returns (tau_c, impostor_frac).
    """
    from cleanlab.count import compute_confident_joint, estimate_latent
    cj = compute_confident_joint(labels=y_cc, pred_probs=P_cc)
    _, _, inv_noise = estimate_latent(confident_joint=cj, labels=y_cc)
    impostor = float(np.clip(1.0 - inv_noise[cls_enc, cls_enc], 0.0, 0.95))
    s_c = pooled[y_cc == cls_enc]
    if len(s_c) == 0:
        return tau_global, impostor
    keep_n = max(int(round(len(s_c) * (1.0 - impostor))), 1)
    tau_c = conformal_quantile(np.sort(s_c)[:keep_n], alpha)
    return min(tau_c, tau_global), impostor


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    # HARD GUARD: a silent CUDA-init failure must not fall back to CPU — TabICL
    # attention at >8k support OOMs host RAM (exit 137) and can take the whole
    # VDI session down. Fail fast instead.
    if device != "cuda" and TOTAL_SUP > 8_000 and os.environ.get("ALLOW_CPU") != "1":
        raise SystemExit(
            f"refusing stage-2 with TOTAL_SUP={TOTAL_SUP:,} on device={device}: "
            "TabICL is O(n^2) in support and OOMs host RAM on CPU. Fix CUDA, "
            "set TOTAL_SUP<=8000, or force with ALLOW_CPU=1.")
    print(f"device={device}  n_folds={N_FOLDS}  K_inner={K_INNER}  "
          f"alpha={ALPHA}  total_sup={TOTAL_SUP}  cls5_force={CLS5_FORCE}  "
          f"dedup_exempt={sorted(DEDUP_EXEMPT_CLASSES)}  clean_mode={CLEAN_MODE}  "
          f"gate_mode={GATE_MODE}  stage2={STAGE2_MODEL}")
    print(f"stable={STABLE_PARQUET.name}")

    stable   = load_parquet(STABLE_PARQUET)
    unstable = load_parquet(UNSTABLE_PARQUET)
    print(f"stable: {len(stable):,} rows   unstable: {len(unstable):,} rows  "
          f"[rss={rss_gb():.1f}G]")

    if EXTRA_FEATURES in ("dem", "lidar"):
        global FEATURE_COLS
        if EXTRA_FEATURES == "dem":
            src, extra_cols = DEM_PARQUET, ["elevation", "slope"]
            sel = "*"
        else:
            src, extra_cols = LIDAR_PARQUET, LIDAR_COLS
            sel = "lon, lat, " + ", ".join(extra_cols)
        extra = (duckdb.sql(f"SELECT {sel} FROM '{src}'").df()
                 .drop_duplicates(subset=["lon", "lat"]))
        stable = stable.merge(extra, on=["lon", "lat"], how="left")
        unstable = unstable.merge(extra, on=["lon", "lat"], how="left")
        n_miss = int(stable[extra_cols[0]].isna().sum()
                     + unstable[extra_cols[0]].isna().sum())
        for frame in (stable, unstable):
            for c in extra_cols:
                frame[c] = frame[c].fillna(extra[c].median()).astype(np.float32)
        FEATURE_COLS = FEATURE_COLS + extra_cols
        print(f"  extra features ({EXTRA_FEATURES}): {extra_cols} joined "
              f"({len(extra):,} locations, {n_miss} rows unmatched->median)")

    # Precompute TRAIN-ONLY cleaning arrays aligned to `stable` row order:
    #   clean_remove[i]   True -> drop row i if it falls in the training fold
    #   clean_labels[i]   the label to USE for training (corrected or original)
    # These are applied per-fold to the training mask only; the test fold uses
    # the original merged labels untouched. Build before fold assignment so the
    # arrays stay aligned through the (no longer mutated) `stable` frame.
    y_pre = merge_classes(stable[TARGET].values)
    clean_remove = np.zeros(len(stable), dtype=bool)
    clean_labels = y_pre.copy()
    # Per-fold (leak-free) modes resolve their mask inside the loop using fold k.
    PERFOLD = CLEAN_MODE.startswith("perfold_")
    # COMBO: centroid REMOVE for class 12 (best for cls12) + perfold_fix RELABEL for all
    # OTHER classes (best mean). cls12 is handled by removal, so perfold-fix's cls12
    # relabels are ignored; non-cls12 rows get perfold-fix's corrected label.
    COMBO = CLEAN_MODE == "combo_centroid_perfold_fix"
    perfold_npz = None
    combo_c12_remove = None
    if PERFOLD or COMBO:
        perfold_npz = np.load(CLEAN_PERFOLD_NPZ)
        print(f"  clean[{CLEAN_MODE}]: per-fold leak-free masks loaded "
              f"(remove/fold={[int(perfold_npz['remove'][k].sum()) for k in range(N_FOLDS)]})")
    if COMBO:
        # clean_stale_class returns a KEEP mask; remove = ~keep (class-12 only).
        combo_c12_remove = ~clean_stale_class(stable, y_pre, CLEAN_TARGET_CLASS)
        print(f"  combo: centroid cls12 remove={int(combo_c12_remove.sum())}")
    elif not PERFOLD and CLEAN_MODE != "off":
        if CLEAN_MODE == "centroid":
            keep = clean_stale_class(stable, y_pre, CLEAN_TARGET_CLASS)
            clean_remove = ~keep
        else:
            npz = np.load(CLEAN_ARTIFACTS_NPZ)
            if not np.array_equal(npz["y_merged"], y_pre):
                raise ValueError("clean artifact label order mismatch; "
                                 "regenerate clean_labels_artifacts.py")
            issue_all = npz["issue_mask_all"]
            issue_c12 = npz["issue_mask_c12"]
            suggested = npz["suggested_all"]
            if CLEAN_MODE == "cleanlab_c12_rm":
                clean_remove = issue_c12.copy()
            elif CLEAN_MODE == "cleanlab_all_rm":
                clean_remove = issue_all.copy()
            elif CLEAN_MODE == "cleanlab_c12_fix":
                clean_labels = np.where(issue_c12, suggested, y_pre)
            elif CLEAN_MODE == "cleanlab_all_fix":
                clean_labels = np.where(issue_all, suggested, y_pre)
            else:
                raise ValueError(f"unknown CLEAN_MODE={CLEAN_MODE}")
        n_rm = int(clean_remove.sum())
        n_fix = int((clean_labels != y_pre).sum())
        print(f"  clean[{CLEAN_MODE}] (train-only): remove={n_rm:,}  "
              f"relabel={n_fix:,}")
    n_dropped = int(clean_remove.sum())
    n_relabelled = int((clean_labels != y_pre).sum())

    # NOTE: stable frame is NOT subset here — cleaning is applied per-fold below.
    y_stable_m   = y_pre
    y_unstable_m = merge_classes(unstable[TARGET].values)
    merged_classes = sorted(set(y_stable_m.tolist()) | set(y_unstable_m.tolist()))
    le = LabelEncoder().fit(np.array(merged_classes))
    n_classes = len(le.classes_)
    decoded   = le.classes_.tolist()
    print(f"classes: {decoded}  n={n_classes}")

    enc5 = int(le.transform([5])[0])

    # Spatial folds via KMeans on stable lon/lat — need lon/lat for stable
    # allyears parquet (same cell_id grid, use cell_id-based GroupKFold instead)
    from sklearn.model_selection import GroupKFold
    groups = stable["cell_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    stable_folds = np.empty(len(stable), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
            gkf.split(stable[FEATURE_COLS], y_stable_m, groups=groups)):
        stable_folds[val_idx] = fold_idx

    # Assign unstable rows to nearest fold by cell_id proximity
    # (unstable has lon/lat; stable allyears has cell_id — use simple mod)
    unstable_fold = np.arange(len(unstable)) % N_FOLDS

    results: list = []
    t_global = time.perf_counter()

    for k in range(N_FOLDS):
        s_test  = (stable_folds == k)
        # TRAIN-ONLY cleaning: drop removed rows from the training fold; use
        # corrected labels for training. Test fold is untouched (original labels).
        if COMBO:
            # class 12: centroid REMOVE (global). non-cls12: perfold-fix RELABEL.
            cur_remove = combo_c12_remove
            fold_labels = perfold_npz["corrected"][k]
            is_c12 = (y_stable_m == CLEAN_TARGET_CLASS)
            cur_labels = np.where(is_c12, y_stable_m, fold_labels)
        elif PERFOLD:
            fold_remove = perfold_npz["remove"][k]          # train-only, this fold
            fold_labels = perfold_npz["corrected"][k]
            if CLEAN_MODE.endswith("_fix"):
                cur_remove = np.zeros(len(stable), dtype=bool)
                cur_labels = fold_labels
            else:  # _rm
                cur_remove = fold_remove
                cur_labels = y_stable_m
        else:
            cur_remove = clean_remove
            cur_labels = clean_labels
        s_train = (~s_test) & (~cur_remove)
        u_train = (unstable_fold != k)

        X_s_tr = stable.loc[s_train, FEATURE_COLS].values.astype(np.float32)
        y_s_tr = le.transform(cur_labels[s_train])
        X_s_te = stable.loc[s_test,  FEATURE_COLS].values.astype(np.float32)
        y_s_te = le.transform(y_stable_m[s_test])
        X_u    = unstable.loc[u_train, FEATURE_COLS].values.astype(np.float32)
        y_u    = le.transform(y_unstable_m[u_train])

        n_train_removed = int((~s_test & cur_remove).sum())
        n_train_fixed = int(((cur_labels != y_stable_m) & s_train).sum())
        print(f"\n=== fold {k}  n_stable_train={len(y_s_tr):,}  "
              f"n_unstable={len(X_u):,}  n_test={len(y_s_te):,}  "
              f"(train_removed={n_train_removed:,} train_relabelled={n_train_fixed:,}) ===")

        # Subsample stable_train for CC calibration to avoid memory issues
        # with 440k rows: cap at 80k (still 5x original)
        rng = np.random.default_rng(SEED + k)
        cc_cap = 80_000
        if len(X_s_tr) > cc_cap:
            cc_idx = rng.choice(len(X_s_tr), size=cc_cap, replace=False)
            X_cc, y_cc = X_s_tr[cc_idx], y_s_tr[cc_idx]
            print(f"  CC calibration subsampled {cc_cap:,} from {len(X_s_tr):,}")
        else:
            X_cc, y_cc = X_s_tr, y_s_tr

        print(f"  Running {K_INNER}-fold CC-APS on {len(X_cc):,} rows...")
        t0 = time.perf_counter()
        pooled, P_cc = cross_conformal_aps(X_cc, y_cc, K_INNER, SEED + k, n_classes)
        print(f"  CC done {time.perf_counter()-t0:.1f}s  [rss={rss_gb():.1f}G]")

        # Final CatBoost on subsampled stable_train (same cc_cap for speed)
        cb_final = make_catboost()
        cb_final.fit(X_cc, y_cc)
        probs_u = cb_final.predict_proba(X_u).astype(np.float64)

        u_test_u = rng.uniform(size=len(X_u))
        tau = conformal_quantile(pooled, ALPHA)
        enc12 = int(le.transform([CLEAN_TARGET_CLASS])[0])
        tau_c12 = impostor_frac = None
        s_mat = aps_score_matrix(probs_u, u_test_u)
        sets_u = s_mat <= tau
        if GATE_MODE == "nc":
            tau_c12, impostor_frac = nc_class_tau(
                pooled, P_cc, y_cc, enc12, ALPHA, tau)
            n_c12_before = int((sets_u[:, enc12]).sum())
            sets_u[:, enc12] = s_mat[:, enc12] <= tau_c12
            print(f"  nc-gate: impostor_frac={impostor_frac:.3f}  "
                  f"τ_c12={tau_c12:.5f} (global τ={tau:.5f})  "
                  f"cls12-in-set {n_c12_before} -> {int(sets_u[:, enc12].sum())}")
        sing   = (sets_u.sum(axis=1) == 1)
        n_kept = int(sing.sum())
        X_u_kept = X_u[sing]
        y_u_kept = sets_u[sing].argmax(axis=1)
        pseudo_acc = float((y_u_kept == y_u[sing]).mean()) if n_kept > 0 else float("nan")
        n_kept_c12 = int((y_u_kept == enc12).sum())

        print(f"  α={ALPHA}  τ={tau:.5f}  kept={n_kept}({100*sing.mean():.1f}%)  "
              f"pseudo_acc={pseudo_acc:.4f}  pseudo_cls12={n_kept_c12}")

        # Deduplicate stable_train to unique locations before stage-2 support.
        # CC calibration used the full long-format data (better τ estimate);
        # stage-2 support uses deduped rows so TabICL sees diverse locations.
        stable_tr_df = stable.loc[s_train].copy()
        stable_tr_df["_y_merged"] = cur_labels[s_train]
        stable_tr_dedup = dedup_to_latest_year(
            stable_tr_df, exempt_classes=DEDUP_EXEMPT_CLASSES)
        X_s_tr_dedup = stable_tr_dedup[FEATURE_COLS].values.astype(np.float32)
        y_s_tr_dedup = le.transform(stable_tr_dedup["_y_merged"].values)
        n_exempt = int(stable_tr_dedup["_y_merged"]
                       .isin(DEDUP_EXEMPT_CLASSES).sum())
        print(f"  stable_train deduped: {len(y_s_tr_dedup):,} rows "
              f"(from {len(y_s_tr):,} long-format; {n_exempt:,} kept multi-year "
              f"for exempt classes {sorted(DEDUP_EXEMPT_CLASSES)})")

        X_aug = np.concatenate([X_s_tr_dedup, X_u_kept])
        y_aug = np.concatenate([y_s_tr_dedup, y_u_kept])
        X_sup, y_sup = biased_subsample(X_aug, y_aug, TOTAL_SUP, enc5, rng)

        # Stage 2: in-context model (TabICL or TabPFNV3) with chunked prediction.
        clear_cuda()
        stage2 = None
        try:
            if STAGE2_MODEL == "tabpfn":
                from tabpfn import TabPFNClassifier
                # ignore_pretraining_limits: support is 25k > TabPFN's 10k soft cap
                # (TabPFN-2.5 / V3 handles up to 50k). n_estimators=16 to match the
                # TabICL ensemble size for a fair head-to-head.
                stage2 = TabPFNClassifier(
                    n_estimators=16, ignore_pretraining_limits=True,
                    random_state=SEED, device=device)
            else:
                from tabicl import TabICLClassifier
                stage2 = TabICLClassifier(
                    n_estimators=16, kv_cache=True,
                    random_state=SEED, verbose=False, device=device)
            t_fit = time.perf_counter()
            stage2.fit(X_sup, y_sup)
            t_fit = time.perf_counter() - t_fit
            # TabPFN-V3 parallelises predict well: 10k batch is ~8x faster than 1k
            # (~10 min/fold vs ~85). TabICL stays at 1k to bound its attention mem.
            pred_chunk = 10_000 if STAGE2_MODEL == "tabpfn" else TABICL_PREDICT_CHUNK
            t_pred = time.perf_counter()
            probs_te = predict_batched(stage2, X_s_te, batch=pred_chunk)
            t_pred = time.perf_counter() - t_pred
            pred_te = probs_te.argmax(axis=1)
            f1m, bal, pc = score_metrics(y_s_te, pred_te, n_classes)
            # Confusion against the NOISY test labels: cm[true_obs, pred].
            # Saved so a post-hoc noise correction (via the transition matrix T)
            # can estimate clean-label F1 without touching individual test rows.
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_s_te, pred_te,
                                  labels=np.arange(n_classes)).astype(int)
            del stage2
            print(f"  F1={f1m:.4f}  bal={bal:.4f}  fit={t_fit:.1f}s  "
                  f"pred={t_pred:.1f}s  [rss={rss_gb():.1f}G]")
            results.append({
                "fold": k, "f1_macro": round(f1m, 4), "bal_acc": round(bal, 4),
                "fit_s": round(t_fit, 2), "pred_s": round(t_pred, 2),
                "n_support": int(len(y_sup)), "alpha": ALPHA,
                "tau": round(tau, 5), "n_kept": n_kept,
                "tau_c12": None if tau_c12 is None else round(tau_c12, 5),
                "impostor_frac": (None if impostor_frac is None
                                  else round(impostor_frac, 4)),
                "n_kept_c12": n_kept_c12,
                "pct_kept": round(100 * sing.mean(), 2),
                "pseudo_acc": round(pseudo_acc, 4),
                "f1_per_class": [round(v, 4) for v in pc],
                "confusion_noisy": cm.tolist(),
            })
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:120]}")
            results.append({"fold": k, "f1_macro": float("nan"),
                            "error": str(e)})
            # Drop the half-built model: a failed fit otherwise keeps tens of
            # GB of host/GPU buffers alive into the next fold (seen: rss 20.5G).
            stage2 = None
        # Free fold-local arrays before the next fold so RSS stays flat
        # across folds instead of stacking three folds' worth of copies.
        del (X_s_tr, X_s_te, X_u, X_cc, y_cc, pooled, P_cc, probs_u, s_mat,
             sets_u, X_u_kept, stable_tr_df, stable_tr_dedup, X_s_tr_dedup,
             X_aug, y_aug, X_sup, y_sup)
        clear_cuda()

    elapsed = time.perf_counter() - t_global
    valid = [r for r in results if not np.isnan(r.get("f1_macro", float("nan")))]
    f1s  = [r["f1_macro"] for r in valid]
    bals = [r["bal_acc"]  for r in valid]
    f1_mean = float(np.mean(f1s)) if f1s else float("nan")
    f1_std  = float(np.std(f1s))  if f1s else float("nan")
    bal_mean = float(np.mean(bals)) if bals else float("nan")

    valid_pcs = [r["f1_per_class"] for r in valid if "f1_per_class" in r]
    pc_mean = (np.array(valid_pcs).mean(axis=0).tolist()
               if valid_pcs else [float("nan")] * n_classes)
    pc_dict = {str(decoded[i]): round(v, 4) for i, v in enumerate(pc_mean)}

    summary = {
        "experiment": "schemeB_allyears_tabicl_deduped",
        "stage2_model": f"{STAGE2_MODEL}_n16",
        "support_deduplication": "lon_lat_latest_year",
        "stable_parquet": STABLE_PARQUET.name,
        "n_stable_rows": len(stable),
        "alpha": ALPHA,
        "total_support": TOTAL_SUP,
        "cls5_force": CLS5_FORCE,
        "dedup_exempt_classes": sorted(DEDUP_EXEMPT_CLASSES),
        "clean_mode": CLEAN_MODE,
        "gate_mode": GATE_MODE,
        "extra_features": EXTRA_FEATURES,
        "n_features": len(FEATURE_COLS),
        "clean_train_only": True,
        "n_dropped": n_dropped,
        "n_relabelled": n_relabelled,
        "merged_classes": decoded,
        "n_classes": n_classes,
        "elapsed_s": round(elapsed, 1),
        "f1_mean": round(f1_mean, 4),
        "f1_std": round(f1_std, 4),
        "bal_mean": round(bal_mean, 4),
        "f1_per_class": pc_dict,
        "reference_f1": 0.6975,  # original schemeB cc_aps_a05
        "per_fold": results,
    }

    out_json = OUT_DIR / f"schemeB_allyears_results{OUT_SUFFIX}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"\n--- Summary ---")
    print(f"  F1 mean={f1_mean:.4f}  std={f1_std:.4f}  bal={bal_mean:.4f}")
    print(f"  vs reference (original stable, cc_aps_a05): 0.6975  "
          f"Δ={f1_mean - 0.6975:+.4f}")
    print(f"\n  Per-class F1:")
    for cls, v in pc_dict.items():
        print(f"    class {cls}: {v:.4f}")
    print(f"\nSaved → {out_json}")


if __name__ == "__main__":
    run()

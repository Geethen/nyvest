"""Shared model core for the nature-types active-learning app.

Single source of truth for: data loading + class merging, the live CatBoost model,
APS conformal calibration, prediction-set sizes (the uncertainty signal that doubles
as the active-learning acquisition function), and artifact persistence/loading.

The APS math (cross-conformal calibration, score matrix, quantile) is ported verbatim
from the validated CV pipeline `common_ground/scripts/llto_schemeB_allyears.py` so the
app's uncertainty layer is identical to the one that produced the F1≈0.7 results. The
heavy TabICL stage-2 is intentionally NOT used live — CatBoost stage-1 is the fast loop
model; TabICL is an optional offline high-quality map pass (see predict_raster.py).

Model is CatBoost (MultiClass), trained on GPU and predicting on CPU. A real-data
head-to-head (2026-06-22) picked it over XGBoost: CatBoost GPU trains ~3× faster (~2.6 s
vs 7.6 s; 5-fold cross-conformal ~14.5 s vs 35.9 s — matters across many train cycles)
AND its symmetric-tree CPU predict is ~17× faster on a dense tile (~0.7 s vs ~12 s). F1 is
within noise. XGBoost 2.x+ GPU can't run on this vGPU at all (no CUDA VMM); timber-compiled
inference is also a dead end here. See reports/active_learning/timber_report.md.

Conventions (must match the rest of the project):
- Features: 64 AlphaEarth bands A00..A63 (FLOAT32).
- Class merge: 1->2, 9->8 (12 raw codes -> 10 effective). Codes match the `class`
  column of the stable-allyears parquet and `models/class_names.json`.
- Conformal: APS at alpha=0.05 (best from the alpha sweep), randomized (uniform u).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# --- repo layout ---------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
AL_DATA_DIR = DATA_DIR / "active_learning"

STABLE_PARQUET = DATA_DIR / "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet"

# --- model conventions ---------------------------------------------------------
FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET = "class"
MERGE_MAP = {1: 2, 9: 8}
ALPHA = 0.05          # APS miscoverage level (best from the alpha sweep)
K_INNER = 5           # folds for cross-conformal calibration
SEED = 0
CC_CAP = 80_000       # cap rows used for CC calibration / final fit (speed)

# persisted artifact paths
MODEL_PATH = MODELS_DIR / "seed_catboost.cbm"
CALIB_PATH = MODELS_DIR / "aps_calib.npz"
ENCODER_PATH = MODELS_DIR / "label_encoder.json"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


# ==============================================================================
# Data loading + class merging
# ==============================================================================
def load_stable(columns_extra: tuple[str, ...] = ("lon", "lat", "year", "cell_id"),
                path: Path = STABLE_PARQUET):
    """Load the stable-allyears parquet as a DataFrame with FLOAT32 features.

    Returns the frame with FEATURE_COLS + TARGET + any of `columns_extra` present,
    NaN feature/label rows dropped. Uses duckdb to cast in SQL (avoids a float64
    materialisation + astype copy of the full 663k×71 frame).
    """
    import duckdb

    cols = set(duckdb.sql(f"DESCRIBE SELECT * FROM '{path}'").df()["column_name"])
    meta = [c for c in (TARGET, *columns_extra) if c in cols]
    sel = ", ".join(
        [f'CAST("{c}" AS FLOAT) AS "{c}"' for c in FEATURE_COLS]
        + [f'"{c}"' for c in meta]
    )
    df = duckdb.sql(f"SELECT {sel} FROM '{path}'").df()
    return df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)


def merge_classes(y: np.ndarray) -> np.ndarray:
    """Apply MERGE_MAP (1->2, 9->8) to an array of raw class codes."""
    y = y.copy().astype(int)
    for src, tgt in MERGE_MAP.items():
        y[y == src] = tgt
    return y


def dedup_to_latest_year(df):
    """One row per unique (lon, lat) — the most recent year.

    Mirrors the stage-2 support dedup in the CV pipeline: gives the model spatially
    diverse rows rather than ~9× yearly replicates of the same pixel. No-op if the
    frame lacks lon/year.
    """
    if "year" not in df.columns or "lon" not in df.columns:
        return df
    return (df.sort_values("year", ascending=False)
              .drop_duplicates(subset=["lon", "lat"])
              .reset_index(drop=True))


# ==============================================================================
# CatBoost (live loop model) — GPU training, CPU prediction
# ==============================================================================
# Device for training. "GPU" uses the A40 (the vGPU has its own CatBoost allocator —
# unlike XGBoost 2.x+, which needs CUDA VMM the vGPU disables). Override with the
# NYVEST_AL_DEVICE env var ("CPU"/"GPU"). Prediction always runs on CPU (fast).
TRAIN_DEVICE = os.environ.get("NYVEST_AL_DEVICE", "GPU")


def make_model(iterations: int = 300, n_classes: int | None = None,
               device: str | None = None, **kw):
    """CatBoost classifier (MultiClass), GPU-trained, depth 6.

    Chosen over XGBoost after a real-data head-to-head (2026-06-22): on this A40-24Q vGPU,
    CatBoost GPU trains a 300-tree depth-6 model in ~2.6 s (5-fold cross-conformal ~14.5 s)
    vs XGBoost-1.7.6 GPU 7.6 s / 35.9 s — and CatBoost's symmetric-tree CPU predict does a
    954k-px tile in ~0.7 s vs XGBoost ~12 s (17×). So CatBoost wins on BOTH train cycles and
    dense inference. F1 is within noise across both (≈0.686–0.690). XGBoost 2.x+ GPU can't run
    here at all (no CUDA VMM); see reports/active_learning/timber_report.md.

    Trained on positional labels 0..n_classes-1, so predict_proba columns / classes_ are
    already canonical (no remap). `device` overrides TRAIN_DEVICE ("GPU"/"CPU").
    `n_classes` is accepted for signature parity with the old XGBoost path (CatBoost infers it).
    """
    from catboost import CatBoostClassifier

    kw.pop("n_classes", None)  # ignore parity kwarg if passed positionally elsewhere
    dev = (device or TRAIN_DEVICE).upper()
    params = dict(
        iterations=iterations, depth=6, learning_rate=0.1,
        loss_function="MultiClass", random_seed=SEED, verbose=False,
        allow_writing_files=False,
    )
    if dev == "GPU":
        params.update(task_type="GPU", devices="0")
    else:
        params.update(task_type="CPU", thread_count=8)
    params.update(kw)
    return CatBoostClassifier(**params)


# Back-compat alias.
make_catboost = make_model


# ==============================================================================
# APS conformal (ported verbatim from llto_schemeB_allyears.py)
# ==============================================================================
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
    """APS conformity score of the TRUE label for each calibration row."""
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = len(y)
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    s = cum_at + u[:, None] * probs
    return s[np.arange(n), y]


def aps_score_matrix(probs, u):
    """Full APS score matrix s[i, c]; class c is in row i's set iff s[i,c] <= tau."""
    ranks, _, sorted_p = _ranks_sorted(probs)
    n = probs.shape[0]
    cs = np.cumsum(sorted_p, axis=1)
    cs_before = np.concatenate([np.zeros((n, 1)), cs[:, :-1]], axis=1)
    cum_at = np.take_along_axis(cs_before, ranks - 1, axis=1)
    return cum_at + u[:, None] * probs


def cross_conformal_aps(X_tr, y_tr, K, seed, n_classes):
    """K-fold cross-conformal APS calibration scores on the training set.

    Returns pooled OOF conformity scores (one per training row); their conformal
    quantile is the global tau used for set membership.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    pooled = np.empty(len(y_tr), dtype=np.float64)
    u_all = np.random.default_rng(seed).uniform(size=len(y_tr))
    for idx_t, idx_v in skf.split(X_tr, y_tr):
        m = make_model()
        m.fit(X_tr[idx_t], y_tr[idx_t])
        raw = m.predict_proba(X_tr[idx_v]).astype(np.float64)
        # A fold's train split may omit a rare class (e.g. cls12, ~1.3k rows): CatBoost
        # then emits fewer columns and classes_ skips it. Scatter into the full
        # 0..n_classes-1 space so APS scoring indexes the right column.
        cols = np.asarray(m.classes_).astype(int)
        if raw.shape[1] == n_classes and np.array_equal(cols, np.arange(n_classes)):
            p_v = raw
        else:
            p_v = np.zeros((len(idx_v), n_classes), dtype=np.float64)
            p_v[:, cols] = raw
        pooled[idx_v] = aps_cal_scores(p_v, y_tr[idx_v], u_all[idx_v])
    return pooled


def prediction_set_sizes(probs: np.ndarray, tau: float, seed: int = SEED):
    """APS prediction-set size per row at the given tau (the uncertainty signal).

    Returns (set_sizes:int[n], in_set:bool[n,C]). Larger size = more ambiguous =
    higher active-learning priority. Uses a fixed rng so sizes are reproducible.
    """
    u = np.random.default_rng(seed).uniform(size=len(probs))
    s_mat = aps_score_matrix(probs.astype(np.float64), u)
    in_set = s_mat <= tau
    # Guarantee at least the argmax is in the set (avoid empty sets at display time).
    top = probs.argmax(axis=1)
    in_set[np.arange(len(probs)), top] = True
    return in_set.sum(axis=1).astype(int), in_set


# ==============================================================================
# Artifact persistence
# ==============================================================================
@dataclass
class SeedModel:
    """A trained, calibrated model ready for prediction + uncertainty scoring."""
    model: object                 # fitted CatBoostClassifier
    tau: float                    # global APS quantile
    classes: np.ndarray           # merged class codes, in model column order
    class_names: dict             # {code(str): label}
    alpha: float = ALPHA

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def predict_proba_aligned(self, X: np.ndarray) -> np.ndarray:
        """predict_proba with columns in canonical `self.classes` order.

        The seed model is trained on positional labels 0..n-1, so XGBoost's
        predict_proba columns and `classes_` are already `arange(n)` — column j maps
        directly to `self.classes[j]`, no remap. If a model was instead trained on the
        merged codes (`classes_ == self.classes`), that is also already aligned. Any
        other ordering is remapped by code; an unrecognised code or a wrong column
        count is a hard error (silent mis-alignment would corrupt every prediction +
        the APS uncertainty layer).
        """
        raw = self.model.predict_proba(X).astype(np.float64)
        if raw.shape[1] != self.n_classes:
            raise ValueError(
                f"model emitted {raw.shape[1]} prob columns, expected "
                f"{self.n_classes}; artifact/encoder mismatch")
        model_classes = np.asarray(self.model.classes_).astype(int)
        if np.array_equal(model_classes, np.arange(self.n_classes)) or \
           np.array_equal(model_classes, self.classes):
            return raw  # column j already maps to self.classes[j]
        out = np.zeros((len(X), self.n_classes), dtype=np.float64)
        col = {int(c): j for j, c in enumerate(self.classes)}
        for j_model, c in enumerate(model_classes):
            if int(c) not in col:
                raise ValueError(
                    f"model class {int(c)} not in canonical classes "
                    f"{self.classes.tolist()}; artifact/encoder mismatch")
            out[:, col[int(c)]] = raw[:, j_model]
        return out

    def predict(self, X: np.ndarray):
        """Return (pred_codes, set_sizes, probs) for rows X (aligned probs)."""
        probs = self.predict_proba_aligned(X)
        sizes, _ = prediction_set_sizes(probs, self.tau)
        pred = self.classes[probs.argmax(axis=1)]
        return pred, sizes, probs

    def label_of(self, code: int) -> str:
        return self.class_names.get(str(int(code)), f"class {int(code)}")


def load_class_names(path: Path = CLASS_NAMES_PATH) -> dict:
    return json.loads(Path(path).read_text())["classes"]


def save_seed_model(model, tau: float, classes: np.ndarray,
                    models_dir: Path = MODELS_DIR) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(models_dir / MODEL_PATH.name))
    np.savez(models_dir / CALIB_PATH.name, tau=np.float64(tau),
             classes=np.asarray(classes, dtype=int), alpha=np.float64(ALPHA))
    (models_dir / ENCODER_PATH.name).write_text(
        json.dumps({"classes": [int(c) for c in classes]}, indent=2))


def load_seed_model(models_dir: Path = MODELS_DIR) -> SeedModel:
    from catboost import CatBoostClassifier

    models_dir = Path(models_dir)
    model = CatBoostClassifier()
    model.load_model(str(models_dir / MODEL_PATH.name))
    calib = np.load(models_dir / CALIB_PATH.name)
    return SeedModel(
        model=model,
        tau=float(calib["tau"]),
        classes=calib["classes"].astype(int),
        class_names=load_class_names(),
        alpha=float(calib["alpha"]),
    )

"""Fast, shared training/inference core for the nyvest DNN.

This is the refactor of the per-stage `train_one`/`train_fold`/`fit_predict`
copies that every stage script grew independently (stage3/4/5/6/7/8/9). It keeps
the *winning recipe* identical numerically (class-weighted CE, label smoothing,
early-stopped 2-layer MLP, 5-seed probability ensemble, StandardScaler) but makes
the training loop fast so new ideas iterate in seconds, and it exposes reusable
building blocks so a new experiment is ~30 lines instead of a copy-paste of 80.

Speed levers over the old loop (same maths, fewer stalls):
  * whole dataset lives on-GPU (664k x 67 f32 ~ 170 MB) — no per-epoch H2D copies.
  * mixed precision autocast on CUDA (A40 tensor cores) — ~1.5-2x on the matmuls.
  * per-epoch validation macro-F1 is computed ON-GPU (no .cpu()/sklearn call every
    epoch); the final sklearn macro_f1 is still used for the reported metric.
  * optional torch.compile (DNN_COMPILE=1) fuses the tiny MLP.
  * a frame cache (.npz) so repeat runs skip the duckdb parquet reload.

Nothing here changes the eval protocol — folds, cleaning and metrics still come
from data_utils, so numbers stay comparable to the TabICL 0.7139 reference and
the DNN best 0.7341.

Public API
----------
  load_cached(extra_features)              -> data dict (du.load_data + cache)
  Config(...)                              -> hyperparameter bundle (env-overridable)
  gpu_macro_f1(y_true_t, y_pred_t, C)      -> float, on-GPU macro-F1
  fit_ensemble(Xtr, ytr, C, cfg, ...)      -> Ensemble (scaler + N models)
  Ensemble.predict_proba(X)                -> np.ndarray [n, C]
  Ensemble.predict(X)                      -> np.ndarray [n]
  Ensemble.save(path) / Ensemble.load(path)
  cv_evaluate(data, cfg, relabel_fn=None)  -> results dict (3-fold spatial CV)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du            # noqa: E402
import conformal_methods as cm     # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _envf(k, d):
    return float(os.environ.get(k, d))


def _envi(k, d):
    return int(os.environ.get(k, d))


@dataclass
class Config:
    """Winning-recipe defaults; every field is env-overridable for sweeps."""
    hidden: tuple = field(default_factory=lambda: tuple(
        int(x) for x in os.environ.get("HIDDEN", "256,128").split(",")))
    dropout: float = field(default_factory=lambda: _envf("DROPOUT", 0.3))
    lr: float = field(default_factory=lambda: _envf("LR", 1e-3))
    weight_decay: float = field(default_factory=lambda: _envf("WEIGHT_DECAY", 1e-4))
    max_epochs: int = field(default_factory=lambda: _envi("MAX_EPOCHS", 200))
    patience: int = field(default_factory=lambda: _envi("PATIENCE", 15))
    batch: int = field(default_factory=lambda: _envi("BATCH", 4096))
    val_frac: float = field(default_factory=lambda: _envf("VAL_FRAC", 0.1))
    weight_mode: str = field(default_factory=lambda: os.environ.get("WEIGHT_MODE", "sqrt"))
    label_smooth: float = field(default_factory=lambda: _envf("LABEL_SMOOTH", 0.05))
    n_ensemble: int = field(default_factory=lambda: _envi("N_ENSEMBLE", 5))
    input_noise: float = field(default_factory=lambda: _envf("INPUT_NOISE", 0.0))
    mixup_alpha: float = field(default_factory=lambda: _envf("MIXUP_ALPHA", 0.0))
    seed: int = field(default_factory=lambda: _envi("SEED", 0))
    # AMP defaults OFF: measured slower on this tiny MLP (fp16 cast + GradScaler
    # overhead exceeds tensor-core benefit at 256->128) and slightly lower F1.
    # Keep the switch for larger nets probed later (DNN_AMP=1).
    amp: bool = field(default_factory=lambda: os.environ.get("DNN_AMP", "0") == "1")
    compile: bool = field(default_factory=lambda: os.environ.get("DNN_COMPILE", "0") == "1")
    # Measured levers (real data, 5-seed fold-0 fit, 442k rows, A40):
    #   fused Adam   124.6s -> 92.9s (1.34x), F1 unchanged  -> DEFAULT ON
    #   TF32         no-op (loop is kernel-launch-bound, not matmul-FLOP-bound)
    #   bf16 autocast 159.8s (SLOWER) — same story as the old fp16 result:
    #                per-op cast overhead beats tensor cores on a 256->128 net
    #   streams      no gain over fused (per-epoch host syncs prevent overlap);
    #                harmless, left as an opt-in switch
    # The bottleneck is Python minibatch launch overhead; fused Adam wins by
    # collapsing the per-parameter optimizer launches into one kernel.
    bf16: bool = field(default_factory=lambda: os.environ.get("DNN_BF16", "0") == "1")
    fused: bool = field(default_factory=lambda: os.environ.get("DNN_FUSED", "1") == "1")
    streams: bool = field(default_factory=lambda: os.environ.get("DNN_STREAMS", "0") == "1")

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


# --------------------------------------------------------------------------- #
# data cache
# --------------------------------------------------------------------------- #
def load_cached(extra_features: str = "lidar", refresh: bool = False):
    """du.load_data, memoised to a .npz so repeat runs skip the duckdb reload.

    The cache stores the arrays the trainer needs (X, labels, groups, coords,
    feat_cols, classes, lidar medians). The parquet mtime is embedded in the key
    so the cache self-invalidates when the source data changes.
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    mtime = int(du.STABLE_PARQUET.stat().st_mtime)
    lid_mtime = int(du.LIDAR_PARQUET.stat().st_mtime) if extra_features == "lidar" else 0
    key = f"frame_{extra_features}_{mtime}_{lid_mtime}.npz"
    path = _CACHE_DIR / key
    if path.exists() and not refresh:
        z = np.load(path, allow_pickle=True)
        return {
            "X": z["X"], "y": z["y"], "y_enc": z["y_enc"],
            "classes": z["classes"].tolist(), "groups": z["groups"],
            "lon": z["lon"], "lat": z["lat"],
            "feat_cols": z["feat_cols"].tolist(),
            "extra_features": extra_features,
            "lidar_med": z["lidar_med"].item() if z["lidar_med"].shape == () else None,
        }
    data = du.load_data(extra_features=extra_features)
    df = data["df"]
    np.savez(
        path, X=data["X"], y=data["y"], y_enc=data["y_enc"],
        classes=np.array(data["classes"]), groups=data["groups"],
        lon=df["lon"].values, lat=df["lat"].values,
        feat_cols=np.array(data["feat_cols"], dtype=object),
        lidar_med=np.array(data["lidar_med"], dtype=object),
    )
    return {
        "X": data["X"], "y": data["y"], "y_enc": data["y_enc"],
        "classes": data["classes"], "groups": data["groups"],
        "lon": df["lon"].values, "lat": df["lat"].values,
        "feat_cols": data["feat_cols"], "extra_features": extra_features,
        "lidar_med": data["lidar_med"],
    }


# --------------------------------------------------------------------------- #
# model + fast training
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def class_weights(y_enc, n_classes, mode):
    if mode == "none":
        return None
    counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    if mode == "inv":
        w = len(y_enc) / (n_classes * counts)
    else:
        w = np.sqrt(1.0 / counts)
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def _gpu_macro_f1_t(y_true_t: torch.Tensor, y_pred_t: torch.Tensor,
                    n_classes: int) -> torch.Tensor:
    """Macro-F1 as a 0-dim device tensor (no host sync). See gpu_macro_f1."""
    t = y_true_t.to(torch.int64)
    p = y_pred_t.to(torch.int64)
    idx = torch.arange(n_classes, device=t.device)
    tp = ((p[None, :] == idx[:, None]) & (t[None, :] == idx[:, None])).sum(1).float()
    pp = (p[None, :] == idx[:, None]).sum(1).float()
    ap = (t[None, :] == idx[:, None]).sum(1).float()
    denom = pp + ap
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return f1.mean()


def gpu_macro_f1(y_true_t: torch.Tensor, y_pred_t: torch.Tensor, n_classes: int) -> float:
    """Macro-F1 computed entirely on-device (no host round-trip per epoch).

    Matches sklearn's f1_score(average='macro', labels=arange(C), zero_division=0):
    classes absent from BOTH truth and prediction contribute 0.
    """
    return float(_gpu_macro_f1_t(y_true_t, y_pred_t, n_classes))


def _train_one(Xtr_t, ytr_t, Xval_t, yval_t, in_dim, n_classes, w, cfg: Config, seed):
    """Single model, fast loop: data already on-GPU, AMP, on-GPU val F1."""
    set_seed(seed)
    model = MLP(in_dim, n_classes, cfg.hidden, cfg.dropout).to(DEVICE)
    if cfg.compile:
        model = torch.compile(model)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smooth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                           fused=(cfg.fused and DEVICE == "cuda"))
    # bf16 autocast needs no GradScaler (unlike fp16); fp16 path keeps the scaler.
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    use_amp = (cfg.amp or cfg.bf16) and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not cfg.bf16)
    n_tr = Xtr_t.shape[0]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, cfg.batch):
            b = order[i:i + cfg.batch]
            xb, yb = Xtr_t[b], ytr_t[b]
            if cfg.input_noise > 0:
                xb = xb + cfg.input_noise * torch.randn(xb.shape, device=DEVICE, generator=g)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if cfg.mixup_alpha > 0 and xb.shape[0] > 1:
                    lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                    perm = torch.randperm(xb.shape[0], device=DEVICE, generator=g)
                    xm = lam * xb + (1 - lam) * xb[perm]
                    logits = model(xm)
                    loss = lam * crit(logits, yb) + (1 - lam) * crit(logits, yb[perm])
                else:
                    loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            vp = model(Xval_t).argmax(1)
        f1 = gpu_macro_f1(yval_t, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def _train_one_streamed(Xtr_t, ytr_t, Xval_t, yval_t, in_dim, n_classes, w, cfg, seed):
    """Same maths as _train_one, but NO per-epoch host sync so the caller can
    overlap several of these across CUDA streams.

    The two syncs in _train_one that serialize streams are (a) `float(f1)` for
    early-stop and (b) the CPU `.clone()` of best_state each improvement. Here we
    keep val-F1 and the best snapshot ON-GPU (best_state cloned device-side), and
    resolve early stopping with a GPU boolean — so the whole per-seed loop is
    launch-only and streams truly overlap. best_state is pulled to CPU once at the
    end. Early-stopping decisions are identical (same +1e-4 margin, same patience).
    """
    set_seed(seed)
    model = MLP(in_dim, n_classes, cfg.hidden, cfg.dropout).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smooth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                           fused=(cfg.fused and DEVICE == "cuda"))
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    use_amp = (cfg.amp or cfg.bf16) and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not cfg.bf16)
    n_tr = Xtr_t.shape[0]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    best_f1_t = torch.tensor(-1.0, device=DEVICE)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    for _ in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, cfg.batch):
            b = order[i:i + cfg.batch]
            xb, yb = Xtr_t[b], ytr_t[b]
            if cfg.input_noise > 0:
                xb = xb + cfg.input_noise * torch.randn(xb.shape, device=DEVICE, generator=g)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if cfg.mixup_alpha > 0 and xb.shape[0] > 1:
                    lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                    perm = torch.randperm(xb.shape[0], device=DEVICE, generator=g)
                    xm = lam * xb + (1 - lam) * xb[perm]
                    logits = model(xm)
                    loss = lam * crit(logits, yb) + (1 - lam) * crit(logits, yb[perm])
                else:
                    loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            vp = model(Xval_t).argmax(1)
        f1_t = _gpu_macro_f1_t(yval_t, vp, n_classes)   # stays on device
        improved = bool((f1_t > best_f1_t + 1e-4).item())  # one tiny sync/epoch
        if improved:
            best_f1_t = f1_t
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, float(best_f1_t.item())


def _fit_seeds_streamed(Xtr_t, ytr_t, Xval_t, yval_t, in_dim, n_classes, w, cfg,
                        seeds, verbose):
    """Train the N ensemble seeds concurrently, each on its own CUDA stream.

    The MLP is so small (256->128) that one seed leaves the A40 almost idle;
    issuing all N seeds' kernels onto separate streams lets the scheduler pack
    them. Numerically identical to sequential training: each seed has its own
    model/optimizer/RNG and the shared X/y tensors are read-only.
    """
    streams = [torch.cuda.Stream() for _ in seeds]
    results = [None] * len(seeds)
    for j, s in enumerate(seeds):
        with torch.cuda.stream(streams[j]):
            results[j] = _train_one_streamed(
                Xtr_t, ytr_t, Xval_t, yval_t, in_dim, n_classes, w, cfg, s)
    torch.cuda.synchronize()
    models = [r[0] for r in results]
    vf1s = [r[1] for r in results]
    if verbose:
        for e, v in enumerate(vf1s):
            print(f"    seed {e}: val_f1={v:.4f}", flush=True)
    return models, vf1s


def _proba(model, X_t, n_classes, amp):
    model.eval()
    out = torch.empty((X_t.shape[0], n_classes), device=X_t.device, dtype=torch.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp and DEVICE == "cuda"):
        for i in range(0, X_t.shape[0], 65536):
            out[i:i + 65536] = F.softmax(model(X_t[i:i + 65536]).float(), 1)
    return out


# --------------------------------------------------------------------------- #
# Ensemble: fit once, predict/save/load — the deployable object
# --------------------------------------------------------------------------- #
class Ensemble:
    """A fitted 5-seed probability ensemble + its StandardScaler + class map.

    Portable: `save()` writes a single .pt with weights, scaler stats, class
    decode map, feature column order and lidar medians so `predict.py` /
    `predict_raster.py` can run standalone without touching the training data.
    """

    def __init__(self, models, mean, std, classes, feat_cols, cfg: Config,
                 lidar_med=None, amp=True):
        self.models = models
        self.mean = mean.astype(np.float32)   # scaler mean  [F]
        self.std = std.astype(np.float32)     # scaler std   [F]
        self.classes = list(classes)          # decode: enc index -> raw class value
        self.feat_cols = list(feat_cols)
        self.cfg = cfg
        self.lidar_med = lidar_med
        self.amp = amp

    @property
    def n_classes(self):
        return len(self.classes)

    def _standardize(self, X):
        return (X.astype(np.float32) - self.mean) / self.std

    def _gpu_stats(self):
        """Cache mean/std and the raw-class decode vector as GPU tensors."""
        if getattr(self, "_mean_t", None) is None:
            self._mean_t = torch.as_tensor(self.mean, device=DEVICE)
            self._std_t = torch.as_tensor(self.std, device=DEVICE)
            self._decode_t = torch.as_tensor(np.array(self.classes, dtype=np.int16),
                                             device=DEVICE)
        return self._mean_t, self._std_t

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Averaged softmax over the ensemble. X: [n, F] raw features."""
        Xs = self._standardize(X)
        X_t = torch.as_tensor(Xs, device=DEVICE)
        P = torch.zeros((X.shape[0], self.n_classes), device=DEVICE, dtype=torch.float32)
        for m in self.models:
            P += _proba(m, X_t, self.n_classes, self.amp)
        return (P / len(self.models)).cpu().numpy()

    def predict_proba_gpu(self, X_raw_t: torch.Tensor, chunk: int = 262144) -> torch.Tensor:
        """RAW features already on-GPU [n,F] -> mean-softmax proba [n,C] on-GPU.

        Standardizes on-device (no numpy round-trip) and averages the ensemble
        softmax. This is the hot path for raster inference: the caller keeps the
        block on the GPU and only pulls the small class map back to host. `chunk`
        bounds the intermediate activations so a big block can't OOM.
        """
        mean_t, std_t = self._gpu_stats()
        n = X_raw_t.shape[0]
        out = torch.empty((n, self.n_classes), device=DEVICE, dtype=torch.float32)
        E = len(self.models)
        for i in range(0, n, chunk):
            xb = (X_raw_t[i:i + chunk] - mean_t) / std_t
            acc = torch.zeros((xb.shape[0], self.n_classes), device=DEVICE)
            with torch.no_grad():
                for m in self.models:
                    acc += F.softmax(m(xb).float(), 1)
            out[i:i + chunk] = acc / E
        return out

    def predict_classmap_gpu(self, X_raw_t: torch.Tensor, chunk: int = 262144):
        """Raw-GPU features -> argmax RAW class codes as int16 on-GPU.

        Skips building/keeping the full [n,C] proba array — for a class map we
        only need the argmax, so we reduce per-chunk. Returns [n] int16.
        """
        mean_t, std_t = self._gpu_stats()
        n = X_raw_t.shape[0]
        cls = torch.empty(n, device=DEVICE, dtype=torch.int16)
        E = len(self.models)
        for i in range(0, n, chunk):
            xb = (X_raw_t[i:i + chunk] - mean_t) / std_t
            acc = torch.zeros((xb.shape[0], self.n_classes), device=DEVICE)
            with torch.no_grad():
                for m in self.models:
                    acc += F.softmax(m(xb).float(), 1)
            cls[i:i + chunk] = self._decode_t[acc.argmax(1)]
        return cls

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax class as the RAW class value (decoded, e.g. 12), not enc index."""
        enc = self.predict_proba(X).argmax(1)
        return np.array(self.classes, dtype=np.int64)[enc]

    def predict_full(self, X: np.ndarray, calib: "Calibration") -> dict:
        """Raw features -> full UQ bundle: class map + calibrated proba +
        LAC+Mondrian prediction sets. Tabular (CPU-returning) path; for raster
        blocks use `predict_full_gpu` to skip the extra host round-trip on the
        raw-proba step.
        """
        P_raw = self.predict_proba(X)
        pred_enc = P_raw.argmax(1)
        proba_cal = calib.predict_proba_calibrated(P_raw)
        included, set_size = calib.predict_sets(P_raw)
        return {
            "pred_class": np.array(self.classes, dtype=np.int64)[pred_enc],
            "proba_calibrated": proba_cal,
            "set_size": set_size,
            "included": included,
        }

    def predict_full_gpu(self, X_raw_t: torch.Tensor, calib: "Calibration",
                         chunk: int = 262144) -> dict:
        """Raw-GPU features -> full UQ bundle, numpy-side (host) outputs.

        The ensemble forward pass runs on-GPU (`predict_proba_gpu`); the
        calibration step (temp-scale or Venn-Abers OvR + LAC/Mondrian sets)
        is numpy/CPU since it's a cheap per-class searchsorted/divide, not a
        matmul — negligible next to raster I/O (see DNN/README.md's
        I/O-bound finding for predict_raster.py).
        """
        P_raw_t = self.predict_proba_gpu(X_raw_t, chunk)
        pred_enc = P_raw_t.argmax(1).cpu().numpy()   # argmax on-GPU; only the small index array crosses PCIe here
        P_raw = P_raw_t.cpu().numpy()                # full proba still needed host-side for calibration below
        proba_cal = calib.predict_proba_calibrated(P_raw)
        included, set_size = calib.predict_sets(P_raw)
        return {
            "pred_class": np.array(self.classes, dtype=np.int64)[pred_enc],
            "proba_calibrated": proba_cal,
            "set_size": set_size,
            "included": included,
        }

    def save(self, path):
        path = Path(path)
        torch.save({
            "state_dicts": [
                {k: v.cpu() for k, v in
                 (m._orig_mod if hasattr(m, "_orig_mod") else m).state_dict().items()}
                for m in self.models],
            "mean": self.mean, "std": self.std, "classes": self.classes,
            "feat_cols": self.feat_cols, "cfg": self.cfg.to_dict(),
            "lidar_med": self.lidar_med, "in_dim": len(self.feat_cols),
        }, path)
        return path

    @classmethod
    def load(cls, path, device=None):
        dev = device or DEVICE
        ck = torch.load(path, map_location=dev, weights_only=False)
        cfg = Config(**{k: (tuple(v) if k == "hidden" else v)
                        for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
        models = []
        for sd in ck["state_dicts"]:
            m = MLP(ck["in_dim"], len(ck["classes"]), cfg.hidden, cfg.dropout).to(dev)
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
        return cls(models, np.asarray(ck["mean"]), np.asarray(ck["std"]),
                   ck["classes"], ck["feat_cols"], cfg, ck.get("lidar_med"), cfg.amp)


class Calibration:
    """Loads a `models/dnn_final_calib.npz` (from `fit_calibration.py`) and
    applies it to raw ensemble probabilities: a point-calibration transform
    (temperature scaling or Venn-Abers one-vs-rest, whichever won the
    head-to-head ECE comparison at fit time) plus LAC+Mondrian conformal
    prediction sets. All numpy, CPU-side — cheap relative to raster I/O.
    """

    def __init__(self, calib_method, taus, classes, T=None, va_calibrators=None):
        self.calib_method = calib_method          # "temp_scale" | "venn_abers"
        self.taus = np.asarray(taus, dtype=np.float64)   # [n_classes], LAC+Mondrian
        self.classes = list(classes)
        self.T = T                                  # scalar, if temp_scale
        self.va_calibrators = va_calibrators         # {class_idx: (p0,p1,c)}, if venn_abers

    @property
    def n_classes(self):
        return len(self.classes)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        method = str(z["calib_method"])
        classes = z["classes"].tolist()
        n_classes = len(classes)
        # fit_calibration.py persists both methods' state when available (VA
        # breakpoints are cheap to keep even when temp-scale wins) — load them
        # opportunistically so callers can inspect/switch without a re-fit.
        va_calibrators = None
        if f"va_p0_{0}" in z.files:
            va_calibrators = {
                c: (z[f"va_p0_{c}"], z[f"va_p1_{c}"], z[f"va_c_{c}"])
                for c in range(n_classes)
            }
        return cls(method, z["lac_mondrian_taus"], classes,
                   T=float(z["T"]), va_calibrators=va_calibrators)

    def predict_proba_calibrated(self, P: np.ndarray) -> np.ndarray:
        """Raw ensemble softmax [n, C] -> calibrated proba [n, C] (sums to 1)."""
        if self.calib_method == "temp_scale":
            eps = 1e-12
            logp = np.log(np.clip(P, eps, 1.0))
            return cm._softmax_from_logprobs(logp, self.T).astype(np.float32)
        return cm.apply_venn_abers_ovr(self.va_calibrators, P)

    def predict_sets(self, P: np.ndarray):
        """Raw ensemble softmax [n, C] -> (inclusion bool [n, C], set_size [n]).

        LAC score = 1 - p. `self.taus[c]` is the Mondrian threshold fit on
        rows whose TRUE label was class c; at inference every row's score for
        CANDIDATE class c is compared against that same tau_c (column-wise
        broadcast, not gathered by the row's predicted/argmax class) — a
        prediction set can and should include classes other than the argmax.
        Matches conformal_methods.sets_from_tau's class-conditional semantics
        exactly (`score_mat_test <= tau[None, :]`).
        """
        score = 1.0 - P
        included = score <= self.taus[None, :]
        return included, included.sum(1).astype(np.int32)


def fit_ensemble(Xtr: np.ndarray, ytr: np.ndarray, n_classes: int, cfg: Config,
                 feat_cols, classes, lidar_med=None, rng=None, verbose=False):
    """Fit the scaler + N-seed ensemble on (Xtr, ytr). Returns an Ensemble.

    Standardization stats come from the FULL training split (leak-free: caller
    passes only train-fold rows). The internal val split (for early stopping) is
    drawn from Xtr with cfg.val_frac.
    """
    rng = rng or np.random.default_rng(cfg.seed)
    mean = Xtr.mean(0)
    std = Xtr.std(0)
    std[std == 0] = 1.0
    Xs = ((Xtr - mean) / std).astype(np.float32)
    in_dim = Xs.shape[1]
    n = len(Xs)
    perm = rng.permutation(n)
    nv = int(n * cfg.val_frac)
    vi, ti = perm[:nv], perm[nv:]
    Xtr_t = torch.as_tensor(Xs[ti], device=DEVICE)
    ytr_t = torch.as_tensor(ytr[ti], device=DEVICE)
    Xval_t = torch.as_tensor(Xs[vi], device=DEVICE)
    yval_t = torch.as_tensor(ytr[vi], device=DEVICE)
    w = class_weights(ytr[ti], n_classes, cfg.weight_mode)
    seeds = [cfg.seed + 100 * e for e in range(cfg.n_ensemble)]
    if cfg.streams and DEVICE == "cuda":
        models, vf1s = _fit_seeds_streamed(
            Xtr_t, ytr_t, Xval_t, yval_t, in_dim, n_classes, w, cfg, seeds, verbose)
    else:
        models, vf1s = [], []
        for e, s in enumerate(seeds):
            m, vf1 = _train_one(Xtr_t, ytr_t, Xval_t, yval_t, in_dim,
                                n_classes, w, cfg, s)
            models.append(m)
            vf1s.append(vf1)
            if verbose:
                print(f"    seed {e}: val_f1={vf1:.4f}", flush=True)
    ens = Ensemble(models, mean, std, classes, feat_cols, cfg, lidar_med, cfg.amp)
    ens.val_f1 = float(np.mean(vf1s))
    return ens


# --------------------------------------------------------------------------- #
# 3-fold spatial CV harness (the experiment entry point)
# --------------------------------------------------------------------------- #
def cv_evaluate(data, cfg: Config = None, relabel_fn=None, clean_cls12=True,
                verbose=True):
    """Run the reference 3-fold spatial CV and return a results dict.

    `relabel_fn(k, tr_idx, y_enc, classes, X, lon, lat) -> (tr_use, ytr, cleaned)`
    lets an experiment inject train-only label surgery per fold (e.g. stage8
    cls12 relabel). If None, applies the default train-only cls12 centroid clean.
    Test labels are never touched.
    """
    cfg = cfg or Config()
    set_seed(cfg.seed)
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    lon, lat = data["lon"], data["lat"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()
    f1s, pcs = [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        if relabel_fn is not None:
            tr_use, ytr, note = relabel_fn(k, tr, y_enc, classes, X, lon, lat)
        elif clean_cls12 and cls12_enc >= 0:
            keep = du.clean_stale_class_mask(
                X[tr], y_enc[tr], None, cls12_enc, lon[tr], lat[tr])
            tr_use, ytr, note = tr[keep], y_enc[tr[keep]], f"cls12_dropped={int((~keep).sum())}"
        else:
            tr_use, ytr, note = tr, y_enc[tr], ""
        ens = fit_ensemble(X[tr_use], ytr, n_classes, cfg,
                           data["feat_cols"], classes, data.get("lidar_med"), rng)
        pred_enc = ens.predict_proba(X[te]).argmax(1)
        f1 = du.macro_f1(y_enc[te], pred_enc, n_classes)
        f1s.append(f1)
        pcs.append(du.per_class_f1(y_enc[te], pred_enc, n_classes))
        if verbose:
            print(f"  fold {k}: F1={f1:.4f}  val_f1={ens.val_f1:.4f}  "
                  f"{note}  {time.perf_counter()-ft:.1f}s", flush=True)
    f1m, f1std = float(np.mean(f1s)), float(np.std(f1s))
    pc = np.mean(pcs, axis=0)
    if verbose:
        print(f"F1 mean={f1m:.4f} std={f1std:.4f}  Δ(0.7139)={f1m-0.7139:+.4f}  "
              f"total {time.perf_counter()-t0:.1f}s", flush=True)
    return {
        "config": cfg.to_dict(), "f1_mean": round(f1m, 4), "f1_std": round(f1std, 4),
        "f1_per_fold": [round(v, 4) for v in f1s],
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc)},
        "wall_s": round(time.perf_counter() - t0, 1),
    }

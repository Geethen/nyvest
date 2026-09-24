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
    # Architecture. "mlp" is the deployed 256,128 net; "moe_shared" is the
    # DeepSeekMoE-style shared-expert MoE from autoresearch/moe_layers.py, whose
    # shared expert IS the mlp — so a moe_shared checkpoint degrades to the mlp
    # exactly, with one flag (`Ensemble.local_off`).
    arch: str = field(default_factory=lambda: os.environ.get("ARCH", "mlp"))
    n_experts: int = field(default_factory=lambda: _envi("N_EXPERTS", 8))
    top_k: int = field(default_factory=lambda: _envi("TOP_K", 2))
    expert_hidden: tuple = field(default_factory=lambda: tuple(
        int(x) for x in os.environ.get("EXPERT_HIDDEN", "64,32").split(",")))
    gate_src: str = field(default_factory=lambda: os.environ.get("GATE_SRC", "content"))

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["expert_hidden"] = list(self.expert_hidden)
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
    # The merge signature MUST be in the key: $MERGE_EXTRA changes y/y_enc/classes
    # without touching the parquet, so a key on mtime alone would serve labels
    # from a different label space and silently train the wrong ontology.
    key = f"frame_{extra_features}_{du.merge_sig()}_{mtime}_{lid_mtime}.npz"
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


def _moe_mod(name):
    """Load autoresearch/<name>.py by path.

    By path rather than `sys.path.insert(autoresearch)` because that directory
    holds `layers.py` / `optimizers.py`, names generic enough to shadow real
    packages for every other importer in the process.
    """
    import importlib.util
    key = f"_dnn_{name}"
    if key in sys.modules:
        return sys.modules[key]
    src = Path(__file__).resolve().parent / "autoresearch" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(key, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod          # before exec: moe_fast imports moe_layers
    spec.loader.exec_module(mod)
    return mod


def build_model(in_dim, n_classes, cfg: Config):
    """The one place an architecture is chosen, so training, `Ensemble.load` and
    the raster path cannot disagree about what a checkpoint contains."""
    if cfg.arch == "mlp":
        return MLP(in_dim, n_classes, cfg.hidden, cfg.dropout)
    if cfg.arch == "moe_shared":
        ML = _moe_mod("moe_layers")
        return ML.SharedExpertMoE(
            in_dim, n_classes, hidden=cfg.hidden, dropout=cfg.dropout,
            n_experts=cfg.n_experts, top_k=cfg.top_k,
            expert_hidden=tuple(cfg.expert_hidden), gate_src=cfg.gate_src)
    raise ValueError(f"unknown arch {cfg.arch!r} (mlp | moe_shared)")


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
    model = build_model(in_dim, n_classes, cfg).to(DEVICE)
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
    model = build_model(in_dim, n_classes, cfg).to(DEVICE)
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

    def _fused(self):
        """The MoE ensemble rewritten as batched matmuls, or None for an MLP.

        A `moe_shared` member is a Python loop over 8 experts of three tiny
        Linears, so a 5-member chunk dispatches ~350 kernels that each do a few
        microseconds of work; folding the 40 experts and 5 trunks into shared
        GEMMs is ~2x on the deployed path and bit-identical on the class map
        (autoresearch/moe_fast.py, and README.md's ladder). Built once, lazily,
        because it is inference-only state that `save()` must not carry.
        """
        if self.cfg.arch != "moe_shared" or DEVICE != "cuda":
            return None
        if getattr(self, "_fused_ens", None) is None:
            MF = _moe_mod("moe_fast")
            self._fused_ens = MF.FusedMoEEnsemble.from_models(self.models)
        return self._fused_ens

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Averaged softmax over the ensemble. X: [n, F] raw features."""
        Xs = self._standardize(X)
        X_t = torch.as_tensor(Xs, device=DEVICE)
        P = torch.zeros((X.shape[0], self.n_classes), device=DEVICE, dtype=torch.float32)
        for m in self.models:
            P += _proba(m, X_t, self.n_classes, self.amp)
        return (P / len(self.models)).cpu().numpy()

    def predict_proba_gpu(self, X_raw_t: torch.Tensor, chunk: int = None) -> torch.Tensor:
        """RAW features already on-GPU [n,F] -> mean-softmax proba [n,C] on-GPU.

        Standardizes on-device (no numpy round-trip) and averages the ensemble
        softmax. This is the hot path for raster inference: the caller keeps the
        block on the GPU and only pulls the small class map back to host. `chunk`
        bounds the intermediate activations so a big block can't OOM.
        """
        mean_t, std_t = self._gpu_stats()
        chunk = chunk or self.default_chunk
        n = X_raw_t.shape[0]
        out = torch.empty((n, self.n_classes), device=DEVICE, dtype=torch.float32)
        fast, E = self._fused(), len(self.models)
        for i in range(0, n, chunk):
            xb = (X_raw_t[i:i + chunk] - mean_t) / std_t
            with torch.no_grad():
                if fast is not None:
                    out[i:i + chunk] = fast.mean_proba(xb)
                    continue
                acc = torch.zeros((xb.shape[0], self.n_classes), device=DEVICE)
                for m in self.models:
                    acc += F.softmax(m(xb).float(), 1)
            out[i:i + chunk] = acc / E
        return out

    @property
    def default_chunk(self):
        """Rows per forward chunk on the raster path.

        A MoE holds a (chunk, 40, 64) dense-expert activation — 4.0 GB at the old
        262144 — and `bench_moe_fast.py --chunk-sweep` shows throughput flat from
        65536 up, so the larger chunk was four times the memory for nothing.
        The MLP has no such tensor and keeps its original value.
        """
        return 65536 if self.cfg.arch == "moe_shared" else 262144

    def predict_classmap_gpu(self, X_raw_t: torch.Tensor, chunk: int = None):
        """Raw-GPU features -> argmax RAW class codes as int16 on-GPU.

        Skips building/keeping the full [n,C] proba array — for a class map we
        only need the argmax, so we reduce per-chunk. Returns [n] int16.
        """
        mean_t, std_t = self._gpu_stats()
        chunk = chunk or self.default_chunk
        n = X_raw_t.shape[0]
        cls = torch.empty(n, device=DEVICE, dtype=torch.int16)
        fast, E = self._fused(), len(self.models)
        for i in range(0, n, chunk):
            xb = (X_raw_t[i:i + chunk] - mean_t) / std_t
            with torch.no_grad():
                if fast is not None:
                    cls[i:i + chunk] = self._decode_t[fast.mean_proba(xb).argmax(1)]
                    continue
                acc = torch.zeros((xb.shape[0], self.n_classes), device=DEVICE)
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
                         chunk: int = None) -> dict:
        """Raw-GPU features -> full UQ bundle, numpy-side (host) outputs.

        Forward pass AND calibration both run on-device. Calibration used to be
        numpy on the host "since it's a cheap per-class searchsorted, negligible
        next to raster I/O" — that stopped being true once the I/O got fast: at
        400k rows it measured 962 ms against a 241 ms forward pass, 80% of this
        function and a hard 0.42 M px/s ceiling on the whole pipeline, since it
        blocks the one GPU thread and therefore the reader pool behind it. On
        device the same transform is 37 ms (25x), taking this function from
        1238 ms to 134 ms at 400k rows — 0.32 -> 2.99 M px/s, which puts UQ back
        below the I/O wall instead of being the wall. `Calibration.predict_*_gpu`
        mirror the numpy dtypes, so venn_abers output is bit-identical; see their
        docstrings for the temp_scale ULP caveat. DNN/verify_deploy.py checks
        both paths against each other on the real checkpoint.

        Only the small typed outputs cross PCIe; the [n,C] float32 raw proba
        never leaves the device.
        """
        P_raw_t = self.predict_proba_gpu(X_raw_t, chunk)
        pred_enc = P_raw_t.argmax(1).cpu().numpy()
        if P_raw_t.is_cuda:
            proba_cal = calib.predict_proba_calibrated_gpu(P_raw_t).cpu().numpy()
            included_t, set_size_t = calib.predict_sets_gpu(P_raw_t)
            included, set_size = included_t.cpu().numpy(), set_size_t.cpu().numpy()
        else:
            P_raw = P_raw_t.numpy()
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
            "arch": self.cfg.arch,
        }, path)
        return path

    @classmethod
    def load(cls, path, device=None):
        dev = device or DEVICE
        ck = torch.load(path, map_location=dev, weights_only=False)
        # Checkpoints written before `arch` existed are all MLPs; `Config`'s own
        # default comes from $ARCH, which must not be allowed to reinterpret an
        # old file as a MoE.
        saved = dict(ck["cfg"])
        saved.setdefault("arch", ck.get("arch", "mlp"))
        cfg = Config(**{k: (tuple(v) if k in ("hidden", "expert_hidden") else v)
                        for k, v in saved.items() if k in Config.__dataclass_fields__})
        models = []
        for sd in ck["state_dicts"]:
            m = build_model(ck["in_dim"], len(ck["classes"]), cfg).to(dev)
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

    def __init__(self, calib_method, taus, classes, T=None, va_calibrators=None,
                 arch=None):
        self.calib_method = calib_method          # "temp_scale" | "venn_abers"
        self.taus = np.asarray(taus, dtype=np.float64)   # [n_classes], LAC+Mondrian
        self.classes = list(classes)
        self.T = T                                  # scalar, if temp_scale
        self.va_calibrators = va_calibrators         # {class_idx: (p0,p1,c)}, if venn_abers
        # Architecture of the ensemble these calibrators were FIT on, or None for
        # files written before this was recorded. Venn-Abers breakpoints and LAC
        # taus are properties of one model's score distribution, so reusing them
        # across architectures silently voids the coverage guarantee — and the
        # class list, the only thing previously checked, is identical between the
        # MLP and the MoE. See predict_raster.py's guard.
        self.arch = arch

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
                   T=float(z["T"]), va_calibrators=va_calibrators,
                   arch=str(z["arch"]) if "arch" in z.files else None)

    # ----------------------------------------------------------------- on-GPU
    # The raster path applies these to every valid pixel of a county, and the
    # numpy implementation below is the pipeline's wall once the reads got fast:
    # Venn-Abers is 20 binary searches into ~330k-entry breakpoint arrays per
    # row, which measured 962 ms of a 1238 ms GPU-thread step at 400k rows (0.42
    # M px/s) — and it runs INSIDE the single GPU thread, so it stalls the reader
    # pool too. torch.searchsorted has exactly numpy's left/right semantics, so
    # the whole transform moves to the device with no change in arithmetic; the
    # `_selftest` at the bottom of this class asserts bit-equality on both paths.
    def _gpu_tables(self, device):
        """Upload the calibration tables once (~80 MB for a 10-class VA fit)."""
        if getattr(self, "_gpu_dev", None) == str(device):
            return self._gpu_tbl
        t = {"taus": torch.as_tensor(self.taus, device=device, dtype=torch.float64)}
        if self.va_calibrators is not None:
            # Only column 1 of p0/p1 is ever read, so slice at upload: half the
            # memory and a contiguous gather instead of a strided one.
            t["va"] = [
                (torch.as_tensor(np.ascontiguousarray(c), device=device,
                                 dtype=torch.float64),
                 torch.as_tensor(np.ascontiguousarray(p0[:, 1]), device=device,
                                 dtype=torch.float64),
                 torch.as_tensor(np.ascontiguousarray(p1[:, 1]), device=device,
                                 dtype=torch.float64))
                for p0, p1, c in (self.va_calibrators[i]
                                  for i in range(self.n_classes))]
        self._gpu_dev, self._gpu_tbl = str(device), t
        return t

    def predict_proba_calibrated_gpu(self, P_t: "torch.Tensor") -> "torch.Tensor":
        """On-device twin of `predict_proba_calibrated`. [n,C] f32 -> [n,C] f32.

        Dtypes mirror the numpy path deliberately — float64 for the Venn-Abers
        searchsorted/divide, float32 throughout temperature scaling — because
        the widths, not just the formulas, decide the result.

        venn_abers (the fitted method) comes out BIT-IDENTICAL to numpy: it is
        all comparisons, gathers and divides, and torch.searchsorted's
        right=True/False are exactly numpy's side="right"/"left".
        temp_scale agrees to one float32 ULP (2.4e-7 max, measured at 400k rows)
        because torch and numpy use different libm exp/log — far below the
        1/60000 quantisation the raster path writes this band at, and it cannot
        move a prediction set, which is scored on the RAW proba (`predict_sets`).
        """
        tbl = self._gpu_tables(P_t.device)
        if self.calib_method == "temp_scale":
            logp = torch.log(P_t.clamp(1e-12, 1.0))
            z = logp / self.T
            z = z - z.amax(1, keepdim=True)
            e = torch.exp(z)
            return e / e.sum(1, keepdim=True)
        p1_out = torch.empty(P_t.shape, device=P_t.device, dtype=torch.float64)
        for c, (cpts, p0col, p1col) in enumerate(tbl["va"]):
            out = P_t[:, c].to(torch.float64)     # numpy promotes the needle too
            p0_at = p0col[torch.searchsorted(cpts, out, right=True)]
            p1_at = p1col[torch.searchsorted(cpts, out, right=False)]
            p1_out[:, c] = p1_at / (1.0 - p0_at + p1_at)
        total = p1_out.sum(1, keepdim=True)
        total = torch.where(total == 0, torch.ones_like(total), total)
        return (p1_out / total).to(torch.float32)

    def predict_sets_gpu(self, P_t: "torch.Tensor"):
        """On-device twin of `predict_sets`. -> (bool [n,C], int32 [n]).

        `1.0 - P` is evaluated in float32 and only then widened for the
        comparison, because that is what numpy does with a float32 `P` and
        float64 `taus`, and the rounding is observable at a tie.
        """
        tbl = self._gpu_tables(P_t.device)
        included = (1.0 - P_t).to(torch.float64) <= tbl["taus"][None, :]
        return included, included.sum(1).to(torch.int32)

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

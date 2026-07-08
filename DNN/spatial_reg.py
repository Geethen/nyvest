"""Spatial regularization — the follow-up the dropout/wd sweep couldn't be.

Diagnosis across scaling_grid.py + reg_sweep.py: the DNN fails by SPATIAL
generalization (val−test gap on GroupKFold cell_id folds), yet dropout / weight
decay / global-noise / mixup all regularize against SAMPLE memorization — they
can't see geography, so they only shrink the gap by dragging val down (confirmed:
raising wd collapses test). This module targets the SPATIAL failure directly with
two mechanisms that neither prior sweep tested:

1. SPATIAL-JITTER augmentation (`jitter_alpha` > 0). Per batch, blend each point's
   embedding toward another random point IN THE SAME CELL AND SAME CLASS:
   x' = (1−λ)·x + λ·x_neighbor,  λ ~ U(0, jitter_alpha).
   This is mixup restricted to a spatial+class neighborhood, so it interpolates
   along the LOCAL manifold (same land-cover, nearby geography — within-cell emb
   std is 0.70× global, verified) instead of globally. It forces the net off exact
   per-location signatures without crossing a class boundary (unlike plain mixup /
   global Gaussian noise, both already shown not to help). Lidar cols (last 3) are
   left unblended by default — they're already median-imputed / terrain, not the
   memorization surface.

2. SPATIAL early-stopping (`spatial_val=True`). The inner val split holds out whole
   CELLS (leave-cells-out inside the train fold), not random rows. Early stopping
   then optimizes for CROSS-CELL transfer — the thing the outer GroupKFold actually
   scores — instead of an in-distribution random val that races to 0.86−0.92 while
   test sits at 0.72. Leak-free: outer test fold is never touched; the spatial inner
   val is carved only from the current train fold's cells.

Everything else = winning recipe (class-weighted CE, label smoothing, sqrt weights,
5-seed... here 3-seed ensemble, StandardScaler, cls12 clean). Protocol identical to
the reference (3-fold GroupKFold cell_id, 64 AE + 3 lidar, leak-free test).

Baselines: scaling/​reg best 256,128 = 0.7304 (random val); reg-sweep overall best
512,256+drop0.5 = 0.7320; DNN reference best 0.7341 (5-seed + cls12 relabel).

Run:
  PY=~/myprojects/recover/.venv/bin/python
  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
    $PY DNN/spatial_reg.py            # writes spatial_reg.json (incremental)

Env: HIDDEN "256,128"  JITTERS "0,0.2,0.4"  SPATIAL_VAL "0,1"  SEEDS 3  FOLDS 3
     N_SPATIAL_VAL_CELLS auto (~10% of train cells)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du            # noqa: E402
import dnn_core as C               # noqa: E402

DEVICE = C.DEVICE


def parse_hidden(s):
    return tuple(int(x) for x in s.split(","))


HIDDEN = parse_hidden(os.environ.get("HIDDEN", "256,128"))
JITTERS = [float(x) for x in os.environ.get("JITTERS", "0,0.2,0.4").split(",")]
SPATIAL_VAL = [x == "1" for x in os.environ.get("SPATIAL_VAL", "0,1").split(",")]
SEEDS = int(os.environ.get("SEEDS", "3"))
FOLDS = int(os.environ.get("FOLDS", "3"))
LR = float(os.environ.get("LR", "1e-3"))
N_EMB = 64   # first 64 cols are AlphaEarth (jitter target); rest are lidar
BASELINE = 0.7304   # 256,128 random-val, full data (scaling grid)


# --------------------------------------------------------------------------- #
# per-cell, per-class neighbor index for spatial jitter
# --------------------------------------------------------------------------- #
def build_neighbor_pools(groups, y_enc):
    """Map (cell, class) -> array of row positions, for O(1) neighbor sampling.
    Only pools with >=2 members can donate a distinct neighbor."""
    pools = {}
    for i, (c, y) in enumerate(zip(groups, y_enc)):
        pools.setdefault((c, y), []).append(i)
    return {k: np.asarray(v) for k, v in pools.items() if len(v) >= 2}


def make_neighbor_map(local_groups, local_y, pools_global_pos, local_pos):
    """For each LOCAL row, precompute a same-cell same-class neighbor's LOCAL index
    (or itself if it has no partner). Vectorized per (cell,class) group.
    `local_pos` maps local idx -> global pos so we can reuse global pools? No —
    neighbors must be sampled WITHIN the training subset, so we build pools on the
    local arrays directly."""
    n = len(local_y)
    nbr = np.arange(n)                       # default: self (no jitter)
    pools = {}
    for i in range(n):
        pools.setdefault((local_groups[i], local_y[i]), []).append(i)
    rng = np.random.default_rng(0)
    for k, idxs in pools.items():
        if len(idxs) < 2:
            continue
        idxs = np.asarray(idxs)
        # each row gets a random OTHER member of its pool
        for i in idxs:
            j = i
            while j == i:
                j = idxs[rng.integers(len(idxs))]
            nbr[i] = j
    return nbr


def spatial_val_split(groups, y_enc, frac_cells=0.1, seed=0):
    """Leave-cells-out inner val: hold out whole cells (spatial), not random rows.
    Returns (train_local_idx, val_local_idx) into the passed arrays."""
    rng = np.random.default_rng(seed)
    cells = np.unique(groups)
    n_val = max(1, int(round(len(cells) * frac_cells)))
    val_cells = set(rng.choice(cells, size=n_val, replace=False).tolist())
    val_mask = np.isin(groups, list(val_cells))
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


# --------------------------------------------------------------------------- #
# training loop with spatial jitter + optional spatial early-stop
# --------------------------------------------------------------------------- #
def train_one_spatial(Xtr, ytr, gtr, in_dim, n_classes, w, seed,
                      jitter_alpha, spatial_val, hidden, lr):
    C.set_seed(seed)
    # inner val split: spatial (leave-cells-out) or random
    if spatial_val:
        ti, vi = spatial_val_split(gtr, ytr, frac_cells=0.1, seed=seed)
    else:
        perm = np.random.default_rng(seed).permutation(len(ytr))
        nv = int(len(ytr) * 0.1)
        vi, ti = perm[:nv], perm[nv:]

    Xtr_t = torch.as_tensor(Xtr[ti], device=DEVICE)
    ytr_t = torch.as_tensor(ytr[ti], device=DEVICE)
    Xval_t = torch.as_tensor(Xtr[vi], device=DEVICE)
    yval_t = torch.as_tensor(ytr[vi], device=DEVICE)

    # precompute same-cell same-class neighbor for each TRAIN-subset row
    nbr = None
    if jitter_alpha > 0:
        nbr = torch.as_tensor(
            make_neighbor_map(gtr[ti], ytr[ti], None, None), device=DEVICE)

    model = C.MLP(in_dim, n_classes, hidden, 0.3).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4,
                           fused=(DEVICE == "cuda"))
    n_tr = Xtr_t.shape[0]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    best_f1, best_state, bad = -1.0, None, 0
    emb_slice = slice(0, N_EMB)
    for _ in range(300):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, 4096):
            b = order[i:i + 4096]
            xb, yb = Xtr_t[b], ytr_t[b]
            if jitter_alpha > 0:
                # blend embedding toward same-cell same-class neighbor
                lam = jitter_alpha * torch.rand(xb.shape[0], 1, device=DEVICE, generator=g)
                xn = Xtr_t[nbr[b]]
                xb = xb.clone()
                xb[:, emb_slice] = (1 - lam) * xb[:, emb_slice] + lam * xn[:, emb_slice]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t).argmax(1)
        f1 = C.gpu_macro_f1(yval_t, vp, n_classes)
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def fit_predict_spatial(Xtr, ytr, gtr, Xte, n_classes, w, hidden, lr,
                        jitter_alpha, spatial_val, seeds):
    """3-seed probability ensemble with spatial jitter + val. Returns pred enc,
    mean inner-val F1."""
    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std == 0] = 1.0
    Xtr_s = ((Xtr - mean) / std).astype(np.float32)
    Xte_s = ((Xte - mean) / std).astype(np.float32)
    Xte_t = torch.as_tensor(Xte_s, device=DEVICE)
    P = torch.zeros((len(Xte_s), n_classes), device=DEVICE)
    vfs = []
    for s in seeds:
        m, vf = train_one_spatial(Xtr_s, ytr, gtr, Xtr_s.shape[1], n_classes, w,
                                  s, jitter_alpha, spatial_val, hidden, lr)
        P += C._proba(m, Xte_t, n_classes, amp=False)
        vfs.append(vf)
    return (P / len(seeds)).argmax(1).cpu().numpy(), float(np.mean(vfs))


def _write(results, t0, done, total):
    out = Path(__file__).resolve().parent / "spatial_reg.json"
    out.write_text(json.dumps({
        "baseline": {"dnn_best_to12fix": 0.7341, "scaling_256_128": 0.7304,
                     "reg_best_512_drop05": 0.7320},
        "grid_axes": {"hidden": list(HIDDEN), "jitters": JITTERS,
                      "spatial_val": SPATIAL_VAL, "seeds": SEEDS, "folds": FOLDS},
        "progress": {"done": done, "total": total},
        "results": results,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=2))
    return out


def main():
    t0 = time.perf_counter()
    data = C.load_cached("lidar")
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    lon, lat = data["lon"], data["lat"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12_enc = classes.index(12) if 12 in classes else -1

    folds = list(du.fold_indices(y_enc, groups, FOLDS))
    grid = [(j, sv) for j in JITTERS for sv in SPATIAL_VAL]
    total = len(grid)
    print(f"spatial-reg: hidden={HIDDEN} {total} cells "
          f"(jitter×spatial_val) × {FOLDS} folds × {SEEDS} seeds\n", flush=True)

    results = []
    for jitter_alpha, spatial_val in grid:
        f1s, vfs, pcs = [], [], []
        for k, tr, te in folds:
            if cls12_enc >= 0:
                keep = du.clean_stale_class_mask(
                    X[tr], y_enc[tr], None, cls12_enc, lon[tr], lat[tr])
                tr = tr[keep]
            Xtr, ytr, gtr = X[tr], y_enc[tr], groups[tr]
            w = C.class_weights(ytr, n_classes, "sqrt")
            seeds = [s * 100 for s in range(SEEDS)]
            pred, vf = fit_predict_spatial(
                Xtr, ytr, gtr, X[te], n_classes, w, HIDDEN, LR,
                jitter_alpha, spatial_val, seeds)
            f1s.append(du.macro_f1(y_enc[te], pred, n_classes))
            vfs.append(vf)
            pcs.append(du.per_class_f1(y_enc[te], pred, n_classes))
        tf1, vf1 = float(np.mean(f1s)), float(np.mean(vfs))
        rec = {
            "jitter": jitter_alpha, "spatial_val": spatial_val,
            "test_f1": round(tf1, 4), "test_f1_std": round(float(np.std(f1s)), 4),
            "val_f1": round(vf1, 4), "gap": round(vf1 - tf1, 4),
            "vs_baseline": round(tf1 - BASELINE, 4),
            "helped": bool(tf1 > BASELINE + 0.001),
            "per_class": {str(c): round(float(v), 4)
                          for c, v in zip(classes, np.mean(pcs, axis=0))},
        }
        results.append(rec)
        flag = "HELP" if rec["helped"] else ("hurt" if rec["vs_baseline"] < -0.001 else "flat")
        print(f"jitter={jitter_alpha:<4} spatial_val={int(spatial_val)}  "
              f"test={tf1:.4f}±{rec['test_f1_std']:.3f}  val={vf1:.4f}  "
              f"gap={rec['gap']:+.3f}  Δbase={rec['vs_baseline']:+.4f} [{flag}]",
              flush=True)
        _write(results, t0, len(results), total)

    _write(results, t0, len(results), total)
    best = max(results, key=lambda r: r["test_f1"])
    print(f"\nbest: {best['test_f1']:.4f}  jitter={best['jitter']} "
          f"spatial_val={int(best['spatial_val'])}  gap={best['gap']:+.3f}")
    print(f"vs baseline 0.7304: {best['test_f1']-0.7304:+.4f}   "
          f"vs reg-best 0.7320: {best['test_f1']-0.7320:+.4f}   "
          f"vs DNN best 0.7341: {best['test_f1']-0.7341:+.4f}")


if __name__ == "__main__":
    main()

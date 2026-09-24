"""ar3 harness: the DEPLOYED moe8 as baseline, scored on a multi-metric card.

Differences from `autoresearch/ar_common.py` (whose training step this reuses
line for line):

  * baseline = moe_shared, 8 routed experts top-2, 9 merged classes (11->2),
    i.e. `models/dnn_final_moe8_merged.pt`'s recipe.
  * a CALIBRATION split of whole cells (10% of each training fold's cells) is
    held out of training in EVERY arm, including the baseline. Temperature and
    conformal thresholds are fitted there. That is the only leak-free way to
    measure the transfer-calibration the deployed Venn-Abers/conformal step
    relies on. It costs every arm the same 10% of cells, so paired deltas are
    unaffected, but absolute F1 sits a little below the 0.7542 reference.
  * metrics: accuracy + calibration + conformal + temporal flips (metrics.py),
    plus a consensus HARD-AREA population when `data/consensus/` exists.
  * hooks beyond ar_common's: `feat_fn` (temporal input features),
    `prep_fn` (per-fold precompute, e.g. year partners for a consistency loss),
    `extra` (add consensus rows to training), and a `post_fn` that sees the test
    rows' location/year metadata (joint multi-year decoding).

Noise floors come from two baseline runs on independent seed sets
(`baseline` at AR_SEED=0, `baseline_s1` at AR_SEED=1).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MERGE_EXTRA", "11:2")
os.environ.setdefault("AR_RESULTS_DIR", str(HERE / "results"))
os.environ.setdefault("N_ENSEMBLE", "3")
sys.path.insert(0, str(HERE.parent / "autoresearch"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
import torch                           # noqa: E402
import torch.nn as nn                  # noqa: E402
import torch.nn.functional as F        # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import ar_common as ac                 # noqa: E402
import data_utils as du                # noqa: E402
import metrics as MX                   # noqa: E402
from trials_local import _moe_builder, post_route_stats  # noqa: E402,F401

DEVICE = ac.DEVICE
SEED = ac.SEED
RESULTS_DIR = ac.RESULTS_DIR
LOGS_DIR = HERE / "logs"
LOGS_DIR.mkdir(exist_ok=True)
CAL_FRAC = 0.10
CONSENSUS_DIR = HERE.parents[1] / "data" / "consensus"
HARD_YEARS = (2018, 2024)      # the two mapped years; hard-area eval rows
HARD_RULE = os.environ.get("HARD_RULE", "gk_plus2")
BASELINE_NAME = "baseline"


@dataclass
class Trial3(ac.Trial):
    feat_fn: Optional[Callable] = None     # (X, meta, feat_cols) -> X'   meta has loc, year
    prep_fn: Optional[Callable] = None     # (ctx) -> None   per-fold precompute
    extra: Optional[dict] = None           # consensus rows to add to TRAIN cells
    post3_fn: Optional[Callable] = None    # (P_te, ctx) -> (P_te', info)
    moe: bool = True                       # False -> plain MLP trunk (control only)


MOE8 = dict(n_experts=8, top_k=2, gate_src="content")


def moe8_build(in_dim, n_classes, ctx):
    return _moe_builder(gate_src="content", balance="none")(in_dim, n_classes, ctx)


def with_moe8_config(cfg):
    """The MoE builder reads n_experts/top_k from trial.config; every ar3 arm
    inherits the deployed 8/2 unless it says otherwise."""
    out = dict(MOE8)
    out.update(cfg or {})
    return out


# ------------------------------------------------------------ data loading
def loc_key(lon, lat):
    """Location id stable across years (float coordinates are exact copies
    across the year rows of one FSCS point)."""
    return pd.util.hash_pandas_object(
        pd.DataFrame({"lon": lon, "lat": lat}), index=False).values


def load_consensus(classes, feat_cols, lidar_med):
    """Hard-area points in the same feature layout as the FSCS frame.

    Returns None if W1 has not delivered yet. Labels are NOT applied here. The
    per-(point, year) consensus under several rules is returned so trials can
    choose, and the hard-area EVAL always uses HARD_RULE.
    """
    pts_p, aef_p = CONSENSUS_DIR / "points.parquet", CONSENSUS_DIR / "aef_long.parquet"
    if not (pts_p.exists() and aef_p.exists()):
        return None
    sys.path.insert(0, str(HERE / "w1_consensus"))
    import crosswalk as CW
    pts = pd.read_parquet(pts_p)
    aef = pd.read_parquet(aef_p)
    df = aef.merge(pts, on="pid", how="inner")
    for c in du.LIDAR_COLS:
        if c in feat_cols:
            df[c] = df[c].fillna(lidar_med[c]).astype(np.float32)
    df = df.dropna(subset=du.EMBED_COLS).reset_index(drop=True)
    remap = {c: i for i, c in enumerate(classes)}
    labels = {}
    for rule in ("gk_plus2", "gk_raw", "ext_majority"):
        y = np.full(len(df), -1, dtype=np.int64)
        for yr in np.unique(df["year"]):
            m = (df["year"] == yr).values
            lab = CW.consensus(df.loc[m], int(yr), rule=rule)[0]
            lab = np.asarray(lab)
            y[m] = [remap.get(int(v), -1) if v is not None and v >= 0 else -1 for v in lab]
        labels[rule] = y
    # Real-change reference: Dynamic World AND Esri each see the SAME
    # 2018->2024 transition (ext_change) or both see none (ext_stable). Only
    # ~275 points qualify for change (mostly snow->bare), so this is a
    # sensitivity check, not a precise rate.
    def _m(pr, yr):
        return np.array([int(v) if isinstance(v, (int, np.integer)) else -1
                         for v in CW.to_merged(pr, pts[f"{pr}_{yr}"].values)])
    d18, d24, e18, e24 = _m("dw", 2018), _m("dw", 2024), _m("esri", 2018), _m("esri", 2024)
    ok = ((d18 >= 0) & (d24 >= 0) & (e18 >= 0) & (e24 >= 0)
          & (pts["dw_2018_frac"].values >= 0.5) & (pts["dw_2024_frac"].values >= 0.5)
          & (d18 == e18) & (d24 == e24))
    ext = pd.DataFrame({"pid": pts["pid"].values,
                        "ext_change": ok & (d18 != d24), "ext_stable": ok & (d18 == d24)})
    df = df.merge(ext, on="pid", how="left")
    X = df[feat_cols].values.astype(np.float32)
    meta = df[["pid", "lon", "lat", "year", "cell_id", "stratum",
               "ext_change", "ext_stable"]].copy()
    meta["loc"] = pd.util.hash_pandas_object(df[["pid"]], index=False).values
    return {"X": X, "labels": labels, "meta": meta}


# ---------------------------------------------------------------- training
def base_ce(out, yb, crit, ctx):
    """The baseline CE for a batch, honouring per-row weights when the arm has
    them. Custom `loss_fn`s must build on this rather than calling crit
    directly, because crit is reduction='none' when row weights exist."""
    loss = crit(out, yb)
    row_w = ctx.get("wtr_t")
    if row_w is None:
        return loss
    b = ctx["batch_idx"]
    wb = row_w[b] * (ctx["w"][yb] if ctx["w"] is not None else 1.0)
    return (loss * row_w[b]).sum() / wb.sum()


def train_member(trial, ctx, seed):
    """ar_common.train_member plus: the batch's row indices in ctx, optional
    per-row weights (consensus rows), identical otherwise."""
    ac.set_seed(seed)
    build = trial.build_fn or (moe8_build if trial.moe else ac.default_mlp)
    model = build(ctx["in_dim"], ctx["n_classes"], ctx).to(DEVICE)
    opt = (trial.opt_fn or ac.default_opt)(model, ctx)
    row_w = ctx.get("wtr_t")
    crit = nn.CrossEntropyLoss(weight=ctx["w"], label_smoothing=trial.label_smooth,
                               reduction="none" if row_w is not None else "mean")
    Xtr_t, ytr_t = ctx["Xtr_t"], ctx["ytr_t"]
    n_tr = Xtr_t.shape[0]
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(ac.MAX_EPOCHS):
        ctx["epoch"] = epoch
        model.train()
        order = torch.randperm(n_tr, device=DEVICE, generator=g)
        for i in range(0, n_tr, ac.BATCH):
            b = order[i:i + ac.BATCH]
            ctx["batch_idx"] = b
            xb, yb = Xtr_t[b], ytr_t[b]
            opt.zero_grad()
            if trial.loss_fn:
                loss = trial.loss_fn(model, xb, yb, crit, ctx)
            else:
                loss = base_ce(ac.logits_of(model, xb), yb, crit, ctx)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = ac.logits_of(model, ctx["Xval_t"]).argmax(1).cpu().numpy()
        f1 = du.macro_f1(ctx["yval_np"], vp, ctx["n_classes"])
        if f1 > best_f1 + 1e-4:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ac.PATIENCE:
                break
    model.load_state_dict(best_state)
    if trial.after_fit:
        model = trial.after_fit(model, ctx)
    return model, best_f1


def run_fold(trial, ctx):
    """N-member ensemble -> averaged probabilities on every eval set."""
    nc = ctx["n_classes"]
    sets = ctx["eval_sets"]
    P = {k: np.zeros((v.shape[0], nc), dtype=np.float64) for k, v in sets.items()}
    vf1s = []
    for e in range(trial.n_ensemble):
        model, vf1 = train_member(trial, ctx, SEED + 100 * e)
        for k, v in sets.items():
            P[k] += ac.predict_probs(model, v, nc)
        vf1s.append(vf1)
        del model
        torch.cuda.empty_cache()
    return {k: v / trial.n_ensemble for k, v in P.items()}, float(np.mean(vf1s))


# -------------------------------------------------------------- the trial
def cal_split(groups_tr, rng):
    """Whole cells -> calibration (never trained on)."""
    cells = np.unique(groups_tr)
    n_cal = max(1, int(round(len(cells) * CAL_FRAC)))
    chosen = rng.choice(cells, size=n_cal, replace=False)
    is_cal = np.isin(groups_tr, chosen)
    return np.flatnonzero(is_cal), np.flatnonzero(~is_cal)


def run_trial(trial, wandb_run=None):
    t0 = time.perf_counter()
    trial = dataclasses.replace(trial, config=with_moe8_config(trial.config)
                                if trial.moe else dict(trial.config))
    print(f"=== {trial.name} [{trial.tier}] ===\nidea       : {trial.idea}\n"
          f"hypothesis : {trial.hypothesis}\nseed={SEED} ens={trial.n_ensemble} "
          f"merge={du.merge_sig()} relabel={ac.RELABEL} moe={trial.moe}")

    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes, feat_cols = data["classes"], data["feat_cols"]
    nc = len(classes)
    lon, lat = df["lon"].values, df["lat"].values
    if ac.RELABEL != "none":
        y_enc, n_ch = du.apply_cls12_relabel(y_enc, classes, ac.RELABEL)
        print(f"relabel={ac.RELABEL}: {n_ch} labels changed")
    meta = df[["lon", "lat", "year", "cell_id"]].copy()
    meta["loc"] = loc_key(lon, lat)
    cons = load_consensus(classes, feat_cols, data["lidar_med"])
    X0 = X  # the stale-class cleaning always sees the 67 deployed features
    if trial.feat_fn:
        X = trial.feat_fn(X, meta, feat_cols)
        if cons is not None:
            cons["X"] = trial.feat_fn(cons["X"], cons["meta"], feat_cols)
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats  {nc} classes; "
          f"consensus={'none' if cons is None else f'{len(cons['X']):,} rows'}")
    if trial.extra and cons is None:
        raise SystemExit("trial needs consensus data but data/consensus/ is missing")

    cls12 = classes.index(12) if 12 in classes else -1
    folds, pcs, vf1s, infos = [], [], [], []
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        # Per-fold streams. The split stream is consumed by NOTHING but the two
        # splits, so every arm gets the identical calibration/val cells on
        # every fold. Anything an arm draws (subsampling, partner selection)
        # comes from `rng`, and cannot shift the splits of later folds.
        split_rng = np.random.default_rng([SEED, k, 0])
        rng = np.random.default_rng([SEED, k, 1])
        # calibration cells first, so the ONLY thing an arm can change is what
        # happens inside the remaining training cells
        cal_rel, fit_rel = cal_split(groups[tr], split_rng)
        cal, fit = tr[cal_rel], tr[fit_rel]
        if ac.CLEAN_CLS12 and cls12 >= 0:
            keep = du.clean_stale_class_mask(X0[fit], y_enc[fit], df.iloc[fit],
                                             cls12, lon[fit], lat[fit])
            fit = fit[keep]
        scaler = StandardScaler().fit(X[fit])
        Xfit = scaler.transform(X[fit]).astype(np.float32)
        v_idx, t_idx = ac.val_split(len(Xfit), groups[fit], split_rng, trial.val_mode)
        tr_rows = fit[t_idx]
        Xtr, ytr = Xfit[t_idx], y_enc[fit][t_idx]
        wtr = None
        tr_meta = meta.iloc[tr_rows].reset_index(drop=True)
        train_cells = np.unique(groups[fit])
        test_cells = np.unique(groups[te])

        eval_sets = {"te": scaler.transform(X[te]).astype(np.float32),
                     "cal": scaler.transform(X[cal]).astype(np.float32)}
        hard_info = None
        n_extra = 0
        if cons is not None:
            cm = cons["meta"]
            in_te = cm["cell_id"].isin(test_cells).values
            hy = cons["labels"][HARD_RULE]
            eval_rows = np.flatnonzero(in_te & cm["year"].isin(HARD_YEARS).values)
            eval_sets["hard"] = scaler.transform(cons["X"][eval_rows]).astype(np.float32)
            hard_info = {"rows": eval_rows, "y": hy[eval_rows],
                         "meta": cm.iloc[eval_rows].reset_index(drop=True)}
            if trial.extra:
                ex = trial.extra
                ey = cons["labels"][ex.get("rule", "gk_plus2")]
                em = (cm["cell_id"].isin(train_cells).values & (ey >= 0)
                      & cm["stratum"].isin(ex.get("strata", ["flip", "uncertain"])).values)
                if ex.get("years"):
                    em &= cm["year"].isin(ex["years"]).values
                erows = np.flatnonzero(em)
                if ex.get("match_n_of"):
                    # control arms add the SAME number of rows as the arm they
                    # control, so "more rows" cannot explain the difference
                    ref = json.loads((RESULTS_DIR / f"{ex['match_n_of']}.json").read_text())
                    n_target = int(ref["fold_info"][k]["n_extra"])
                    erows = rng.choice(erows, size=min(n_target, len(erows)), replace=False)
                n_extra = len(erows)
                Xe = scaler.transform(cons["X"][erows]).astype(np.float32)
                Xtr = np.vstack([Xtr, Xe])
                ytr = np.concatenate([ytr, ey[erows]])
                ew = float(trial.config.get("weight", ex.get("weight", 1.0)))
                wtr = np.concatenate([np.ones(len(t_idx)), np.full(len(erows), ew)])
                tr_meta = pd.concat([tr_meta, cm.iloc[erows][["lon", "lat", "year", "cell_id", "loc"]]],
                                    ignore_index=True)
                print(f"  fold {k}: + {len(erows):,} consensus rows "
                      f"({ex.get('rule', 'gk_plus2')}, {ex.get('strata')})")

        ctx = {
            "fold": k, "n_classes": nc, "classes": classes, "in_dim": Xtr.shape[1],
            "trial": trial, "rng": rng, "scaler": scaler, "feat_cols": feat_cols,
            "Xtr_t": torch.tensor(Xtr, device=DEVICE),
            "ytr_t": torch.tensor(ytr, device=DEVICE),
            "ytr_np": ytr, "tr_meta": tr_meta,
            "Xval_t": torch.tensor(Xfit[v_idx], device=DEVICE),
            "yval_np": y_enc[fit][v_idx],
            "groups_tr": groups[tr_rows],
            "lidar_cols": [feat_cols.index(c) for c in du.LIDAR_COLS if c in feat_cols],
            "n_gate_extra": 0,
            "te_meta": meta.iloc[te].reset_index(drop=True), "y_te": y_enc[te],
        }
        if wtr is not None and not np.all(wtr == 1.0):
            ctx["wtr_t"] = torch.tensor(wtr, dtype=torch.float32, device=DEVICE)
        ctx["counts"] = np.bincount(ytr, minlength=nc).astype(np.float64)
        ctx["w"] = ac.class_weights(ytr, nc, trial.weight_mode)
        ctx["eval_sets"] = {kk: torch.tensor(v, device=DEVICE) for kk, v in eval_sets.items()}
        if trial.prep_fn:
            trial.prep_fn(ctx)

        if trial.fold_fn:
            P, vf1 = trial.fold_fn(ctx)
        else:
            P, vf1 = run_fold(trial, ctx)
        info = {"n_extra": n_extra}
        P_raw_te = P["te"]
        if trial.post3_fn:
            ctx["P_all"] = P
            ctx["y_cal"] = y_enc[cal]
            # the calibration cells get the SAME post-processing, else the
            # temperature / conformal thresholds are fitted on raw probabilities
            # and applied to smoothed ones
            P["cal"], _ = trial.post3_fn(P["cal"], {**ctx, "te_meta": meta.iloc[cal].reset_index(drop=True)})
            P["te"], info_post = trial.post3_fn(P["te"], ctx)
            info.update(info_post)
            info["f1_raw_argmax"] = MX.macro_f1(y_enc[te], P_raw_te.argmax(1), nc)

        hard = None
        if hard_info is not None:
            lab = hard_info["y"] >= 0
            hard = {"P": P["hard"][lab], "y": hard_info["y"][lab],
                    "pred_all": P["hard"].argmax(1), "meta_all": hard_info["meta"]}
        fm, pc = MX.fold_metrics(P["te"], y_enc[te], ctx["te_meta"], P["cal"], y_enc[cal],
                                 nc, hard=hard)
        fm["val_f1"] = vf1
        folds.append(fm)
        pcs.append(pc)
        infos.append(info)
        print(f"  fold {k}: " + "  ".join(f"{m}={fm[m]:.4f}" for m in
              ("f1", "nll", "ece", "conf_size", "flip_1720", "flip_1824") if m in fm)
              + (f"  hard_f1={fm['hard_f1']:.4f}" if "hard_f1" in fm else "")
              + f"  ({time.perf_counter() - ft:.0f}s)")
        if wandb_run is not None:
            wandb_run.log({"fold": k, **{f"fold_{m}": v for m, v in fm.items()
                                          if isinstance(v, float)}})
        for key in ("Xtr_t", "ytr_t", "Xval_t", "eval_sets", "wtr_t", "P_all"):
            ctx.pop(key, None)
        torch.cuda.empty_cache()

    rec = {
        "name": trial.name, "tier": trial.tier, "idea": trial.idea,
        "hypothesis": trial.hypothesis, "provenance": trial.provenance,
        "seed": SEED, "n_ensemble": trial.n_ensemble, "is_bound": trial.is_bound, "merge": du.merge_sig(),
        "config": trial.config, "extra": trial.extra,
        "folds": folds, "fold_info": infos,
        "mean": {m: float(np.nanmean([f[m] for f in folds])) for m in folds[0]
                 if isinstance(folds[0][m], (float, int))},
        "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, np.mean(pcs, 0))},
        "f1_per_class_per_fold": [[round(float(v), 4) for v in pc] for pc in pcs],
        "runtime_s": round(time.perf_counter() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if trial.name != BASELINE_NAME and (RESULTS_DIR / f"{BASELINE_NAME}.json").exists():
        rec.update(scorecard(rec, classes))
    out = RESULTS_DIR / f"{trial.name}.json"
    out.write_text(json.dumps(rec, indent=2, default=float))
    print(json.dumps({k: rec.get(k) for k in ("mean", "verdict", "improves", "regresses")},
                     indent=1, default=float))
    print(f"saved -> {out}  ({rec['runtime_s']}s)")
    return rec


# -------------------------------------------------------------- scorecard
def noise_floors():
    """Per-metric floor = 2 × RMS of the per-fold paired delta between two
    baseline runs on independent seed sets. Macro-F1 additionally keeps the
    0.003 bar from the previous round. None until `baseline_s1` exists."""
    b0, b1 = RESULTS_DIR / "baseline.json", RESULTS_DIR / "baseline_s1.json"
    if not (b0.exists() and b1.exists()):
        return None
    r0, r1 = json.loads(b0.read_text()), json.loads(b1.read_text())
    fl = {}
    for m in MX.HIGHER_BETTER:
        d = [f1[m] - f0[m] for f0, f1 in zip(r0["folds"], r1["folds"])
             if m in f0 and m in f1 and np.isfinite(f0[m]) and np.isfinite(f1[m])]
        if d:
            fl[m] = 2.0 * float(np.sqrt(np.mean(np.square(d))))
    fl["f1"] = max(fl.get("f1", 0.0), 0.003)
    return fl


def scorecard(rec, classes):
    base = json.loads((RESULTS_DIR / f"{BASELINE_NAME}.json").read_text())
    floors = noise_floors()
    card, improves, regresses = {}, [], []
    for m, hb in MX.HIGHER_BETTER.items():
        if m not in rec["folds"][0] or m not in base["folds"][0]:
            continue
        d = [f[m] - b[m] for f, b in zip(rec["folds"], base["folds"])]
        if not all(np.isfinite(d)):
            continue
        dm = float(np.mean(d))
        good = [(x > 0) == hb for x in d]
        fl = floors.get(m) if floors else None
        v = "n/a-no-floor"
        if fl is not None:
            if all(good) and abs(dm) > fl:
                v = "better"
                improves.append(m)
            elif not any(good) and abs(dm) > fl:
                v = "worse"
                regresses.append(m)
            else:
                v = "tie"
        card[m] = {"delta_mean": round(dm, 5), "delta_per_fold": [round(x, 5) for x in d],
                   "floor": None if fl is None else round(fl, 5), "verdict": v}
    pcd = {str(c): round(rec["f1_per_class"][str(c)] - base["f1_per_class"][str(c)], 4)
           for c in classes}
    verdict = ("upper-bound" if rec.get("is_bound") else
               "ADOPTABLE" if improves and not regresses else
               "tradeoff" if improves and regresses else
               "regresses" if regresses else "tie")
    return {"card": card, "improves": improves, "regresses": regresses,
            "verdict": verdict, "delta_per_class": pcd}

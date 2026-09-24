"""Compare area estimators (classical, PPI, PPI++, stratified, cross-PPI) on a
population where the true class areas are KNOWN.

Population: the uniform random pixel sample built by area_sample.py (~100k valid
2024-map pixels), each with the deployed map class, its 9 calibrated class
probabilities (pcal), AEF+lidar features, and the grunnkart reference class.
Truth = the population's reference class shares. Every replicate draws a small
"labelled" sample from it (what a field/photo-interpretation campaign would
collect), runs every estimator, and scores bias, RMSE, CI coverage and CI width
against the truth. Replicates share their draws across estimators (paired).

Estimators (per reference class k; estimand = share of pixels with Y = k):
  map_count     pixel counting of the hard map (no CI; shows the map's bias)
  classical     labelled-sample proportion, CLT interval
  ppi_hard      PPI, lambda=1, predictor 1[map == k]          (difference estimator)
  ppi_soft      PPI, lambda=1, predictor pcal_k
  ppipp_soft    PPI++ (power-tuned lambda), predictor pcal_k
  olofsson      stratified estimator, strata = map class (Olofsson et al. 2014);
                post-stratified under SRS
  stratppi      stratified PPI++ (Fisch et al. 2024), strata = map class, pcal_k
                predictor, lambda tuned per stratum
  xppi_aef      cross-PPI: LightGBM on AEF+lidar trained ONLY on the labelled
                sample, K-fold (Zrnic & Candes 2024) — no pretrained model used
  xppi_pcal     cross-PPI on the 9 pcal probabilities: learns a recalibration of
                the deployed model from the labels

Designs: SRS (all estimators) and a map-stratified random sample with
sqrt(W_h) allocation, >= 30 per stratum (olofsson / stratppi, which are the
estimators that design is built for).

Then one "headline" run on the whole AOI: a single labelled draw, with the
pretrained estimators using the exact wall-to-wall map totals as the unlabelled
side (area_map_totals_2024.json).

Run:
    ~/myprojects/recover/.venv/bin/python DNN/area_ppi.py            # full
    ~/myprojects/recover/.venv/bin/python DNN/area_ppi.py --quick    # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ppi_py import classical_mean_ci, crossppi_mean_ci, ppi_mean_ci
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dnn_paths import REPO_DIR, RESULTS_DIR, result_path  # noqa: E402

warnings.filterwarnings("ignore")

MAP_CLASSES = [2, 3, 4, 5, 6, 7, 8, 10, 12]
REF_MERGE = {1: 2, 9: 8, 11: 2}               # deployed 9-class ontology
NAMES = {2: "bare+sparse", 3: "crop", 4: "forest", 5: "grassland", 6: "scrub",
         7: "wetland", 8: "water", 10: "built", 12: "snow/ice", 13: "other"}
FEATS_AEF = [f"A{i:02d}" for i in range(64)] + ["elevation", "tri", "tch"]
FEATS_PCAL = [f"pcal_{c}" for c in MAP_CLASSES]
ALPHA = 0.05
Z = norm.ppf(1 - ALPHA / 2)


# --------------------------------------------------------------------------- #
def load_population(year):
    df = pd.read_parquet(REPO_DIR / "data" / f"area_sample_{year}.parquet")
    gk = df["grunnkart"].replace(REF_MERGE).astype(int)
    n0 = int((gk == 0).sum())
    df = df[gk > 0].reset_index(drop=True)          # no grunnkart -> no reference
    df["ref"] = gk[gk > 0].values
    return df, n0


def onehot(v, classes):
    return (np.asarray(v)[:, None] == np.asarray(classes)[None, :]).astype(float)


def ci(est, se):
    return est - Z * se, est + Z * se


# --------------------------------------------------------------------------- #
# Stratified estimators (own implementation; ppi_py 0.2.3 has no stratified PPI)
# --------------------------------------------------------------------------- #
def olofsson(YL, hL, W):
    """Stratified estimator of class shares. YL (n,C) one-hot reference, hL (n,)
    stratum index, W (H,) stratum weights. Strata with < 2 labels are imputed as
    'map is right' with zero variance (a real survey would collapse them)."""
    C = YL.shape[1]
    est, var = np.zeros(C), np.zeros(C)
    for h, w in enumerate(W):
        m = hL == h
        nh = m.sum()
        if nh < 2:
            est[h] += w                              # map class h == ref column h
            continue
        p = YL[m].mean(0)
        est += w * p
        var += w ** 2 * p * (1 - p) / (nh - 1)
    return est, np.sqrt(var)


def stratppi(YL, FL, hL, FU, hU, W):
    """Stratified PPI++ (Fisch et al. 2024): per-stratum power-tuned PPI mean,
    combined with known stratum weights W."""
    C = YL.shape[1]
    est, var = np.zeros(C), np.zeros(C)
    for h, w in enumerate(W):
        mL, mU = hL == h, hU == h
        nh, Nh = mL.sum(), mU.sum()
        if nh < 2:
            est[h] += w
            continue
        Y, F, Fu = YL[mL], FL[mL], FU[mU]
        vF = np.concatenate([F, Fu]).var(0)
        cov = ((Y - Y.mean(0)) * (F - F.mean(0))).sum(0) / (nh - 1)
        lam = np.clip(np.where(vF > 0, cov / ((1 + nh / Nh) * vF + 1e-300), 0), 0, 1)
        th = lam * Fu.mean(0) + (Y - lam * F).mean(0)
        v = lam ** 2 * Fu.var(0) / Nh + (Y - lam * F).var(0, ddof=1) / nh
        est += w * th
        var += w ** 2 * v
    return est, np.sqrt(var)


# --------------------------------------------------------------------------- #
# Cross-PPI
# --------------------------------------------------------------------------- #
def _lgbm(nfeat):
    from lightgbm import LGBMClassifier
    small = nfeat <= 12
    return LGBMClassifier(n_estimators=150 if small else 250, learning_rate=0.05,
                          num_leaves=7 if small else 15, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=1.0, n_jobs=8, verbose=-1)


def cross_predict(XL, yL, XU, classes, K=5, seed=0):
    """K-fold: OOF probas on the labelled set (n,C) and each fold-model's probas
    on the unlabelled set (C, N, K). Classes absent from a training fold get 0."""
    n, C = len(yL), len(classes)
    col = {c: j for j, c in enumerate(classes)}
    fold = np.random.default_rng(seed).permutation(n) % K
    P_L = np.zeros((n, C))
    P_U = np.zeros((C, len(XU), K))
    for k in range(K):
        tr, te = fold != k, fold == k
        ytr = yL[tr]
        present = np.unique(ytr)
        if len(present) == 1:                          # degenerate fold
            j = col[present[0]]
            P_L[te, j] = 1
            P_U[j, :, k] = 1
            continue
        m = _lgbm(XL.shape[1]).fit(XL[tr], ytr)
        cols = [col[c] for c in m.classes_]
        P_L[np.ix_(np.flatnonzero(te), cols)] = m.predict_proba(XL[te])
        P_U[cols, :, k] = m.predict_proba(XU).T
    return P_L, P_U


def xppi(YL, P_L, P_U):
    lo, hi, est = [], [], []
    for j in range(YL.shape[1]):
        l, h = crossppi_mean_ci(YL[:, j], P_L[:, j], P_U[j], alpha=ALPHA)
        lo.append(float(np.squeeze(l))); hi.append(float(np.squeeze(h)))
        est.append(float(P_U[j].mean() + (YL[:, j] - P_L[:, j]).mean()))
    return np.array(est), np.array(lo), np.array(hi)


# --------------------------------------------------------------------------- #
def ppi_lib(YL, FL, FU, lam):
    """ppi_py per class (1-D calls so lambda is tuned per class)."""
    lo, hi, est = [], [], []
    for j in range(YL.shape[1]):
        l, h = ppi_mean_ci(YL[:, j], FL[:, j], FU[:, j], alpha=ALPHA, lam=lam)
        lo.append(float(np.squeeze(l))); hi.append(float(np.squeeze(h)))
    lo, hi = np.array(lo), np.array(hi)
    return (lo + hi) / 2, lo, hi


def pred_matrix(df, classes, kind):
    """(N, C) predictor aligned to the reference class columns."""
    out = np.zeros((len(df), len(classes)))
    for j, c in enumerate(classes):
        if c in MAP_CLASSES:
            out[:, j] = (df["map_class"].values == c) if kind == "hard" else df[f"pcal_{c}"].values
    return out


def draw(rng, pop_h, n, design, W):
    N = len(pop_h)
    if design == "srs":
        return rng.choice(N, n, replace=False)
    # floor of min(30, n/2H) per stratum, the rest by sqrt(W): total stays == n
    H = len(W)
    floor = min(30, n // (2 * H))
    alloc = floor + np.floor(np.sqrt(W) / np.sqrt(W).sum() * (n - H * floor)).astype(int)
    alloc[np.argmax(W)] += n - alloc.sum()
    idx = []
    for h, a in enumerate(alloc):
        members = np.flatnonzero(pop_h == h)
        idx.append(rng.choice(members, min(a, len(members)), replace=False))
    return np.concatenate(idx)


def simulate(df, classes, ns, reps, reps_cross, seed, do_cross=True):
    N = len(df)
    Y = onehot(df["ref"].values, classes)
    truth = Y.mean(0)
    F_hard = pred_matrix(df, classes, "hard")
    F_soft = pred_matrix(df, classes, "soft")
    h = np.array([MAP_CLASSES.index(c) for c in df["map_class"].values])
    W = np.bincount(h, minlength=len(MAP_CLASSES)) / N
    X_aef = df[FEATS_AEF].values.astype(np.float32)
    X_pcal = df[FEATS_PCAL].values.astype(np.float32)
    y = df["ref"].values
    rows = []
    timing = {}

    def rec(design, n, r, method, est, lo=None, hi=None):
        for j, c in enumerate(classes):
            rows.append({"design": design, "n": n, "rep": r, "method": method,
                         "cls": c, "est": est[j],
                         "lo": np.nan if lo is None else lo[j],
                         "hi": np.nan if hi is None else hi[j]})

    for design in ["srs", "strat"]:
        for n in ns:
            rng = np.random.default_rng([seed, n, design == "strat"])
            t0 = time.time()
            for r in range(reps):
                L = draw(rng, h, n, design, W)
                mask = np.ones(N, bool); mask[L] = False
                U = np.flatnonzero(mask)
                YL = Y[L]
                est, se = olofsson(YL, h[L], W)
                rec(design, n, r, "olofsson", est, *ci(est, se))
                est, se = stratppi(YL, F_soft[L], h[L], F_soft[U], h[U], W)
                rec(design, n, r, "stratppi", est, *ci(est, se))
                if design != "srs":
                    continue
                rec(design, n, r, "map_count", F_hard.mean(0))
                lo, hi = np.array([[float(np.squeeze(v)) for v in classical_mean_ci(YL[:, j], alpha=ALPHA)]
                                   for j in range(YL.shape[1])]).T
                rec(design, n, r, "classical", YL.mean(0), lo, hi)
                for name, F, lam in [("ppi_hard", F_hard, 1), ("ppi_soft", F_soft, 1),
                                     ("ppipp_soft", F_soft, None)]:
                    rec(design, n, r, name, *ppi_lib(YL, F[L], F[U], lam))
                if do_cross and r < reps_cross:
                    Usub = rng.choice(U, min(20000, len(U)), replace=False)
                    for name, X in [("xppi_aef", X_aef), ("xppi_pcal", X_pcal)]:
                        tc = time.time()
                        P_L, P_U = cross_predict(X[L], y[L], X[Usub], classes, seed=r)
                        rec(design, n, r, name, *xppi(YL, P_L, P_U))
                        timing.setdefault(f"{name}_n{n}", []).append(time.time() - tc)
            print(f"  {design} n={n}: {reps} reps in {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res["truth"] = res["cls"].map(dict(zip(classes, truth)))
    return res, truth, W, {k: float(np.mean(v)) for k, v in timing.items()}


def summarise(res):
    res = res.assign(err=res.est - res.truth, width=res.hi - res.lo,
                     cover=(res.lo <= res.truth) & (res.truth <= res.hi))
    g = res.groupby(["design", "n", "method", "cls"])
    s = pd.DataFrame({
        "truth": g.truth.first(),
        "bias": g.err.mean(),
        "rmse": np.sqrt(g.err.apply(lambda e: (e ** 2).mean())),
        "coverage": g.cover.mean(),
        "width": g.width.mean(),
        "reps": g.size(),
    }).reset_index()
    s.loc[s.method == "map_count", ["coverage", "width"]] = np.nan
    return s


# --------------------------------------------------------------------------- #
def headline(df, classes, totals, n, seed):
    """One labelled draw of size n; the pretrained estimators get the EXACT
    wall-to-wall map totals as the unlabelled side (N = every valid pixel)."""
    rng = np.random.default_rng([seed, 999])
    L = rng.choice(len(df), n, replace=False)
    YL = onehot(df["ref"].values[L], classes)
    Nmap = totals["n_valid_px"]
    km2 = Nmap * totals["pixel_area_m2"] / 1e6
    Wmap = np.array([totals["hard_counts"][str(c)] for c in MAP_CLASSES]) / Nmap
    mean_soft = np.array([totals["pcal_sums"].get(str(c), 0.0) / Nmap if c in MAP_CLASSES
                          else 0.0 for c in classes])
    mean_hard = np.array([totals["hard_counts"].get(str(c), 0) / Nmap if c in MAP_CLASSES
                          else 0.0 for c in classes])
    F_hard = pred_matrix(df, classes, "hard")[L]
    F_soft = pred_matrix(df, classes, "soft")[L]
    hL = np.array([MAP_CLASSES.index(c) for c in df["map_class"].values[L]])
    out = {}

    def put(name, est, se):
        out[name] = {"est_km2": est * km2, "lo_km2": (est - Z * se) * km2,
                     "hi_km2": (est + Z * se) * km2}

    out["map_count"] = {"est_km2": mean_hard * km2}
    put("classical", YL.mean(0), YL.std(0, ddof=1) / np.sqrt(n))
    for name, F, mu in [("ppi_hard", F_hard, mean_hard), ("ppi_soft", F_soft, mean_soft)]:
        rect = YL - F                                  # N ~ 8e8: imputed variance ~ 0
        put(name, mu + rect.mean(0), rect.std(0, ddof=1) / np.sqrt(n))
    # PPI++: lambda from the labelled sample (N >> n so the (1+n/N) factor is 1)
    vF = F_soft.var(0, ddof=1)
    cov = ((YL - YL.mean(0)) * (F_soft - F_soft.mean(0))).sum(0) / (n - 1)
    lam = np.clip(np.where(vF > 0, cov / np.maximum(vF, 1e-12), 0), 0, 1)
    rect = YL - lam * F_soft
    put("ppipp_soft", lam * mean_soft + rect.mean(0), rect.std(0, ddof=1) / np.sqrt(n))
    put("olofsson", *olofsson(YL, hL, Wmap))
    # stratified PPI / cross-PPI: unlabelled = the rest of the random pixel sample
    # (an SRS of the AOI); stratum weights are still the exact wall-to-wall ones
    U = np.setdiff1d(np.arange(len(df)), L)
    hAll = np.array([MAP_CLASSES.index(c) for c in df["map_class"].values])
    F_all = pred_matrix(df, classes, "soft")
    put("stratppi", *stratppi(YL, F_soft, hL, F_all[U], hAll[U], Wmap))
    y = df["ref"].values
    for name, cols in [("xppi_aef", FEATS_AEF), ("xppi_pcal", FEATS_PCAL)]:
        X = df[cols].values.astype(np.float32)
        P_L, P_U = cross_predict(X[L], y[L], X[U], classes, seed=seed)
        est, lo, hi = xppi(YL, P_L, P_U)
        out[name] = {"est_km2": est * km2, "lo_km2": lo * km2, "hi_km2": hi * km2}
    return {k: {kk: [round(float(x), 1) for x in vv] for kk, vv in v.items()}
            for k, v in out.items()}, km2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--ns", default="300,1000,3000")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--reps-cross", type=int, default=100)
    ap.add_argument("--headline-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--headline-only", action="store_true",
                    help="reuse the saved simulation; recompute only the AOI headline")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.reps_cross, a.ns = 5, 2, "300"

    df, n_nogk = load_population(a.year)
    classes = sorted(set(df["ref"]) | set(MAP_CLASSES))
    totals = json.loads((RESULTS_DIR / f"area_map_totals_{a.year}.json").read_text())
    print(f"population {len(df):,} px with grunnkart ({n_nogk:,} without, dropped); "
          f"classes {classes}", flush=True)

    tag = "_quick" if a.quick else ""
    if a.headline_only:
        out = json.loads(result_path(f"area_ppi_{a.year}{tag}.json").read_text())
        out["headline"], _ = headline(df, classes, totals, out["headline_n"], a.seed)
        result_path(f"area_ppi_{a.year}{tag}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out["headline"]["stratppi"]))
        return
    ns = [int(x) for x in a.ns.split(",")]
    res, truth, W, timing = simulate(df, classes, ns, a.reps, a.reps_cross, a.seed)
    summ = summarise(res)
    summ.to_csv(result_path(f"area_ppi_sim_{a.year}{tag}.csv"), index=False)

    head, km2 = headline(df, classes, totals, a.headline_n, a.seed)
    fscs = pd.read_parquet(REPO_DIR / "data" /
                           "grunnkart_nyvest_fscs_stable_allyears_alphaearth.parquet",
                           columns=["class", "lon", "lat"]).drop_duplicates(["lon", "lat"])
    fscs_share = (fscs["class"].replace(REF_MERGE).value_counts(normalize=True)
                  .reindex(classes).fillna(0).values)
    out = {
        "year": a.year, "alpha": ALPHA, "classes": classes,
        "names": {str(c): NAMES.get(c, str(c)) for c in classes},
        "population_px": len(df), "population_no_grunnkart_px": n_nogk,
        "aoi_valid_km2": km2,
        "truth_share": [float(x) for x in truth],
        "map_stratum_weights": {str(c): float(w) for c, w in zip(MAP_CLASSES, W)},
        "fscs_training_share": [float(x) for x in fscs_share],
        "reps": a.reps, "reps_cross": a.reps_cross, "ns": ns,
        "cross_fit_seconds_per_rep": timing,
        "headline_n": a.headline_n, "headline": head,
    }
    result_path(f"area_ppi_{a.year}{tag}.json").write_text(json.dumps(out, indent=2))

    # console digest: per method, averaged over classes
    s = summ.dropna(subset=["width"]).copy()
    base = s[s.method == "classical"].set_index(["n", "cls"]).width
    s = s[s.design == "srs"].join(base.rename("w_cls"), on=["n", "cls"])
    s["ess_gain"] = (s.w_cls / s.width) ** 2
    print(s.groupby(["n", "method"]).agg(coverage=("coverage", "mean"),
                                        min_cov=("coverage", "min"),
                                        ess_gain=("ess_gain", "median"),
                                        abs_bias=("bias", lambda b: b.abs().mean()))
          .round(3).to_string())
    print(f"-> {result_path(f'area_ppi_{a.year}{tag}.json')}")


if __name__ == "__main__":
    main()

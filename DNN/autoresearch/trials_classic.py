"""Classical references: what the features are worth without a deep net.

Every number in this round is a delta against the deployed MLP, which answers
"is mechanism X better than what we ship" and never answers "how much of the
0.7414 is the model at all". Three rounds of ties are much easier to read once
the floor is on the page:

  probe_linear   multinomial logistic regression on the same 67 standardized
                 features. AlphaEarth bands are already a learned embedding, and
                 the standard way to report what an embedding carries is a linear
                 probe. Whatever this scores is the part of the task that needed
                 no nonlinearity at all; the MLP's margin over it is the entire
                 value of the representation the network learns on top.

  rf_tuned       a tuned random forest — the model this project would otherwise
                 have reached for, and the family (CatBoost) the DNN replaced.
                 Axis-aligned splits on the same columns, tuned on the same
                 leak-free inner split the MLP early-stops on.

Both are held to the harness's rules rather than sklearn habits:

  * the SAME GroupKFold folds, the same cls12 relabel, the same stale-class
    cleaning, the same StandardScaler fitted on the training fold only —
    `run_trial` does all of it before `fold_fn` is called, so the only thing
    that differs from the baseline record is the estimator.
  * sqrt-inverse-frequency weights, applied as sample weights, because the
    target is macro-F1 and the deployed net is trained with exactly those
    weights. An unweighted forest is a different experiment and would understate
    the family on the weak classes.
  * hyperparameters chosen on the INNER VAL SPLIT, never on the test fold. The
    inner split is the same 10% the MLP early-stops on, so both models get one
    look at held-out data and neither gets to see the held-out region.

Neither is expected to win. A reference is not a candidate: it is the number
that says how much of the score the deployed recipe is actually responsible for.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import ar_common as ac
import data_utils as du

QUEUE_CLASSIC = ["probe_linear", "rf_tuned"]

# Tuning grids. Deliberately small and stated here rather than buried in the
# fold function: a "tuned" baseline whose grid is unpublished is not a baseline
# anyone can check.
C_GRID = [0.01, 1.0, 100.0]
RF_GRID = [
    {"max_features": 8, "min_samples_leaf": 1},     # sqrt(67), sklearn's default
    {"max_features": 8, "min_samples_leaf": 5},
    {"max_features": 16, "min_samples_leaf": 1},
    {"max_features": 16, "min_samples_leaf": 5},
    {"max_features": 32, "min_samples_leaf": 1},
    {"max_features": 32, "min_samples_leaf": 5},
]
RF_TUNE_TREES = 150      # cheap forests to rank the grid
RF_FINAL_TREES = 500     # the reported model
N_JOBS = 8


def _arrays(ctx):
    """The harness keeps its splits on the GPU; sklearn needs them on the host."""
    return (ctx["Xtr_t"].cpu().numpy(), ctx["ytr_np"],
            ctx["Xval_t"].cpu().numpy(), ctx["yval_np"],
            ctx["Xte_t"].cpu().numpy())


def _sample_weight(ctx):
    """Per-row weights from the harness's own class weights, so 'sqrt weighting'
    means the same thing here as it does in the cross-entropy."""
    w = ctx["w"]
    if w is None:
        return None
    return w.detach().cpu().numpy()[ctx["ytr_np"]]


def _pick(cands):
    """argmax over (score, config) with the whole grid printed — the tuning trace
    is part of the result, not a side effect."""
    for score, tag in cands:
        print(f"      {tag:44s} val macro-F1 {score:.4f}")
    best = max(cands, key=lambda c: c[0])
    print(f"      -> chose {best[1]} ({best[0]:.4f})")
    return best


def fold_probe_linear(ctx):
    """Multinomial logistic regression, C chosen on the inner val.

    lbfgs on standardized features; `max_iter` is generous because a probe that
    stopped early would understate the linear ceiling, which is the one thing
    this trial exists to measure. C=100 is effectively unregularised and C=0.01
    is heavily shrunk, so the grid brackets the regime rather than sampling
    around a guess.
    """
    Xtr, ytr, Xval, yval, Xte = _arrays(ctx)
    sw = _sample_weight(ctx)
    n_classes = ctx["n_classes"]

    cands, fitted = [], {}
    for C in C_GRID:
        t0 = time.perf_counter()
        clf = LogisticRegression(C=C, max_iter=600, tol=1e-3)
        clf.fit(Xtr, ytr, sample_weight=sw)
        f1 = du.macro_f1(yval, clf.predict(Xval), n_classes)
        cands.append((f1, f"C={C}"))
        fitted[f"C={C}"] = clf
        print(f"      (C={C} fitted in {time.perf_counter() - t0:.0f}s, "
              f"converged={clf.n_iter_[0] < 600})")
    best_f1, tag = _pick(cands)
    clf = fitted[tag]

    # Probabilities on the class columns the estimator actually saw. A fold can
    # legitimately be missing a class after cleaning, and silently returning a
    # narrower matrix would misalign every downstream argmax.
    P_te = np.zeros((Xte.shape[0], n_classes))
    P_te[:, clf.classes_] = clf.predict_proba(Xte)
    return P_te, best_f1, {"chosen": tag,
                           "val_f1_by_config": {t: round(s, 4) for s, t in cands}}


def fold_rf_tuned(ctx):
    """Random forest: rank the grid with small forests on the inner val, then
    refit the winner at full size on the whole training fold.

    Ranking with 150 trees and reporting with 500 is the standard split of a
    tuning budget — forest ranking is stable in n_estimators long before the
    score has converged — and it keeps the grid affordable on an 8-core box.
    The refit uses train+val because, unlike the MLP, a forest has no early stop
    that needs the val rows held back; the val split has already done its one
    job by the time the winner is known.
    """
    Xtr, ytr, Xval, yval, Xte = _arrays(ctx)
    sw = _sample_weight(ctx)
    n_classes = ctx["n_classes"]

    cands = []
    for g in RF_GRID:
        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=RF_TUNE_TREES, n_jobs=N_JOBS,
                                    random_state=ac.SEED, **g)
        rf.fit(Xtr, ytr, sample_weight=sw)
        f1 = du.macro_f1(yval, rf.predict(Xval), n_classes)
        tag = f"max_features={g['max_features']},min_samples_leaf={g['min_samples_leaf']}"
        cands.append((f1, tag))
        print(f"      ({tag} in {time.perf_counter() - t0:.0f}s)")
        del rf
    best_f1, tag = _pick(cands)
    g = RF_GRID[[t for _, t in cands].index(tag)]

    X_all = np.concatenate([Xtr, Xval])
    y_all = np.concatenate([ytr, yval])
    sw_all = None if sw is None else ctx["w"].detach().cpu().numpy()[y_all]
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=RF_FINAL_TREES, n_jobs=N_JOBS,
                                random_state=ac.SEED, **g)
    rf.fit(X_all, y_all, sample_weight=sw_all)
    print(f"      refit {RF_FINAL_TREES} trees on {len(y_all):,} rows "
          f"in {time.perf_counter() - t0:.0f}s")

    P_te = np.zeros((Xte.shape[0], n_classes))
    P_te[:, rf.classes_] = rf.predict_proba(Xte)
    # val_f1 reported is the TUNING score (150 trees, val-held-out). Scoring the
    # refit model on val would be scoring it on its own training rows, and that
    # number would sit in the same column as the MLP's honest early-stop val F1.
    info = {"chosen": tag, "final_trees": RF_FINAL_TREES,
            "val_f1_by_config": {t: round(s, 4) for s, t in cands},
            "mean_depth": round(float(np.mean([e.get_depth()
                                               for e in rf.estimators_])), 1),
            "total_nodes": int(sum(e.tree_.node_count for e in rf.estimators_))}
    return P_te, best_f1, info


def _selftest(verbose=True):
    """Pre-flight assertions on the two things a reference can get silently
    wrong: leaking the test fold, and returning a probability matrix that does
    not line up with the harness's class encoding.

    Run on synthetic data in a couple of seconds:  python trials_classic.py
    """
    import torch

    rng = np.random.default_rng(0)
    n, d, C = 3000, 12, 5
    X = rng.normal(size=(n, d)).astype(np.float32)
    # a class the TRAINING split never contains, so the column-alignment claim
    # is actually exercised rather than asserted on a case that cannot happen
    y = rng.integers(0, C - 1, size=n)
    w = np.linspace(0.5, 2.0, C)
    ok = []

    def chk(name, cond, detail=""):
        ok.append(bool(cond))
        if verbose:
            print(f"  [{'ok ' if cond else 'FAIL'}] {name} {detail}")

    class _Poison:
        """Anything that touches the test labels blows up here rather than
        quietly producing an optimistic number."""
        def __getitem__(self, k):
            raise AssertionError("fold_fn read the TEST labels")
        def __len__(self):
            raise AssertionError("fold_fn read the TEST labels")

    ctx = {
        "n_classes": C, "trial": None,
        "Xtr_t": torch.tensor(X[:2000]), "ytr_np": y[:2000],
        "Xval_t": torch.tensor(X[2000:2600]), "yval_np": y[2000:2600],
        "Xte_t": torch.tensor(X[2600:]),
        "y_te": _Poison(),
        "w": torch.tensor(w, dtype=torch.float32),
    }

    sw = _sample_weight(ctx)
    chk("sample weights follow the row's own class",
        sw.shape == (2000,) and np.allclose(sw, w[y[:2000]]))

    global C_GRID, RF_GRID, RF_TUNE_TREES, RF_FINAL_TREES
    keep = (C_GRID, RF_GRID, RF_TUNE_TREES, RF_FINAL_TREES)
    C_GRID = [1.0]
    RF_GRID = [{"max_features": 4, "min_samples_leaf": 5}]
    RF_TUNE_TREES, RF_FINAL_TREES = 8, 12
    try:
        for name, fn in (("probe_linear", fold_probe_linear),
                         ("rf_tuned", fold_rf_tuned)):
            # the assertion here is that this call RETURNS: ctx["y_te"] is a
            # _Poison, so any read of the test labels raises out of fn()
            P, vf1, info = fn(ctx)
            chk(f"{name}: no test-label access", True)
            chk(f"{name}: P has one column per harness class",
                P.shape == (n - 2600, C), f"got {P.shape}")
            chk(f"{name}: unseen class {C - 1} gets zero mass, not a shifted column",
                float(P[:, C - 1].max()) == 0.0)
            chk(f"{name}: rows are a distribution",
                np.allclose(P.sum(1), 1.0, atol=1e-6))
            chk(f"{name}: val F1 is a real score", 0.0 <= vf1 <= 1.0, f"{vf1:.3f}")
            chk(f"{name}: tuning trace recorded",
                "chosen" in info and "val_f1_by_config" in info)
    finally:
        C_GRID, RF_GRID, RF_TUNE_TREES, RF_FINAL_TREES = keep

    print(f"\n{sum(ok)}/{len(ok)} passed")
    return all(ok), ok


def register(_add, _Trial):
    """Called from trials.py so these land in the same TRIALS registry that
    run.py, resolve() and the loop already understand."""

    _add(_Trial(
        name="probe_linear", tier="reference",
        idea="multinomial logistic regression on the same 67 standardized "
             "features, C tuned on the inner val",
        hypothesis="AlphaEarth is already a learned embedding, and the linear "
                   "probe is the standard way to say what an embedding carries "
                   "on its own. Whatever this scores is the share of the task "
                   "that needed no nonlinearity; the deployed MLP's margin over "
                   "it is the whole value of the representation the network "
                   "builds on top, and every tie in this round is a failure to "
                   "add to THAT margin, not to the score",
        provenance="linear-probe evaluation protocol, standard in "
                   "representation learning (Alain & Bengio 2016)",
        fold_fn=fold_probe_linear, n_ensemble=1,
        config={"C_grid": ",".join(str(c) for c in C_GRID), "max_iter": 600}))

    _add(_Trial(
        name="rf_tuned", tier="reference",
        idea="random forest, 6-cell grid ranked on the inner val and refit at "
             "500 trees — the classical model this project would otherwise ship",
        hypothesis="the DNN replaced a CatBoost+TabICL stack, so 'a deep net was "
                   "needed here' is an assumption this round has been resting on "
                   "without ever measuring it inside this harness. A tuned "
                   "forest on identical folds, identical weights and identical "
                   "features either supports it or does not. Trees also fail "
                   "differently from an MLP under spatial shift — axis-aligned "
                   "splits extrapolate to a held-out region worse than a smooth "
                   "decision surface — so the gap should be widest exactly where "
                   "this round has been stuck",
        provenance="Breiman 2001; the deployed-family control, cf. "
                   "benchmark_tabular.py",
        fold_fn=fold_rf_tuned, n_ensemble=1,
        config={"tune_trees": RF_TUNE_TREES, "final_trees": RF_FINAL_TREES,
                "grid_size": len(RF_GRID)}))


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(0 if _selftest()[0] else 1)

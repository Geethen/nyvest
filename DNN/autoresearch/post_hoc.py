"""Decision-rule experiments: change WHERE argmax falls, not what the net learns.

These run on the saved probabilities from `baseline_gval` (which held its inner
val out by whole cell_ids), so they cost seconds rather than GPU-hours and every
rule is scored on the SAME frozen probabilities — a perfectly paired comparison
between rules, with `baseline_gval`'s own plain argmax as the internal control.

Why this class of intervention is worth its own file: macro-F1 is not maximised
by the argmax of a calibrated posterior. The Bayes rule for accuracy is argmax p;
the Bayes rule for macro-F1 is a per-class THRESHOLD rule, and the gap between
them grows with class imbalance. Everything else in this program tries to make
the model know more. These trials only ask whether the model already knows
enough and is being asked the wrong question — and by construction they cannot
change what the net learned, only how its output is read.

Leak discipline: every rule is fitted on the inner val (train-fold cells the
models never saw) and applied unchanged to the test fold. `dr_oracle_offsets`
deliberately breaks that rule to measure HEADROOM and is reported as an upper
bound, never as a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ar_common as ac

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_utils as du  # noqa: E402

SOURCE = "baseline_gval"
GRID = np.arange(-3.0, 3.01, 0.1)


def _f1(y, logp, off, n_classes):
    return du.macro_f1(y, (logp + off).argmax(1), n_classes)


def fit_offsets(logp, y, n_classes, rounds=4):
    """Coordinate ascent on per-class additive log-offsets to maximise macro-F1.

    The plug-in view: with a fixed posterior, the macro-F1-optimal decision rule
    is argmax_c (log p_c + b_c) for some per-class b. There is no closed form
    because each class's optimal threshold depends on every other class's, so
    coordinate ascent on the actual metric is the honest way to find b.
    """
    off = np.zeros(n_classes)
    best = _f1(y, logp, off, n_classes)
    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            cur = off[c]
            scores = []
            for d in GRID:
                off[c] = cur + d
                scores.append(_f1(y, logp, off, n_classes))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                best, off[c], improved = scores[j], cur + GRID[j], True
            else:
                off[c] = cur
        if not improved:
            break
    return off, best


def em_target_prior(P, src_prior, iters=100, tol=1e-7):
    """Saerens-Latinne-Decaestecker (Neural Computation, 2002) EM for prior shift.

    Estimates the TARGET region's class prior from unlabelled target
    probabilities alone, then re-weights the posterior by pi_target/pi_source.
    Old idea, exactly on point here: the documented failure mode is that a
    held-out geographic fold has a different land-cover MIX from the training
    regions, and this is the closed-form correction for a pure prior shift
    (transductive — uses target features, never target labels).

    `src_prior` MUST be the prior the posteriors are actually calibrated to. The
    first version of this passed the empirical LABEL frequencies, which is wrong
    here: the net trains with sqrt class weights, so its posterior already
    carries a flattened implicit prior. Dividing by the steeper label prior
    re-boosted the rare classes on every iteration and the recursion ran away to
    a degenerate prior (99.3% snow/ice by iteration 5, macro-F1 0.0004). Passing
    the model's own mean predicted probability fixes it — and is the quantity the
    derivation actually calls for.
    """
    pi = src_prior.copy()
    for _ in range(iters):
        W = P * (pi / src_prior)[None, :]
        W /= W.sum(1, keepdims=True)
        new = W.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    if pi.max() > 0.75 and src_prior.max() < 0.5:
        # A prior this degenerate is divergence, not a discovery about the region.
        raise RuntimeError(f"EM diverged: pi_max={pi.max():.3f} from "
                           f"src_max={src_prior.max():.3f}")
    return pi


def fewshot_region(z, base, ctrl, classes, n_folds, fracs=(0.002, 0.01, 0.05)):
    """How many labels in a NEW region would it take to unlock the oracle gain?

    The oracle rule wins +0.0064 but needs the whole test fold's labels; fitting
    on held-out cells wins nothing because the optimal per-class thresholds are
    region-specific. That leaves a practical question rather than a dead end: the
    thresholds are only 10 numbers, so how much labelling in a target region buys
    them?

    Protocol: fit the offsets on a random FRACTION of the test fold's labelled
    rows, then score macro-F1 on the ROWS THAT WERE NOT USED — so the reported
    number is honest out-of-sample within the target region. The control is the
    same probabilities read by plain argmax on the same evaluation rows.
    """
    n_classes = len(classes)
    rng = np.random.default_rng(0)
    out = []
    for frac in fracs:
        f1s, ctrls, ns = [], [], []
        for k in range(n_folds):
            P = z[f"fold{k}_P"].astype(np.float64)
            y = z[f"fold{k}_y"]
            logp = np.log(P + 1e-12)
            idx = rng.permutation(len(y))
            n_fit = int(len(y) * frac)
            fit_i, ev_i = idx[:n_fit], idx[n_fit:]
            off, _ = fit_offsets(logp[fit_i], y[fit_i], n_classes)
            f1s.append(du.macro_f1(y[ev_i], (logp[ev_i] + off).argmax(1), n_classes))
            ctrls.append(du.macro_f1(y[ev_i], logp[ev_i].argmax(1), n_classes))
            ns.append(n_fit)
        gain = float(np.mean(f1s) - np.mean(ctrls))
        rec = {
            "name": f"dr_fewshot_{frac:g}".replace(".", "p"),
            "tier": "decision-rule",
            "idea": f"per-class offsets fitted on {frac:.1%} of the TARGET region's "
                    f"labels (~{int(np.mean(ns)):,} points/fold), scored on the "
                    f"held-out remainder",
            "hypothesis": "the optimal decision rule is region-specific and cannot be "
                          "transferred from training regions — but it is only 10 "
                          "numbers, so a small labelling budget inside the target "
                          "region may recover it",
            "provenance": "few-shot / on-site calibration of a plug-in F-measure rule",
            "relabel": ac.RELABEL, "transductive": False,
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "f1_per_fold": [round(v, 4) for v in f1s],
            "f1_per_class": base["f1_per_class"],
            "delta_mean": round(gain, 4),
            "delta_per_fold": [round(f1s[i] - ctrls[i], 4) for i in range(n_folds)],
            "baseline_per_fold": [round(v, 4) for v in ctrls],
            "baseline_mean": round(float(np.mean(ctrls)), 4),
            "control": "same probabilities, plain argmax, same evaluation rows",
            "delta_vs_control_mean": round(gain, 4),
            "all_folds_positive": all(f1s[i] > ctrls[i] for i in range(n_folds)),
            "verdict": ("WIN" if (all(f1s[i] > ctrls[i] for i in range(n_folds))
                                  and gain > 0.003) else
                        "tie" if abs(gain) <= 0.003 else "LOSS"),
            "n_labels_per_fold": ns,
            "val_mode": "target-region", "weight_mode": "sqrt", "n_ensemble": 5,
            "config": {"frac": frac}, "runtime_s": 0.0,
        }
        (ac.RESULTS_DIR / f"{rec['name']}.json").write_text(json.dumps(rec, indent=2))
        out.append(rec)
        print(f"{rec['name']:20s} {int(np.mean(ns)):>6,} labels/fold -> "
              f"F1 {np.mean(ctrls):.4f} -> {np.mean(f1s):.4f}  ({gain:+.4f}) "
              f"{rec['verdict']}")
    return out


def main():
    src = ac.RESULTS_DIR / f"{SOURCE}.json"
    npz = ac.RESULTS_DIR / f"{SOURCE}_probs.npz"
    if not (src.exists() and npz.exists()):
        raise SystemExit(f"need {SOURCE} results + probs first")
    base = ac.load_baseline()
    ctrl = json.loads(src.read_text())
    z = np.load(npz)
    classes = [int(c) for c in base["f1_per_class"]]
    n_classes = len(classes)
    n_folds = len(ctrl["f1_per_fold"])

    rules = {
        "dr_prior_tune": "per-class log-offsets fitted by coordinate ascent on the "
                         "spatial inner val (plug-in macro-F1 rule)",
        "dr_logitadj_tau": "single-parameter prior adjustment log p - tau*log(prior), "
                           "tau swept on the spatial inner val",
        "dr_priorem": "Saerens EM target-prior estimate from the target fold's own "
                      "unlabelled probabilities (transductive)",
        "dr_oracle_offsets": "UPPER BOUND ONLY: the same per-class offsets fitted "
                             "directly on the test fold — measures how much macro-F1 "
                             "a perfect decision rule could ever recover",
    }
    out = {k: {"f1": [], "pc": [], "params": []} for k in rules}

    for k in range(n_folds):
        P = z[f"fold{k}_P"].astype(np.float64)
        y = z[f"fold{k}_y"]
        Pv = z[f"fold{k}_Pval"].astype(np.float64)
        yv = z[f"fold{k}_yval"]
        logp, logpv = np.log(P + 1e-12), np.log(Pv + 1e-12)
        # The model's IMPLICIT source prior — its mean predicted probability on
        # held-out in-distribution rows. Not the label frequencies: sqrt class
        # weighting means the two differ substantially (e.g. snow/ice 0.076 vs
        # 0.008), and every rule here is a correction TO the posterior, so it has
        # to be expressed relative to the prior the posterior actually carries.
        prior = Pv.mean(0)
        prior /= prior.sum()

        off, vbest = fit_offsets(logpv, yv, n_classes)
        out["dr_prior_tune"]["params"].append([round(float(v), 2) for v in off])
        _push(out["dr_prior_tune"], y, (logp + off).argmax(1), n_classes)

        taus = np.arange(0.0, 1.01, 0.05)
        vs = [du.macro_f1(yv, (logpv - t * np.log(prior)).argmax(1), n_classes)
              for t in taus]
        tau = float(taus[int(np.argmax(vs))])
        out["dr_logitadj_tau"]["params"].append(tau)
        _push(out["dr_logitadj_tau"], y, (logp - tau * np.log(prior)).argmax(1), n_classes)

        pi_t = em_target_prior(P, prior)
        out["dr_priorem"]["params"].append([round(float(v), 4) for v in pi_t])
        _push(out["dr_priorem"], y, (logp + np.log(pi_t / prior)).argmax(1), n_classes)

        off_o, _ = fit_offsets(logp, y, n_classes)
        out["dr_oracle_offsets"]["params"].append([round(float(v), 2) for v in off_o])
        _push(out["dr_oracle_offsets"], y, (logp + off_o).argmax(1), n_classes)

        print(f"fold {k}: control={ctrl['f1_per_fold'][k]:.4f} " +
              " ".join(f"{r.replace('dr_','')}={out[r]['f1'][-1]:.4f}" for r in rules))

    recs = []
    for name, desc in rules.items():
        f1s = out[name]["f1"]
        pc = np.mean(out[name]["pc"], axis=0)
        deltas = [f1s[i] - base["f1_per_fold"][i] for i in range(n_folds)]
        ctrl_d = [f1s[i] - ctrl["f1_per_fold"][i] for i in range(n_folds)]
        dmean = float(np.mean(deltas))
        all_pos = all(d > 0 for d in deltas)
        oracle = name == "dr_oracle_offsets"
        verdict = ("upper-bound" if oracle else
                   "WIN" if (all_pos and dmean > 0.003) else
                   "tie" if abs(dmean) <= 0.003 else "LOSS")
        pc_delta = np.array([pc[i] - base["f1_per_class"][str(c)]
                             for i, c in enumerate(classes)])
        rec = {
            "name": name, "tier": "decision-rule", "idea": desc,
            "hypothesis": "macro-F1's Bayes rule is a per-class threshold rule, not "
                          "argmax of the posterior; if the gap is real it is free, "
                          "since the network is untouched",
            "provenance": ("Saerens et al. 2002" if name == "dr_priorem" else
                           "Koyejo et al. 2014 / Lipton et al. 2014 (plug-in F-measure)"),
            "relabel": ac.RELABEL, "transductive": name == "dr_priorem",
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "f1_per_fold": [round(v, 4) for v in f1s],
            "f1_per_class": {str(c): round(float(v), 4) for c, v in zip(classes, pc)},
            "baseline_per_fold": base["f1_per_fold"],
            "baseline_mean": base["f1_mean"],
            "delta_per_fold": [round(d, 4) for d in deltas],
            "delta_mean": round(dmean, 4),
            "delta_vs_control_mean": round(float(np.mean(ctrl_d)), 4),
            "control": f"{SOURCE} plain argmax {ctrl['f1_mean']}",
            "all_folds_positive": all_pos, "verdict": verdict,
            "delta_per_class": {str(c): round(float(v), 4)
                                for c, v in zip(classes, pc_delta)},
            "params_per_fold": out[name]["params"],
            "val_mode": "group", "weight_mode": "sqrt", "n_ensemble": 5,
            "config": {}, "runtime_s": 0.0,
            **ac._tradeoff_report(pc_delta, classes),
        }
        (ac.RESULTS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2))
        recs.append(rec)
        print(f"{name:20s} F1={rec['f1_mean']:.4f} Δ={dmean:+.4f} "
              f"(vs same-probs control {rec['delta_vs_control_mean']:+.4f}) "
              f"{verdict}  no_tradeoff={rec['no_tradeoff']}")

    recs += fewshot_region(z, base, ctrl, classes, n_folds)
    _log_wandb(recs, classes)


def _push(slot, y, pred, n_classes):
    slot["f1"].append(du.macro_f1(y, pred, n_classes))
    slot["pc"].append(du.per_class_f1(y, pred, n_classes))


def _log_wandb(recs, classes):
    try:
        import wandb
    except ImportError:
        return
    for rec in recs:
        run = wandb.init(project=ac.WANDB_PROJECT, name=rec["name"],
                         group="decision-rule", job_type="decision-rule",
                         reinit=True, tags=["decision-rule"],
                         notes=f"{rec['idea']}\n\nHYPOTHESIS: {rec['hypothesis']}",
                         config={"trial": rec["name"], "tier": "decision-rule",
                                 "idea": rec["idea"], "hypothesis": rec["hypothesis"],
                                 "provenance": rec["provenance"],
                                 "source_probs": SOURCE})
        summary = {k: v for k, v in rec.items()
                   if isinstance(v, (int, float, bool, str))}
        for c, v in rec["f1_per_class"].items():
            summary[f"f1_class_{c}"] = v
        for c, v in rec["delta_per_class"].items():
            summary[f"delta_class_{c}"] = v
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()

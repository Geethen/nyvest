"""The autoresearch loop: schedule trials, react to results, keep the page fresh.

Three rounds, each conditioned on the last:

  round 1  the pre-registered QUEUE — one mechanism per trial, breadth first.
  round 1b decision rules, as soon as `baseline_gval` has produced probabilities
           (they are near-free and read those probabilities, so they run the
           moment their input exists rather than waiting for the queue).
  round 2  follow-ups generated FROM round-1 results: sweep the hyperparameter of
           anything that scored above the noise floor, and combine mechanisms
           that won on different hooks to test whether the gains are additive.

The loop is resumable — a trial whose results JSON already exists is skipped — so
it can be killed and restarted without losing work.

Run:  python loop.py            (full program, 3 GPU workers)
      python loop.py --workers 2 --round 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import ar_common as ac
import trials as T

HERE = Path(__file__).resolve().parent
PY = sys.executable
STATE = ac.RESULTS_DIR / "_loop_state.json"

# A follow-up is only worth GPU time if the mechanism showed signal. This is the
# threshold for "not noise, worth a sweep" — deliberately BELOW the 0.003 win bar,
# because round 2 exists to find out whether a promising direction can be pushed
# over that bar, not to celebrate near-misses.
FOLLOWUP_FLOOR = 0.001

# Pre-registered sweeps, so round 2 cannot be accused of fishing: the axis for
# each mechanism is fixed here BEFORE round 1 reports.
SWEEPS = {
    "pair_margin": ["pair_margin@margin=2.0", "pair_margin@lam=1.5",
                    "pair_margin@margin=0.5,lam=1.0"],
    "vs_loss": ["vs_loss@gamma=0.3,tau=1.5", "vs_loss@gamma=0.1,tau=0.75"],
    "ldam_drw": ["ldam_drw@max_margin=0.3", "ldam_drw@max_margin=0.7"],
    "hier_c2f": ["hier_c2f@label_smooth=0.0"],
    "hier_finewt": ["hier_finewt@fine_weight=4.0"],
    "hier_specialist": ["hier_specialist@n_ensemble=3"],
    "tabm": ["tabm@k=32"],
    "kan": ["kan@degree=8"],
    "numemb_lidar": ["numemb_lidar@sigma=0.1", "numemb_lidar@sigma=2.0"],
    "numemb_all": ["numemb_all@sigma=0.1"],
    "opt_muon": ["opt_muon@muon_lr=0.005", "opt_muon@muon_lr=0.05"],
    "opt_lion": ["opt_lion@lion_lr=3e-5"],
    "softf1": ["softf1@lam=3.0"],
    "crt": ["crt@crt_steps=1500,crt_lr=1e-4"],
    "tta_cbst": ["tta_cbst@quantile=0.5", "tta_cbst@quantile=0.1"],
    "balanced_softmax": [],
    "selfdistill": ["selfdistill@alpha=0.2"],
    "tau_norm": [],
    "act_gelu": [], "act_mish": [], "act_prelu": [], "act_snake": [],
    "act_xielu": [], "opt_schedulefree": [], "ctrl_noweight": [],
}

# Hook occupancy — two mechanisms can only be combined if they use different
# hooks. Everything else would silently become a third, untested mechanism.
HOOKS = {"build_fn", "loss_fn", "opt_fn", "after_fit", "fold_fn"}


def done(name):
    return (ac.RESULTS_DIR / f"{name}.json").exists()


def result(name):
    p = ac.RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def spec_name(spec):
    return T.resolve(spec).name


def launch(spec):
    log = ac.LOGS_DIR / f"{spec_name(spec)}.log"
    f = open(log, "w")
    p = subprocess.Popen([PY, "-u", str(HERE / "run.py"), spec], cwd=HERE,
                         stdout=f, stderr=subprocess.STDOUT)
    p._logfile, p._spec = f, spec
    return p


def refresh_page():
    subprocess.run([PY, "-u", str(HERE / "build_artifact.py")], cwd=HERE,
                   capture_output=True)


def run_pool(specs, workers, label):
    """Run `specs` with at most `workers` concurrent GPU processes."""
    todo = [s for s in specs if not done(spec_name(s))]
    if not todo:
        print(f"[{label}] nothing to do")
        return
    print(f"[{label}] {len(todo)} trials, {workers} workers: "
          f"{', '.join(spec_name(s) for s in todo)}")
    running, queue = [], list(todo)
    while queue or running:
        while queue and len(running) < workers:
            s = queue.pop(0)
            print(f"[{label}] start {spec_name(s)}  ({len(queue)} queued)")
            running.append(launch(s))
        time.sleep(10)
        for p in list(running):
            if p.poll() is None:
                continue
            p._logfile.close()
            running.remove(p)
            rec = result(spec_name(p._spec))
            if rec is None:
                print(f"[{label}] FAILED {spec_name(p._spec)} "
                      f"(rc={p.returncode}) — see logs/")
            else:
                print(f"[{label}] done {rec['name']:28s} F1={rec['f1_mean']:.4f} "
                      f"Δ={rec.get('delta_mean', 0):+.4f} {rec.get('verdict', '')}"
                      f"{'  NO-TRADEOFF' if rec.get('no_tradeoff') else ''}")
            refresh_page()
            _save_state()


def _save_state():
    # Diagnostics (diag_locality.json) share results/ with trial records but have
    # no name or f1_mean. Filter on the trial shape rather than the filename, so
    # a future diagnostic cannot take the whole loop down between trials again.
    recs = [r for r in (json.loads(p.read_text())
                        for p in sorted(ac.RESULTS_DIR.glob("*.json"))
                        if not p.name.startswith("_"))
            if isinstance(r, dict) and "f1_mean" in r and "name" in r]
    STATE.write_text(json.dumps({
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_done": len(recs),
        "best": sorted([(r["f1_mean"], r["name"]) for r in recs], reverse=True)[:5],
    }, indent=2))


def plan_round2():
    """Generate follow-ups from round-1 results. Two kinds:

    sweeps   — for any mechanism above the follow-up floor, walk the ONE
               hyperparameter axis registered for it in SWEEPS above.
    combos   — the cross product of the best mechanisms that occupy DIFFERENT
               hooks. Combining is the only way to distinguish "these two fix
               different errors" from "these two fix the same error twice".
    """
    recs = [r for r in (result(n) for n in T.TRIALS) if r and "delta_mean" in r]
    recs.sort(key=lambda r: r["delta_mean"], reverse=True)
    promising = [r for r in recs if r["delta_mean"] >= FOLLOWUP_FLOOR]
    print(f"[round2] {len(promising)} mechanisms above the {FOLLOWUP_FLOOR} floor: "
          + ", ".join(f"{r['name']}({r['delta_mean']:+.4f})" for r in promising))

    specs = []
    for r in promising:
        specs += SWEEPS.get(r["name"], [])

    # combos: best per hook, pairwise, only where hooks do not collide
    by_hook = {}
    for r in promising:
        t = T.TRIALS.get(r["name"])
        if t is None:
            continue
        used = tuple(sorted(h for h in HOOKS if getattr(t, h) is not None))
        if not used:
            used = ("none",)
        by_hook.setdefault(used, []).append(r)
    heads = [v[0]["name"] for v in by_hook.values()]
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            a, b = T.TRIALS[heads[i]], T.TRIALS[heads[j]]
            if any(getattr(a, h) is not None and getattr(b, h) is not None
                   for h in HOOKS):
                continue
            specs.append(f"combo:{heads[i]}+{heads[j]}")
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--round", type=int, default=0, help="0 = all rounds")
    args = ap.parse_args()

    if not done("baseline"):
        print("[round0] baseline (everything is scored against it)")
        run_pool(["baseline"], 1, "round0")
        if not done("baseline"):
            raise SystemExit("baseline failed — see logs/baseline.log")

    if args.round in (0, 1):
        # baseline_gval first and alone: the decision-rule trials read its
        # probabilities, so getting it out of the way unblocks round 1b early.
        run_pool(["baseline_gval"], 1, "round1a")
        if done("baseline_gval"):
            print("[round1b] decision rules (post-hoc, seconds)")
            subprocess.run([PY, "-u", str(HERE / "post_hoc.py")], cwd=HERE)
            refresh_page()
        rest = [n for n in T.QUEUE if n not in ("baseline", "baseline_gval")]
        run_pool(rest, args.workers, "round1")

    if args.round in (0, 2):
        specs = plan_round2()
        run_pool(specs, args.workers, "round2")

    refresh_page()
    _save_state()
    print("[loop] complete")
    print(json.dumps(json.loads(STATE.read_text()), indent=2))


if __name__ == "__main__":
    main()

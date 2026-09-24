"""ar3 loop: screen -> follow up -> confirm. Resumable (a finished record is skipped).

  round 1   pre-registered QUEUE at N_ENSEMBLE=3, seed set 0, paired vs `baseline`.
  round 2   SWEEPS (axes fixed here, before round 1 reports) for every arm that
            improved at least one metric, plus the CONSENSUS arms once
            data/consensus/ exists.
  confirm   every arm whose verdict is ADOPTABLE or tradeoff, re-run at
            N_ENSEMBLE=5 on FRESH seed sets (1 and 2) in results_confirm/, with
            its own baseline pair. This is the result to believe: the last round's
            best arm halved on a fresh seed.

Run:  python loop3.py [--workers 2] [--round 1|2|3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
SCREEN_DIR = HERE / "results"
CONFIRM_DIR = HERE / "results_confirm"
LOGS = HERE / "logs"
LOGS.mkdir(exist_ok=True)

QUEUE = [
    # temporal
    "tmp_consist", "tmp_pair", "tmp_ctx", "tmp_ctx_mean",
    "tmp_joint", "tmp_hmm", "tmp_pool",
    # calibration / confidence
    "cal_ls0", "cal_focal", "cal_logitnorm", "cal_ens5",
]
CONSENSUS_QUEUE = ["cons_add", "cons_add_w05", "ctrl_hard_gk", "ctrl_rand_cons",
                   "cons_ext_majority"]
SWEEPS = {
    "tmp_consist": ["tmp_consist@lam=0.3", "tmp_consist@lam=3.0"],
    "tmp_joint": ["tmp_joint@eps=0.002", "tmp_joint@eps=0.05"],
    "tmp_hmm": ["tmp_hmm@eps=0.001", "tmp_hmm@eps=0.02", "tmp_hmm@eps=0.05"],
    "cal_focal": ["cal_focal@gamma=1.0", "cal_focal@gamma=5.0"],
    "cal_logitnorm": ["cal_logitnorm@tau=0.01", "cal_logitnorm@tau=0.1"],
    "cons_add": ["cons_add@weight=0.25"],
}
# combos of the round-1 adoptable arms, fixed after round 1 reported (so NOT
# pre-registered — they are treated as exploratory until confirmed)
COMBOS = ["combo:tmp_pair+cal_ls0", "combo:tmp_ctx_mean+cal_ls0",
          "combo:tmp_pair+tmp_joint", "combo:tmp_ctx_mean+tmp_joint"]


def rec_name(spec):
    if spec.startswith("combo:"):
        return "combo_" + "_".join(rec_name(x) for x in spec[6:].split("+"))
    return spec.replace("@", "__").replace("=", "").replace(",", "_").replace(".", "p")


def result(spec, d=SCREEN_DIR):
    p = d / f"{rec_name(spec)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def launch(spec, d, seed, n_ens):
    env = dict(os.environ, AR_RESULTS_DIR=str(d), AR_SEED=str(seed),
               N_ENSEMBLE=str(n_ens), WANDB_MODE=os.environ.get("WANDB_MODE", "offline"))
    tag = "" if d == SCREEN_DIR else "confirm_"
    f = open(LOGS / f"{tag}{rec_name(spec)}.log", "w")
    p = subprocess.Popen([PY, "-u", str(HERE / "run3.py"), spec], cwd=HERE, env=env,
                         stdout=f, stderr=subprocess.STDOUT)
    p._f, p._spec, p._d = f, spec, d
    return p


def pool(jobs, workers, label):
    """jobs: list of (spec, dir, seed, n_ens)."""
    todo = [j for j in jobs if result(j[0], j[1]) is None]
    print(f"[{label}] {len(todo)} to run: {[j[0] for j in todo]}", flush=True)
    running = []
    while todo or running:
        while todo and len(running) < workers:
            running.append(launch(*todo.pop(0)))
        time.sleep(20)
        for p in list(running):
            if p.poll() is None:
                continue
            p._f.close()
            running.remove(p)
            r = result(p._spec, p._d)
            if r is None:
                print(f"[{label}] FAILED {p._spec} rc={p.returncode}", flush=True)
            else:
                print(f"[{label}] done {p._spec:28s} f1={r['mean']['f1']:.4f} "
                      f"verdict={r.get('verdict')} +{r.get('improves')} "
                      f"-{r.get('regresses')}", flush=True)
        leaderboard()


def leaderboard():
    rows = []
    for p in sorted(SCREEN_DIR.glob("*.json")):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and "mean" in r:
            rows.append({"name": r["name"], "verdict": r.get("verdict", "baseline"),
                         "improves": r.get("improves"), "regresses": r.get("regresses"),
                         **{k: round(v, 4) for k, v in r["mean"].items()}})
    (SCREEN_DIR / "_leaderboard.json").write_text(json.dumps(rows, indent=1))


def promising(d=SCREEN_DIR):
    out = []
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict):  # _leaderboard.json is a list
            continue
        if r.get("verdict") in ("ADOPTABLE", "tradeoff") and not r.get("is_bound"):
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--round", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))

    base = [("baseline", SCREEN_DIR, 0, 3), ("baseline_s1", SCREEN_DIR, 1, 3)]
    pool(base, 2, "round0")

    if a.round in (0, 1):
        pool([(s, SCREEN_DIR, 0, 3) for s in QUEUE], a.workers, "round1")

    if a.round in (0, 2):
        specs = []
        for r in promising():
            specs += SWEEPS.get(r["name"], [])
        specs += COMBOS
        if os.environ.get("AR3_CONSENSUS") == "1":  # arms defined only once W1 data is reviewed
            specs = CONSENSUS_QUEUE + specs
        pool([(s, SCREEN_DIR, 0, 3) for s in specs], a.workers, "round2")

    if a.round in (0, 3):
        CONFIRM_DIR.mkdir(exist_ok=True)
        # confirm baselines use seed sets 1 and 2: 1 is the one the arms re-run
        # on, 2 exists only to give this directory its own noise floor
        pool([("baseline", CONFIRM_DIR, 1, 5), ("baseline_s1", CONFIRM_DIR, 2, 5)],
             2, "confirm0")
        # records store the resolved NAME; map back to the SPEC run3 understands
        known = QUEUE + CONSENSUS_QUEUE + COMBOS + [x for v in SWEEPS.values() for x in v]
        spec_of = {rec_name(x): x for x in known}
        top = [spec_of.get(r["name"], r["name"]) for r in promising()]
        pool([(n, CONFIRM_DIR, 1, 5) for n in top], a.workers, "confirm")
    leaderboard()
    print("[loop3] complete", flush=True)


if __name__ == "__main__":
    main()

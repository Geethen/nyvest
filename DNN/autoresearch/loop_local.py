"""The locality round's autoresearch loop.

Reuses loop.py's pool, page refresh and state machinery; what is different is
the plan. Round 3a runs the eleven mechanisms; round 3b generates follow-ups
from what they report.

The SWEEPS below are written down BEFORE round 3a reports, which is the only
thing that separates a follow-up from fishing. Two of them are controls rather
than sweeps, and they are the ones that decide what a win would mean:

  moe_soup@n_regions=1   identical extra fine-tuning with no regional split, so
                         a soup gain has to be locality and not extra epochs.
  moe_shared@zero_init=0 routed experts starting from random instead of from
                         zero, which is what every earlier MoE here did. If the
                         zero-init arm wins and this one loses, "start at the
                         global model" was the load-bearing part.

Run:  python loop_local.py               (all of it, 3 GPU workers)
      python loop_local.py --round 1     (mechanisms only)
"""

from __future__ import annotations

import argparse
import json

import ar_common as ac
import loop as LP
import trials as T

# ---------------------------------------------------------- pre-registered
SWEEPS = {
    # granularity: DeepSeekMoE's claim is that MANY SMALL experts beat few large
    # ones, because a fine partition lets a row combine specialists instead of
    # picking one. 4 -> 8 -> 16 walks exactly that axis at fixed total capacity.
    "moe_shared": ["moe_shared@n_experts=8,top_k=2",
                   "moe_shared@n_experts=16,top_k=4",
                   "moe_shared@expert_h1=128,expert_h2=64",
                   "moe_shared@zero_init=0"],
    "moe_terrain": ["moe_terrain@n_experts=8,top_k=2"],
    "moe_lora": ["moe_lora@rank=16", "moe_lora@n_experts=8,top_k=2"],
    # how much of the model the datastore is allowed to replace, and how many
    # neighbours a decision rests on
    "knn_retrieval": ["knn_retrieval@k=8", "knn_retrieval@k=128"],
    "mod_depth": ["mod_depth@capacity=0.25", "mod_depth@capacity=0.75"],
    "film_hyper": ["film_hyper@ctx_src=content", "film_hyper@hyper_hidden=128"],
    "moe_soup": ["moe_soup@n_regions=1", "moe_soup@n_regions=8"],
    "moe_lossfree": ["moe_lossfree@bias_gamma=0.01"],
    "moe_auxbal": ["moe_auxbal@aux_lam=0.001"],
    "moe_echoice": [], "moe_geo": [],
}

# The controls run whatever happens — they are how the round stays
# interpretable when (as is likely) everything ties.
ALWAYS = ["ctrl_capacity", "bound_region_oracle", "moe_soup@n_regions=1",
          "moe_shared@zero_init=0"]


def plan_followups():
    recs = [LP.result(n) for n in T.QUEUE_LOCAL]
    recs = [r for r in recs if r and "delta_mean" in r]
    recs.sort(key=lambda r: r["delta_mean"], reverse=True)
    promising = [r for r in recs if r["delta_mean"] >= LP.FOLLOWUP_FLOOR]
    print(f"[3b] {len(promising)}/{len(recs)} above the {LP.FOLLOWUP_FLOOR} "
          "follow-up floor: "
          + (", ".join(f"{r['name']}({r['delta_mean']:+.4f})" for r in promising)
             or "none"))

    specs = list(ALWAYS)
    for r in promising:
        specs += SWEEPS.get(r["name"], [])

    # If nothing cleared the floor the round is a null, and the informative
    # follow-up is not another sweep of a flat mechanism — it is the granularity
    # axis on the flagship plus the two controls, which is what tells us whether
    # the null is "locality does nothing" or "this much locality does nothing".
    if not promising:
        specs += ["moe_shared@n_experts=16,top_k=4"]

    # combos across different hooks, minus the pairs whose semantics are not
    # defined: retrieval needs a representation it knows how to read, and two
    # fold overrides cannot both own the fold.
    best_build = next((r["name"] for r in recs
                       if T.TRIALS[r["name"]].build_fn is not None), None)
    if best_build and LP.result("knn_retrieval"):
        kn = LP.result("knn_retrieval")
        if kn["delta_mean"] >= LP.FOLLOWUP_FLOOR and recs[0]["name"] == best_build:
            specs.append(f"combo:{best_build}+knn_retrieval")

    seen, out = set(), []
    for s in specs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--round", type=int, default=0, help="0 = both, 1 = mechanisms")
    args = ap.parse_args()

    if not LP.done("baseline"):
        raise SystemExit("no baseline in results/ — run loop.py first")

    if args.round in (0, 1):
        LP.run_pool(T.QUEUE_LOCAL, args.workers, "3a")
    if args.round in (0, 2):
        LP.run_pool(plan_followups(), args.workers, "3b")

    LP.refresh_page()
    LP._save_state()
    recs = [LP.result(n) for n in T.QUEUE_LOCAL]
    recs = [r for r in recs if r and "delta_mean" in r]
    print("\n[locality] round complete")
    for r in sorted(recs, key=lambda r: -r["delta_mean"]):
        print(f"  {r['name']:16s} F1={r['f1_mean']:.4f} Δ={r['delta_mean']:+.4f} "
              f"{r['verdict']:5s} folds={r['delta_per_fold']}")
    print(json.dumps(json.loads(LP.STATE.read_text()), indent=2))


if __name__ == "__main__":
    main()

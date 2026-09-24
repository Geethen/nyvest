"""Re-derive stored verdicts from stored numbers.

The verdict rule originally sent any mean above the +0.003 bar that was NOT
positive on every fold to LOSS, so strongly POSITIVE results were published as
losses (moe_shared@n_experts=8 at +0.0040, @n_experts=16 at +0.0033). ar_common
now emits `split` for that case, but a loop process that imported the old module
keeps writing the old label until it restarts — so this recomputes the verdict
for every record from its own delta_mean / all_folds_positive / is-bound status.

Pure relabelling: no score is touched and nothing is re-run.

    python fix_verdicts.py [--dry-run]
"""

from __future__ import annotations

import json
import sys

import ar_common as ac

BOUNDS = {"bound_region_oracle", "dr_oracle_offsets"}


def verdict_for(rec):
    if rec["name"] == "baseline":
        return "baseline"
    if rec.get("verdict") == "upper-bound" or rec["name"] in BOUNDS:
        return "upper-bound"
    d = rec["delta_mean"]
    if rec.get("all_folds_positive") and d > 0.003:
        return "WIN"
    if abs(d) <= 0.003:
        return "tie"
    return "split" if d > 0.003 else "LOSS"


def main():
    dry = "--dry-run" in sys.argv
    changed = 0
    for p in sorted(ac.RESULTS_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        rec = json.loads(p.read_text())
        if not isinstance(rec, dict) or "delta_mean" not in rec or "name" not in rec:
            continue
        want = verdict_for(rec)
        if want != rec.get("verdict"):
            print(f"  {rec['name']:36s} {rec.get('verdict'):>11s} -> {want:<11s} "
                  f"(Δ{rec['delta_mean']:+.4f} folds {rec['delta_per_fold']})")
            if not dry:
                rec["verdict"] = want
                p.write_text(json.dumps(rec, indent=2))
            changed += 1
    print(f"{'would relabel' if dry else 'relabelled'} {changed} record(s)")


if __name__ == "__main__":
    main()

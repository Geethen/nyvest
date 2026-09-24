"""Backfill inner-val F1 into result JSONs written before the harness stored it.

The val-minus-test gap is the load-bearing number for the locality round, and
every earlier round printed it to the log without recording it. Parsing the logs
recovers it exactly — these are the same numbers the run reported, not a re-run
or an estimate — so the panel can span the whole program instead of only the
trials that happened to run after the field was added.
"""

from __future__ import annotations

import json
import re

import ar_common as ac

FOLD = re.compile(r"^  fold (\d+): F1=([\d.]+).*?\(val=([\d.]+)", re.M)


def main():
    n = 0
    for p in sorted(ac.RESULTS_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name == "diag_locality.json":
            continue
        rec = json.loads(p.read_text())
        if "val_f1_per_fold" in rec:
            continue
        log = ac.LOGS_DIR / f"{rec['name']}.log"
        if not log.exists():
            continue
        m = FOLD.findall(log.read_text())
        if len(m) != len(rec["f1_per_fold"]):
            continue
        # only trust the log if its test scores match the record exactly
        if any(abs(float(t) - f) > 1e-4 for (_, t, _), f in zip(m, rec["f1_per_fold"])):
            print(f"  skip {rec['name']}: log does not match the record")
            continue
        vals = [round(float(v), 4) for _, _, v in m]
        rec["val_f1_per_fold"] = vals
        rec["val_f1_mean"] = round(sum(vals) / len(vals), 4)
        p.write_text(json.dumps(rec, indent=2))
        n += 1
    print(f"backfilled val F1 into {n} records")


if __name__ == "__main__":
    main()

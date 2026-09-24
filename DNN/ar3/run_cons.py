"""Consensus round: own baseline pair (hard-area + real-change metrics need a
baseline that saw data/consensus/), then the consensus arms and their controls.
cons_add must finish before the matched-count controls start."""
from pathlib import Path
import loop3 as L

D = L.HERE / "results_cons"
D.mkdir(exist_ok=True)
L.pool([("baseline", D, 0, 3), ("baseline_s1", D, 1, 3)], 2, "cons0")
L.pool([("cons_add", D, 0, 3), ("cons_add_w05", D, 0, 3), ("cons_ext_majority", D, 0, 3)], 3, "cons1")
L.pool([("ctrl_hard_gk", D, 0, 3), ("ctrl_rand_cons", D, 0, 3)], 2, "cons2")
print("[cons] complete", flush=True)

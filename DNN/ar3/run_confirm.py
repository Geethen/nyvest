"""Confirmation: the round-1/2 candidates at N_ENSEMBLE=5 on FRESH seed sets.
Baselines at seeds 1 and 2 give this directory its own noise floor; arms run
at seed 1. data/consensus/ is present, so hard-area and real-change metrics
are measured here too — the check the stable-point metrics cannot make."""
import loop3 as L

D = L.CONFIRM_DIR
D.mkdir(exist_ok=True)
L.pool([("baseline", D, 1, 5), ("baseline_s1", D, 2, 5)], 2, "confirm0")
ARMS = ["combo:tmp_ctx_mean+cal_ls0", "combo:tmp_pair+cal_ls0", "tmp_ctx_mean",
        "tmp_pair", "cal_ls0", "tmp_hmm", "tmp_joint"]
L.pool([(a, D, 1, 5) for a in ARMS], 3, "confirm")
print("[confirm] complete", flush=True)

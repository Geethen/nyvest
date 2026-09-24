"""Run one ar3 trial:  python run3.py <trial|trial@k=v,...>   (AR_SEED, N_ENSEMBLE via env)"""
import os
import sys
import traceback

import ar3_common as A
import trials3 as T


def main():
    if len(sys.argv) < 2:
        for n, t in T.TRIALS.items():
            print(f"  {n:24s} [{t.tier}] {t.idea}")
        raise SystemExit(1)
    trial = T.resolve(sys.argv[1])
    run = None
    if os.environ.get("WANDB_MODE") != "disabled":
        import wandb
        run = wandb.init(project="nyvest-dnn-ar3", name=f"{trial.name}_s{A.SEED}",
                         group=trial.tier, reinit=True,
                         notes=f"{trial.idea}\n\nHYPOTHESIS: {trial.hypothesis}",
                         config={"trial": trial.name, "seed": A.SEED,
                                 "n_ensemble": trial.n_ensemble, **trial.config})
    try:
        rec = A.run_trial(trial, wandb_run=run)
    except Exception:
        traceback.print_exc()
        if run:
            run.finish(exit_code=1)
        raise
    if run:
        run.summary.update({f"mean_{k}": v for k, v in rec["mean"].items()})
        run.summary["verdict"] = rec.get("verdict", "baseline")
        run.finish()


if __name__ == "__main__":
    main()

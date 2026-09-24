"""Run one trial by name, log it to Weights & Biases, save the record.

    python run.py pair_margin
    WANDB_MODE=offline python run.py kan
"""

from __future__ import annotations

import sys
import traceback

import ar_common as ac
import trials as T


def main():
    if len(sys.argv) < 2:
        print("usage: run.py <trial|trial@k=v,...|combo:a+b>\navailable:")
        for n, t in T.TRIALS.items():
            print(f"  {n:20s} [{t.tier}] {t.idea}")
        raise SystemExit(1)
    trial = T.resolve(sys.argv[1])

    import wandb
    run = wandb.init(
        project=ac.WANDB_PROJECT, name=trial.name, group=trial.tier,
        job_type=trial.tier, reinit=True,
        tags=[trial.tier] + (["transductive"] if trial.transductive else []),
        notes=f"{trial.idea}\n\nHYPOTHESIS: {trial.hypothesis}"
              + (f"\n\nPROVENANCE: {trial.provenance}" if trial.provenance else ""),
        config={
            "trial": trial.name, "tier": trial.tier, "idea": trial.idea,
            "hypothesis": trial.hypothesis, "provenance": trial.provenance,
            "val_mode": trial.val_mode, "weight_mode": trial.weight_mode,
            "n_ensemble": trial.n_ensemble, "label_smooth": trial.label_smooth,
            "transductive": trial.transductive,
            "relabel": ac.RELABEL, "hidden": list(ac.HIDDEN),
            "dropout": ac.DROPOUT, "lr": ac.LR, "weight_decay": ac.WEIGHT_DECAY,
            "batch": ac.BATCH, "patience": ac.PATIENCE,
            **trial.config,
        })
    try:
        rec = ac.run_trial(trial, wandb_run=run)
    except Exception:
        traceback.print_exc()
        run.summary["failed"] = True
        run.finish(exit_code=1)
        raise

    summary = {k: v for k, v in rec.items()
               if isinstance(v, (int, float, bool, str))}
    for c, v in rec["f1_per_class"].items():
        summary[f"f1_class_{c}"] = v
    for c, v in rec.get("delta_per_class", {}).items():
        summary[f"delta_class_{c}"] = v
    run.summary.update(summary)

    # A per-class table makes the tradeoff visible at a glance in the UI.
    if "delta_per_class" in rec:
        tbl = wandb.Table(columns=["class", "kind", "f1", "delta_vs_baseline"])
        for c, v in rec["f1_per_class"].items():
            kind = ("weak" if int(c) in ac.WEAK_CLASSES else
                    "strong" if int(c) in ac.STRONG_CLASSES else "mid")
            tbl.add_data(int(c), kind, v, rec["delta_per_class"][c])
        run.log({"per_class": tbl})
    run.finish()


if __name__ == "__main__":
    main()

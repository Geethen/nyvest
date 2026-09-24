#!/usr/bin/env bash
# ONE sequential runner for everything remaining. No wait-loops, no pgrep.
# Previous approach spawned 4 concurrent runners that all polled overlapping
# patterns, released together, and collided on one GPU (two duplicate
# exp_combine.py workers racing on the same output file). A single script that
# runs jobs in order cannot have that failure mode.
set -u
cd "$(dirname "$0")/../.."
PY=~/myprojects/recover/.venv/bin/python
L=DNN/research/logs

j() { local name=$1; shift; local script=$1; shift
  echo "### START $name  $(date +%H:%M:%S)"
  env "$@" $PY -u "$script" > $L/$name.log 2>&1
  local rc=$?
  echo "### END   $name rc=$rc  $(grep -oP 'F1 mean=\K[0-9.]+' $L/$name.log | head -1)"
}

# --- tau sweep: tau=1 was an untuned-hyperparameter artifact (cls12 prior 0.002
#     gets +6.2 logits vs +1.3 for common classes). Baseline already applies sqrt
#     class weights, so expect SMALL tau to be best.
j logitadj_post_tau0.25 DNN/research/exp_logitadj.py MODE=post TAU=0.25
j logitadj_post_tau0.5  DNN/research/exp_logitadj.py MODE=post TAU=0.5
# --- ensembling: the one lever that has ever worked here
j ens_logitavg   DNN/research/exp_ensemble.py EXP=logitavg
j ens_tempscale  DNN/research/exp_ensemble.py EXP=tempscale
j ens_entropy    DNN/research/exp_ensemble.py EXP=entropy
j ens_bigens     DNN/research/exp_ensemble.py EXP=bigens N_ENS=15
# --- heterogeneous ensembles (must run AFTER ens_bigens: it is the
#     size-matched control that separates diversity from member count)
j combine_arch   DNN/research/exp_combine.py VARIANT=arch N_PER=3
j combine_all    DNN/research/exp_combine.py VARIANT=all  N_PER=3
# --- diagnostic last: oracle bound on any per-row routing scheme
j diag_ensemble  DNN/research/diag_ensemble.py
echo "### ALL COMPLETE $(date +%H:%M:%S)"

#!/usr/bin/env bash
# Stage-3 lever ablation. Each run drops ONE lever from the winning config
# (mixup=0.2, noise=0.1, 5-seed ensemble, cls12 clean) and writes its own JSON.
set -e
PY=~/myprojects/recover/.venv/bin/python
cd "$(dirname "$0")/.."
run() {  # name  ENV...
  name="$1"; shift
  echo "=== ablation: $name ==="
  env "$@" OUT_TAG="$name" $PY DNN/ablation_one.py "$name"
}
run no_mixup     MIXUP_ALPHA=0
run no_noise     INPUT_NOISE=0
run no_clean     CLEAN_CLS12=0
run single_seed  N_ENSEMBLE=1
run no_reg       MIXUP_ALPHA=0 INPUT_NOISE=0
echo "=== ablation done ==="

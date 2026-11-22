#!/bin/bash
# Optional second worker: complete seed 101 and proxy-dynamics runs early.
# Shared Trainer file locks prevent duplicate execution if the main worker catches up.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
source experiments/07_literature_strict/power_guard.sh
mkdir -p outputs/literature_strict/logs
for method in long_random long_mid long_fd long_fd_mid pool_random; do
  ensure_power || exit 2
  printf '%s backfill starting %s seed 101\n' "$(date '+%F %T')" "$method"
  "$EXPERIMENT_PYTHON" experiments/04_len_forgetting/train.py \
    --selection "results/literature_strict/selections/${method}__s101.json" \
    --training-seed 101 --output results/literature_strict/runs \
    --checkpoint-root outputs/literature_strict/checkpoints \
    >> "outputs/literature_strict/logs/${method}__s101.log" 2>&1
  printf '%s backfill completed %s seed 101\n' "$(date '+%F %T')" "$method"
done
bash experiments/07_literature_strict/run_gsnr.sh

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
source experiments/07_literature_strict/power_guard.sh
mkdir -p outputs/literature_strict/logs
for seed in 17 42 101; do
  ensure_power || exit 2
  printf '%s starting long_gsnr seed %s\n' "$(date '+%F %T')" "$seed"
  "$EXPERIMENT_PYTHON" experiments/04_len_forgetting/train.py \
    --selection "results/literature_strict/selections/long_gsnr__s${seed}.json" \
    --training-seed "$seed" --output results/literature_strict/runs \
    --checkpoint-root outputs/literature_strict/checkpoints \
    >> "outputs/literature_strict/logs/long_gsnr__s${seed}.log" 2>&1
  printf '%s completed long_gsnr seed %s\n' "$(date '+%F %T')" "$seed"
done

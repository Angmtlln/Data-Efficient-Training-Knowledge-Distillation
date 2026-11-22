#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
source experiments/07_literature_strict/power_guard.sh
mkdir -p outputs/literature_strict/logs
for seed in 17 42 101; do
  for method in long_random long_mid long_fd long_fd_mid pool_random; do
    ensure_power || exit 2
    printf '%s starting %s seed %s\n' "$(date '+%F %T')" "$method" "$seed"
    "$EXPERIMENT_PYTHON" experiments/04_len_forgetting/train.py \
      --selection "results/literature_strict/selections/${method}__s${seed}.json" \
      --training-seed "$seed" --output results/literature_strict/runs \
      --checkpoint-root outputs/literature_strict/checkpoints \
      >> "outputs/literature_strict/logs/${method}__s${seed}.log" 2>&1
    printf '%s completed %s seed %s\n' "$(date '+%F %T')" "$method" "$seed"
  done
done

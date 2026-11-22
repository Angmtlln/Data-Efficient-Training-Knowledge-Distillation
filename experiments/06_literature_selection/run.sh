#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p outputs/literature_selection/logs
for seed in 17 42 101; do
  for method in long_random long_mid long_fd long_fd_mid pool_random; do
    battery=$(pmset -g batt)
    if ! printf '%s\n' "$battery" | grep -q 'AC Power'; then
      echo 'Stopped before the next job: external power is absent.' >&2
      exit 2
    fi
    printf '%s starting %s seed %s\n' "$(date '+%F %T')" "$method" "$seed"
    "$EXPERIMENT_PYTHON" experiments/04_len_forgetting/train.py \
      --selection "results/literature_selection/selections/${method}__s${seed}.json" \
      --training-seed "$seed" --output results/literature_selection/runs \
      --checkpoint-root outputs/literature_selection/checkpoints \
      >> "outputs/literature_selection/logs/${method}__s${seed}.log" 2>&1
    printf '%s completed %s seed %s\n' "$(date '+%F %T')" "$method" "$seed"
  done
done

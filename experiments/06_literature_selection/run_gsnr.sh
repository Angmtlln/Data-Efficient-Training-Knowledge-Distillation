#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p outputs/literature_selection/logs
for seed in 17 42 101; do
  battery=$(pmset -g batt)
  if ! printf '%s\n' "$battery" | grep -q 'AC Power'; then
    echo 'Stopped before next job: external power is absent.' >&2
    exit 2
  fi
  printf '%s starting long_gsnr seed %s\n' "$(date '+%F %T')" "$seed"
  "$EXPERIMENT_PYTHON" experiments/04_len_forgetting/train.py \
    --selection "results/literature_selection/selections/long_gsnr__s${seed}.json" \
    --training-seed "$seed" --output results/literature_selection/runs \
    --checkpoint-root outputs/literature_selection/checkpoints \
    >> "outputs/literature_selection/logs/long_gsnr__s${seed}.log" 2>&1
  printf '%s completed long_gsnr seed %s\n' "$(date '+%F %T')" "$seed"
done

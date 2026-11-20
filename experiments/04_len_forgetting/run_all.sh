#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p outputs/len_forgetting/logs
for seed in 17 42 101; do
  for method in len8_matched_random len8_forgetting len8 random; do
    echo "$(date '+%F %T') Starting $method seed $seed"
    "$EXPERIMENT_PYTHON" experiments/04_len_forgetting/train.py \
      --selection "results/len_forgetting/selections/${method}__s${seed}.json" \
      --training-seed "$seed" >> "outputs/len_forgetting/logs/${method}__s${seed}.log" 2>&1 || {
        echo "FAILED $method seed $seed, see outputs/len_forgetting/logs/${method}__s${seed}.log" >&2
        exit 1
      }
    echo "$(date '+%F %T') Completed $method seed $seed"
  done
done

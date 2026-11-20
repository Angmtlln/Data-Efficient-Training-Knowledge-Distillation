#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
worker="${1:?worker index}"
workers="${WORKER_COUNT:-2}"
if ! [[ "$workers" =~ ^[1-9][0-9]*$ && "$worker" =~ ^[0-9]+$ ]] || (( worker >= workers )); then
  echo 'worker must be in [0, WORKER_COUNT)' >&2; exit 2
fi
: "${EXPERIMENT_PYTHON:=python}"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p outputs/proxy_mixture/logs
job=0
for seed in 17 42 101; do
  for method in mix_random mix_hard mix_uncertain mix_mid mix_easy mix_rare mix_diverse; do
    current=$job
    job=$((job+1))
    if (( current % workers != worker )); then continue; fi
    printf '%s worker %s starting %s seed %s\n' "$(date '+%F %T')" "$worker" "$method" "$seed"
    "$EXPERIMENT_PYTHON" experiments/04_len_forgetting/train.py \
      --selection "results/proxy_mixture/selections/${method}__s${seed}.json" \
      --training-seed "$seed" --output results/proxy_mixture/runs \
      --checkpoint-root outputs/proxy_mixture/checkpoints \
      >> "outputs/proxy_mixture/logs/${method}__s${seed}.log" 2>&1 || {
        printf 'FAILED %s seed %s; check log\n' "$method" "$seed" >&2; exit 1;
      }
    "$EXPERIMENT_PYTHON" experiments/05_proxy_mixture/prune_completed.py --method "$method" --seed "$seed"
    printf '%s worker %s completed %s seed %s\n' "$(date '+%F %T')" "$worker" "$method" "$seed"
  done
done

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
: "${ORIGINAL_SST_TEST:?set ORIGINAL_SST_TEST to the frozen public SST binary test file}"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/eval_original_sst.py --test "$ORIGINAL_SST_TEST" --check-only
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/eval_original_sst.py --test "$ORIGINAL_SST_TEST"
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/analyze_original_sst.py --test "$ORIGINAL_SST_TEST"
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/report.py

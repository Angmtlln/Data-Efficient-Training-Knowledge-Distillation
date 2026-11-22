#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EXPERIMENT_PYTHON:=python}"
: "${IMDB_SOURCE:?set IMDB_SOURCE to the frozen local IMDb test dataset}"
bash experiments/07_literature_strict/run.sh
bash experiments/07_literature_strict/run_gsnr.sh
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/analyze.py
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/eval_imdb_holdout.py --source "$IMDB_SOURCE" --check-only
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/eval_imdb_holdout.py --source "$IMDB_SOURCE"
"$EXPERIMENT_PYTHON" experiments/07_literature_strict/analyze_imdb_holdout.py --source "$IMDB_SOURCE"

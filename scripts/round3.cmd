@echo off
rem Round 3: final hybrid, extra seeds for the top strategies, cheap scores for CCS.
rem Run from the project root:  scripts\round3.cmd
setlocal

echo.
echo ===== [1/5] cheap scores: 1 epoch on random 10%%, score the whole pool =====
uv run python -m src.analysis.cheap_scores || goto :error

echo.
echo ===== [2/5] merge cheap scores into data/scores.parquet =====
uv run python -m src.analysis.scores > nul || goto :error

echo.
echo ===== [3/5] subsets on cheap scores =====
uv run python -m src.selection.strategies --seeds 0 1 2 --only ccs_b10_cheap ccs_len8_cheap || goto :error

echo.
echo ===== [4/5] training: hybrid (5 seeds), top-5 seeds 3-4, cheap CCS (3 seeds) =====
uv run python scripts/run_experiments.py --strategies ccs_len8 --seeds 0 1 2 3 4 || goto :error
uv run python scripts/run_experiments.py --strategies len_ge8 clean_len8 low_ppl ccs_b10 kcenter --seeds 3 4 || goto :error
uv run python scripts/run_experiments.py --strategies ccs_b10_cheap ccs_len8_cheap --seeds 0 1 2 || goto :error

echo.
echo ===== [5/5] report =====
uv run python -m src.analysis.report || goto :error

echo.
echo DONE
exit /b 0

:error
echo.
echo FAILED at the step above (exit code %errorlevel%)
exit /b 1

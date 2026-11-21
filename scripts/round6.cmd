@echo off
rem Round 6: source-sentence selection (nested phrases) and LLM-as-judge quality.
rem Run from the project root:  scripts\round6.cmd
setlocal

echo.
echo ===== [1/4] scores: source sentences, LLM judge (Qwen2.5-3B, ~45 min) =====
uv run python -m src.analysis.nested || goto :error
if not exist data\llm_scores.parquet uv run python -m src.analysis.llm_scores || goto :error
uv run python -m src.analysis.scores || goto :error

echo.
echo ===== [2/4] subsets =====
uv run python -m src.selection.strategies --only roots one_per_sent one_per_sent_len8 all_per_sent roots_mix50 --seeds 0 1 2 3 4 || goto :error
uv run python -m src.selection.strategies --only llm_quality_top llm_quality_top_bal llm_top_bal llm_filter_bal llm_clean llm_filter_len8_bal --seeds 0 1 2 3 4 || goto :error

echo.
echo ===== [3/4] training =====
uv run python scripts/run_experiments.py --strategies roots one_per_sent one_per_sent_len8 all_per_sent roots_mix50 --seeds 0 1 2 3 4 || goto :error
uv run python scripts/run_experiments.py --strategies llm_quality_top llm_quality_top_bal llm_top_bal llm_filter_bal llm_clean llm_filter_len8_bal --seeds 0 1 2 || goto :error

echo.
echo ===== [4/4] report =====
uv run python -m src.analysis.report || goto :error

echo.
echo DONE
exit /b 0

:error
echo.
echo FAILED at the step above (exit code %errorlevel%)
exit /b 1

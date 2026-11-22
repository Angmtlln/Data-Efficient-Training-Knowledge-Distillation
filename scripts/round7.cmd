@echo off
rem Round 7: knowledge distillation on 10%% subsets (random, len_ge8, roots), 3 seeds each.
rem Teachers: llm (Qwen zero-shot, no labels), self (ModernBERT on the same subset), oof (ModernBERT on 2/3 of the pool,
rem reference upper bound). Run from the project root:  scripts\round7.cmd
setlocal

echo.
echo ===== [1/4] teachers: llm, oof =====
uv run python -m src.analysis.teachers || goto :error

echo.
echo ===== [2/4] self-distillation teachers: retrain the baselines and save their probabilities =====
uv run python scripts/run_experiments.py --strategies random len_ge8 roots --seeds 0 1 2 --tag teacher -- --save-probs || goto :error

echo.
echo ===== [3/4] students =====
uv run python scripts/run_experiments.py --strategies random len_ge8 roots --seeds 0 1 2 --tag kd_llm -- --teacher llm || goto :error
uv run python scripts/run_experiments.py --strategies random len_ge8 roots --seeds 0 1 2 --tag kd_self -- --teacher self_{subset} || goto :error
uv run python scripts/run_experiments.py --strategies random len_ge8 roots --seeds 0 1 2 --tag kd_oof -- --teacher oof || goto :error
rem no gold labels at all: train on the LLM's probabilities only
uv run python scripts/run_experiments.py --strategies random roots --seeds 0 1 2 --tag llm_only -- --teacher llm --alpha 0 || goto :error

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

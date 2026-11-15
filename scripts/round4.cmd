@echo off
rem Round 4: domain-matching selection (domain classifier, DSIR) and class-balanced CCS.
rem Run from the project root:  scripts\round4.cmd
setlocal

echo.
echo ===== [1/2] training: domain_clf (5 seeds), dsir, ccs_b10_bal, ccs_b10_cheap_bal (3 seeds) =====
uv run python scripts/run_experiments.py --strategies domain_clf --seeds 0 1 2 3 4 || goto :error
uv run python scripts/run_experiments.py --strategies dsir ccs_b10_bal ccs_b10_cheap_bal --seeds 0 1 2 || goto :error

echo.
echo ===== [2/2] report =====
uv run python -m src.analysis.report || goto :error

echo.
echo DONE
exit /b 0

:error
echo.
echo FAILED at the step above (exit code %errorlevel%)
exit /b 1

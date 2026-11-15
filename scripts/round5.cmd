@echo off
rem Round 5: budget sweep (1%%, 5%%, 20%%) for random, len_ge8, domain_clf, ccs_b10_cheap.
rem The 10%% points are already in results.csv. Run from the project root:  scripts\round5.cmd
setlocal

echo.
echo ===== [1/2] training: 4 strategies x 3 budgets x 3 seeds =====
uv run python scripts/run_experiments.py --strategies random_b01 len_ge8_b01 domain_clf_b01 ccs_b10_cheap_b01 random_b05 len_ge8_b05 domain_clf_b05 ccs_b10_cheap_b05 random_b20 len_ge8_b20 domain_clf_b20 ccs_b10_cheap_b20 --seeds 0 1 2 || goto :error

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

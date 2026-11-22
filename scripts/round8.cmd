@echo off
rem Round 8: consolidate roots. Same-environment baselines, more seeds, length / source controls, budget sweep.
rem Run from the project root:  scripts\round8.cmd   (~44 runs, ~1.5 h; full-data runs take ~10 min each)
setlocal

echo.
echo ===== [1/6] subsets =====
uv run python -m src.selection.strategies --only roots len_ge8 random --seeds 5 6 7 8 9 || goto :error
uv run python -m src.selection.strategies --only len_match longest_child --seeds 0 1 2 3 4 || goto :error
uv run python -m src.selection.strategies --only roots --budget 0.01 --seeds 0 1 2 || goto :error
uv run python -m src.selection.strategies --only roots --budget 0.05 --seeds 0 1 2 || goto :error
uv run python -m src.selection.strategies --only roots --budget 0.20 --seeds 0 1 2 || goto :error

echo.
echo ===== [2/6] controls: full sentences or just their length / source diversity? =====
uv run python scripts/run_experiments.py --strategies len_match longest_child --seeds 0 1 2 3 4 || goto :error

echo.
echo ===== [3/6] seeds 5-9: roots vs len_ge8 vs random =====
uv run python scripts/run_experiments.py --strategies roots len_ge8 random --seeds 5 6 7 8 9 || goto :error

echo.
echo ===== [4/6] budget sweep for roots (1, 5, 20%%; 20%% = all roots + one phrase per other sentence) =====
uv run python scripts/run_experiments.py --strategies roots_b01 roots_b05 roots_b20 --seeds 0 1 2 || goto :error

echo.
echo ===== [5/6] baselines in the current environment =====
rem domain_clf re-runs tell whether the 2026-11-15 evening runs (domain_clf, budget sweep) are from this environment
uv run python scripts/run_experiments.py --strategies domain_clf --seeds 0 1 2 --tag rerun || goto :error
uv run python scripts/run_experiments.py --strategies random len_ge8 --seeds 3 4 --tag rerun || goto :error
uv run python scripts/run_experiments.py --strategies full --seeds 1 2 || goto :error
uv run python scripts/run_experiments.py --strategies full --seeds 0 --tag rerun || goto :error

echo.
echo ===== [6/6] report =====
uv run python -m src.analysis.report || goto :error
uv run python -m src.analysis.same_env || goto :error

echo.
echo DONE
exit /b 0

:error
echo.
echo FAILED at the step above (exit code %errorlevel%)
exit /b 1

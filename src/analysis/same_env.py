"""Compare strategies on runs from one software environment only.

    python -m src.analysis.same_env [--since 2026-11-21]
Training is bit-exact within an environment but not across them: after the uv.lock update of 2026-11-15
the same subset and seed move test acc by up to 0.9 pp. Early baselines are therefore re-trained with
--tag rerun (or --tag teacher); here every (subset, seed) is represented by its latest run, and runs older
than --since are dropped. Outputs results/same_env.csv.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from src.config import RESULTS_CSV, RESULTS_DIR

BASE_TAGS = ["", "teacher", "rerun"]   # tags that train with the plain loss


def env_check(res):
    """Runs repeated in the current environment vs their first run: zero diff = same environment."""
    first = res[res.tag == ""]
    again = res[res.tag.isin(["teacher", "rerun"])]
    m = again.merge(first, on=["subset", "seed"], suffixes=("", "_first"))
    if m.empty:
        return
    m["diff_pp"] = (m.test_acc - m.test_acc_first) * 100
    print("Re-runs vs first run (exact match = the first run is from the current environment):")
    print(m[["subset", "seed", "timestamp_first", "test_acc_first", "test_acc", "diff_pp"]]
          .sort_values("timestamp_first").round(4).to_string(index=False))


def compare(runs, names, ref):
    a = runs[runs.strategy == ref].test_acc
    return {n: (runs[runs.strategy == n].test_acc.mean() - a.mean()) * 100 for n in names}, \
        {n: ttest_ind(runs[runs.strategy == n].test_acc, a, equal_var=False).pvalue if n != ref else np.nan
         for n in names}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2026-11-21", help="drop runs older than this ISO timestamp")
    args = p.parse_args()

    res = pd.read_csv(RESULTS_CSV)
    res["tag"] = res.tag.fillna("")
    res = res[res.tag.isin(BASE_TAGS)]
    env_check(res)

    runs = res.sort_values("timestamp").groupby(["subset", "seed"]).tail(1)
    runs = runs[runs.timestamp >= args.since].copy()
    runs["strategy"] = runs.subset.str.replace(r"_s\d+$", "", regex=True)

    table = runs.groupby("strategy").agg(n=("seed", "size"), test_acc=("test_acc", "mean"),
                                         std=("test_acc", "std"), val_acc=("val_acc", "mean")).reset_index()
    names = table.strategy.tolist()
    for ref in ("random", "roots", "len_ge8"):
        if ref in names:
            d, pv = compare(runs, names, ref)
            table[f"d_vs_{ref}_pp"], table[f"p_vs_{ref}"] = table.strategy.map(d), table.strategy.map(pv)
    if {"random", "full"} <= set(names):
        acc = table.set_index("strategy").test_acc
        table["gap_closed_%"] = (table.test_acc - acc["random"]) / (acc["full"] - acc["random"]) * 100
    table = table.sort_values("test_acc", ascending=False)
    table.to_csv(RESULTS_DIR / "same_env.csv", index=False)
    print(f"\nRuns since {args.since}, latest run per (subset, seed):")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

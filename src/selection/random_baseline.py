"""Random and class-stratified random 10% subsets of the pool.

    python -m src.selection.random_baseline --seeds 0 1 2 3 4
"""
import argparse

import numpy as np

from src.config import BUDGET_FRACTION
from src.data import load_splits, save_subset


def random_subset(n_pool, budget, seed):
    return np.random.default_rng(seed).choice(n_pool, size=budget, replace=False)


def stratified_subset(labels, budget, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    out = []
    for c in np.unique(labels):
        cls_idx = np.flatnonzero(labels == c)
        k = round(budget * len(cls_idx) / len(labels))
        out.append(rng.choice(cls_idx, size=k, replace=False))
    return np.concatenate(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    pool = load_splits()["pool"]
    budget = int(len(pool) * BUDGET_FRACTION)
    for s in args.seeds:
        save_subset(f"random_s{s}", random_subset(len(pool), budget, s))
        save_subset(f"stratified_s{s}", stratified_subset(pool["label"], budget, s))
    print(f"saved random/stratified subsets of size {budget} for seeds {args.seeds}")

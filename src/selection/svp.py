"""One-shot SVP core-set selection using a shared proxy.

Run `python -m src.selection.svp --method uncertainty --seeds 0 1 2`.
For k-center, first generate data/svp_embeddings.npy with src.analysis.svp_proxy.
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.config import BUDGET_FRACTION, DATA_DIR, SUBSETS_DIR


def uncertainty(gold_probability):
    """Binary entropy, invariant to whether the gold class is 0 or 1."""
    p = np.asarray(gold_probability, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def kcenter(x, budget, seed):
    """Exact farthest-first traversal with Euclidean distance, O(n*k*d)."""
    x = np.asarray(x, dtype=np.float32, order="C")
    if x.ndim != 2 or not np.isfinite(x).all() or not 0 < budget <= len(x):
        raise ValueError("invalid embeddings or budget")
    norms = np.einsum('ij,ij->i', x, x)
    selected = np.empty(budget, dtype=np.int64)
    distances = np.full(len(x), np.inf, dtype=np.float32)
    current = int(np.random.default_rng(seed).integers(len(x)))
    for i in range(budget):
        selected[i] = current
        d = np.maximum(norms + norms[current] - 2 * (x @ x[current]), 0)
        np.minimum(distances, d, out=distances)
        distances[selected[:i + 1]] = -np.inf
        if i + 1 < budget:
            current = int(np.argmax(distances))
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['uncertainty', 'kcenter'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--budget', type=float, default=BUDGET_FRACTION)
    args = p.parse_args()
    df = pd.read_parquet(DATA_DIR / 'scores.parquet').sort_values('pool_idx')
    ids = df.pool_idx.to_numpy()
    if not np.array_equal(ids, np.arange(len(df))):
        raise ValueError('scores must contain every pool_idx exactly once')
    budget = int(len(df) * args.budget)
    if not 0 < budget <= len(df):
        raise ValueError('budget must select between 1 and all pool examples')
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    if args.method == 'uncertainty':
        proxy_path = DATA_DIR / 'svp_proxy_scores.parquet'
        if proxy_path.exists():
            proxy = pd.read_parquet(proxy_path).sort_values('pool_idx')
            if not np.array_equal(proxy.pool_idx.to_numpy(), ids):
                raise ValueError('proxy scores do not align with pool')
            gold = proxy.gold_prob_cheap.to_numpy()
        else:
            gold = df.gold_prob_cheap.to_numpy()
        scores = uncertainty(gold)
        np.save(DATA_DIR / 'svp_uncertainty_scores.npy', scores)
        # Deterministic tie-breaking; this is one fixed proxy, not three independent proxies.
        order = np.argsort(-scores, kind='stable')[:budget]
    else:
        x = np.load(DATA_DIR / 'svp_embeddings.npy', mmap_mode='r')
        if len(x) != len(df):
            raise ValueError('embedding rows must correspond to pool_idx')
    for seed in args.seeds:
        idx = order if args.method == 'uncertainty' else kcenter(x, budget, seed)
        path = SUBSETS_DIR / f'svp_{args.method}_s{seed}.npy'
        np.save(path, ids[idx])
        print(f'{path}: {len(idx)} unique={len(np.unique(idx))}, positive={df.label.iloc[idx].mean():.4f}')
    print(f'selection elapsed: {time.perf_counter() - t:.2f}s')


if __name__ == '__main__':
    main()

"""Additional one-shot core-set experiments using exported SVP artifacts."""
import argparse
from pathlib import Path
import time
import numpy as np
from src.selection.svp import kcenter, uncertainty
from src.config import SUBSETS_DIR


def select_forgetting(dynamics, budget, seed):
    """Epoch-level approximation: never learned first, then most forgotten, random ties."""
    correct = np.asarray(dynamics['gold_prob']) > .5
    if correct.ndim != 2 or correct.shape[0] < 2:
        raise ValueError('Need at least two passes over the pool')
    n = correct.shape[1]
    ids = np.asarray(dynamics['pool_idx'])
    if not np.array_equal(np.sort(ids), np.arange(n)) or not 0 < budget <= n:
        raise ValueError('Invalid pool indices or budget')
    forget = (correct[:-1] & ~correct[1:]).sum(axis=0)
    never = ~correct.any(axis=0)
    priority = np.where(never, np.inf, forget)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), -priority))[:budget]
    return ids[order], never, forget


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['forget_epoch', 'normalized_cls', 'embeddings'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    p.add_argument('--budget', type=float, default=.1)
    p.add_argument('--normalize', action='store_true')
    args = p.parse_args()
    if args.method == 'forget_epoch':
        data = np.load(args.input); n = len(data['pool_idx'])
    else:
        data = np.load(args.input); n = len(data)
        if data.ndim != 2 or not np.isfinite(data).all(): raise ValueError('Invalid embedding matrix')
        if args.normalize or args.method == 'normalized_cls':
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            if np.any(norms < 1e-12): raise ValueError('Cannot normalize zero embeddings')
            data = np.ascontiguousarray(data / norms, dtype=np.float32)
    budget = int(n * args.budget)
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        start = time.perf_counter()
        if args.method == 'forget_epoch':
            idx, never, forgotten = select_forgetting(data, budget, seed)
            print('never learned:', never.sum(), 'forgot at least once:', (forgotten > 0).sum(), flush=True)
        else:
            idx = kcenter(data, budget, seed)
        np.save(SUBSETS_DIR / f'{args.name}_s{seed}.npy', idx)
        print(args.name, seed, len(idx), f'{time.perf_counter()-start:.1f} sec', flush=True)

if __name__ == '__main__': main()

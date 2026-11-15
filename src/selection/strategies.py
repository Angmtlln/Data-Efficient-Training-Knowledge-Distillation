"""Subset selection strategies. Each returns `budget` pool indices.

    python -m src.selection.strategies --seeds 0 1 2          # all strategies
    python -m src.selection.strategies --only kcenter clean    # a few
Saves results/subsets/<strategy>_s<seed>.npy
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.config import BUDGET_FRACTION, DATA_DIR
from src.data import save_subset
from src.selection import coreset


def _random_from(candidates, budget, rng):
    candidates = np.asarray(candidates)
    if len(candidates) < budget:
        raise ValueError(f"only {len(candidates)} candidates for budget {budget}")
    return rng.choice(candidates, size=budget, replace=False)


def _top(scores, budget, largest=True):
    order = np.argsort(scores, kind="stable")
    return order[-budget:] if largest else order[:budget]


# ---------- quality ----------
def longest(df, budget, rng):
    # tie-break randomly so that seeds give different subsets among equal lengths
    return _top(df.n_words.to_numpy() + rng.random(len(df)) * 1e-3, budget)


def len_ge8(df, budget, rng):
    return _random_from(np.flatnonzero(df.n_words >= 8), budget, rng)


def low_ppl(df, budget, rng):
    return _random_from(np.flatnonzero(df.ppl <= df.ppl.median()), budget, rng)


def clean(df, budget, rng):
    return _random_from(np.flatnonzero(~df.label_issue), budget, rng)


# ---------- diversity ----------
def kmeans_balanced(df, budget, rng):
    groups = [np.flatnonzero(df.cluster == c) for c in np.unique(df.cluster)]
    per = budget // len(groups)
    out = [rng.choice(g, size=min(per, len(g)), replace=False) for g in groups]
    chosen = np.concatenate(out)
    rest = np.setdiff1d(np.arange(len(df)), chosen)
    return np.concatenate([chosen, rng.choice(rest, size=budget - len(chosen), replace=False)])


def kcenter(df, budget, rng):
    """Greedy k-center (Sener & Savarese, 2018) on MiniLM embeddings."""
    x = torch.from_numpy(np.load(DATA_DIR / "emb_pool.npy").astype(np.float32)).cuda()
    first = int(rng.integers(len(df)))
    chosen = [first]
    min_dist = torch.cdist(x, x[first:first + 1]).squeeze(1)
    for _ in range(budget - 1):
        nxt = int(torch.argmax(min_dist))
        chosen.append(nxt)
        min_dist = torch.minimum(min_dist, torch.cdist(x, x[nxt:nxt + 1]).squeeze(1))
    return np.array(chosen)


# ---------- importance (dataset cartography) ----------
def hard(df, budget, rng):
    return _top(df.confidence.to_numpy(), budget, largest=False)


def easy(df, budget, rng):
    return _top(df.confidence.to_numpy(), budget, largest=True)


def ambiguous(df, budget, rng):
    return _top(df.variability.to_numpy(), budget, largest=True)


def class_balanced(strategy):
    """Apply `strategy` within each class, keeping the pool's label proportions."""
    def wrapped(df, budget, rng):
        labels = df.label.to_numpy()
        out = []
        for c in np.unique(labels):
            cls = np.flatnonzero(labels == c)
            k = round(budget * len(cls) / len(labels))
            local = strategy(df.iloc[cls].reset_index(drop=True), k, rng)
            out.append(cls[local])
        idx = np.concatenate(out)
        return idx[:budget]
    return wrapped


# ---------- hybrids ----------
def restrict(mask_fn, strategy):
    """Run `strategy` only on the pool rows where mask_fn(df) is True."""
    def wrapped(df, budget, rng):
        keep = np.flatnonzero(mask_fn(df))
        return keep[strategy(df.iloc[keep].reset_index(drop=True), budget, rng)]
    return wrapped


def rank_agg(df, budget, rng):
    """Unified score: mean percentile rank of length, label cleanliness (OOF gold prob) and variability."""
    pct = lambda s: s.rank(pct=True)
    score = (pct(df.n_words) + pct(df.oof_gold_prob) + pct(df.variability)) / 3
    return _top(score.to_numpy() + rng.random(len(df)) * 1e-6, budget)


def mix(strategy, frac):
    """`frac` of the budget from `strategy`, the rest uniformly at random from the remaining rows."""
    def wrapped(df, budget, rng):
        k = int(budget * frac)
        first = strategy(df, k, rng)
        rest = np.setdiff1d(np.arange(len(df)), first)
        return np.concatenate([first, rng.choice(rest, size=budget - k, replace=False)])
    return wrapped


not_noisy = lambda df: ~df.label_issue
not_noisy_len5 = lambda df: ~df.label_issue & (df.n_words >= 5)
not_noisy_len8 = lambda df: ~df.label_issue & (df.n_words >= 8)

STRATEGIES = dict(
    longest=longest, len_ge8=len_ge8, low_ppl=low_ppl, clean=clean,
    kmeans=kmeans_balanced, kcenter=kcenter,
    hard=hard, easy=easy, ambiguous=ambiguous,
    hard_bal=class_balanced(hard), easy_bal=class_balanced(easy), ambiguous_bal=class_balanced(ambiguous),
    clean_kmeans=restrict(not_noisy, kmeans_balanced),
    clean_ambiguous_bal=restrict(not_noisy, class_balanced(ambiguous)),
    clean_len5_kmeans=restrict(not_noisy_len5, kmeans_balanced),
    rank_agg=class_balanced(rank_agg),
    clean_len8=restrict(not_noisy_len8, len_ge8),
    mix_ambig20=restrict(not_noisy, mix(class_balanced(ambiguous), 0.2)),
    # literature coreset methods (src/selection/coreset.py)
    ccs_b03=coreset.ccs(beta=0.03),
    ccs_b10=coreset.ccs(beta=0.10),
    moderate=coreset.moderate,
    d2=coreset.d2(beta=0.0),
    d2_b10=coreset.d2(beta=0.10),
)
NEEDS_OOF = {"clean", "clean_kmeans", "clean_ambiguous_bal", "clean_len5_kmeans", "rank_agg",
             "clean_len8", "mix_ambig20"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--only", nargs="+", default=None)
    args = p.parse_args()

    df = pd.read_parquet(DATA_DIR / "scores.parquet")
    budget = int(len(df) * BUDGET_FRACTION)
    for name in args.only or STRATEGIES:
        if name in NEEDS_OOF and "label_issue" not in df:
            print(f"skip {name}: run cv_predict + scores first")
            continue
        for s in args.seeds:
            idx = STRATEGIES[name](df, budget, np.random.default_rng(s))
            assert len(np.unique(idx)) == budget
            save_subset(f"{name}_s{s}", idx)
        sub = df.iloc[idx]
        print(f"{name:10s} words {sub.n_words.mean():5.1f} | pos {sub.label.mean():.3f} | "
              f"conf {sub.confidence.mean():.3f} | clusters {sub.cluster.nunique():3d} | test_sim {sub.test_sim.mean():.3f}")


if __name__ == "__main__":
    main()

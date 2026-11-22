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
from src.selection import coreset, domain, dsir, llm, nested


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
    # same draw as random_baseline.random_subset, so random_s<seed> files are reproduced exactly
    random=lambda df, budget, rng: rng.choice(len(df), size=budget, replace=False),
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
    # final hybrid: domain filter first, then coverage-centric selection
    ccs_len8=restrict(lambda df: df.n_words >= 8, coreset.ccs(beta=0.10)),
    # scalability: the same selection on cheap scores (src/analysis/cheap_scores.py)
    ccs_b10_cheap=coreset.ccs(beta=0.10, score="el2n_cheap"),
    ccs_len8_cheap=restrict(lambda df: df.n_words >= 8, coreset.ccs(beta=0.10, score="el2n_cheap")),
    # is the CCS gain a class-balance effect?
    ccs_b10_bal=class_balanced(coreset.ccs(beta=0.10)),
    ccs_b10_cheap_bal=class_balanced(coreset.ccs(beta=0.10, score="el2n_cheap")),
    # principled version of the length filter: importance resampling towards unlabeled target sentences
    dsir=dsir.dsir,
    domain_clf=domain.domain_clf,
    # LLM-as-judge quality (src/selection/llm.py); top-k / filters within each class
    llm_quality_top=llm.llm_quality_top,
    llm_quality_top_bal=class_balanced(llm.llm_quality_top),
    llm_top_bal=class_balanced(llm.llm_top),
    llm_filter_bal=class_balanced(llm.llm_filter),
    llm_clean=llm.llm_clean,
    llm_filter_len8_bal=restrict(lambda df: df.n_words >= 8, class_balanced(llm.llm_filter)),
    # source sentence of each phrase (src/selection/nested.py): length vs source diversity
    roots=nested.roots,
    one_per_sent=nested.one_per_sent(),
    one_per_sent_len8=nested.one_per_sent(min_words=8),
    all_per_sent=nested.all_per_sent,
    roots_mix50=nested.roots_mix(0.5),
    # is it full sentences or just their length? (round 8)
    len_match=nested.len_match,
    longest_child=nested.longest_child,
)
NEEDS_OOF = {"clean", "clean_kmeans", "clean_ambiguous_bal", "clean_len5_kmeans", "rank_agg",
             "clean_len8", "mix_ambig20"}
NEEDS_CHEAP = {"ccs_b10_cheap", "ccs_len8_cheap", "ccs_b10_cheap_bal"}
NEEDS_COLUMN = {name: "llm_quality" for name in STRATEGIES if name.startswith("llm_")} | {
    name: "sent_id" for name in ("roots", "one_per_sent", "one_per_sent_len8", "all_per_sent", "roots_mix50",
                                 "len_match", "longest_child")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--budget", type=float, default=BUDGET_FRACTION,
                   help="fraction of the pool; non-default budgets are saved as <name>_b<pct>_s<seed>")
    args = p.parse_args()

    df = pd.read_parquet(DATA_DIR / "scores.parquet")
    budget = int(len(df) * args.budget)
    suffix = "" if args.budget == BUDGET_FRACTION else f"_b{round(args.budget * 100):02d}"
    for name in args.only or STRATEGIES:
        if name in NEEDS_OOF and "label_issue" not in df:
            print(f"skip {name}: run cv_predict + scores first")
            continue
        if name in NEEDS_CHEAP and "el2n_cheap" not in df:
            print(f"skip {name}: run cheap_scores + scores first")
            continue
        if name in NEEDS_COLUMN and NEEDS_COLUMN[name] not in df:
            print(f"skip {name}: column {NEEDS_COLUMN[name]} missing, run llm_scores / nested + scores first")
            continue
        for s in args.seeds:
            idx = STRATEGIES[name](df, budget, np.random.default_rng(s))
            assert len(np.unique(idx)) == budget
            save_subset(f"{name}{suffix}_s{s}", idx)
        sub = df.iloc[idx]
        print(f"{name + suffix:10s} n {budget} | words {sub.n_words.mean():5.1f} | pos {sub.label.mean():.3f} | "
              f"conf {sub.confidence.mean():.3f} | clusters {sub.cluster.nunique():3d} | test_sim {sub.test_sim.mean():.3f}")


if __name__ == "__main__":
    main()

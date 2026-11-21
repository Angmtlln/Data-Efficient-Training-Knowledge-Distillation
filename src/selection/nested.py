"""Selection by source sentence (src/analysis/nested.py): separate phrase length from source diversity.

  roots              random full sentences
  one_per_sent       one random phrase from each of `budget` random sentences (max source diversity)
  one_per_sent_len8  the same, among phrases of >= 8 words
  all_per_sent       all phrases of random sentences until the budget is filled (min source diversity)
  roots_mix         `frac` full sentences + one phrase from each of other sentences
Phrases shared by several sentences (n_parents > 1, mostly short and generic) are never picked by the
per-sentence strategies.
"""
import numpy as np


def roots(df, budget, rng):
    return rng.choice(np.flatnonzero(df.is_root), size=budget, replace=False)


def _one_per_sentence(df, sents, rng, mask=None):
    """For each sentence id in `sents`, one random row of df with that sent_id (and mask)."""
    keep = (df.sent_id >= 0) if mask is None else (df.sent_id >= 0) & mask
    rows = np.flatnonzero(keep)
    shuffled = rows[rng.permutation(len(rows))]
    # first occurrence in a random order = a uniformly random phrase of each sentence
    _, pos = np.unique(df.sent_id.to_numpy()[shuffled], return_index=True)
    pick = shuffled[pos]
    by_sent = dict(zip(df.sent_id.to_numpy()[pick], pick))
    return np.array([by_sent[s] for s in sents if s in by_sent])


def one_per_sent(min_words=0):
    """Round robin over sentences: one random phrase from each, then a second one, ... until the budget."""
    def strategy(df, budget, rng):
        rows = np.flatnonzero((df.sent_id >= 0) & (df.n_words >= min_words))
        rows = rows[rng.permutation(len(rows))]
        sent = df.sent_id.to_numpy()[rows]
        # rank of each row within its sentence in the random order: 0 for the first phrase drawn, ...
        order = np.argsort(sent, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(sent[order])) + 1]
        rank = np.empty(len(rows), dtype=np.int64)
        rank[order] = np.arange(len(rows)) - np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
        # all rank-0 rows first (random order among them), then rank 1, ...
        return rows[np.lexsort((rng.random(len(rows)), rank))[:budget]]
    return strategy


def all_per_sent(df, budget, rng):
    sent = df.sent_id.to_numpy()
    out = []
    for s in rng.permutation(np.unique(sent[sent >= 0])):
        out.extend(np.flatnonzero(sent == s))
        if len(out) >= budget:
            break
    return np.array(out[:budget])   # the last sentence is cut to fit the budget


def roots_mix(frac):
    def strategy(df, budget, rng):
        k = int(budget * frac)
        full = rng.choice(np.flatnonzero(df.is_root), size=k, replace=False)
        has_phrase = np.unique(df.sent_id[(df.sent_id >= 0) & ~df.is_root])
        others = np.setdiff1d(has_phrase, df.sent_id.to_numpy()[full])
        sents = rng.choice(others, size=budget - k, replace=False)
        return np.concatenate([full, _one_per_sentence(df, sents, rng, mask=~df.is_root)])
    return strategy

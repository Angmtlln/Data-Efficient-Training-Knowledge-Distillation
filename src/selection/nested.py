"""Selection by source sentence (src/analysis/nested.py): separate phrase length from source diversity.

  roots              random full sentences; above 6 670 (all of them) filled with one phrase per other sentence
  len_match          control for roots: non-root phrases with the same length distribution (~2 per sentence)
  longest_child      control for roots: the longest proper sub-phrase of each of `budget` random sentences,
                     one per sentence like roots but shorter (15.7 vs 19.4 words)
  one_per_sent       one random phrase from each of `budget` random sentences (max source diversity)
  one_per_sent_len8  the same, among phrases of >= 8 words
  all_per_sent       all phrases of random sentences until the budget is filled (min source diversity)
  roots_mix         `frac` full sentences + one phrase from each of other sentences
Phrases shared by several sentences (n_parents > 1, mostly short and generic) are never picked by the
per-sentence strategies.
"""
import numpy as np


def roots(df, budget, rng):
    full = np.flatnonzero(df.is_root)
    if budget <= len(full):
        return rng.choice(full, size=budget, replace=False)
    rest = one_per_sent()(df[~df.is_root].reset_index(drop=True), budget - len(full), rng)
    return np.concatenate([full, np.flatnonzero(~df.is_root)[rest]])


def len_match(df, budget, rng):
    """Non-root phrases drawn to the length histogram of a random roots subset. Lengths with too few non-roots
    (long ones) borrow from the next shorter length. Long phrases come ~2 per sentence: with one per sentence
    the length cannot be matched (that control is longest_child)."""
    want = df.n_words.to_numpy()[roots(df, budget, rng)]
    lengths = df.n_words.to_numpy()
    out, carry = [], 0
    for n in range(want.max(), 0, -1):
        k = (want == n).sum() + carry
        cand = np.flatnonzero(~df.is_root & (lengths == n))
        take = min(k, len(cand))
        out.append(rng.choice(cand, size=take, replace=False))
        carry = k - take
    return np.concatenate(out)


def longest_child(df, budget, rng):
    sub = df[(df.sent_id >= 0) & ~df.is_root]
    # longest phrase per sentence, ties broken at random
    order = np.lexsort((rng.random(len(sub)), sub.n_words.to_numpy()))
    longest = sub.iloc[order].groupby("sent_id").tail(1).index.to_numpy()
    return rng.choice(longest, size=budget, replace=False)


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

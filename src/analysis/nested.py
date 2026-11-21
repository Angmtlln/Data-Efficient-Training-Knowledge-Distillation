"""Recover the source sentence of every pool phrase.

SST-2 train is the set of parse-tree phrases of ~6.9k sentences, so phrases are nested inside each other.
A *root* is a phrase that is not a contiguous token span of any other phrase (pool + val): roots are the full
sentences. A phrase's parents are the roots that contain it; 94% of pool phrases have exactly one.

    python -m src.analysis.nested
Output: data/nested.parquet with
  sent_id    id of the unique parent sentence, -1 if the phrase occurs in several sentences
  n_parents  number of roots containing the phrase
  is_root    the phrase is a full sentence
  root_len   length in words of the parent sentence (NaN when sent_id == -1)
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.data import load_splits


def spans(words, proper=False):
    n = len(words)
    for a in range(n):
        for b in range(a + 1, n + 1):
            if not (proper and b - a == n):
                yield " ".join(words[a:b])


def main():
    splits = load_splits()
    pool = [s.strip() for s in splits["pool"]["sentence"]]
    texts = pool + [s.strip() for s in splits["val"]["sentence"]]
    first = {}
    for i, t in enumerate(texts):
        first.setdefault(t, i)
    words = [t.split() for t in texts]

    contained = set()
    for w in words:
        contained.update(first[s] for s in spans(w, proper=True) if s in first)
    roots = [i for i in set(first.values()) if i not in contained]

    parents = defaultdict(set)
    for r in roots:
        for s in spans(words[r]):
            if s in first:
                parents[first[s]].add(r)

    rows = []
    for t in pool:
        par = parents[first[t]]
        sent = next(iter(par)) if len(par) == 1 else -1
        rows.append((sent, len(par), first[t] in par, len(words[sent]) if sent >= 0 else np.nan))
    df = pd.DataFrame(rows, columns=["sent_id", "n_parents", "is_root", "root_len"])
    df.insert(0, "pool_idx", splits["pool"]["pool_idx"])
    df.to_parquet(DATA_DIR / "nested.parquet")

    print(f"roots (full sentences): {len(roots)}, of them in pool: {df.is_root.sum()}")
    print(f"pool phrases with a unique parent: {(df.sent_id >= 0).mean():.3f}")
    print(f"phrases per sentence: {df[df.sent_id >= 0].groupby('sent_id').size().describe().round(1).to_dict()}")


if __name__ == "__main__":
    main()

"""Replace up to 2% of a roots-only subset with overlapping phrases of opposite sentiment."""

from collections import defaultdict
import html
import unicodedata

import numpy as np


def tokens(text):
    value = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    for old, new in [("-lrb-", "("), ("-rrb-", ")"), ("-lsb-", "["), ("-rsb-", "]"),
                     ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("``", '"'), ("''", '"')]:
        value = value.replace(old, new)
    return tuple(value.split())


def contrast_subset(texts, labels, is_root, base, seed):
    """Keep each selected parent, add its opposite-label span and preserve class counts."""
    labels, base = np.asarray(labels), np.asarray(base)
    rows = [tokens(text) for text in texts]
    lengths = np.array([len(row) for row in rows])
    by_text = defaultdict(list)
    for idx in np.flatnonzero(~is_root & (lengths >= 8)):
        by_text[rows[idx]].append(idx)

    rng = np.random.default_rng(seed)
    candidates = []
    for parent in base:
        row = rows[parent]
        # Keep the longest qualifying span of this parent. Tokens must be contiguous.
        match = None
        for size in range(len(row) - 1, max(8, (3 * len(row) + 4) // 5) - 1, -1):
            matches = [i for start in range(len(row) - size + 1)
                       for i in by_text.get(row[start:start + size], ())
                       if labels[i] != labels[parent]]
            if matches:
                match = int(rng.choice(matches))
                break
        if match is not None:
            candidates.append((parent, match))

    limit = round(len(base) * 0.02)
    chosen = []
    used_texts = set()
    for k in rng.permutation(len(candidates)):
        parent, child = candidates[k]
        if rows[child] not in used_texts:
            chosen.append((parent, child))
            used_texts.add(rows[child])
            if len(chosen) >= limit:
                break

    # Match the removed root to the added phrase by label and closest word count.
    positions = np.arange(len(base))
    available = ~np.isin(base, [parent for parent, child in chosen])
    selected = base.copy()
    audit = []
    for parent, child in chosen:
        options = positions[available & (labels[base] == labels[child])]
        distances = np.abs(lengths[base[options]] - lengths[child])
        slot = rng.choice(options[distances == distances.min()])
        available[slot] = False
        selected[slot] = child
        audit.append((int(parent), int(child), int(base[slot])))
    return selected, audit

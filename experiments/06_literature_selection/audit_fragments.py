"""Quantify shared eight-token fragments; shared n-grams do not prove common origin."""
from collections import defaultdict
import json
import re

import numpy as np
import prepare as base


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def main():
    ds, _ = base.read_data()
    pool = [words(t) for t in ds['pool']['sentence']]
    eligible = np.load(base.ROOT / 'scores.npz')['eligible']
    results = {}
    for split in ('val', 'test'):
        lookup = defaultdict(set)
        for j, text in enumerate(ds[split]['sentence']):
            row = words(text)
            for i in range(len(row) - 7):
                lookup[tuple(row[i:i + 8])].add(j)
        matches = set()
        matched_pool = set()
        for i, row in enumerate(pool):
            if not eligible[i]:
                continue
            hit = set()
            for j in range(len(row) - 7):
                hit.update(lookup.get(tuple(row[j:j + 8]), ()))
            if hit:
                matches.update(hit)
                matched_pool.add(i)
        results[split] = {'examples_with_shared_8gram': len(matches), 'evaluation_examples': len(ds[split]),
                          'eligible_pool_with_shared_8gram': len(matched_pool),
                          'caveat': 'Shared 8-token fragments can be boilerplate; not proof of a shared source.'}
    (base.ROOT / 'fragment_overlap_audit.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

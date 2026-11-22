"""Infer candidate root sentences from pool + inner val, then audit eligibility.

Inspired by the collaborator's round-6 source-sentence analysis. This lexical
containment heuristic is not an authoritative parse tree or parent mapping.
"""
import json
import time

import numpy as np
import prepare as base


def main():
    start = time.time()
    ds, _ = base.read_data()
    pool = ds['pool']['sentence']
    texts = list(pool) + ds['val']['sentence']
    tuples = [tuple(text.strip().split()) for text in texts]
    unique = set(tuples)
    contained = set()
    for row in unique:
        for start_idx in range(len(row)):
            for end_idx in range(start_idx + 1, len(row) + 1):
                if start_idx == 0 and end_idx == len(row):
                    continue
                fragment = row[start_idx:end_idx]
                if fragment in unique:
                    contained.add(fragment)
    roots = unique - contained
    mask = np.array([row in roots for row in tuples[:len(pool)]])
    scores = np.load(base.ROOT / 'scores.npz')
    eligible = scores['eligible']
    lengths = np.asarray([len(s.split()) for s in pool])
    labels = np.asarray(ds['pool']['label'])
    candidates = np.flatnonzero(mask & eligible)
    target = base.ROOT / 'inferred_roots.npz'
    np.savez_compressed(target, is_root=mask)
    report = {'method': 'contiguous whitespace-token containment in original pool + val, no labels',
              'caveat': 'Heuristic roots are not authoritative SST parse trees or guaranteed full sentences.',
              'pool_root_rows': int(mask.sum()), 'eligible_root_rows': len(candidates),
              'eligible_long_root_rows': int((lengths[candidates] >= 8).sum()),
              'eligible_root_positive_fraction': float(labels[candidates].mean()),
              'eligible_root_mean_words': float(lengths[candidates].mean()),
              'root_mask_sha256': base.sha(target), 'seconds': time.time() - start}
    (base.ROOT / 'inferred_roots_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Infer a source proxy through lexical containment and audit selected source diversity."""
from collections import defaultdict
import json
import time

import numpy as np
import prepare as base
from analyze import METHODS, SEEDS


def main():
    start = time.time()
    ds, _ = base.read_data()
    texts = list(ds['pool']['sentence']) + list(ds['val']['sentence'])
    rows = [tuple(t.strip().split()) for t in texts]
    unique = set(rows)
    contained = set()
    for words in unique:
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                if i == 0 and j == len(words):
                    continue
                frag = words[i:j]
                if frag in unique:
                    contained.add(frag)
    roots = sorted(unique - contained)
    root_set = set(roots)
    parent = defaultdict(set)
    for root_id, words in enumerate(roots):
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                frag = words[i:j]
                if frag in unique:
                    parent[frag].add(root_id)
    pool_count = len(ds['pool'])
    counts = np.asarray([len(parent[row]) for row in rows[:pool_count]], dtype=np.int32)
    ids = np.asarray([next(iter(parent[row])) if len(parent[row]) == 1 else -1 for row in rows[:pool_count]], dtype=np.int32)
    val_ids = np.asarray([next(iter(parent[row])) if len(parent[row]) == 1 else -1 for row in rows[pool_count:]], dtype=np.int32)
    inferred_mask = np.load(base.ROOT / 'inferred_roots.npz')['is_root']
    own_mask = np.asarray([row in root_set for row in rows[:pool_count]])
    if not np.array_equal(inferred_mask, own_mask):
        raise ValueError('Source audit disagrees with stored root mask')
    np.savez_compressed(base.ROOT / 'source_parent_proxy.npz', parent_id=ids, parent_count=counts, val_parent_id=val_ids)
    val_parents = set(val_ids[val_ids >= 0].tolist())
    result = []
    for method in METHODS:
        for seed in SEEDS:
            selection = json.loads((base.ROOT / 'selections' / f'{method}__s{seed}.json').read_text())
            indices = np.asarray(selection['indices'])
            unique_parents = ids[indices][ids[indices] >= 0]
            result.append({'name': method, 'seed': seed, 'parent_unique_assignment_fraction': float((ids[indices] >= 0).mean()),
                           'distinct_uniquely_assigned_parents': int(len(np.unique(unique_parents))),
                           'shared_with_uniquely_assigned_val_parent': int(sum(int(i) in val_parents for i in ids[indices] if i >= 0)),
                           'inferred_root_fraction': float(inferred_mask[indices].mean())})
    out = {'definition': 'Unlabelled contiguous-whitespace-token containment inferred from pool + inner val',
           'caveat': 'Lexical candidates are not verified parse trees and short phrases can have several parents.',
           'seconds': time.time() - start, 'roots_total': len(roots),
           'pool_unique_parent_fraction': float((ids >= 0).mean()),
           'val_unique_parent_fraction': float((val_ids >= 0).mean()),
           'val_distinct_unique_parents': len(val_parents), 'selections': result}
    (base.ROOT / 'source_diversity_audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'selections'}, indent=2))


if __name__ == '__main__':
    main()

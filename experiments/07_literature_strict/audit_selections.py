"""Inspect label, length, token cost and overlap without model results."""
import json
from pathlib import Path

from datasets import load_from_disk
import numpy as np

import prepare as base

METHODS = ('pool_random', 'long_random', 'long_mid', 'long_fd', 'long_fd_mid', 'long_gsnr',
           'long_roots_matched', 'long_roots_max')


def main():
    ds, meta = base.read_data()
    tokens = load_from_disk(str(base.REPO / 'experiments/01_cpu_selection/data/tokenized'))['pool']['input_ids']
    token_len = np.asarray([len(t) for t in tokens])
    text = ds['pool']['sentence']
    lengths = np.asarray([len(t.split()) for t in text])
    labels = np.asarray(ds['pool']['label'])
    rows, overlaps = [], []
    for seed in base.SEEDS:
        selections = {}
        for name in METHODS:
            m = json.loads((base.ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
            checksum = m.pop('sha256')
            if base.digest(m) != checksum or m['pool_sha256'] != meta['splits']['pool']['sha256']:
                raise ValueError(f'Selection mismatch {name} {seed}')
            ids = np.asarray(m['indices'], dtype=int)
            selections[name] = ids
            rows.append({'name': name, 'seed': seed, 'n': len(ids), 'positive_fraction': float(labels[ids].mean()),
                         'mean_words': float(lengths[ids].mean()), 'mean_tokens_truncated': float(token_len[ids].mean()),
                         'tokens_total_per_epoch': int(token_len[ids].sum()),
                         'tokenized_at_max64_fraction': float((token_len[ids] >= 64).mean())})
        for idx, name in enumerate(METHODS):
            for other in METHODS[idx + 1:]:
                overlaps.append({'seed': seed, 'a': name, 'b': other,
                                 'intersection': int(np.intersect1d(selections[name], selections[other]).size)})
    (base.ROOT / 'selection_cost_audit.json').write_text(json.dumps({'selections': rows, 'overlap': overlaps}, indent=2) + '\n')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()

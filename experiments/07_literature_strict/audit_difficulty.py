"""Audit proxy difficulty of selected target positions and preserve comparison scope."""
import json
import numpy as np

import prepare as base
from analyze import METHODS, SEEDS


def main():
    ds, _ = base.read_data()
    labels = np.asarray(ds['pool']['label'])
    lengths = np.asarray([len(s.split()) for s in ds['pool']['sentence']])
    scores = np.load(base.ROOT / 'scores.npz')
    gold = scores['gold_probability']
    rows = []
    for seed in SEEDS:
        _, background, _ = base.select(labels, lengths, scores['eligible'], gold, scores['fd_bin'], seed)
        for name in METHODS:
            manifest = json.loads((base.ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
            ids = np.asarray(manifest['indices'])
            all_sample = name in ('pool_random', 'long_roots_max')
            target = ids if all_sample else np.setdiff1d(ids, background)
            rows.append({'name': name, 'seed': seed, 'scope': 'full sample' if all_sample else '1634 selected positions',
                         'n': len(target), 'mean_gold_probability': float(gold[target].mean()),
                         'proxy_disagreement_fraction': float((gold[target] < .5).mean())})
    out = {'caveat': 'OOF proxy disagreement is not a verified annotation error. Full-sample rows differ in scope from target-only rows.',
           'selections': rows}
    (base.ROOT / 'difficulty_audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

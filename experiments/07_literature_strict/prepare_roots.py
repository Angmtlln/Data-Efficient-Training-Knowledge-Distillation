"""Create pre-registered root-enriched controls on the strict pool."""
from collections import Counter
import json

import numpy as np
import prepare as base


def main():
    ds, meta = base.read_data()
    labels = np.asarray(ds['pool']['label'])
    lengths = np.asarray([len(text.split()) for text in ds['pool']['sentence']])
    length_bin = np.minimum(lengths // 4, 8)
    scores = np.load(base.ROOT / 'scores.npz')
    mask_path = base.ROOT / 'inferred_roots.npz'
    roots = np.load(mask_path)['is_root']
    eligible = scores['eligible'] & (lengths >= 8)
    root_candidates = np.flatnonzero(eligible & roots)
    other_candidates = np.flatnonzero(eligible & ~roots)
    if len(root_candidates) > base.N:
        raise ValueError('Unexpected root count; revisit protocol')
    audit = []
    for seed in base.SEEDS:
        originals, bg, quota = base.select(labels, lengths, scores['eligible'], scores['gold_probability'], scores['fd_bin'], seed)
        control = json.loads((base.ROOT / 'selections' / f'long_random__s{seed}.json').read_text())
        if not np.array_equal(originals['long_random'], control['indices']):
            raise ValueError('Baseline mismatch')
        rng = np.random.default_rng(200000 + seed)
        avail = np.setdiff1d(root_candidates, bg)
        target = []
        for (label, group), count in sorted(quota.items()):
            ids = avail[(labels[avail] == label) & (length_bin[avail] == group)]
            if len(ids) < count:
                raise ValueError(f'Root capacity below matched quota: seed {seed} {label} {group}')
            target.extend(rng.choice(ids, count, replace=False).tolist())
        selections = {'long_roots_matched': np.sort(np.concatenate([bg, target])),
                      'long_roots_max': np.sort(np.concatenate([root_candidates,
                          rng.choice(other_candidates, base.N - len(root_candidates), replace=False)]))}
        for name, ids in selections.items():
            if len(ids) != base.N or len(np.unique(ids)) != base.N or not eligible[ids].all():
                raise ValueError(f'Invalid {name}')
            if name == 'long_roots_matched':
                q = Counter(zip(labels[target].tolist(), length_bin[target].tolist()))
                if q != quota or not np.isin(bg, ids).all() or not roots[target].all():
                    raise ValueError('Matched conditions failed')
            manifest = {'name': name, 'selection_seed': seed, 'indices': ids.tolist(), 'n': base.N, 'ratio': .1,
                        'pool_sha256': meta['splits']['pool']['sha256'],
                        'provenance': {'method': 'inferred whole sentences; see PROTOCOL_ROOTS.md',
                                       'root_mask_sha256': base.sha(mask_path),
                                       'eligibility_scores_sha256': base.sha(base.ROOT / 'scores.npz')}}
            manifest['sha256'] = base.digest(manifest)
            (base.ROOT / 'selections' / f'{name}__s{seed}.json').write_text(json.dumps(manifest, indent=2) + '\n')
            audit.append({'name': name, 'seed': seed, 'n_roots': int(roots[ids].sum()),
                          'mean_words': float(lengths[ids].mean()), 'positive_fraction': float(labels[ids].mean()),
                          'shared_background': int(np.intersect1d(ids, bg).size)})
    (base.ROOT / 'root_selections_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()

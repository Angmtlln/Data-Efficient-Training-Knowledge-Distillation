"""Build matched long-text mixtures from frozen proxy and lexical scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'results/proxy_mixture'
sys.path.insert(0, str(REPO / 'experiments/01_cpu_selection/code'))
from common import read_data, digest

N = 6534
DOSE = 1634
SEEDS = (17, 42, 101)
METHODS = ('mix_random', 'mix_hard', 'mix_uncertain', 'mix_mid', 'mix_easy', 'mix_rare', 'mix_diverse')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratified_mixture(labels, lengths, p_gold, rare, clusters, seed, n=N, dose=DOSE):
    labels = np.asarray(labels)
    lengths = np.asarray(lengths)
    p_gold = np.asarray(p_gold)
    rare = np.asarray(rare)
    clusters = np.asarray(clusters)
    if not (len(labels) == len(lengths) == len(p_gold) == len(rare) == len(clusters)) or not (0 < dose < n):
        raise ValueError('Invalid input shapes/budget')
    if not np.isfinite(p_gold).all() or not np.isfinite(rare).all() or (p_gold < 0).any() or (p_gold > 1).any():
        raise ValueError('Invalid scores')

    rng = np.random.default_rng(seed)
    bins = np.minimum(lengths // 4, 8)
    eligible = np.flatnonzero(lengths >= 8)
    if len(eligible) < n:
        raise ValueError('Insufficient long candidates')
    if (clusters[eligible] < 0).any():
        raise ValueError('Missing cluster assignments')
    background = rng.choice(eligible, n - dose, replace=False)
    available = np.setdiff1d(eligible, background, assume_unique=False)
    slots = rng.choice(available, dose, replace=False)

    def strata(indices):
        return Counter(zip(labels[indices].tolist(), bins[indices].tolist()))

    quota = strata(slots)
    targets = {method: [] for method in METHODS}
    for (label, length_bin), count in sorted(quota.items()):
        candidates = available[(labels[available] == label) & (bins[available] == length_bin)]
        if len(candidates) < count:
            raise ValueError(f'Insufficient candidates in {(label, length_bin)}')
        random_ids = slots[(labels[slots] == label) & (bins[slots] == length_bin)]
        targets['mix_random'].extend(random_ids.tolist())

        # Resolve tied scores with a seeded random order.
        scores = [
            ('mix_hard', p_gold[candidates]),
            ('mix_uncertain', np.abs(p_gold[candidates] - .5)),
            ('mix_mid', np.abs(p_gold[candidates] - np.median(p_gold[candidates]))),
            ('mix_easy', -p_gold[candidates]),
            ('mix_rare', -rare[candidates]),
        ]
        for name, score in scores:
            ranked = np.lexsort((rng.random(len(candidates)), score))
            targets[name].extend(candidates[ranked[:count]].tolist())

        # Take turns across nonempty clusters, without replacing examples.
        buckets = {}
        for cluster_id in np.unique(clusters[candidates]):
            group = candidates[clusters[candidates] == cluster_id]
            buckets[int(cluster_id)] = rng.permutation(group).tolist()
        order = rng.permutation(np.asarray(sorted(buckets), dtype=int)).tolist()
        selected = []
        while len(selected) < count:
            for cluster_id in order:
                if buckets[cluster_id]:
                    selected.append(buckets[cluster_id].pop())
                if len(selected) == count:
                    break
        targets['mix_diverse'].extend(selected)

    result = {}
    for name in METHODS:
        target = np.asarray(targets[name], dtype=np.int64)
        all_ids = np.concatenate([background, target])
        if len(all_ids) != n or len(np.unique(all_ids)) != n or strata(target) != quota or np.any(lengths[all_ids] < 8):
            raise ValueError(f'Invalid selection {name}')
        result[name] = np.sort(all_ids)
    return result, background, quota


def main():
    dataset, meta = read_data()
    pool = dataset['pool']
    labels = np.asarray(pool['label'])
    lengths = np.asarray([len(text.split()) for text in pool['sentence']])
    proxy_meta = json.loads((ROOT / 'proxy/metadata.json').read_text())
    if proxy_meta['pool_sha256'] != meta['splits']['pool']['sha256']:
        raise ValueError('Proxy pool hash mismatch')
    lexical_path = ROOT / 'proxy/lexical_features.npz'
    if sha(lexical_path) != proxy_meta['lexical_sha256']:
        raise ValueError('Lexical hash mismatch')
    lexical = np.load(lexical_path)
    clusters, rare = lexical['cluster'], lexical['rare_idf']
    output = ROOT / 'selections'
    output.mkdir(parents=True, exist_ok=True)
    audit = []
    for seed in SEEDS:
        score_path = ROOT / 'proxy' / f'gold_probability_s{seed}.npz'
        if sha(score_path) != next(row['sha256'] for row in proxy_meta['seeds'] if row['seed'] == seed):
            raise ValueError('Score hash mismatch')
        p_gold = np.load(score_path)['p_gold']
        selections, background, quota = stratified_mixture(labels, lengths, p_gold, rare, clusters, seed)
        for name, indices in selections.items():
            manifest = {
                'name': name, 'selection_seed': seed, 'indices': indices.tolist(), 'n': N, 'ratio': .1,
                'pool_sha256': meta['splits']['pool']['sha256'],
                'provenance': {
                    'selection': '4900 shared random long background + 1634 matched class/length slots',
                    'oof_score_seed': seed, 'oof_score_sha256': sha(score_path), 'lexical_sha256': sha(lexical_path),
                    'labels': 'entire pool available to OOF proxy and selector', 'long_definition': 'len(text.split()) >= 8',
                },
            }
            manifest['sha256'] = digest(manifest)
            (output / f'{name}__s{seed}.json').write_text(json.dumps(manifest, indent=2) + '\n')
            target = np.setdiff1d(indices, background, assume_unique=False)
            audit.append({
                'name': name, 'seed': seed, 'n': len(indices), 'background_n': len(background), 'target_n': len(target),
                'positive': int(labels[indices].sum()), 'mean_words': float(lengths[indices].mean()),
                'target_oof_gold_mean': float(p_gold[target].mean()),
                'target_oof_wrong_fraction': float((p_gold[target] < .5).mean()),
                'target_rare_idf_mean': float(rare[target].mean()),
                'target_cluster_count': int(len(np.unique(clusters[target]))),
                'target_cluster_largest_share': float(np.max(np.bincount(clusters[target])) / len(target)),
                'target_strata': [{'label': int(a), 'length_bin': int(b), 'count': int(v)}
                                  for (a, b), v in sorted(quota.items())],
            })
    (ROOT / 'selection_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    for row in audit:
        print(row['seed'], row['name'], 'pos', row['positive'], 'gold', round(row['target_oof_gold_mean'], 3),
              'proxy_wrong', round(row['target_oof_wrong_fraction'], 3), 'clusters', row['target_cluster_count'], flush=True)


if __name__ == '__main__':
    main()

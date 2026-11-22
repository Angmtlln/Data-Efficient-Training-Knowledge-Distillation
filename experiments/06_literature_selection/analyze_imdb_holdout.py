"""Verify the external predictions and summarize after all evaluations are complete."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score

import prepare as base
from analyze import METHODS, SEEDS
sys.path.insert(0, str(base.REPO / 'experiments/05_proxy_mixture'))
from prepare_imdb import clean


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((base.ROOT / 'imdb_holdout_manifest.json').read_text())
    ds = load_from_disk(str(args.source))
    ids = manifest['indices']
    labels = np.asarray([int(ds[i]['label']) for i in ids])
    texts = [clean(ds[i]['text']) for i in ids]
    if base.digest(list(zip(ids, labels.tolist(), texts))) != manifest['data_sha256']:
        raise ValueError('IMDb source mismatch')
    for name in METHODS:
        for seed in SEEDS:
            runid = f'{name}__ss{seed}__ts{seed}'
            if not (base.ROOT / 'imdb_holdout_runs' / runid / 'metrics.json').is_file():
                raise ValueError(f'Incomplete evaluation: {runid}')
    results = {}
    for name in METHODS:
        for seed in SEEDS:
            runid = f'{name}__ss{seed}__ts{seed}'
            run = base.ROOT / 'imdb_holdout_runs' / runid
            info = json.loads((run / 'metrics.json').read_text())
            selected = json.loads((base.ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
            if info['data_sha256'] != manifest['data_sha256'] or info['selection_sha256'] != selected['sha256']:
                raise ValueError(f'Hash mismatch: {runid}')
            with (run / 'predictions.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            pred = np.asarray([int(r['prediction']) for r in rows])
            if not (np.array_equal(ids, [int(r['imdb_idx']) for r in rows]) and np.array_equal(labels, [int(r['label']) for r in rows])):
                raise ValueError(f'Rows mismatch: {runid}')
            acc = float(accuracy_score(labels, pred))
            f1 = float(f1_score(labels, pred, average='macro'))
            if abs(acc - info['accuracy']) > 1e-12 or abs(f1 - info['macro_f1']) > 1e-12:
                raise ValueError(f'Metric mismatch: {runid}')
            results[(name, seed)] = {'accuracy': acc, 'macro_f1': f1}
    summary = []
    for name in METHODS:
        acc = np.asarray([results[(name, s)]['accuracy'] for s in SEEDS])
        baseline = np.asarray([results[('long_random', s)]['accuracy'] for s in SEEDS])
        summary.append({'method': name, 'accuracy_mean': float(acc.mean()), 'accuracy_sd': float(acc.std(ddof=1)),
                        'accuracy_by_seed': acc.tolist(), 'macro_f1_mean': float(np.mean([results[(name, s)]['macro_f1'] for s in SEEDS])),
                        'delta_vs_long_random_pp': ((acc - baseline) * 100).tolist()})
    (base.ROOT / 'imdb_holdout_analysis.json').write_text(json.dumps({'summary': summary, 'sample': manifest}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Recompute all original-SST results and verify prediction/label alignment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import audit_original_sst as audit
import prepare as base
from analyze import METHODS, SEEDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', type=Path, required=True)
    args = p.parse_args()
    m = json.loads((base.ROOT / 'original_sst_manifest.json').read_text())
    if hashlib.sha256(args.test.read_bytes()).hexdigest() != m['test_sha256']:
        raise ValueError('Source mismatch')
    records = audit.read(args.test)
    labels = np.asarray([label for _, label in records])
    if base.digest([(i, text, label) for i, (text, label) in enumerate(records)]) != m['data_sha256']:
        raise ValueError('Normalized records mismatch')
    clean = np.asarray(m['clean_indices'])
    results = {}
    for name in METHODS:
        for seed in SEEDS:
            runid = f'{name}__ss{seed}__ts{seed}'
            run = base.ROOT / 'original_sst_runs' / runid
            info = json.loads((run / 'metrics.json').read_text())
            selection = json.loads((base.ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
            if info['data_sha256'] != m['data_sha256'] or info['selection_sha256'] != selection['sha256']:
                raise ValueError(f'Hash mismatch {runid}')
            with (run / 'predictions.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            pred = np.asarray([int(r['prediction']) for r in rows])
            if not (np.array_equal(np.arange(len(records)), [int(r['original_index']) for r in rows]) and
                    np.array_equal(labels, [int(r['label']) for r in rows])):
                raise ValueError(f'Prediction rows mismatch {runid}')
            values = {'accuracy': float(accuracy_score(labels, pred)),
                      'macro_f1': float(f1_score(labels, pred, average='macro')),
                      'clean_accuracy': float(accuracy_score(labels[clean], pred[clean])),
                      'clean_macro_f1': float(f1_score(labels[clean], pred[clean], average='macro'))}
            if any(abs(info[k] - v) > 1e-12 for k, v in values.items()):
                raise ValueError(f'Metric mismatch {runid}')
            results[(name, seed)] = values
    summary = []
    for name in METHODS:
        acc = np.asarray([results[(name, seed)]['accuracy'] for seed in SEEDS])
        clean_acc = np.asarray([results[(name, seed)]['clean_accuracy'] for seed in SEEDS])
        control = np.asarray([results[('long_random', seed)]['accuracy'] for seed in SEEDS])
        summary.append({'method': name, 'accuracy_mean': float(acc.mean()), 'accuracy_sd': float(acc.std(ddof=1)),
                        'accuracy_by_seed': acc.tolist(), 'macro_f1_mean': float(np.mean([results[(name, s)]['macro_f1'] for s in SEEDS])),
                        'clean_accuracy_mean': float(clean_acc.mean()), 'clean_accuracy_by_seed': clean_acc.tolist(),
                        'delta_vs_long_random_pp': ((acc - control) * 100).tolist()})
    (base.ROOT / 'original_sst_analysis.json').write_text(json.dumps({'summary': summary, 'manifest': m}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

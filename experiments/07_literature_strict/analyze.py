"""Recompute all pre-registered metrics only after the complete series exists."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / 'results/literature_strict'
sys.path.insert(0, str(REPO / 'experiments/01_cpu_selection/code'))
from common import digest, read_data

SEEDS = (17, 42, 101)
METHODS = ('long_random', 'long_mid', 'long_fd', 'long_fd_mid', 'pool_random', 'long_gsnr')


def main():
    ds, meta = read_data()
    truth = np.asarray(ds['test']['label'])
    ids = np.asarray(ds['test']['orig_idx'])
    lengths = np.asarray([len(s.split()) for s in ds['test']['sentence']])
    all_runs = {}
    # Require the complete series before looking at any results.
    for seed in SEEDS:
        for name in METHODS:
            runid = f'{name}__ss{seed}__ts{seed}'
            directory = ROOT / 'runs' / runid
            for path in (directory / 'metrics.json', directory / 'predictions.csv'):
                if not path.is_file():
                    raise ValueError(f'Incomplete series: {path}')
    for seed in SEEDS:
        for name in METHODS:
            runid = f'{name}__ss{seed}__ts{seed}'
            run = ROOT / 'runs' / runid
            info = json.loads((run / 'metrics.json').read_text())
            manifest = json.loads((ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
            checksum = manifest.pop('sha256')
            if digest(manifest) != checksum or info['selection_sha256'] != checksum:
                raise ValueError(f'Selection mismatch: {runid}')
            with (run / 'predictions.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            pred = np.asarray([int(r['prediction']) for r in rows])
            if not (np.array_equal(ids, [int(r['orig_idx']) for r in rows]) and
                    np.array_equal(truth, [int(r['label']) for r in rows])):
                raise ValueError(f'Test rows do not match snapshot: {runid}')
            acc = float(accuracy_score(truth, pred))
            f1 = float(f1_score(truth, pred, average='macro'))
            if abs(acc - info['metrics']['accuracy']) > 1e-12 or abs(f1 - info['metrics']['macro_f1']) > 1e-12:
                raise ValueError(f'Metric mismatch: {runid}')
            masks = {'short_lt8': lengths < 8, 'length_8_15': (lengths >= 8) & (lengths < 16),
                     'length_16_plus': lengths >= 16, 'negative': truth == 0, 'positive': truth == 1}
            slices = {key: {'n': int(mask.sum()), 'accuracy': float(accuracy_score(truth[mask], pred[mask]))}
                      for key, mask in masks.items() if mask.any()}
            all_runs[(name, seed)] = {'method': name, 'seed': seed, 'accuracy': acc, 'macro_f1': f1,
                                     'seconds': info['seconds'], 'slices': slices}
    summary = []
    for name in METHODS:
        runs = [all_runs[(name, seed)] for seed in SEEDS]
        acc = np.asarray([r['accuracy'] for r in runs])
        baseline = np.asarray([all_runs[('long_random', seed)]['accuracy'] for seed in SEEDS])
        summary.append({'method': name, 'n_seeds': len(SEEDS), 'accuracy_mean': float(acc.mean()),
                        'accuracy_sd': float(acc.std(ddof=1)), 'accuracy_by_seed': acc.tolist(),
                        'macro_f1_mean': float(np.mean([r['macro_f1'] for r in runs])),
                        'delta_vs_long_random_pp': ((acc - baseline) * 100).tolist(),
                        'seconds': [r['seconds'] for r in runs]})
    out = {'dataset': meta, 'summary': summary, 'runs': list(all_runs.values()),
           'caveat': 'GLUE validation reused for exploratory comparisons; 3 correlated seeds, no multiplicity correction.'}
    (ROOT / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

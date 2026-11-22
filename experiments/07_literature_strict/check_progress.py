"""Verify completed run metadata and artifacts without inspecting model quality."""
import csv
import json
from pathlib import Path

from safetensors import safe_open
import numpy as np
import prepare as base
from analyze import METHODS, SEEDS


def main():
    ds, meta = base.read_data()
    completed, missing = [], []
    labels = np.asarray(ds['test']['label'])
    ids = np.asarray(ds['test']['orig_idx'])
    for name in METHODS:
        for seed in SEEDS:
            runid = f'{name}__ss{seed}__ts{seed}'
            directory = base.ROOT / 'runs' / runid
            if not (directory / 'metrics.json').exists():
                missing.append(runid)
                continue
            info = json.loads((directory / 'metrics.json').read_text())
            manifest = json.loads((base.ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
            sha = manifest.pop('sha256')
            if (base.digest(manifest) != sha or info['selection_sha256'] != sha or
                    manifest['pool_sha256'] != meta['splits']['pool']['sha256'] or info['pilot'] or info['split'] != 'test'):
                raise ValueError(f'Invalid metadata: {runid}')
            with (directory / 'predictions.csv').open(newline='') as stream:
                predictions = list(csv.DictReader(stream))
            if not (np.array_equal(ids, [int(r['orig_idx']) for r in predictions]) and
                    np.array_equal(labels, [int(r['label']) for r in predictions])):
                raise ValueError(f'Prediction IDs/labels mismatch: {runid}')
            ckpt = base.REPO / 'outputs/literature_strict/checkpoints' / runid / Path(info['best_checkpoint']).name
            with safe_open(ckpt / 'model.safetensors', framework='pt', device='cpu') as file:
                if len(file.keys()) != 138:
                    raise ValueError(f'Unexpected model tensors: {runid}')
                for key in file.keys():
                    file.get_slice(key).get_shape()
            completed.append(runid)
    print(json.dumps({'n_completed': len(completed), 'n_missing': len(missing), 'completed': completed, 'missing': missing}, indent=2))


if __name__ == '__main__':
    main()

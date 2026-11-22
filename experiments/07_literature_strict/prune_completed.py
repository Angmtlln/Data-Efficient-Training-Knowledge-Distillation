"""Remove resumable optimizer/interim files only for verified completed study runs.

Keep the selected best checkpoint with config, model weights, tokenizer and trainer state.
"""
import json
from pathlib import Path
import shutil

import prepare as base


def prune(runid):
    results = base.ROOT / 'runs' / runid
    if not (results / 'predictions.csv').is_file() or not (results / 'metrics.json').is_file():
        raise ValueError(f'Incomplete result {runid}')
    info = json.loads((results / 'metrics.json').read_text())
    expected_id = f"{info['name']}__ss{info['selection_seed']}__ts{info['training_seed']}"
    if runid != expected_id or info['pilot'] or info['split'] != 'test':
        raise ValueError(f'Run identity mismatch {runid}')
    parent = (base.REPO / 'outputs/literature_strict/checkpoints').resolve()
    unresolved_root = parent / runid
    if unresolved_root.is_symlink():
        raise ValueError('Refusing to prune symlink root')
    root = unresolved_root.resolve()
    if root.parent != parent:
        raise ValueError('Checkpoint root escapes experiment directory')
    # The saved path belongs to the training machine; only its name is portable.
    best = (root / Path(info['best_checkpoint']).name).resolve()
    if best.parent != root or not (best / 'config.json').is_file() or not (best / 'model.safetensors').is_file():
        raise ValueError(f'Invalid checkpoint {runid}')
    folders = list(root.glob('checkpoint-*'))
    for folder in folders:
        if folder.is_symlink():
            raise ValueError('Refusing to prune symlink')
    optimizer = best / 'optimizer.pt'
    if optimizer.is_symlink():
        raise ValueError('Refusing to prune symlink optimizer')
    for folder in folders:
        if folder.resolve() != best:
            shutil.rmtree(folder)
    if optimizer.exists():
        optimizer.unlink()
    print('Retained best checkpoint and removed temporary optimizer/intermediate checkpoint:', runid)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('runid')
    args = p.parse_args()
    if not args.runid or Path(args.runid).name != args.runid:
        raise ValueError('Invalid run ID')
    prune(args.runid)


if __name__ == '__main__':
    main()

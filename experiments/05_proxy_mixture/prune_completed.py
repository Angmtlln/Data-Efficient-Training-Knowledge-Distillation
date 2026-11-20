"""Keep best-model inference files, remove large completed-run optimizer/interim checkpoints.

Only touches this experiment's own output directory after a verified complete result exists.
"""
import argparse
import json
from pathlib import Path
import shutil

REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args()
    runid=f'{a.method}__ss{a.seed}__ts{a.seed}'
    root=(REPO/'outputs/proxy_mixture/checkpoints'/runid).resolve()
    results=REPO/'results/proxy_mixture/runs'/runid
    if not (results/'predictions.csv').is_file() or not (results/'metrics.json').is_file():raise ValueError('Cannot prune incomplete run')
    result=json.loads((results/'metrics.json').read_text())
    if result['name']!=a.method or result['selection_seed']!=a.seed or result['pilot']:raise ValueError('Run identity mismatch')
    best=Path(result['best_checkpoint']).resolve()
    if best.parent!=root or not (best/'config.json').is_file() or not (best/'model.safetensors').is_file():
        raise ValueError('Invalid best checkpoint')
    for folder in root.glob('checkpoint-*'):
        if folder.resolve()!=best:shutil.rmtree(folder)
    optimizer=best/'optimizer.pt'
    if optimizer.exists():optimizer.unlink()
    print('Preserved best model; removed completed-run optimizer and other checkpoints:',runid,flush=True)

if __name__=='__main__':main()

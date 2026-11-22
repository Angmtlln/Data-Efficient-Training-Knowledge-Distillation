"""Evaluate all completed target checkpoints on the frozen new IMDb stress test."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import prepare as base
from analyze import METHODS, SEEDS
from hash_checkpoints import file_hash

sys.path.insert(0, str(base.REPO / 'experiments/05_proxy_mixture'))
from prepare_imdb import clean, REV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((base.ROOT / 'imdb_holdout_manifest.json').read_text())
    source = load_from_disk(str(args.source))
    ids = manifest['indices']
    truth = [int(source[i]['label']) for i in ids]
    texts = [clean(source[i]['text']) for i in ids]
    if manifest['revision'] != REV or base.digest(list(zip(ids, truth, texts))) != manifest['data_sha256']:
        raise ValueError('IMDb sample or revision mismatch')
    old = json.loads((base.REPO / 'results/proxy_mixture/imdb_manifest.json').read_text())
    if len(ids) != 2000 or len(set(ids)) != 2000 or set(ids) & set(old['indices']):
        raise ValueError('IMDb sample overlap or size mismatch')
    jobs = []
    integrity = json.loads((base.ROOT / 'checkpoint_integrity.json').read_text())
    for name in METHODS:
        for seed in SEEDS:
            runid = f'{name}__ss{seed}__ts{seed}'
            run = base.ROOT / 'runs' / runid
            info = json.loads((run / 'metrics.json').read_text())
            ckpt = base.REPO / 'outputs/literature_strict/checkpoints' / runid / Path(info['best_checkpoint']).name
            if info['pilot'] or info['split'] != 'test' or not (ckpt / 'model.safetensors').is_file():
                raise ValueError(f'Invalid checkpoint: {runid}')
            if integrity[runid]['selection_sha256'] != info['selection_sha256'] or file_hash(ckpt / 'model.safetensors') != integrity[runid]['sha256']:
                raise ValueError(f'Checkpoint checksum mismatch: {runid}')
            jobs.append((runid, info, ckpt))
    print('Validated sample and', len(jobs), 'completed checkpoints', flush=True)
    if args.check_only:
        return
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else None)
    if device is None:
        raise RuntimeError('CUDA or MPS required')
    tok = AutoTokenizer.from_pretrained(str(base.REPO / 'experiments/01_cpu_selection/data/tokenizer'), local_files_only=True)
    encoded = tok(texts, truncation=True, max_length=64)
    features = [{k: v[i] for k, v in encoded.items()} for i in range(len(ids))]
    loader = DataLoader(features, batch_size=32, shuffle=False, collate_fn=DataCollatorWithPadding(tok))
    out = base.ROOT / 'imdb_holdout_runs'
    out.mkdir(exist_ok=True)
    for runid, info, ckpt in jobs:
        target = out / runid
        target.mkdir(exist_ok=True)
        metric_path = target / 'metrics.json'
        if metric_path.exists():
            saved = json.loads(metric_path.read_text())
            if saved['data_sha256'] != manifest['data_sha256'] or saved['selection_sha256'] != info['selection_sha256']:
                raise ValueError('Saved evaluation mismatch')
            continue
        start = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), attn_implementation='sdpa').to(device).eval()
        parts = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                parts.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        prob = np.concatenate(parts)
        pred = prob.argmax(1)
        with (target / 'predictions.csv').open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['imdb_idx', 'label', 'prediction', 'p_positive'])
            writer.writerows(zip(ids, truth, pred.tolist(), prob[:, 1].tolist()))
        result = {'name': info['name'], 'selection_seed': info['selection_seed'], 'selection_sha256': info['selection_sha256'],
                  'data_sha256': manifest['data_sha256'], 'n': len(ids), 'max_length': 64,
                  'accuracy': float(accuracy_score(truth, pred)), 'macro_f1': float(f1_score(truth, pred, average='macro')),
                  'seconds': time.time() - start, 'checkpoint': str(ckpt), 'device': device}
        metric_path.write_text(json.dumps(result, indent=2) + '\n')
        print(runid, 'accuracy', result['accuracy'], flush=True)
        del model
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

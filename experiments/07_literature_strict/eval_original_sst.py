"""Evaluate all frozen checkpoints on the published original SST binary test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import audit_original_sst as audit
import prepare as base
from analyze import METHODS, SEEDS
from hash_checkpoints import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((base.ROOT / 'original_sst_manifest.json').read_text())
    if hashlib.sha256(args.test.read_bytes()).hexdigest() != manifest['test_sha256']:
        raise ValueError('Source file checksum mismatch')
    rows = audit.read(args.test)
    if len(rows) != manifest['n'] or base.digest([(i, text, label) for i, (text, label) in enumerate(rows)]) != manifest['data_sha256']:
        raise ValueError('Normalized source records mismatch')
    jobs = []
    integrity = json.loads((base.ROOT / 'checkpoint_integrity.json').read_text())
    for name in METHODS:
        for seed in SEEDS:
            runid = f'{name}__ss{seed}__ts{seed}'
            info = json.loads((base.ROOT / 'runs' / runid / 'metrics.json').read_text())
            ckpt = base.REPO / 'outputs/literature_strict/checkpoints' / runid / Path(info['best_checkpoint']).name
            if info['pilot'] or info['split'] != 'test' or not (ckpt / 'model.safetensors').is_file():
                raise ValueError(f'Invalid checkpoint {runid}')
            if integrity[runid]['selection_sha256'] != info['selection_sha256'] or file_hash(ckpt / 'model.safetensors') != integrity[runid]['sha256']:
                raise ValueError(f'Checkpoint checksum mismatch: {runid}')
            jobs.append((runid, info, ckpt))
    print('Verified source and', len(jobs), 'checkpoints', flush=True)
    if args.check_only:
        return
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else None)
    if device is None:
        raise RuntimeError('CUDA or MPS required')
    tokenizer = AutoTokenizer.from_pretrained(str(base.REPO / 'experiments/01_cpu_selection/data/tokenizer'), local_files_only=True)
    texts = [t for t, _ in rows]
    labels = np.asarray([label for _, label in rows])
    encoded = tokenizer(texts, truncation=True, max_length=64)
    features = [{key: values[i] for key, values in encoded.items()} for i in range(len(rows))]
    loader = DataLoader(features, batch_size=32, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
    clean = np.asarray(manifest['clean_indices'])
    parent = base.ROOT / 'original_sst_runs'
    parent.mkdir(exist_ok=True)
    for runid, info, ckpt in jobs:
        out = parent / runid
        out.mkdir(exist_ok=True)
        metric_file = out / 'metrics.json'
        if metric_file.exists():
            prior = json.loads(metric_file.read_text())
            if prior['data_sha256'] != manifest['data_sha256'] or prior['selection_sha256'] != info['selection_sha256']:
                raise ValueError(f'Saved result mismatch {runid}')
            continue
        started = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), attn_implementation='sdpa').to(device).eval()
        parts = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
                parts.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs = np.concatenate(parts)
        pred = probs.argmax(1)
        with (out / 'predictions.csv').open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['original_index', 'label', 'prediction', 'p_positive'])
            writer.writerows(zip(range(len(rows)), labels.tolist(), pred.tolist(), probs[:, 1].tolist()))
        result = {'name': info['name'], 'selection_seed': info['selection_seed'], 'selection_sha256': info['selection_sha256'],
                  'data_sha256': manifest['data_sha256'], 'n': len(rows), 'n_clean': len(clean),
                  'accuracy': float(accuracy_score(labels, pred)), 'macro_f1': float(f1_score(labels, pred, average='macro')),
                  'clean_accuracy': float(accuracy_score(labels[clean], pred[clean])),
                  'clean_macro_f1': float(f1_score(labels[clean], pred[clean], average='macro')),
                  'max_length': 64, 'seconds': time.time() - started, 'device': device, 'checkpoint': str(ckpt)}
        metric_file.write_text(json.dumps(result, indent=2) + '\n')
        print(runid, 'accuracy', result['accuracy'], flush=True)
        del model
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

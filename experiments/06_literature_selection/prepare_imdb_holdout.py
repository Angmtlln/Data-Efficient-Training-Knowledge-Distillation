"""Freeze a new IMDb test sample disjoint from series 05 before evaluating models."""
import json
from pathlib import Path
import sys

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

import prepare as base

REPO = base.REPO
sys.path.insert(0, str(REPO / 'experiments/05_proxy_mixture'))
from prepare_imdb import clean, REV, tokens


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    imdb = load_from_disk(str(args.source))
    old = json.loads((REPO / 'results/proxy_mixture/imdb_manifest.json').read_text())
    if old['revision'] != REV:
        raise ValueError('IMDb revision mismatch')
    labels = np.asarray(imdb['label'])
    old_mask = np.zeros(len(labels), dtype=bool)
    old_mask[old['indices']] = True
    rng = np.random.default_rng(20261122)
    ids = sorted(np.concatenate([rng.choice(np.flatnonzero((labels == y) & ~old_mask), 1000, replace=False) for y in (0, 1)]).tolist())
    texts = [clean(imdb[i]['text']) for i in ids]
    truth = [int(labels[i]) for i in ids]
    ds, meta = base.read_data()
    sst2_text = [s for split in ('pool', 'val', 'test') for s in ds[split]['sentence']]
    normalized = {' '.join(tokens(s)) for s in sst2_text}
    exact = sum(' '.join(tokens(s)) in normalized for s in texts)
    grams = set()
    for text in sst2_text:
        row = tokens(text)
        grams.update(tuple(row[i:i + 8]) for i in range(len(row) - 7))
    overlap = 0
    for text in texts:
        row = tokens(text)
        overlap += any(tuple(row[i:i + 8]) in grams for i in range(len(row) - 7))
    tok = AutoTokenizer.from_pretrained(str(REPO / 'experiments/01_cpu_selection/data/tokenizer'), local_files_only=True)
    lens = [len(row) for row in tok(texts, add_special_tokens=True, truncation=False, verbose=False)['input_ids']]
    out = {'source': 'stanfordnlp/imdb', 'revision': REV, 'split': 'test', 'seed': 20261122,
           'n': len(ids), 'indices': ids, 'excluded_previous_sample': True,
           'data_sha256': base.digest(list(zip(ids, truth, texts))), 'sst2_pool_sha256': meta['splits']['pool']['sha256'],
           'normalized_exact_sst2_any_split': exact, 'containing_any_8token_ngram_from_sst2': overlap,
           'mean_tokens': float(np.mean(lens)), 'over_64_tokens': int(sum(length > 64 for length in lens)),
           'max_length': 64, 'pretraining_contamination': 'unknown'}
    path = base.ROOT / 'imdb_holdout_manifest.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'indices'}, indent=2))


if __name__ == '__main__':
    main()

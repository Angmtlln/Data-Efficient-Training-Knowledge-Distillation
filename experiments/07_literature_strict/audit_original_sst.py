"""Audit a public SST binary test mirror before any model evaluation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from datasets import Dataset
import numpy as np
import prepare as base


def read(path):
    records = []
    for line in path.read_text().splitlines():
        label, text = line.split(' ', 1)
        if label not in ('0', '1'):
            raise ValueError('Invalid label')
        # The mirror uses Penn Treebank bracket tokens and escaped slashes;
        # GLUE normalizes them back to ordinary punctuation.
        text = text.replace('-lrb-', '(').replace('-rrb-', ')')
        text = re.sub(r'\\+/', '/', text)
        records.append((text, int(label)))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--glue-test-arrow', type=Path, required=True)
    args = p.parse_args()
    dev, test = read(args.dev), read(args.test)
    ds, meta = base.read_data()
    raw_glue_test = Dataset.from_file(str(args.glue_test_arrow))
    if len(dev) != 872 or len(test) != 1821 or len(raw_glue_test) != 1821:
        raise ValueError('Unexpected source sizes')
    dev_lookup = defaultdict(set)
    for text, label in dev:
        dev_lookup[base.norm(text)].add(label)
    glue_dev = list(zip(ds['test']['sentence'], ds['test']['label']))
    dev_match = [base.norm(text) in dev_lookup for text, _ in glue_dev]
    dev_consistent = [int(label) in dev_lookup.get(base.norm(text), set()) for text, label in glue_dev]
    test_lookup = defaultdict(set)
    for text, label in test:
        test_lookup[base.norm(text)].add(label)
    test_input_match = sum(base.norm(text) in test_lookup for text in raw_glue_test['sentence'])
    pool_texts = ds['pool']['sentence']
    eligible = np.load(base.ROOT / 'scores.npz')['eligible']
    pool_norm = {base.norm(pool_texts[i]) for i in np.flatnonzero(eligible)}
    pool_grams = set()
    for i in np.flatnonzero(eligible):
        pool_grams.update(base.eightgrams(pool_texts[i]))
    exact_pool = sum(base.norm(text) in pool_norm for text, _ in test)
    gram_pool = sum(bool(base.eightgrams(text) & pool_grams) for text, _ in test)
    clean_ids = [i for i, (text, _) in enumerate(test)
                 if base.norm(text) not in pool_norm and not (base.eightgrams(text) & pool_grams)]
    previous_eval = {base.norm(text) for split in ('val', 'test') for text in ds[split]['sentence']}
    report = {'source': 'AcademiaSinicaNLPLab/sentiment_dataset/data/stsa.binary.{dev,test}',
              'test_blob_sha1': 'bb5d021fbc75d277a92c6a0fb1b0786984776007',
              'dev_blob_sha1': 'a80b081639ae8ac506f0285de4af86d276e8f983',
              'dev_sha256': hashlib.sha256(args.dev.read_bytes()).hexdigest(),
              'test_sha256': hashlib.sha256(args.test.read_bytes()).hexdigest(),
              'dev_size': len(dev), 'dev_exact_text_match_glue_validation': sum(dev_match),
              'dev_exact_text_and_label_match_glue_validation': sum(dev_consistent),
              'test_size': len(test), 'test_class_counts': dict(Counter(label for _, label in test)),
              'test_exact_text_match_glue_unlabeled_test': test_input_match,
              'test_unique_normalized': len(test_lookup),
              'test_ambiguous_normalized_labels': sum(len(v) > 1 for v in test_lookup.values()),
              'test_exact_with_eligible_pool': exact_pool,
              'test_shared_8gram_with_eligible_pool': gram_pool,
              'test_clean_of_pool_exact_or_shared_8gram': len(clean_ids),
              'test_exact_with_prior_evaluation': sum(base.norm(t) in previous_eval for t, _ in test),
              'pool_sha256': meta['splits']['pool']['sha256'],
              'limitation': 'Public third-party SST binary mirror, not an official GLUE leaderboard submission; pretraining exposure unknown.'}
    (base.ROOT / 'original_sst_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = {'source': report['source'], 'test_sha256': report['test_sha256'],
                'test_blob_sha1': report['test_blob_sha1'], 'n': len(test),
                'data_sha256': base.digest([(i, text, label) for i, (text, label) in enumerate(test)]),
                'clean_indices': clean_ids, 'n_clean': len(clean_ids),
                'clean_rule': 'exclude exact normalized text or any shared 8-token gram with eligible pool',
                'pool_sha256': meta['splits']['pool']['sha256']}
    (base.ROOT / 'original_sst_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

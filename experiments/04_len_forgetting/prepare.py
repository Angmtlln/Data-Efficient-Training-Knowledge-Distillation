"""Prepare paired SST-2 subsets without using evaluation labels or scores.

Uses the dataset and OOF TF-IDF forgetting scores archived by series 01.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'experiments/01_cpu_selection/code'))
from common import read_data, digest

N = 6534
DOSE = 327
SEEDS = (17, 42, 101)


def make_subsets(labels, lengths, scores, seed, n=N, dose=DOSE):
    labels, lengths, scores = np.asarray(labels), np.asarray(lengths), np.asarray(scores)
    if not (len(labels) == len(lengths) == len(scores)) or not set(np.unique(labels)) <= {0, 1}:
        raise ValueError('Input shapes or labels are invalid')
    if not 0 < dose < n:
        raise ValueError('Invalid dose')
    rng = np.random.default_rng(seed)
    bins = np.minimum(lengths // 4, 8)
    eligible = np.flatnonzero(lengths >= 8)
    target_pool = eligible[scores[eligible] > 0]
    if len(target_pool) < dose or len(eligible) < n:
        raise ValueError(f'Insufficient candidates: long={len(eligible)}, nonzero={len(target_pool)}, dose={dose}')
    # Largest forgetting score first, randomly resolve ties with the selection seed.
    target = target_pool[np.lexsort((rng.random(len(target_pool)), -scores[target_pool]))[:dose]]
    matches = []
    for label, length_bin in sorted(set(zip(labels[target].tolist(), bins[target].tolist()))):
        count = int(((labels[target] == label) & (bins[target] == length_bin)).sum())
        candidates = eligible[(labels[eligible] == label) & (bins[eligible] == length_bin) & (scores[eligible] == 0)]
        if len(candidates) < count:
            raise ValueError(f'Insufficient zero-score matches for label={label}, length_bin={length_bin}: {len(candidates)} < {count}')
        matches.extend(rng.choice(candidates, size=count, replace=False).tolist())
    matches = np.asarray(matches, dtype=np.int64)
    available = np.setdiff1d(eligible, np.concatenate([target, matches]), assume_unique=False)
    if len(available) < n - dose:
        raise ValueError('Insufficient background candidates')
    background = rng.choice(available, size=n-dose, replace=False)
    sets = {
        'random': rng.choice(len(labels), size=n, replace=False),
        'len8': rng.choice(eligible, size=n, replace=False),
        'len8_matched_random': np.concatenate([background, matches]),
        'len8_forgetting': np.concatenate([background, target]),
    }
    for name, ids in sets.items():
        if len(ids) != n or len(np.unique(ids)) != n or ids.min() < 0 or ids.max() >= len(labels):
            raise ValueError(f'Invalid subset: {name}')
        if name != 'random' and not np.all(lengths[ids] >= 8):
            raise ValueError(f'Length constraint broken: {name}')
    h = lambda ids: Counter(zip(labels[ids].tolist(), bins[ids].tolist()))
    if h(target) != h(matches):
        raise ValueError('Target and matched control strata differ')
    return sets, dict(long_candidates=int(len(eligible)), nonzero_long_candidates=int(len(target_pool)),
                      dose=dose, background_size=int(len(background)),
                      targeted_score_histogram={str(k): int(v) for k,v in sorted(Counter(scores[target].tolist()).items())},
                      paired_strata=[{'label':int(k[0]),'length_bin':int(k[1]),'count':int(v)} for k,v in sorted(h(target).items())])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--output', type=Path, default=REPO/'results/len_forgetting/selections')
    args=parser.parse_args()
    ds, snapshot=read_data()
    archived=json.loads((REPO/'results/inputs/part1/snapshot.json').read_text())
    if snapshot['splits']['pool']['sha256'] != archived['splits']['pool']['sha256']:
        raise ValueError('Pool differs from proxy score provenance')
    pool=ds['pool']; labels=np.asarray(pool['label']); lengths=np.asarray([len(x.split()) for x in pool['sentence']])
    args.output.mkdir(parents=True, exist_ok=True)
    audits=[]
    for seed in args.seeds:
        score_file=REPO/f'results/inputs/part1/forgetting_s{seed}.npz'
        scores=np.load(score_file)['forgetting']
        subsets, audit=make_subsets(labels, lengths, scores, seed)
        for name, indices in subsets.items():
            ids=sorted(map(int, indices))
            value=dict(name=name, selection_seed=seed, indices=ids, n=N, ratio=0.1,
                       pool_sha256=snapshot['splits']['pool']['sha256'],
                       provenance=dict(score_seed=seed, score_sha256=hashlib.sha256(score_file.read_bytes()).hexdigest(),
                                       method='3-fold held-out TF-IDF forgetting, 8 epochs; long text >=8 whitespace words',
                                       dose=DOSE, background='shared in matched arms',
                                       label_budget='all pool labels available to proxy and selection'))
            value['sha256']=digest(value)
            (args.output/f'{name}__s{seed}.json').write_text(json.dumps(value, indent=2)+'\n')
            a=np.asarray(ids); audits.append(dict(seed=seed, name=name, n=len(a),
                positive=int(labels[a].sum()), mean_words=float(lengths[a].mean()),
                nonzero_forgetting=int((scores[a]>0).sum()), **audit))
    (args.output.parent/'selection_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    print(json.dumps(audits,indent=2), flush=True)

if __name__=='__main__': main()

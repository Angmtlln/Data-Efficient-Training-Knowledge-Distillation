"""Construct auditable SCDP-inspired matched long-text selections."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / 'results/literature_strict'
sys.path.insert(0, str(REPO / 'experiments/01_cpu_selection/code'))
from common import digest, read_data

SEEDS = (17, 42, 101)
N, DOSE = 6534, 1634
METHODS = ('long_random', 'long_mid', 'long_fd', 'long_fd_mid')


def norm(text):
    return ' '.join(text.casefold().split())


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def eightgrams(text):
    row = words(text)
    return {tuple(row[i:i + 8]) for i in range(len(row) - 7)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate_scores(texts, labels, eligible):
    start = time.time()
    y = np.asarray(labels)
    ids = np.flatnonzero(eligible)
    groups = np.array([norm(texts[i]) for i in ids])
    gold = np.full(len(y), np.nan, dtype=np.float64)
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261122)
    fold_results = []
    for fold, (tr, te) in enumerate(folds.split(ids, y[ids], groups)):
        train, held = ids[tr], ids[te]
        if set(groups[tr]) & set(groups[te]):
            raise ValueError('OOF groups intersect')
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        x = vectorizer.fit_transform([texts[i] for i in train])
        z = vectorizer.transform([texts[i] for i in held])
        clf = SGDClassifier(loss='log_loss', alpha=1e-5, random_state=20261122 + fold, average=True)
        rng = np.random.default_rng(20261122 + fold)
        for _ in range(8):
            order = rng.permutation(len(train))
            clf.partial_fit(x[order], y[train][order], classes=np.array([0, 1]))
        probs = clf.predict_proba(z)
        gold[held] = probs[np.arange(len(held)), y[held]]
        fold_results.append({'fold': fold, 'n': len(held), 'accuracy': float((probs.argmax(1) == y[held]).mean())})
    assert np.isfinite(gold[ids]).all() and np.isnan(gold[~eligible]).all()
    proxy_time = time.time() - start

    start = time.time()
    long_ids = np.array([i for i in ids if len(texts[i].split()) >= 8], dtype=np.int64)
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=50000, dtype=np.float64)
    matrix = vec.fit_transform([texts[i] for i in long_ids])
    norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    center = np.asarray(matrix.mean(axis=0)).ravel()
    converged = False
    for iteration in range(200):
        distances = np.sqrt(np.maximum(norms + center @ center - 2 * (matrix @ center), 1e-24))
        weights = 1.0 / np.maximum(distances, 1e-12)
        next_center = np.asarray(matrix.T @ weights).ravel() / weights.sum()
        shift = np.linalg.norm(next_center - center)
        center = next_center
        if shift < 1e-8:
            converged = True
            break
    if not converged:
        raise RuntimeError('Geometric median did not converge')
    distances = np.sqrt(np.maximum(norms + center @ center - 2 * (matrix @ center), 0))
    fd = np.full(len(y), np.nan, dtype=np.float64)
    fd[long_ids] = distances
    edges = np.linspace(distances.min(), distances.max(), 11)
    bins = np.full(len(y), -1, dtype=np.int8)
    bins[long_ids] = np.clip(np.searchsorted(edges[1:-1], distances, side='right'), 0, 9)
    return gold, fd, bins, {'proxy_seconds': proxy_time, 'folds': fold_results,
                           'fd_seconds': time.time() - start, 'fd_iterations': iteration + 1,
                           'fd_shift': float(shift), 'fd_edges': edges.tolist(),
                           'fd_counts': np.bincount(bins[long_ids], minlength=10).tolist(),
                           'long_count': len(long_ids)}


def balanced_bins(ids, bins, count, rng):
    """Water-fill equal quotas over occupied bins, randomise residual ties."""
    buckets = {int(b): rng.permutation(ids[bins[ids] == b]).tolist() for b in np.unique(bins[ids])}
    if any(b < 0 for b in buckets) or sum(map(len, buckets.values())) < count:
        raise ValueError('Invalid FD buckets')
    result = []
    while len(result) < count:
        active = [b for b in sorted(buckets) if buckets[b]]
        if not active:
            raise ValueError('Not enough FD candidates')
        for b in rng.permutation(active):
            result.append(buckets[int(b)].pop())
            if len(result) == count:
                break
    return result


def select(labels, lengths, eligible, gold, bins, seed, n=N, dose=DOSE):
    y, lengths = np.asarray(labels), np.asarray(lengths)
    rng = np.random.default_rng(seed)
    long_ids = np.flatnonzero(eligible & (lengths >= 8))
    background = rng.choice(long_ids, n - dose, replace=False)
    available = np.setdiff1d(long_ids, background)
    random_slots = rng.choice(available, dose, replace=False)
    length_bins = np.minimum(lengths // 4, 8)
    key = lambda a: Counter(zip(y[a].tolist(), length_bins[a].tolist()))
    quota = key(random_slots)
    targets = {name: [] for name in METHODS}
    for (label, length_bin), count in sorted(quota.items()):
        candidates = available[(y[available] == label) & (length_bins[available] == length_bin)]
        targets['long_random'].extend(random_slots[(y[random_slots] == label) & (length_bins[random_slots] == length_bin)].tolist())
        order = np.lexsort((rng.random(len(candidates)), np.abs(gold[candidates] - np.median(gold[candidates]))))
        ranked = candidates[order]
        targets['long_mid'].extend(ranked[:count].tolist())
        targets['long_fd'].extend(balanced_bins(candidates, bins, count, rng))
        mid_pool = ranked[:max(count, (len(candidates) + 1) // 2)]
        targets['long_fd_mid'].extend(balanced_bins(mid_pool, bins, count, rng))
    selections = {}
    for name, target in targets.items():
        arr = np.sort(np.concatenate([background, np.asarray(target, dtype=np.int64)]))
        if len(arr) != n or len(np.unique(arr)) != n or not eligible[arr].all() or key(np.asarray(target)) != quota:
            raise ValueError(f'Invalid selection {name}')
        selections[name] = arr
    selections['pool_random'] = np.sort(rng.choice(np.flatnonzero(eligible), n, replace=False))
    return selections, background, quota


def main():
    ds, meta = read_data()
    texts = ds['pool']['sentence']
    labels = np.asarray(ds['pool']['label'])
    lengths = np.array([len(text.split()) for text in texts])
    forbidden_val = {norm(s) for s in ds['val']['sentence']}
    forbidden_test = {norm(s) for s in ds['test']['sentence']}
    evaluation_grams = set()
    for split in ('val', 'test'):
        for sentence in ds[split]['sentence']:
            evaluation_grams.update(eightgrams(sentence))
    exact_eligible = np.array([norm(t) not in forbidden_val and norm(t) not in forbidden_test for t in texts])
    fragment_eligible = np.array([not (eightgrams(t) & evaluation_grams) for t in texts])
    eligible = exact_eligible & fragment_eligible
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'selections').mkdir(exist_ok=True)
    gold, fd, bins, info = calculate_scores(texts, labels, eligible)
    score_path = ROOT / 'scores.npz'
    np.savez_compressed(score_path, eligible=eligible, gold_probability=gold, frequency_distance=fd, fd_bin=bins)
    audit = []
    for seed in SEEDS:
        selections, bg, quota = select(labels, lengths, eligible, gold, bins, seed)
        for name, arr in selections.items():
            manifest = {'name': name, 'selection_seed': seed, 'indices': arr.tolist(), 'n': N, 'ratio': .1,
                        'pool_sha256': meta['splits']['pool']['sha256'],
                        'provenance': {'method': 'series 06 SCDP-inspired, see PROTOCOL.md',
                                       'scores_sha256': sha(score_path), 'label_budget': 'entire eligible pool'}}
            manifest['sha256'] = digest(manifest)
            (ROOT / 'selections' / f'{name}__s{seed}.json').write_text(json.dumps(manifest, indent=2) + '\n')
            target = np.setdiff1d(arr, bg) if name != 'pool_random' else arr
            audit.append({'method': name, 'seed': seed, 'n': len(arr), 'background_n': int(len(np.intersect1d(arr, bg))),
                          'positive_fraction': float(labels[arr].mean()), 'mean_words': float(lengths[arr].mean()),
                          'target_mean_gold': float(np.nanmean(gold[target])),
                          'target_proxy_disagreement': float(np.nanmean(gold[target] < .5)),
                          'target_fd_bins': np.bincount(bins[target][bins[target] >= 0], minlength=10).tolist(),
                          'group_quotas': [[int(a), int(b), int(v)] for (a, b), v in sorted(quota.items())]})
    info.update({'pool_sha256': meta['splits']['pool']['sha256'], 'score_sha256': sha(score_path),
                 'excluded_val': sum(norm(s) in forbidden_val for s in texts),
                 'excluded_test': sum(norm(s) in forbidden_test for s in texts),
                 'excluded_shared_8gram': int((~fragment_eligible).sum()),
                 'excluded_total': int((~eligible).sum()), 'eligible_count': int(eligible.sum())})
    (ROOT / 'audit.json').write_text(json.dumps({'scores': info, 'selections': audit}, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()

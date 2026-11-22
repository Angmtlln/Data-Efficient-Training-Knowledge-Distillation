"""OOF linear-classifier gradient dynamics, inspired by G-SNR (EMNLP 2026)."""
import json
from pathlib import Path
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

import prepare as base


def main():
    started = time.time()
    ds, meta = base.read_data()
    texts = ds['pool']['sentence']
    y = np.asarray(ds['pool']['label'])
    lengths = np.asarray([len(s.split()) for s in texts])
    scores = np.load(base.ROOT / 'scores.npz')
    eligible = scores['eligible']
    ids = np.flatnonzero(eligible)
    groups = np.asarray([base.norm(texts[i]) for i in ids])
    early = np.full((3, len(y)), np.nan)
    late = np.full_like(early, np.nan)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261122)
    for fold, (tr, te) in enumerate(splitter.split(ids, y[ids], groups)):
        if set(groups[tr]) & set(groups[te]):
            raise ValueError('Proxy train/heldout overlap')
        train, held = ids[tr], ids[te]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        x = vec.fit_transform([texts[i] for i in train])
        z = vec.transform([texts[i] for i in held])
        gradient_scale = np.sqrt(np.asarray(z.multiply(z).sum(axis=1)).ravel() + 1)
        for member, seed in enumerate((11, 23, 37)):
            clf = SGDClassifier(loss='log_loss', alpha=1e-5, random_state=seed + fold, average=True)
            rng = np.random.default_rng(seed + fold)
            for epoch in range(8):
                order = rng.permutation(len(train))
                clf.partial_fit(x[order], y[train][order], classes=np.array([0, 1]))
                if epoch in (0, 7):
                    p = clf.predict_proba(z)[np.arange(len(held)), y[held]]
                    (early if epoch == 0 else late)[member, held] = (1 - p) * gradient_scale
    if not np.isfinite(early[:, ids]).all() or not np.isfinite(late[:, ids]).all():
        raise ValueError('Missing OOF gradients')
    score = np.full(len(y), np.nan)
    e, l = early[:, ids].mean(0), late[:, ids].mean(0)
    variance = late[:, ids].var(0)
    score[ids] = ((e - l) / (e + .001)) / (variance + .001)
    path = base.ROOT / 'gsnr_scores.npz'
    np.savez_compressed(path, early=early, late=late, score=score)
    audit = []
    for seed in base.SEEDS:
        original, background, quota = base.select(y, lengths, eligible, scores['gold_probability'], scores['fd_bin'], seed)
        prior = json.loads((base.ROOT / 'selections' / f'long_random__s{seed}.json').read_text())
        if not np.array_equal(original['long_random'], prior['indices']):
            raise ValueError('Control differs')
        available = np.setdiff1d(np.flatnonzero(eligible & (lengths >= 8)), background)
        length_bin = np.minimum(lengths // 4, 8)
        rng = np.random.default_rng(100000 + seed)
        selected = []
        for (label, group), count in sorted(quota.items()):
            candidates = available[(y[available] == label) & (length_bin[available] == group)]
            order = np.lexsort((rng.random(len(candidates)), -score[candidates]))
            selected.extend(candidates[order[:count]].tolist())
        result = np.sort(np.concatenate([background, np.asarray(selected, dtype=int)]))
        if len(result) != base.N or len(np.unique(result)) != base.N or not eligible[result].all():
            raise ValueError('Selection failed')
        manifest = {'name': 'long_gsnr', 'selection_seed': seed, 'indices': result.tolist(), 'n': base.N,
                    'ratio': .1, 'pool_sha256': meta['splits']['pool']['sha256'],
                    'provenance': {'method': 'linear-proxy G-SNR-inspired, see PROTOCOL_GSNR.md',
                                   'scores_sha256': base.sha(path), 'label_budget': 'entire eligible pool'}}
        manifest['sha256'] = base.digest(manifest)
        (base.ROOT / 'selections' / f'long_gsnr__s{seed}.json').write_text(json.dumps(manifest, indent=2) + '\n')
        audit.append({'seed': seed, 'selected_score_mean': float(np.mean(score[selected])),
                      'selected_gold_mean': float(np.mean(scores['gold_probability'][selected])),
                      'selected_proxy_disagreement': float(np.mean(scores['gold_probability'][selected] < .5)),
                      'shared_background': len(background), 'target': len(selected)})
    (base.ROOT / 'gsnr_audit.json').write_text(json.dumps({'seconds': time.time() - started,
        'score_sha256': base.sha(path), 'score_quantiles': np.quantile(score[ids], [0, .1, .5, .9, 1]).tolist(),
        'selection': audit}, indent=2) + '\n')
    print(json.dumps({'seconds': time.time() - started, 'selection': audit}, indent=2), flush=True)


if __name__ == '__main__':
    main()

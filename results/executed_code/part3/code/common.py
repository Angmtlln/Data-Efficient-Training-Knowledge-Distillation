from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read_data():
    from datasets import load_from_disk
    ds = load_from_disk(str(ROOT / 'data/sst2'))
    meta = json.loads((ROOT / 'data/snapshot.json').read_text())
    for name in ['pool','val','test']:
        rows = list(zip(ds[name]['orig_idx'], ds[name]['sentence'], ds[name]['label']))
        if digest(rows) != meta['splits'][name]['sha256']:
            raise ValueError(f'Dataset content mismatch: {name}')
    return ds, meta

def quotas(labels, n):
    import numpy as np
    classes, counts = np.unique(labels, return_counts=True)
    raw = counts / counts.sum() * n
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw-q), kind='stable')[:n-q.sum()]: q[i] += 1
    return dict(zip(map(int, classes), map(int, q)))

def ranked(labels, scores, n, seed, eligible=None):
    import numpy as np
    labels, scores = np.asarray(labels), np.asarray(scores)
    eligible = np.arange(len(labels)) if eligible is None else np.asarray(eligible)
    rng = np.random.default_rng(seed)
    out = []
    for label, count in quotas(labels, n).items():
        ids = eligible[labels[eligible] == label]
        if len(ids) < count: raise ValueError('Insufficient class capacity')
        out.extend(ids[np.lexsort((rng.random(len(ids)), -scores[ids]))[:count]].tolist())
    return sorted(out)

def manifest(name, ids, seed, meta, provenance):
    ids = sorted(map(int, ids))
    assert len(ids) == len(set(ids)) == 6534
    assert min(ids) >= 0 and max(ids) < 65349
    value = dict(name=name, selection_seed=seed, indices=ids, n=len(ids), ratio=.1,
                 pool_sha256=meta['splits']['pool']['sha256'], provenance=provenance)
    value['sha256'] = digest(value)
    dump(ROOT / 'selections' / f'{name}__s{seed}.json', value)

def environment():
    import importlib.metadata as im
    packages = {}
    for name in ['torch','datasets','numpy','scikit-learn','transformers','accelerate','sentence-transformers']:
        try: packages[name] = im.version(name)
        except im.PackageNotFoundError: pass
    return dict(python=sys.version, platform=platform.platform(), packages=packages, timestamp=time.time())

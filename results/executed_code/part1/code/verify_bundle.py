"""Check portable artifacts before uploading or GPU training."""
import json
from pathlib import Path
from common import ROOT,read_data,digest
from datasets import load_from_disk


def main():
    integrity=ROOT/'bundle_manifest.json'
    if integrity.exists():
        from checkpoints import sha
        for name,expected_hash in json.loads(integrity.read_text()).items():
            assert sha(ROOT/name)==expected_hash, f'File integrity mismatch: {name}'
    ds,meta=read_data()
    encoded=load_from_disk(str(ROOT/'data/tokenized'))
    tokenization=json.loads((ROOT/'data/tokenization.json').read_text())
    assert tokenization['snapshot']==meta
    for k in ds:
        assert len(encoded[k])==len(ds[k])
        assert encoded[k]['label']==ds[k]['label']
        assert max(map(len,encoded[k]['input_ids']))<=64
    manifests=list((ROOT/'selections').glob('*.json'))
    groups=json.loads((ROOT/'experiment_groups.json').read_text())
    expected={f'{name}__s{seed}.json' for methods in groups.values() for name in methods for seed in [17,42,101]}
    assert {p.name for p in manifests}==expected, 'Missing or unexpected selection files'
    for path in manifests:
        m=json.loads(path.read_text()); sha=m.pop('sha256');assert sha==digest(m),path
        assert m['pool_sha256']==meta['splits']['pool']['sha256']
        ids=m['indices'];assert len(ids)==len(set(ids))==6534
        assert min(ids)>=0 and max(ids)<65349
    if (ROOT/'data/skill_roles.json').exists():
        roles=json.loads((ROOT/'data/skill_roles.json').read_text())
        excluded=set(roles['proxy_train'])
        for ids in roles['references'].values():
            assert not excluded & set(ids)
            excluded.update(ids)
        for path in manifests:
            m=json.loads(path.read_text())
            if m['name'].startswith(('skill_','negation_','contrast_')):
                assert not set(m['indices']) & excluded
    print(f'OK: {len(manifests)} selections, data hashes, labels, bounds, tokenization and reference isolation')

if __name__=='__main__':main()

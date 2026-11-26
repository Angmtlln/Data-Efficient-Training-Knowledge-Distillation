import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from common import read_data,digest
if __name__=='__main__':
    expected=json.loads((ROOT/'SHA256.json').read_text())
    for name,sha in expected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,f'Changed file: {name}'
    ds,meta=read_data()
    for seed in [1103,1201,1301]:
        values=[]
        for name in ['roots','mixed_independent','mixed_paired','mixed_rank']:
            d=json.loads((ROOT/'selections'/f'{name}__s{seed}.json').read_text());sha=d.pop('sha256');assert digest(d)==sha
            assert d['pool_sha256']==meta['splits']['pool']['sha256']
            ids=d['indices'];assert len(ids)==len(set(ids))==6534
            for a,b in d['pairs']:assert a in ids and b in ids and ds['pool'][a]['label']!=ds['pool'][b]['label']
            values.append(d)
        assert values[1]['indices']==values[2]['indices']==values[3]['indices']
        assert values[1]['pairs']==values[2]['pairs']==values[3]['pairs']
        assert len(set(values[0]['indices'])&set(values[1]['indices']))==6423
    print('Verified source, code, data hashes and all 12 selections')

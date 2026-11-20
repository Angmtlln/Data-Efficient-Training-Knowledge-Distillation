"""Verify matching, deterministic construction, source hashes and input provenance."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1];ROOT=REPO/'results/proxy_mixture'
sys.path.insert(0,str(HERE))
from prepare import stratified_mixture,sha,METHODS
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'))
from common import read_data,digest

class SelectionTests(unittest.TestCase):
    def test_matched_real_selections(self):
        ds,snapshot=read_data();y=np.asarray(ds['pool']['label']);lengths=np.asarray([len(t.split()) for t in ds['pool']['sentence']])
        md=json.loads((ROOT/'proxy/metadata.json').read_text());self.assertEqual(md['pool_sha256'],snapshot['splits']['pool']['sha256'])
        self.assertEqual(sha(ROOT/'proxy/lexical_features.npz'),md['lexical_sha256'])
        f=np.load(ROOT/'proxy/lexical_features.npz'); bins=np.minimum(lengths//4,8)
        hist=lambda a:Counter(zip(y[a].tolist(),bins[a].tolist()))
        for seed in (17,42,101):
            path=ROOT/'proxy'/f'gold_probability_s{seed}.npz';self.assertEqual(sha(path),next(x['sha256'] for x in md['seeds'] if x['seed']==seed))
            p=np.load(path)['p_gold'];generated,bg,q=stratified_mixture(y,lengths,p,f['rare_idf'],f['cluster'],seed)
            reference=None
            for name in METHODS:
                obj=json.loads((ROOT/'selections'/f'{name}__s{seed}.json').read_text());expected=obj.pop('sha256')
                self.assertEqual(digest(obj),expected);self.assertEqual(obj['pool_sha256'],snapshot['splits']['pool']['sha256'])
                ids=np.asarray(obj['indices']);self.assertTrue(np.array_equal(ids,generated[name]));self.assertEqual(len(ids),6534)
                self.assertEqual(len(np.unique(ids)),6534);self.assertTrue(np.all(lengths[ids]>=8));self.assertTrue(set(bg)<=set(ids))
                hist_this=hist(ids)
                if reference is None:reference=hist_this
                else:self.assertEqual(hist_this,reference)
                self.assertEqual(hist(np.setdiff1d(ids,bg)),q)

if __name__=='__main__':unittest.main()

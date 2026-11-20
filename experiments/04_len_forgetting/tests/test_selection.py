"""Exercise pair matching and validate the actual archived selections."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
from prepare import make_subsets
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'))
from common import read_data, digest

class SelectionTests(unittest.TestCase):
    def test_small_handmade_pair(self):
        labels=np.tile([0,1],60);lengths=np.full(120,9);scores=np.zeros(120,dtype=int)
        scores[:10]=1
        subsets,meta=make_subsets(labels,lengths,scores,7,n=30,dose=6)
        c=set(subsets['len8_matched_random']);t=set(subsets['len8_forgetting'])
        self.assertEqual(len(c & t),24)
        self.assertEqual(sum(scores[list(t-c)]),6)
        self.assertEqual(sum(scores[list(c-t)]),0)
        self.assertEqual(sum(labels[list(t-c)]),sum(labels[list(c-t)]))
        self.assertEqual(meta['dose'],6)

    def test_actual_selections(self):
        ds,meta=read_data(); y=np.asarray(ds['pool']['label']); words=np.array([len(s.split()) for s in ds['pool']['sentence']])
        for seed in [17,42,101]:
            scores=np.load(REPO/f'results/inputs/part1/forgetting_s{seed}.npz')['forgetting']
            records={}
            for name in ['random','len8','len8_matched_random','len8_forgetting']:
                obj=json.loads((REPO/f'results/len_forgetting/selections/{name}__s{seed}.json').read_text())
                sha=obj.pop('sha256');self.assertEqual(sha,digest(obj));self.assertEqual(obj['pool_sha256'],meta['splits']['pool']['sha256'])
                ids=np.asarray(obj['indices']); self.assertEqual(len(ids),6534);self.assertEqual(len(set(ids)),6534)
                if name!='random':self.assertTrue(np.all(words[ids]>=8))
                records[name]=set(ids)
            control=records['len8_matched_random']; treatment=records['len8_forgetting']
            self.assertEqual(len(control & treatment),6207)
            old=np.asarray(sorted(control-treatment));new=np.asarray(sorted(treatment-control))
            self.assertTrue(np.all(scores[old]==0)); self.assertTrue(np.all(scores[new]>0))
            for label in [0,1]:
                for b in range(2,9):
                    self.assertEqual(int(np.sum((y[old]==label)&(np.minimum(words[old]//4,8)==b))),
                                     int(np.sum((y[new]==label)&(np.minimum(words[new]//4,8)==b))))

if __name__=='__main__':unittest.main()

"""Check the fixed external sample against the pinned, locally available IMDb data."""
import json
import os
from pathlib import Path
import sys
import unittest
from datasets import load_from_disk

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
sys.path.insert(0,str(HERE));from prepare_imdb import clean,REV
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'));from common import digest

class ExternalManifestTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('IMDB_TEST_PATH'),'set IMDB_TEST_PATH to a verified local test split')
    def test_fixed_sample(self):
        data=load_from_disk(os.environ['IMDB_TEST_PATH'])
        manifest=json.loads((REPO/'results/proxy_mixture/imdb_manifest.json').read_text());ids=manifest['indices']
        self.assertEqual(manifest['revision'],REV);self.assertEqual(len(ids),2000);self.assertEqual(len(set(ids)),2000)
        labels=[int(data[i]['label']) for i in ids];texts=[clean(data[i]['text']) for i in ids]
        self.assertEqual(labels.count(0),1000);self.assertEqual(labels.count(1),1000)
        self.assertEqual(digest(list(zip(ids,labels,texts))),manifest['data_sha256'])

if __name__=='__main__':unittest.main()

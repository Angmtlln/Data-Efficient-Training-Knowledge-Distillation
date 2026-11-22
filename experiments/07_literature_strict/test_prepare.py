import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('study06_prepare', HERE / 'prepare.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SelectionTests(unittest.TestCase):
    def test_waterfill_caps_and_reproducibility(self):
        bins = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2])
        ids = np.arange(len(bins))
        result = mod.balanced_bins(ids, bins, 7, np.random.default_rng(42))
        self.assertEqual(result, mod.balanced_bins(ids, bins, 7, np.random.default_rng(42)))
        self.assertEqual(len(set(result)), 7)
        self.assertEqual(np.bincount(bins[result]).tolist(), [2, 3, 2])

    def test_real_selections_match_protocol(self):
        ds, meta = mod.read_data()
        scores = np.load(mod.ROOT / 'scores.npz')
        labels = np.asarray(ds['pool']['label'])
        lengths = np.array([len(t.split()) for t in ds['pool']['sentence']])
        forbidden = {mod.norm(s) for split in ('val', 'test') for s in ds[split]['sentence']}
        forbidden_grams = set()
        for split in ('val', 'test'):
            for text in ds[split]['sentence']:
                forbidden_grams.update(mod.eightgrams(text))
        for idx, text in enumerate(ds['pool']['sentence']):
            if scores['eligible'][idx]:
                self.assertFalse(mod.eightgrams(text) & forbidden_grams)
        for seed in mod.SEEDS:
            selected, background, quotas = mod.select(labels, lengths, scores['eligible'], scores['gold_probability'], scores['fd_bin'], seed)
            for name, ids in selected.items():
                manifest = json.loads((mod.ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
                expected = manifest.pop('sha256')
                self.assertEqual(mod.digest(manifest), expected)
                self.assertEqual(manifest['pool_sha256'], meta['splits']['pool']['sha256'])
                np.testing.assert_array_equal(ids, manifest['indices'])
                self.assertEqual(len(set(ids)), mod.N)
                self.assertFalse(any(mod.norm(ds['pool'][int(i)]['sentence']) in forbidden for i in ids))
                if name != 'pool_random':
                    self.assertTrue(np.isin(background, ids).all())
                    self.assertTrue((lengths[ids] >= 8).all())
                    self.assertEqual(len(np.setdiff1d(ids, background)), mod.DOSE)
                    target = np.setdiff1d(ids, background)
                    from collections import Counter
                    actual = Counter(zip(labels[target].tolist(), np.minimum(lengths[target] // 4, 8).tolist()))
                    self.assertEqual(actual, quotas)

    def test_gsnr_uses_same_background_and_quotas(self):
        ds, meta = mod.read_data()
        original_scores = np.load(mod.ROOT / 'scores.npz')
        path = mod.ROOT / 'gsnr_scores.npz'
        gsnr = np.load(path)
        labels = np.asarray(ds['pool']['label'])
        lengths = np.array([len(t.split()) for t in ds['pool']['sentence']])
        eligible = original_scores['eligible']
        self.assertTrue(np.isfinite(gsnr['score'][eligible]).all())
        from collections import Counter
        for seed in mod.SEEDS:
            _, bg, quota = mod.select(labels, lengths, eligible, original_scores['gold_probability'], original_scores['fd_bin'], seed)
            manifest = json.loads((mod.ROOT / 'selections' / f'long_gsnr__s{seed}.json').read_text())
            expected = manifest.pop('sha256')
            self.assertEqual(mod.digest(manifest), expected)
            self.assertEqual(manifest['pool_sha256'], meta['splits']['pool']['sha256'])
            self.assertEqual(manifest['provenance']['scores_sha256'], mod.sha(path))
            ids = np.asarray(manifest['indices'])
            self.assertEqual(len(set(ids)), mod.N)
            self.assertTrue(eligible[ids].all())
            self.assertTrue((lengths[ids] >= 8).all())
            self.assertTrue(np.isin(bg, ids).all())
            target = np.setdiff1d(ids, bg)
            self.assertEqual(Counter(zip(labels[target].tolist(), np.minimum(lengths[target] // 4, 8).tolist())), quota)

    def test_root_enrichment_and_strict_eligibility(self):
        ds, meta = mod.read_data()
        scores = np.load(mod.ROOT / 'scores.npz')
        roots = np.load(mod.ROOT / 'inferred_roots.npz')['is_root']
        labels = np.asarray(ds['pool']['label'])
        lengths = np.asarray([len(t.split()) for t in ds['pool']['sentence']])
        from collections import Counter
        for seed in mod.SEEDS:
            _, bg, quota = mod.select(labels, lengths, scores['eligible'], scores['gold_probability'], scores['fd_bin'], seed)
            for name in ('long_roots_matched', 'long_roots_max'):
                manifest = json.loads((mod.ROOT / 'selections' / f'{name}__s{seed}.json').read_text())
                checksum = manifest.pop('sha256')
                self.assertEqual(mod.digest(manifest), checksum)
                self.assertEqual(manifest['pool_sha256'], meta['splits']['pool']['sha256'])
                ids = np.asarray(manifest['indices'])
                self.assertEqual(len(set(ids)), mod.N)
                self.assertTrue(scores['eligible'][ids].all())
                self.assertTrue((lengths[ids] >= 8).all())
                if name == 'long_roots_matched':
                    self.assertTrue(np.isin(bg, ids).all())
                    target = np.setdiff1d(ids, bg)
                    self.assertTrue(roots[target].all())
                    self.assertEqual(Counter(zip(labels[target].tolist(), np.minimum(lengths[target] // 4, 8).tolist())), quota)
                else:
                    self.assertEqual(int(roots[ids].sum()), int((roots & scores['eligible'] & (lengths >= 8)).sum()))


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

import prune_completed as tool


class PruningTests(unittest.TestCase):
    def test_completed_only_and_preserves_best_weights(self):
        with tempfile.TemporaryDirectory() as d:
            prior = tool.base.ROOT, tool.base.REPO
            repo = Path(d)
            tool.base.REPO = repo
            tool.base.ROOT = repo / 'results/literature_strict'
            try:
                runid = 'long_random__ss17__ts17'
                run = tool.base.ROOT / 'runs' / runid
                run.mkdir(parents=True)
                ckpts = repo / 'outputs/literature_strict/checkpoints' / runid
                best, old = ckpts / 'checkpoint-2', ckpts / 'checkpoint-1'
                best.mkdir(parents=True)
                old.mkdir()
                for file in ('config.json', 'model.safetensors', 'optimizer.pt'):
                    (best / file).write_text('test')
                (run / 'metrics.json').write_text(json.dumps({'name': 'long_random', 'selection_seed': 17,
                    'training_seed': 17, 'pilot': False, 'split': 'test',
                    'best_checkpoint': '/old/machine/checkpoint-2'}))
                with self.assertRaises(ValueError):
                    tool.prune(runid)
                self.assertTrue(old.exists())
                (run / 'predictions.csv').write_text('test')
                tool.prune(runid)
                self.assertTrue((best / 'model.safetensors').exists())
                self.assertFalse((best / 'optimizer.pt').exists())
                self.assertFalse(old.exists())
            finally:
                tool.base.ROOT, tool.base.REPO = prior


if __name__ == '__main__':
    unittest.main()

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


class CheckpointHashTests(unittest.TestCase):
    def test_uses_study_root_and_hashes_contents(self):
        study = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('strict_hash', study / 'hash_checkpoints.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.ROOT, study.parents[1] / 'results/literature_strict')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            path.write_bytes(b'model-weights')
            self.assertEqual(module.file_hash(path), hashlib.sha256(b'model-weights').hexdigest())

    def test_cached_evaluation_requires_matching_weights(self):
        study = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('strict_hash', study / 'hash_checkpoints.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        saved = {'data_sha256': 'data', 'selection_sha256': 'selection', 'checkpoint_sha256': 'weights'}
        self.assertTrue(module.evaluation_matches(saved, 'data', 'selection', 'weights'))
        self.assertFalse(module.evaluation_matches(saved, 'data', 'selection', 'other'))
        self.assertFalse(module.evaluation_matches(saved, 'other', 'selection', 'weights'))
        self.assertFalse(module.evaluation_matches(saved, 'data', 'other', 'weights'))
        del saved['checkpoint_sha256']
        self.assertFalse(module.evaluation_matches(saved, 'data', 'selection', 'weights'))


if __name__ == '__main__':
    unittest.main()

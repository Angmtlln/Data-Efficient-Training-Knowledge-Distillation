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


if __name__ == '__main__':
    unittest.main()

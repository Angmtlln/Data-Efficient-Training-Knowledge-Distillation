import tempfile
from pathlib import Path
import unittest

import audit_original_sst


class OriginalSSTTests(unittest.TestCase):
    def test_mirror_normalization_and_binary_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'sst.txt'
            source.write_text('1 good -lrb- yes -rrb- \\/ good\n0 bad\n')
            self.assertEqual(audit_original_sst.read(source), [('good ( yes ) / good', 1), ('bad', 0)])
            source.write_text('2 neutral\n')
            with self.assertRaises(ValueError):
                audit_original_sst.read(source)


if __name__ == '__main__':
    unittest.main()

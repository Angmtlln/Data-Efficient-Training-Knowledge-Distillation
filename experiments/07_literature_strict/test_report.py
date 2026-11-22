import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('literature_strict_report', Path(__file__).resolve().parent / 'report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_report_combines_three_complete_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = report.base.ROOT
            report.base.ROOT = root
            try:
                entries = [{'method': name, 'accuracy_mean': .9, 'accuracy_sd': .01, 'macro_f1_mean': .89,
                            'clean_accuracy_mean': .901, 'delta_vs_long_random_pp': [0., .1, -.1]}
                           for name in report.METHODS]
                documents = {
                    'analysis.json': {'summary': entries},
                    'original_sst_analysis.json': {'summary': entries},
                    'imdb_holdout_analysis.json': {'summary': entries, 'sample': {'n': 2000,
                        'normalized_exact_sst2_any_split': 0, 'containing_any_8token_ngram_from_sst2': 3,
                        'over_64_tokens': 1952}},
                    'audit.json': {'scores': {'eligible_count': 61199, 'excluded_total': 4150}},
                    'selection_cost_audit.json': {'selections': [{'name': name, 'tokens_total_per_epoch': 100000,
                                                                  'mean_words': 12., 'positive_fraction': .5}
                                                                  for name in report.METHODS]},
                    'inferred_roots_audit.json': {'eligible_long_root_rows': 5419},
                    'source_diversity_audit.json': {'selections': [{'name': name, 'inferred_root_fraction': .2,
                        'distinct_uniquely_assigned_parents': 3500, 'shared_with_uniquely_assigned_val_parent': 1000}
                        for name in report.METHODS for _ in report.SEEDS]},
                    'difficulty_audit.json': {'selections': [{'name': name, 'scope': 'sample',
                        'mean_gold_probability': .8, 'proxy_disagreement_fraction': .05}
                        for name in report.METHODS for _ in report.SEEDS]},
                }
                for name, doc in documents.items():
                    (root / name).write_text(json.dumps(doc))
                report.main()
                text = (root / 'REPORT.md').read_text()
                self.assertIn('SST-2 test', text)
                self.assertIn('IMDb', text)
                self.assertIn('90.00 ± 1.00', text)
                self.assertTrue((root / 'comparison.png').stat().st_size > 1000)
            finally:
                report.base.ROOT = previous


if __name__ == '__main__':
    unittest.main()

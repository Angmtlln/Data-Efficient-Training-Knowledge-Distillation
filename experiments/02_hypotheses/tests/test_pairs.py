"""Synthetic unit fixtures only; these never enter experiment datasets."""
import unittest
from collections import Counter
from pairs import tokens, mine_pairs, disjoint_pairs, build_selections


class PairTests(unittest.TestCase):
    def test_negation_is_preserved(self):
        self.assertNotEqual(tokens('this is good'), tokens('this is not good'))
        texts = ['the acting in this film is good', 'the acting in this film is not good']
        edges, _ = mine_pairs(texts, [1, 0], max_span=1)
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0]['removed'], edges[0]['added']}, {'', 'not'})

    def test_conflicting_duplicates_excluded(self):
        edges, stats = mine_pairs(['the acting in this film is good'] * 2, [0, 1])
        self.assertEqual(edges, [])
        self.assertEqual(stats['normalized_conflicting_texts'], 1)

    def test_no_same_label_pair(self):
        edges, _ = mine_pairs(['the acting in this film is good', 'the acting in this film is bad'], [1, 1])
        self.assertEqual(edges, [])

    def test_controls_budget_matching_and_reproducibility(self):
        texts = [f'the acting in film number {i} is good' for i in range(200)]
        labels = [i % 2 for i in range(200)]
        edges = [dict(a=2*i, b=2*i+1, similarity=.9) for i in range(10)]
        result, chosen = build_selections(texts, labels, edges, 20, .4, 17)
        self.assertEqual(result, build_selections(texts, labels, edges, 20, .4, 17)[0])
        ref = Counter(labels[i] for i in result['pairs'])
        for ids in result.values():
            self.assertEqual(len(set(ids)), 20)
            self.assertEqual(Counter(labels[i] for i in ids), ref)
        for edge in chosen:
            self.assertEqual(sum(i in result['broken_pairs'] for i in [edge['a'], edge['b']]), 1)
        self.assertFalse(set(result['pairs']) & set(result['matched_random']))

    def test_insufficient_pairs_fails(self):
        with self.assertRaises(ValueError):
            build_selections(['a'] * 20, [0, 1] * 10, [], 10, .2, 0)

    def test_disjoint(self):
        edges = [dict(a=0, b=1, similarity=.9), dict(a=0, b=2, similarity=.8)]
        self.assertEqual(len(disjoint_pairs(edges, 1)), 1)


if __name__ == '__main__':
    unittest.main()

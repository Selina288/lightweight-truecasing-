import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from size_matched_genre import aggregate, context_indices, sample_indices, sampled_resources
from logreg import features


class EqualBudgetTests(unittest.TestCase):
    def rows(self):
        return [dict(document_id=g, sentence_id='s', genre=g, split='train',
                     original_token=w, lowercase_input=w.lower(), casing_label=label)
                for g in ('answers', 'email')
                for w, label in [('First', 'initial_uppercase'), ('iPhone', 'mixed_case'),
                                 ('last', 'lowercase'), ('.', 'uncased')]]

    def test_exact_budget_pool_determinism_and_validation(self):
        rows = self.rows()
        for condition, expected in [('in_genre', 'answers'), ('other_genres', 'email')]:
            chosen = sample_indices(rows, 'answers', condition, 2, 11)
            self.assertEqual(len(set(chosen)), 2)
            self.assertEqual({rows[i]['genre'] for i in chosen}, {expected})
            self.assertTrue(all(rows[i]['casing_label'] != 'uncased' for i in chosen))
            self.assertEqual(chosen, sample_indices(rows, 'answers', condition, 2, 11))
        with self.assertRaises(ValueError):
            sample_indices(rows, 'answers', 'in_genre', 4, 11)

    def test_original_context_and_sample_only_decoder(self):
        rows = self.rows()
        self.assertEqual(features(rows, 2, 'context_char')['previous'], 'iphone')
        self.assertEqual(context_indices(rows, [2]), {1, 2, 3})
        self.assertEqual(context_indices(rows, [0]), {0, 1})
        vocabulary, mixed = sampled_resources(rows, [0, 2])
        self.assertEqual(vocabulary, {'first', 'last'})
        self.assertEqual(mixed, {})
        before = features(rows, 2, 'context_char')
        rows[1].update(original_token='SECRET', casing_label='all_uppercase', upos='PROPN')
        self.assertEqual(before, features(rows, 2, 'context_char'))

    def test_paired_gaps_use_within_seed_differences(self):
        rows = []
        for seed, a, b in [(11, .8, .6), (22, .5, .6)]:
            for condition, value in [('in_genre', a), ('other_genres', b)]:
                rows.append(dict(target_genre='answers', condition=condition, sampling_seed=seed,
                    training_cased_tokens=2, cased_tokens=10, macro_f1=value,
                    exact_accuracy=value, capitalization_required_accuracy=value))
        summary, gaps = aggregate(rows)
        self.assertEqual(len(summary), 2)
        self.assertAlmostEqual(gaps[0]['macro_f1_gap_mean'], .05)
        self.assertEqual(gaps[0]['macro_f1_positive_pairs'], 1)
        self.assertAlmostEqual(gaps[0]['macro_f1_gap_std'], .2121320343559643)


if __name__ == '__main__':
    unittest.main()

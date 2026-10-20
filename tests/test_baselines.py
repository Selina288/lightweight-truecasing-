import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baselines import Scores, fit_lexicon, predict


class BaselineTests(unittest.TestCase):
    def test_lexicon_and_unseen(self):
        rows = [dict(lowercase_input='apple', original_token=x)
                for x in ('Apple', 'Apple', 'apple')]
        lexicon = fit_lexicon(rows)
        self.assertEqual(predict('most_frequent_case', 'apple', False, lexicon), 'Apple')
        self.assertEqual(predict('most_frequent_case', 'unseen', False, lexicon), 'unseen')
        self.assertEqual(predict('sentence_initial', 'hello', True, lexicon), 'Hello')
        self.assertEqual(predict('sentence_initial', 'hello', False, lexicon), 'hello')

    def test_surface_and_class_scores_differ(self):
        score = Scores()
        for gold, pred in [('apple', 'apple'), ('Apple', 'apple'),
                           ('USA', 'USA'), ('iPhone', 'Iphone'), ('.', '.')]:
            score.add(gold, pred)
        result = score.summary()
        self.assertEqual(result['cased_tokens'], 4)
        self.assertEqual(result['exact_accuracy'], 0.5)
        self.assertAlmostEqual(result['macro_f1'], (2/3 + 0 + 1 + 0)/4)
        self.assertAlmostEqual(result['capitalization_required_accuracy'], 1/3)
        # Same class does not guarantee the mixed-case spelling is correct.
        mixed = Scores()
        mixed.add('iPhone', 'ipHone')
        self.assertEqual(mixed.summary()['exact_accuracy'], 0)
        self.assertEqual(mixed.metrics()[-1]['f1'], 1)


if __name__ == '__main__':
    unittest.main()

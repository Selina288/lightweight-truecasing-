import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from error_analysis import transition,diagnostics


class ErrorAnalysisTests(unittest.TestCase):
    def test_pair_transitions(self):
        self.assertEqual(transition('Apple','Apple','Apple'),'both_correct')
        self.assertEqual(transition('Apple','apple','Apple'),'corrected_by_C')
        self.assertEqual(transition('Apple','Apple','apple'),'regressed_by_C')
        self.assertEqual(transition('Apple','APPLE','apple'),'both_wrong')

    def test_exact_error_can_have_correct_class(self):
        flags=diagnostics('iPhone','ipHone',False,False)
        self.assertIn('same_class_wrong_surface',flags)
        self.assertIn('gold_mixed_case',flags)
        self.assertIn('unseen_in_training',flags)
        self.assertNotIn('undercapitalized',flags)

    def test_observable_flags_do_not_assert_cause(self):
        flags=diagnostics('Apple','apple',True,True)
        self.assertIn('undercapitalized',flags)
        self.assertIn('first_token',flags)
        self.assertFalse(any('entity' in f or 'brand' in f for f in flags))


if __name__=='__main__':
    unittest.main()

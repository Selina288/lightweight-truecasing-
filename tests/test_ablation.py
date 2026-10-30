import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from logreg import FEATURE_SETS, features


class AblationTests(unittest.TestCase):
    def rows(self):
        return [dict(document_id='d',sentence_id='s',lowercase_input=w,original_token=w)
                for w in ['the','iphone','works']]

    def test_feature_sets_are_nested(self):
        rows=self.rows()
        a,b,c=[features(rows,1,setting) for setting in FEATURE_SETS]
        self.assertEqual({k:b[k] for k in a},a)
        self.assertEqual({k:c[k] for k in b},b)
        self.assertNotIn('previous',a)
        self.assertEqual(b['previous'],'the')
        self.assertEqual(c['prefix2'],'ip')
        self.assertEqual(c['suffix3'],'one')
        grams={k:v for k,v in c.items() if k.startswith('char_ngram=')}
        self.assertIn('char_ngram=^ip',grams)
        self.assertAlmostEqual(sum(v*v for v in grams.values()),1.0)

    def test_lexical_ignores_neighbor_identity(self):
        rows=self.rows()
        before=features(rows,1,'lexical')
        context_before=features(rows,1,'context')
        rows[0]['lowercase_input']='an'
        self.assertEqual(features(rows,1,'lexical'),before)
        self.assertNotEqual(features(rows,1,'context'),context_before)

    def test_all_variants_ignore_gold(self):
        rows=self.rows()
        before=[features(rows,1,s) for s in FEATURE_SETS]
        for row in rows:
            row.update(original_token='SECRET',casing_label='all_uppercase',upos='PROPN')
        self.assertEqual(before,[features(rows,1,s) for s in FEATURE_SETS])


if __name__=='__main__':
    unittest.main()

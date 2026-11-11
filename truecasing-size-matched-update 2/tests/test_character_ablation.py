import csv
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from character_ablation import family_features, load_train_dev
from logreg import features


class CharacterFamilyTests(unittest.TestCase):
    def rows(self):
        return [dict(document_id='d', sentence_id='s', lowercase_input=w,
                     original_token=w, casing_label='lowercase') for w in ['an', 'iphone', 'works']]

    def test_families_preserve_context_and_recombine_exactly(self):
        rows = self.rows()
        base = features(rows, 1, 'context')
        affix = family_features(rows, 1, 'context_affixes')
        grams = family_features(rows, 1, 'context_ngrams')
        self.assertEqual(affix | grams, features(rows, 1, 'context_char'))
        self.assertEqual(set(affix) & set(grams), set(base))
        self.assertTrue(all(affix[k] == grams[k] == v for k,v in base.items()))
        self.assertTrue(any(k.startswith('prefix') for k in affix))
        self.assertFalse(any(k.startswith('char_ngram=') for k in affix))
        self.assertFalse(any(k.startswith(('prefix','suffix')) for k in grams))
        self.assertAlmostEqual(sum(v*v for k,v in grams.items() if k.startswith('char_ngram=')), 1)

    def test_gold_and_pos_cannot_change_features(self):
        rows = self.rows()
        before = [family_features(rows,1,v) for v in ['context_affixes','context_ngrams']]
        for row in rows:
            row.update(original_token='SECRET', casing_label='mixed_case', upos='PROPN')
        self.assertEqual(before,[family_features(rows,1,v) for v in ['context_affixes','context_ngrams']])

    def test_only_train_dev_are_required_and_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ['train','dev']:
                with (root/f'tokens_{split}.tsv').open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=['split','document_id'],delimiter='\t')
                    writer.writeheader();writer.writerow(dict(split=split,document_id=split))
            train,dev=load_train_dev(root)
            self.assertEqual((len(train),len(dev)),(1,1))
            p=root/'tokens_dev.tsv';p.write_text('split\tdocument_id\ndev\ttrain\n')
            with self.assertRaises(ValueError):load_train_dev(root)


if __name__ == '__main__':
    unittest.main()

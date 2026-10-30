import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossgenre import partition, fit_source
from preprocess import classify_casing, EXPECTED_GENRES


def row(genre,split,word,document=None):
    return dict(genre=genre,split=split,document_id=document or f'{genre}-{split}',
        sentence_id=f'{genre}-{split}-1',token_id='1',original_token=word,
        lowercase_input=word.lower(),casing_label=classify_casing(word))


class CrossGenreTests(unittest.TestCase):
    def test_target_train_and_other_dev_are_excluded(self):
        train=[row(g,'train','Example') for g in EXPECTED_GENRES]
        dev=[row(g,'dev','Development') for g in EXPECTED_GENRES]
        for target in EXPECTED_GENRES:
            source, evaluation=partition(train,dev,target)
            self.assertEqual({r['genre'] for r in source},EXPECTED_GENRES-{target})
            self.assertEqual({r['genre'] for r in evaluation},{target})
            self.assertTrue(all(r['split']=='train' for r in source))
            self.assertTrue(all(r['split']=='dev' for r in evaluation))

    def test_document_overlap_is_rejected(self):
        with self.assertRaises(AssertionError):
            partition([row('email','train','Word',document='same')],
                      [row('reviews','dev','Word',document='same')],'reviews')

    def test_source_fit_cannot_memorize_target_only_word(self):
        target='reviews'
        source=[]
        for g in sorted(EXPECTED_GENRES-{target}):
            for word in ['apple','Monday','USA','iPhone']:
                source.extend([row(g,'train',word)]*4)
        train=source+[row(target,'train','TargetOnly')]
        dev=[row(target,'dev','TargetOnly')]
        selected,evaluation=partition(train,dev,target)
        for variant in ['context','context_char']:
            bundle,audit=fit_source(selected,variant)
            self.assertNotIn('targetonly',bundle['vocabulary'])
            self.assertNotIn('targetonly',bundle['mixed_lexicon'])
            self.assertNotIn('word=targetonly',bundle['vectorizer'].vocabulary_)
            self.assertNotIn(target,audit['training_genres'])
            self.assertTrue(audit['converged'])


if __name__=='__main__':
    unittest.main()

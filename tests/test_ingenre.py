import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ingenre import partition_in_genre
from crossgenre import fit_source
from preprocess import EXPECTED_GENRES,classify_casing


def row(g,split,word,document=None):
    return dict(genre=g,split=split,document_id=document or g+'-'+split,sentence_id=g+'-'+split+'-1',
        token_id='1',original_token=word,lowercase_input=word.lower(),casing_label=classify_casing(word))


class InGenreTests(unittest.TestCase):
    def test_only_target_training_and_target_dev_are_selected(self):
        train=[row(g,'train','Word') for g in EXPECTED_GENRES]
        dev=[row(g,'dev','Example') for g in EXPECTED_GENRES]
        for target in EXPECTED_GENRES:
            a,b=partition_in_genre(train,dev,target)
            self.assertEqual(len(a),1);self.assertEqual(len(b),1)
            self.assertEqual(a[0]['genre'],target);self.assertEqual(a[0]['split'],'train')
            self.assertEqual(b[0]['genre'],target);self.assertEqual(b[0]['split'],'dev')

    def test_overlap_and_empty_partitions_rejected(self):
        with self.assertRaises(AssertionError):
            partition_in_genre([row('email','train','I','same')],[row('email','dev','I','same')],'email')
        with self.assertRaises(ValueError):
            partition_in_genre([row('email','train','I')],[row('reviews','dev','I')],'email')

    def test_other_genre_word_cannot_enter_vocab(self):
        train=[row('email','train',w) for w in ['apple','Monday','USA','iPhone']]*4
        train += [row('reviews','train','OutsideOnly')]
        selected,_=partition_in_genre(train,[row('email','dev','OutsideOnly')],'email')
        for variant in ('context','context_char'):
            model,audit=fit_source(selected,variant)
            self.assertEqual(audit['training_genres'],['email'])
            self.assertNotIn('outsideonly',model['vocabulary'])
            self.assertNotIn('word=outsideonly',model['vectorizer'].vocabulary_)
            self.assertNotIn('outsideonly',model['mixed_lexicon'])


if __name__=='__main__':unittest.main()

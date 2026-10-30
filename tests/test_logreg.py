import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from logreg import features, reconstruct, evaluate, solver_matrix, predict_rows
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import tempfile


class LogregTests(unittest.TestCase):
    def rows(self):
        return [dict(document_id='d',sentence_id=s,lowercase_input=w,original_token=g,genre='email')
                for s,w,g in [('1','i','I'),('1','like','like'),('2','apple','Apple')]]

    def test_features_ignore_gold(self):
        rows=self.rows()
        before=[features(rows,i) for i in range(len(rows))]
        for row in rows:
            row['original_token']='LEAKED'
            row['casing_label']='mixed_case'
            row['upos']='PROPN'
        self.assertEqual(before,[features(rows,i) for i in range(len(rows))])
        self.assertEqual(before[0]['previous'],'<BOS>')
        self.assertEqual(before[1]['next'],'<EOS>')
        self.assertEqual(before[2]['previous'],'<BOS>')

    def test_surface_reconstruction(self):
        self.assertEqual(reconstruct('iphone','mixed_case',{'iphone':'iPhone'}),'iPhone')
        self.assertEqual(reconstruct('unseen','mixed_case',{}),'unseen')
        self.assertEqual(reconstruct('usa','all_uppercase',{}),'USA')
        self.assertEqual(reconstruct('i','initial_uppercase',{}),'I')

    def test_train_save_reload_predict(self):
        rows = self.rows()*4
        vectorizer = DictVectorizer(dtype=float)
        x = solver_matrix(vectorizer.fit_transform(features(rows,i) for i in range(len(rows))))
        classifier = LogisticRegression(solver='saga',random_state=42,max_iter=1000,tol=0.01)
        classifier.fit(x,['initial_uppercase','lowercase','initial_uppercase']*4)
        bundle = dict(vectorizer=vectorizer,classifier=classifier,mixed_lexicon={})
        expected = predict_rows(bundle,self.rows())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib'
            joblib.dump(bundle,path)
            self.assertEqual(predict_rows(joblib.load(path),self.rows()),expected)
        self.assertEqual(len(expected[0]),3)

    def test_grouped_scores(self):
        grouped=evaluate(self.rows(),['I','like','apple'],{'i','like'})
        self.assertEqual(grouped['ALL'].summary()['exact_accuracy'],2/3)
        self.assertEqual(grouped['seen'].summary()['exact_accuracy'],1)
        self.assertEqual(grouped['unseen'].summary()['exact_accuracy'],0)
        self.assertEqual(grouped['genre:email'].summary()['cased_tokens'],3)


if __name__=='__main__':
    unittest.main()

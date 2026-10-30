"""Compare lexical, context and context+character models on train/dev only.
Run: python src/ablation.py
"""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from baselines import LABELS, read_tokens, fit_lexicon, write_tsv
from logreg import FEATURE_SETS, features, solver_matrix, predict_rows, evaluate

PARAMETERS = dict(solver='saga', C=1.0, max_iter=500, tol=0.001, random_state=42)


def run(data_dir, output_dir, model_dir):
    train_path, dev_path = data_dir/'tokens_train.tsv', data_dir/'tokens_dev.tsv'
    train, dev = list(read_tokens(train_path)), list(read_tokens(dev_path))
    lexicon = fit_lexicon(train)
    mixed = fit_lexicon(r for r in train if r['casing_label']=='mixed_case')
    indices = [i for i,r in enumerate(train) if r['casing_label'] in LABELS]
    targets = [train[i]['casing_label'] for i in indices]
    output_dir.mkdir(parents=True,exist_ok=True)
    model_dir.mkdir(parents=True,exist_ok=True)
    summary, per_class, records = [], [], []
    # This loop fits every vocabulary, decoder and model exclusively on train.
    for feature_set in FEATURE_SETS:
        start = time.time()
        print('Training:',feature_set,flush=True)
        vectorizer = DictVectorizer(dtype=float)
        x = solver_matrix(vectorizer.fit_transform(features(train,i,feature_set) for i in indices))
        classifier = LogisticRegression(**PARAMETERS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            classifier.fit(x,targets)
        converged = not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        if not converged:
            raise RuntimeError(f'{feature_set} did not converge; no completed comparison written.')
        bundle = dict(vectorizer=vectorizer,classifier=classifier,mixed_lexicon=mixed,
                      vocabulary=set(lexicon),feature_set=feature_set,
                      feature_version='ablation_v1')
        outputs, labels = predict_rows(bundle,dev)
        joblib.dump(bundle,model_dir/f'{feature_set}.joblib')
        for group,score in sorted(evaluate(dev,outputs,lexicon).items()):
            summary.append(dict(system=feature_set,split='dev',group=group,**score.summary()))
            per_class.extend(dict(system=feature_set,split='dev',group=group,**m) for m in score.metrics())
        prediction_rows = [dict(document_id=r['document_id'],sentence_id=r['sentence_id'],
            token_id=r['token_id'],genre=r['genre'],input=r['lowercase_input'],gold=r['original_token'],
            predicted_label=label,predicted_form=out) for r,label,out in zip(dev,labels,outputs)]
        write_tsv(data_dir/f'ablation_{feature_set}_dev_predictions.tsv',prediction_rows)
        record = dict(feature_set=feature_set,features=x.shape[1],
                      iterations=classifier.n_iter_.tolist(),converged=converged,
                      seconds=round(time.time()-start,2))
        records.append(record)
        main_score = next(r for r in summary if r['system']==feature_set and r['group']=='ALL')
        print(feature_set,'macro-F1:',round(main_score['macro_f1'],4),
              'accuracy:',round(main_score['exact_accuracy'],4),record,flush=True)
        del x, bundle, classifier, vectorizer
    write_tsv(output_dir/'dev_comparison.tsv',summary)
    write_tsv(output_dir/'dev_per_class.tsv',per_class)
    metadata = dict(split='dev',test_used=False,parameters=PARAMETERS,
        training_cased_tokens=len(indices),runs=records,python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path,dev_path)},
        control='Same train/dev, classifier parameters, labels, decoder and scoring; features alone vary.',
        character_features='Prefixes/suffixes length 1-4; boundary-marked binary character 2-4 grams with per-token L2 normalization.',
        scoring='Four-class macro-F1 from reconstructed forms; uncased tokens excluded.',
        caution='One fixed-parameter development comparison, not final test or held-out-genre results.')
    (output_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/ablation'))
    parser.add_argument('--model-dir',type=Path,default=Path('models/ablation'))
    args=parser.parse_args()
    run(args.data_dir,args.output_dir,args.model_dir)

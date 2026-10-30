"""Train a contextual truecaser on official train; evaluate only official dev.

Run from project root: python3 src/logreg.py
"""
import argparse
import json
import platform
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from baselines import LABELS, SYSTEMS, Scores, fit_lexicon, predict, read_tokens, write_tsv
from preprocess import classify_casing


def solver_matrix(x):
    # SAGA requires 32-bit sparse index arrays; this corpus is far below the limit.
    x = x.tocsr()
    if max(x.shape) >= 2**31 or x.nnz >= 2**31:
        raise ValueError("Sparse matrix exceeds 32-bit solver limits")
    x.indices = x.indices.astype("int32")
    x.indptr = x.indptr.astype("int32")
    return x


def sentence_key(row):
    return row['document_id'], row['sentence_id']


def is_first(rows, i):
    return i == 0 or sentence_key(rows[i]) != sentence_key(rows[i-1])


FEATURE_SETS = ('lexical', 'context', 'context_char')


def features(rows, i, feature_set='context'):
    """Features use only lowercase inputs and sentence boundaries, never gold."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(feature_set)
    word = rows[i]['lowercase_input']
    first = is_first(rows, i)
    result = dict(word=word, sentence_initial=float(first), length_scaled=len(word)/20.0)
    if feature_set in ('context', 'context_char'):
        last = i+1 == len(rows) or sentence_key(rows[i]) != sentence_key(rows[i+1])
        result['previous'] = '<BOS>' if first else rows[i-1]['lowercase_input']
        result['next'] = '<EOS>' if last else rows[i+1]['lowercase_input']
    if feature_set == 'context_char':
        for n in range(1,5):
            if len(word) >= n:
                result[f'prefix{n}'] = word[:n]
                result[f'suffix{n}'] = word[-n:]
        # Boundary-marked binary character 2-4 grams, L2-normalized per token.
        marked = '^'+word+'$'
        grams = {marked[j:j+n] for n in (2,3,4) for j in range(len(marked)-n+1)}
        scale = len(grams)**-0.5 if grams else 1.0
        for gram in grams:
            result['char_ngram='+gram] = scale
    return result


def initial_capital(word):
    for i, char in enumerate(word):
        if char.lower() != char.upper():
            return word[:i]+char.upper()+word[i+1:]
    return word


def reconstruct(word, label, mixed_lexicon):
    if label == 'lowercase':
        return word
    if label == 'initial_uppercase':
        return initial_capital(word)
    if label == 'all_uppercase':
        return word.upper()
    if label == 'mixed_case':
        return mixed_lexicon.get(word, word)
    raise ValueError(label)


def predict_rows(bundle, rows):
    indices = [i for i,r in enumerate(rows) if classify_casing(r['lowercase_input']) != 'uncased']
    outputs = [r['lowercase_input'] for r in rows]
    labels = ['uncased'] * len(rows)
    if indices:
        x = solver_matrix(bundle['vectorizer'].transform(features(rows,i,bundle.get('feature_set','context')) for i in indices))
        predictions = bundle['classifier'].predict(x)
        for i, label in zip(indices,predictions):
            labels[i] = str(label)
            outputs[i] = reconstruct(rows[i]['lowercase_input'],label,bundle['mixed_lexicon'])
    return outputs, labels


def evaluate(rows, outputs, vocabulary):
    grouped = defaultdict(Scores)
    for row, output in zip(rows,outputs):
        for group in ('ALL','genre:'+row['genre'],
                      'seen' if row['lowercase_input'] in vocabulary else 'unseen'):
            grouped[group].add(row['original_token'],output)
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/logreg'))
    parser.add_argument('--model',type=Path,default=Path('models/logreg.joblib'))
    args = parser.parse_args()
    start = time.time()
    train = list(read_tokens(args.data_dir/'tokens_train.tsv'))
    dev = list(read_tokens(args.data_dir/'tokens_dev.tsv'))
    lexicon = fit_lexicon(train)
    mixed = fit_lexicon(r for r in train if r['casing_label']=='mixed_case')
    indices = [i for i,r in enumerate(train) if r['casing_label'] in LABELS]
    vectorizer = DictVectorizer(dtype=float)
    x = solver_matrix(vectorizer.fit_transform(features(train,i) for i in indices))
    y = [train[i]['casing_label'] for i in indices]
    classifier = LogisticRegression(solver='saga',C=1.0,max_iter=500,tol=0.001,random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        classifier.fit(x,y)
    converged = not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    bundle = dict(vectorizer=vectorizer,classifier=classifier,mixed_lexicon=mixed,
                  vocabulary=set(lexicon),feature_version='word_context_position_length_v1')
    outputs, labels = predict_rows(bundle,dev)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.model.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,args.model)
    summaries, classes = [], []
    for system in (*SYSTEMS,'logistic_regression'):
        predictions = outputs if system=='logistic_regression' else [
            predict(system,r['lowercase_input'],is_first(dev,i),lexicon) for i,r in enumerate(dev)]
        for group,score in sorted(evaluate(dev,predictions,lexicon).items()):
            summaries.append(dict(system=system,split='dev',group=group,**score.summary()))
            classes.extend(dict(system=system,split='dev',group=group,**m) for m in score.metrics())
    write_tsv(args.output_dir/'dev_comparison.tsv',summaries)
    write_tsv(args.output_dir/'dev_per_class.tsv',classes)
    # Prediction text is kept under the ignored processed-data directory.
    prediction_rows = [dict(document_id=r['document_id'],sentence_id=r['sentence_id'],
        token_id=r['token_id'],genre=r['genre'],input=r['lowercase_input'],
        gold=r['original_token'],predicted_label=label,predicted_form=output)
        for r,label,output in zip(dev,labels,outputs)]
    write_tsv(args.data_dir/'logreg_dev_predictions.tsv',prediction_rows)
    metadata = dict(train_file='tokens_train.tsv',evaluation_file='tokens_dev.tsv',
        test_used=False,training_cased_tokens=len(indices),feature_count=x.shape[1],
        features=['current lowercase word','previous lowercase word','next lowercase word',
                  'sentence initial position','scaled word length'],
        gold_POS_used=False,vectorizer_fit='train only',mixed_case_lexicon='train only',
        solver='saga',C=1.0,max_iter=500,tolerance=0.001,random_state=42,
        iterations=classifier.n_iter_.tolist(),converged=converged,
        sklearn_version=sklearn.__version__,python_version=platform.python_version(),
        elapsed_seconds=round(time.time()-start,2),
        scoring='Four-class macro-F1 derived from reconstructed forms; uncased tokens excluded')
    (args.output_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for row in summaries:
        if row['group']=='ALL':
            print(f"{row['system']}: accuracy={row['exact_accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, capitalization accuracy={row['capitalization_required_accuracy']:.4f}",flush=True)
    print('Converged:',converged,'; iterations:',classifier.n_iter_,'; test not read.',flush=True)
    if not converged:
        raise RuntimeError('Model did not converge; inspect metadata before reporting results.')


if __name__=='__main__':
    main()

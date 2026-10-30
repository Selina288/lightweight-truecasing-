"""Five leave-one-genre-out development evaluations; no test data are read.
Run after src/ablation.py: python src/crossgenre.py
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

from ablation import PARAMETERS
from baselines import LABELS, fit_lexicon, read_tokens, write_tsv
from logreg import features, solver_matrix, predict_rows, evaluate
from preprocess import EXPECTED_GENRES

VARIANTS = ('context','context_char')


def partition(train, dev, target):
    if target not in EXPECTED_GENRES:
        raise ValueError(target)
    source = [r for r in train if r['genre'] != target]
    evaluation = [r for r in dev if r['genre'] == target]
    if not source or not evaluation:
        raise ValueError('Empty training or evaluation partition')
    assert all(r['split']=='train' and r['genre']!=target for r in source)
    assert all(r['split']=='dev' and r['genre']==target for r in evaluation)
    assert not {r['document_id'] for r in source} & {r['document_id'] for r in evaluation}
    return source, evaluation


def fit_source(source, variant):
    lexicon = fit_lexicon(source)
    mixed = fit_lexicon(r for r in source if r['casing_label']=='mixed_case')
    indices = [i for i,r in enumerate(source) if r['casing_label'] in LABELS]
    vectorizer = DictVectorizer(dtype=float)
    x = solver_matrix(vectorizer.fit_transform(features(source,i,variant) for i in indices))
    classifier = LogisticRegression(**PARAMETERS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        classifier.fit(x,[source[i]['casing_label'] for i in indices])
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        raise RuntimeError('Source-only classifier did not converge')
    bundle = dict(vectorizer=vectorizer,classifier=classifier,mixed_lexicon=mixed,
        vocabulary=set(lexicon),feature_set=variant,feature_version='ablation_v1')
    audit = dict(training_cased_tokens=len(indices),feature_count=x.shape[1],
        training_documents=len({r['document_id'] for r in source}),
        training_genres=sorted({r['genre'] for r in source}),
        vocabulary_size=len(lexicon),mixed_lexicon_size=len(mixed),
        iterations=classifier.n_iter_.tolist(),converged=True)
    return bundle,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/crossgenre'))
    parser.add_argument('--model-dir',type=Path,default=Path('models/crossgenre'))
    parser.add_argument('--reference-dir',type=Path,default=Path('models/ablation'))
    parser.add_argument('--reference-metadata',type=Path,default=Path('results/ablation/run_metadata.json'))
    args=parser.parse_args()
    train_path,dev_path=args.data_dir/'tokens_train.tsv',args.data_dir/'tokens_dev.tsv'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path,dev_path)}
    metadata=json.loads(args.reference_metadata.read_text())
    assert metadata['data_sha256']==hashes,'Reference dataset mismatch; rerun ablation.py'
    assert metadata['parameters']==PARAMETERS,'Reference hyperparameter mismatch'
    assert all(r['converged'] for r in metadata['runs'])
    references={v:joblib.load(args.reference_dir/f'{v}.joblib') for v in VARIANTS}
    for v,b in references.items():
        assert b['feature_set']==v
        assert all(b['classifier'].get_params()[k]==value for k,value in PARAMETERS.items())
    train,dev=list(read_tokens(train_path)),list(read_tokens(dev_path))
    assert {r['genre'] for r in train} == EXPECTED_GENRES
    assert all(r['split']=='train' for r in train) and all(r['split']=='dev' for r in dev)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.model_dir.mkdir(parents=True,exist_ok=True)
    summaries,classes,runs=[],[],[]
    for target in sorted(EXPECTED_GENRES):
        source,target_dev=partition(train,dev,target)
        # Both conditions' seen/unseen scores use this SAME source-only vocabulary.
        source_vocabulary=set(r['lowercase_input'] for r in source)
        for variant in VARIANTS:
            print('Training held-out:',target,variant,flush=True)
            started=time.time()
            bundle,audit=fit_source(source,variant)
            assert bundle['vocabulary']==source_vocabulary
            assert target not in audit['training_genres'] and len(audit['training_genres'])==4
            joblib.dump(bundle,args.model_dir/f'{target}_{variant}.joblib')
            for condition,model in [('all_genres',references[variant]),('held_out',bundle)]:
                outputs,labels=predict_rows(model,target_dev)
                grouped=evaluate(target_dev,outputs,source_vocabulary)
                for group,score in grouped.items():
                    if group.startswith('genre:'):
                        continue
                    prefix=dict(target_genre=target,system=variant,condition=condition,split='dev',group=group)
                    summaries.append(dict(**prefix,**score.summary()))
                    classes.extend(dict(**prefix,**m) for m in score.metrics())
                token_rows=[dict(document_id=r['document_id'],sentence_id=r['sentence_id'],
                    token_id=r['token_id'],genre=r['genre'],input=r['lowercase_input'],gold=r['original_token'],
                    seen_in_source=r['lowercase_input'] in source_vocabulary,
                    predicted_label=label,predicted_form=out)
                    for r,label,out in zip(target_dev,labels,outputs)]
                write_tsv(args.data_dir/f'crossgenre_{target}_{variant}_{condition}_dev_predictions.tsv',token_rows)
            record=dict(target_genre=target,variant=variant,**audit,
                evaluation_cased_tokens=grouped['ALL'].n,evaluation_documents=len({r['document_id'] for r in target_dev}),
                seconds=round(time.time()-started,2))
            runs.append(record)
            pair=[r for r in summaries if r['target_genre']==target and r['system']==variant and r['group']=='ALL']
            print(target,variant,[(r['condition'],round(r['macro_f1'],4)) for r in pair],flush=True)
    changes=[]
    for row in summaries:
        if row['condition']!='held_out':
            continue
        ref=next(r for r in summaries if r['condition']=='all_genres' and all(r[k]==row[k]
            for k in ('target_genre','system','group')))
        assert row['cased_tokens']==ref['cased_tokens']
        changes.append(dict(target_genre=row['target_genre'],system=row['system'],group=row['group'],
            cased_tokens=row['cased_tokens'],all_genres_macro_f1=ref['macro_f1'],held_out_macro_f1=row['macro_f1'],
            macro_f1_drop=ref['macro_f1']-row['macro_f1'],
            all_genres_exact_accuracy=ref['exact_accuracy'],held_out_exact_accuracy=row['exact_accuracy'],
            exact_accuracy_drop=ref['exact_accuracy']-row['exact_accuracy']))
    write_tsv(args.output_dir/'dev_summary.tsv',summaries)
    write_tsv(args.output_dir/'dev_per_class.tsv',classes)
    write_tsv(args.output_dir/'dev_transfer_gaps.tsv',changes)
    audit=dict(test_used=False,evaluation_split='dev',parameters=PARAMETERS,data_sha256=hashes,
        sklearn_version=sklearn.__version__,python_version=platform.python_version(),runs=runs,
        all_genre_reference='Pretrained ablation context/context_char models; metadata and data hashes checked.',
        source_only_resources='Each fold fits vectorizer, vocabulary, mixed-case decoder and classifier on four source genres only.',
        subgroup_definition='Both conditions use the held-out fold source vocabulary to define identical seen/unseen subsets.',
        limitations='Training size and genre coverage both change; gaps are not isolated causal genre effects. No test data or significance testing.')
    (args.output_dir/'run_metadata.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Completed 10 source-only fits. Test not read.',flush=True)


if __name__=='__main__':
    main()

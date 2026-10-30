"""Complete in-genre-only training and compare all three genre conditions on dev.
Run after ablation.py and crossgenre.py: python src/ingenre.py
No test data are read. Parameters are fixed, not tuned on the target dev set.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import sklearn
from ablation import PARAMETERS
from baselines import read_tokens,write_tsv
from crossgenre import fit_source,VARIANTS
from logreg import predict_rows,evaluate
from preprocess import EXPECTED_GENRES


def partition_in_genre(train,dev,target):
    if target not in EXPECTED_GENRES:raise ValueError(target)
    selected=[r for r in train if r['genre']==target]
    evaluation=[r for r in dev if r['genre']==target]
    if not selected or not evaluation:raise ValueError('Empty in-genre partition')
    assert all(r['split']=='train' for r in selected)
    assert all(r['split']=='dev' for r in evaluation)
    assert not {r['document_id'] for r in selected}&{r['document_id'] for r in evaluation}
    return selected,evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/ingenre'))
    parser.add_argument('--model-dir',type=Path,default=Path('models/ingenre'))
    parser.add_argument('--all-reference',type=Path,default=Path('models/ablation'))
    parser.add_argument('--held-reference',type=Path,default=Path('models/crossgenre'))
    args=parser.parse_args()
    paths=[args.data_dir/'tokens_train.tsv',args.data_dir/'tokens_dev.tsv']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name in ('ablation','crossgenre'):
        meta=json.loads((Path('results')/name/'run_metadata.json').read_text())
        assert meta['data_sha256']==hashes,'Reference data mismatch'
        assert meta['parameters']==PARAMETERS,'Reference parameter mismatch'
        assert all(r['converged'] for r in meta['runs'])
    train,dev=map(lambda p:list(read_tokens(p)),paths)
    assert {r['genre'] for r in train}==EXPECTED_GENRES
    assert all(r['split']=='train' for r in train) and all(r['split']=='dev' for r in dev)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.model_dir.mkdir(parents=True,exist_ok=True)
    summary,classes,runs=[],[],[]
    for target in sorted(EXPECTED_GENRES):
        selected,evaluation=partition_in_genre(train,dev,target)
        # A fixed target-train vocabulary defines the SAME subset in all three conditions.
        target_vocab={r['lowercase_input'] for r in selected}
        for variant in VARIANTS:
            print('Training in-genre:',target,variant,flush=True)
            started=time.time()
            bundle,audit=fit_source(selected,variant)
            assert audit['training_genres']==[target] and bundle['vocabulary']==target_vocab
            model_path=args.model_dir/f'{target}_{variant}.joblib'
            joblib.dump(bundle,model_path)
            references={
                'all_genres':args.all_reference/f'{variant}.joblib',
                'held_out':args.held_reference/f'{target}_{variant}.joblib',
                'in_genre':model_path,
            }
            model_hashes={}
            for condition,path in references.items():
                model=bundle if condition=='in_genre' else joblib.load(path)
                assert model['feature_set']==variant
                assert all(model['classifier'].get_params()[k]==v for k,v in PARAMETERS.items())
                expected_vocab={r['lowercase_input'] for r in train if
                    condition=='all_genres' or (r['genre']==target if condition=='in_genre' else r['genre']!=target)}
                assert model['vocabulary']==expected_vocab
                model_hashes[condition]=hashlib.sha256(path.read_bytes()).hexdigest()
                outputs,labels=predict_rows(model,evaluation)
                grouped=evaluate(evaluation,outputs,target_vocab)
                for group,score in grouped.items():
                    if group.startswith('genre:'):continue
                    fixed_group={'seen':'seen_in_target_train','unseen':'unseen_in_target_train'}.get(group,group)
                    prefix=dict(target_genre=target,system=variant,condition=condition,split='dev',group=fixed_group)
                    summary.append(dict(**prefix,**score.summary()))
                    classes.extend(dict(**prefix,**m) for m in score.metrics())
                predictions=[dict(document_id=r['document_id'],sentence_id=r['sentence_id'],token_id=r['token_id'],
                    genre=r['genre'],input=r['lowercase_input'],gold=r['original_token'],
                    seen_in_target_train=r['lowercase_input'] in target_vocab,
                    predicted_label=label,predicted_form=out) for r,label,out in zip(evaluation,labels,outputs)]
                write_tsv(args.data_dir/f'ingenre_{target}_{variant}_{condition}_dev_predictions.tsv',predictions)
            runs.append(dict(target_genre=target,variant=variant,**audit,
                evaluation_cased_tokens=grouped['ALL'].n,evaluation_documents=len({r['document_id'] for r in evaluation}),
                model_sha256=model_hashes,seconds=round(time.time()-started,2)))
            print(target,variant,[(r['condition'],round(r['macro_f1'],4)) for r in summary
                if r['target_genre']==target and r['system']==variant and r['group']=='ALL'],flush=True)
    write_tsv(args.output_dir/'dev_three_conditions.tsv',summary)
    write_tsv(args.output_dir/'dev_per_class.tsv',classes)
    table=[]
    for target in sorted(EXPECTED_GENRES):
        for variant in VARIANTS:
            matched={c:next(r for r in summary if r['target_genre']==target and r['system']==variant
                and r['condition']==c and r['group']=='ALL') for c in ('all_genres','in_genre','held_out')}
            assert len({r['cased_tokens'] for r in matched.values()})==1
            row=dict(target_genre=target,system=variant,cased_tokens=matched['in_genre']['cased_tokens'])
            for c,r in matched.items():
                row[c+'_macro_f1']=r['macro_f1'];row[c+'_exact_accuracy']=r['exact_accuracy']
            table.append(row)
    write_tsv(args.output_dir/'dev_overview.tsv',table)
    metadata=dict(test_used=False,models_newly_trained=10,parameters=PARAMETERS,data_sha256=hashes,runs=runs,
        sklearn_version=sklearn.__version__,python_version=platform.python_version(),
        training_policy='In-genre vectorizer, decoder, vocabulary and classifier fitted only on target genre train.',
        subset_definition='All three conditions share target-train vocabulary for identical seen/unseen subsets. This differs from crossgenre report source-vocabulary subsets.',
        scope='Fixed-parameter exploratory dev comparison, no condition-specific tuning.',
        limitation='Training sizes differ across all, in-genre and held-out conditions; domain and data-size effects are confounded.')
    (args.output_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Completed ten in-genre fits and three-condition dev comparison. Test not read.',flush=True)


if __name__=='__main__':main()

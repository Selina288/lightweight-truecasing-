"""Evaluate frozen models without fitting. Default dev reproduces recorded experiments.
Run dev before test: python src/final_evaluate.py --split dev
Then: python src/final_evaluate.py --split test
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import joblib
import sklearn
from baselines import SYSTEMS,fit_lexicon,read_tokens,predict,write_tsv
from logreg import evaluate,predict_rows,is_first
from preprocess import EXPECTED_GENRES

VARIANTS=('lexical','context','context_char')


def table(path):return list(read_tokens(path))


def verify_manifest(root,manifest):
    for name,sha in manifest['artifacts'].items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=sha:raise ValueError('Frozen artifact mismatch: '+name)


def verify_dev(root,main,genre):
    old_baselines=table(root/'results/baselines/dev_summary.tsv')
    old_ablation=table(root/'results/ablation/dev_comparison.tsv')
    old_three=table(root/'results/ingenre/dev_three_conditions.tsv')
    numeric=['cased_tokens','exact_accuracy','macro_f1','capitalization_required_tokens','capitalization_required_accuracy']
    n=0
    for row in main:
        old=next(r for r in (old_baselines if row['system'] in SYSTEMS else old_ablation)
                 if r['system']==row['system'] and r['group']==row['group'])
        for k in numeric:
            if abs(float(row[k])-float(old[k]))>1e-12:raise ValueError('Dev main mismatch: '+str(row))
        n+=1
    for row in genre:
        if row['subset_basis']!='target_train':continue
        old=next(r for r in old_three if all(r[k]==row[k] for k in ('target_genre','system','condition','group')))
        for k in numeric:
            if abs(float(row[k])-float(old[k]))>1e-12:raise ValueError('Dev genre mismatch: '+str(row))
        n+=1
    return n


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('dev','test'),default='dev')
    args=parser.parse_args()
    root=Path('.')
    manifest_path=root/'docs/FINAL_FROZEN_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    verify_manifest(root,manifest)
    dev_check=root/'results/final_dev/verification.json'
    if args.split=='test':
        previous=json.loads(dev_check.read_text())
        if not previous['matched_existing_results']:raise ValueError('Dev replication required first')
        if previous['frozen_manifest_sha256']!=hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('Dev replication used a different frozen manifest')
    outdir=root/f'results/final_{args.split}'
    if (outdir/'run_metadata.json').exists():
        raise ValueError('Completed evaluation already exists; preserve it instead of silently overwriting.')
    train=table(root/'data/processed/tokens_train.tsv')
    evaluation_path=root/f'data/processed/tokens_{args.split}.tsv'
    rows=table(evaluation_path)
    assert all(r['split']=='train' for r in train)
    assert all(r['split']==args.split for r in rows)
    assert not {r['document_id'] for r in train}&{r['document_id'] for r in rows}
    assert {r['genre'] for r in rows}==EXPECTED_GENRES
    lexicon=fit_lexicon(train)
    main_summary,main_classes=[],[]
    full_predictions={}
    predictions_dir=root/'data/processed'/f'final_{args.split}_predictions'
    predictions_dir.mkdir(parents=True,exist_ok=True)
    model_hashes={}

    def save_predictions(name,selected,outputs):
        assert len(selected)==len(outputs)
        write_tsv(predictions_dir/f'{name}.tsv',[
            dict(document_id=r['document_id'],sentence_id=r['sentence_id'],token_id=r['token_id'],genre=r['genre'],
                 input=r['lowercase_input'],gold=r['original_token'],predicted_form=p)
            for r,p in zip(selected,outputs)])

    for system in (*SYSTEMS,*VARIANTS):
        if system in SYSTEMS:
            outputs=[predict(system,r['lowercase_input'],is_first(rows,i),lexicon) for i,r in enumerate(rows)]
        else:
            path=root/f'models/ablation/{system}.joblib'
            model=joblib.load(path)
            assert model['vocabulary']==set(lexicon)
            outputs,_=predict_rows(model,rows)
            model_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        full_predictions[system]=outputs
        save_predictions('all_'+system,rows,outputs)
        for group,score in sorted(evaluate(rows,outputs,lexicon).items()):
            prefix=dict(system=system,split=args.split,group=group)
            main_summary.append(dict(**prefix,**score.summary()))
            main_classes.extend(dict(**prefix,**m) for m in score.metrics())
        result=next(r for r in main_summary if r['system']==system and r['group']=='ALL')
        print(args.split,system,round(result['exact_accuracy'],4),round(result['macro_f1'],4),flush=True)

    genre_summary,genre_classes=[],[]
    for target in sorted(EXPECTED_GENRES):
        selected=[r for r in rows if r['genre']==target]
        target_vocab={r['lowercase_input'] for r in train if r['genre']==target}
        source_vocab={r['lowercase_input'] for r in train if r['genre']!=target}
        indices=[i for i,r in enumerate(rows) if r['genre']==target]
        for variant in ('context','context_char'):
            for condition in ('all_genres','in_genre','held_out'):
                if condition=='all_genres':outputs=[full_predictions[variant][i] for i in indices]
                else:
                    folder='ingenre' if condition=='in_genre' else 'crossgenre'
                    path=root/f'models/{folder}/{target}_{variant}.joblib'
                    model=joblib.load(path)
                    assert model['vocabulary']==(target_vocab if condition=='in_genre' else source_vocab)
                    outputs,_=predict_rows(model,selected)
                    model_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                save_predictions(f'{target}_{variant}_{condition}',selected,outputs)
                bases=[('target_train',target_vocab)]
                if condition!='in_genre':bases.append(('source_train',source_vocab))
                for basis,vocab in bases:
                    for group,score in sorted(evaluate(selected,outputs,vocab).items()):
                        if group.startswith('genre:'):continue
                        if group!='ALL':group=group+'_in_'+basis
                        prefix=dict(target_genre=target,system=variant,condition=condition,split=args.split,
                                    subset_basis=basis,group=group)
                        genre_summary.append(dict(**prefix,**score.summary()))
                        genre_classes.extend(dict(**prefix,**m) for m in score.metrics())
    matched=verify_dev(root,main_summary,genre_summary) if args.split=='dev' else None
    outdir.mkdir(parents=True,exist_ok=True)
    for name,data in [('all_systems_summary.tsv',main_summary),('all_systems_per_class.tsv',main_classes),
                      ('genre_conditions_summary.tsv',genre_summary),('genre_conditions_per_class.tsv',genre_classes)]:
        write_tsv(outdir/name,data)
    overview=[]
    for target in sorted(EXPECTED_GENRES):
        for variant in ('context','context_char'):
            record=dict(target_genre=target,system=variant)
            for condition in ('all_genres','in_genre','held_out'):
                r=next(r for r in genre_summary if r['target_genre']==target and r['system']==variant
                    and r['condition']==condition and r['subset_basis']=='target_train' and r['group']=='ALL')
                record[condition+'_macro_f1']=r['macro_f1'];record[condition+'_exact_accuracy']=r['exact_accuracy']
            overview.append(record)
    write_tsv(outdir/'genre_overview.tsv',overview)
    ranking=sorted([r for r in main_summary if r['group']=='ALL'],key=lambda r:(-r['macro_f1'],-r['exact_accuracy']))
    metadata=dict(split=args.split,models_retrained=False,test_used=args.split=='test',
        model_selected_before_test='context_char',primary_metric='Four-class macro-F1 of reconstructed outputs',
        tiebreaker='Exact surface-form accuracy',ranking=[r['system'] for r in ranking],
        evaluation_sha256=hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        frozen_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),model_sha256=model_hashes,
        gold_policy='Observed corpus casing; no gold editing; uncased tokens excluded',
        sklearn_version=sklearn.__version__,python_version=platform.python_version(),
        team_example_confirmation='Pending; not inferred from assistant verification',
        caution='Exploratory dev-informed model selection; final test is not a source for tuning. Genre and training-size effects confounded.')
    (outdir/'verification.json').write_text(json.dumps(dict(matched_existing_results=args.split=='dev',
        matching_dev_rows=matched,frozen_manifest_sha256=metadata['frozen_manifest_sha256']),indent=2)+'\n')
    (outdir/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Completed',args.split,'without model fitting. Dev matching rows:',matched,flush=True)


if __name__=='__main__':main()

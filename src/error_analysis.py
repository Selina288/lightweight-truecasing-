"""Summarize paired B/C development errors without reading test data.
Run: python src/error_analysis.py
Token examples stay in ignored data/processed; aggregate results can be shared.
"""
import argparse
import csv
import hashlib
from collections import Counter,defaultdict
from pathlib import Path
from baselines import LABELS,write_tsv,fit_lexicon,read_tokens
from preprocess import classify_casing,EXPECTED_GENRES


def key(row):
    return row['document_id'],row['sentence_id'],row['token_id']


def load_predictions(path):
    rows=list(read_tokens(path))
    result={key(r):r for r in rows}
    if len(rows)!=len(result):
        raise ValueError('Duplicate prediction IDs')
    return result


def transition(gold,b,c):
    if b==gold and c==gold:return 'both_correct'
    if b!=gold and c==gold:return 'corrected_by_C'
    if b==gold and c!=gold:return 'regressed_by_C'
    return 'both_wrong'


def diagnostics(gold,pred,seen,first):
    """Observable error properties, not explanations or entity annotations."""
    a,b=classify_casing(gold),classify_casing(pred)
    flags=[]
    if a=='lowercase' and b!='lowercase':flags.append('overcapitalized')
    if a!='lowercase' and b=='lowercase':flags.append('undercapitalized')
    if a==b:flags.append('same_class_wrong_surface')
    if a=='mixed_case':flags.append('gold_mixed_case')
    if a=='all_uppercase':flags.append('gold_all_uppercase')
    if not seen:flags.append('unseen_in_training')
    if first:flags.append('first_token')
    return flags


def collect(data_dir):
    train=list(read_tokens(data_dir/'tokens_train.tsv'))
    dev=list(read_tokens(data_dir/'tokens_dev.tsv'))
    assert all(r['split']=='train' for r in train) and all(r['split']=='dev' for r in dev)
    full_vocab=set(r['lowercase_input'] for r in train)
    vocabs={g:set(r['lowercase_input'] for r in train if r['genre']!=g) for g in EXPECTED_GENRES}
    sentences=defaultdict(list)
    first_ids=set()
    for r in dev:
        sid=(r['document_id'],r['sentence_id'])
        if not sentences[sid]:first_ids.add(key(r))
        sentences[sid].append(r)
    conditions={}
    conditions['all_genres']={v:load_predictions(data_dir/f'ablation_{v}_dev_predictions.tsv') for v in ('context','context_char')}
    conditions['held_out']={v:{} for v in ('context','context_char')}
    for g in sorted(EXPECTED_GENRES):
        for v in ('context','context_char'):
            part=load_predictions(data_dir/f'crossgenre_{g}_{v}_held_out_dev_predictions.tsv')
            assert not set(part)&set(conditions['held_out'][v])
            conditions['held_out'][v].update(part)
    ids={key(r) for r in dev}
    records=[]
    for condition,predictions in conditions.items():
        assert all(set(p)==ids for p in predictions.values())
        for r in dev:
            gold=r['original_token']
            if classify_casing(gold)=='uncased':continue
            k=key(r)
            b,c=(predictions[v][k] for v in ('context','context_char'))
            assert b['gold']==c['gold']==gold
            assert b['input']==c['input']==r['lowercase_input']
            vocab=full_vocab if condition=='all_genres' else vocabs[r['genre']]
            sentence=sentences[(r['document_id'],r['sentence_id'])]
            pos=next(i for i,x in enumerate(sentence) if key(x)==k)
            window=sentence[max(0,pos-4):pos+5]
            records.append(dict(condition=condition,**{f:r[f] for f in ['document_id','sentence_id','token_id','genre']},
                input=r['lowercase_input'],gold=gold,gold_class=classify_casing(gold),
                B=b['predicted_form'],C=c['predicted_form'],
                seen=r['lowercase_input'] in vocab,first=k in first_ids,
                transition=transition(gold,b['predicted_form'],c['predicted_form']),
                context=' '.join('['+x['original_token']+']' if key(x)==k else x['original_token'] for x in window)))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/error_analysis'))
    args=parser.parse_args()
    records=collect(args.data_dir)
    errors=defaultdict(Counter);changes=defaultdict(Counter);confusion=Counter();flags=Counter()
    for r in records:
        changes[(r['condition'],r['genre'])][r['transition']]+=1
        for system in ('B','C'):
            prediction=r[system];wrong=prediction!=r['gold']
            groups=['ALL','genre:'+r['genre'],'gold_class:'+r['gold_class'],
                    'seen' if r['seen'] else 'unseen','first' if r['first'] else 'nonfirst']
            for group in groups:
                counter=errors[(r['condition'],system,group)]
                counter['tokens']+=1;counter['errors']+=wrong
            confusion[(r['condition'],system,r['gold_class'],classify_casing(prediction))]+=1
            if wrong:
                for flag in diagnostics(r['gold'],prediction,r['seen'],r['first']):
                    flags[(r['condition'],system,flag)]+=1
    summary=[dict(condition=c,system=s,group=g,tokens=n['tokens'],errors=n['errors'],
                  error_rate=n['errors']/n['tokens']) for (c,s,g),n in sorted(errors.items())]
    transitions=[]
    for (condition,genre),count in sorted(changes.items()):
        transitions.append(dict(condition=condition,genre=genre,
            **{k:count[k] for k in ('both_correct','corrected_by_C','regressed_by_C','both_wrong')},
            net_errors_removed=count['corrected_by_C']-count['regressed_by_C']))
    matrices=[dict(condition=c,system=s,gold_class=g,predicted_class=p,count=confusion[(c,s,g,p)])
        for c in ('all_genres','held_out') for s in ('B','C') for g in LABELS for p in LABELS]
    diagnostic_rows=[dict(condition=c,system=s,diagnostic=f,count=n)
        for (c,s,f),n in sorted(flags.items())]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in [('dev_error_summary.tsv',summary),('dev_B_C_transitions.tsv',transitions),
                      ('dev_confusion.tsv',matrices),('dev_diagnostics.tsv',diagnostic_rows)]:
        write_tsv(args.output_dir/name,rows)
    # Deterministic stratified sample for review; no claim that it estimates prevalence.
    buckets=defaultdict(list)
    for r in records:
        if r['transition']!='both_correct':
            buckets[(r['condition'],r['genre'],r['transition'])].append(r)
    sample=[]
    for bucket,rows in sorted(buckets.items()):
        ranked=sorted(rows,key=lambda r:hashlib.sha256('|'.join(map(str,key(r))).encode()).hexdigest())
        sample.extend(ranked[:3])
    write_tsv(args.data_dir/'error_review_candidates.tsv',sample)
    print('Cased token-condition records:',len(records))
    for row in summary:
        if row['group']=='ALL':print(row)
    print('Review candidates:',len(sample),'(local processed-data directory). Test not read.')


if __name__=='__main__':
    main()

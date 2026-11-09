"""Verify 19 retained review examples against dev, training vocabulary and saved models.
No test data are read. Run from project root: python src/verify_review.py
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import joblib
from baselines import read_tokens
from logreg import predict_rows
from error_analysis import key,transition
from preprocess import classify_casing


def main():
    root=Path('.')
    path=root/'results/error_analysis/dev_reviewed_examples.tsv'
    reviewed=list(read_tokens(path))
    assert len(reviewed)==19
    assert len({(r['condition'],*key(r)) for r in reviewed})==19
    train=list(read_tokens(root/'data/processed/tokens_train.tsv'))
    dev=list(read_tokens(root/'data/processed/tokens_dev.tsv'))
    assert all(r['split']=='train' for r in train) and all(r['split']=='dev' for r in dev)
    sentences=defaultdict(list)
    for r in dev:sentences[(r['document_id'],r['sentence_id'])].append(r)
    cache={};model_hashes={};details=[]
    for number,r in enumerate(reviewed,1):
        sentence=sentences[(r['document_id'],r['sentence_id'])]
        index=next(i for i,x in enumerate(sentence) if key(x)==key(r))
        gold=sentence[index]
        assert gold['original_token']==r['gold'] and gold['lowercase_input']==r['input']
        assert gold['genre']==r['genre'] and classify_casing(r['gold'])==r['gold_class']
        vocab={x['lowercase_input'] for x in train if r['condition']=='all_genres' or x['genre']!=r['genre']}
        assert (r['seen']=='True')==(r['input'] in vocab)
        assert (r['first']=='True')==(index==0)
        assert r['team_confirmed']=='No','Team review must not be inferred by this script.'
        for system,variant in [('B','context'),('C','context_char')]:
            model=(root/f'models/ablation/{variant}.joblib' if r['condition']=='all_genres' else
                   root/f'models/crossgenre/{r["genre"]}_{variant}.joblib')
            if str(model) not in cache:
                cache[str(model)]=joblib.load(model)
                model_hashes[str(model)]=hashlib.sha256(model.read_bytes()).hexdigest()
            predicted,_=predict_rows(cache[str(model)],sentence)
            assert predicted[index]==r[system],(number,system,predicted[index],r[system])
        assert r['transition']==transition(r['gold'],r['B'],r['C'])
        details.append(dict(example=number,condition=r['condition'],genre=r['genre'],
            token_id=r['token_id'],surface_and_models_verified=True,seen_status_verified=True,
            position_verified=True,transition_verified=True,team_confirmed=False))
    result=dict(review_count=len(details),test_used=False,gold_changed=False,models_retrained=False,
        annotation_review='Second assistant context review; not human/team confirmation.',
        reviewed_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_sha256=model_hashes,examples=details)
    (root/'results/error_analysis/review_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('All 19 review examples verified against dev, source vocabulary and reloaded model predictions.')
    print('Team confirmation remains pending. No test read; no labels or models changed.')


if __name__=='__main__':main()

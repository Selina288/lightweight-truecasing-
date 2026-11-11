"""Audit the completed local equal-budget run without opening any test tokens."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import joblib
from sklearn.metrics import accuracy_score, f1_score

from baselines import LABELS, read_tokens
from character_ablation import load_train_dev, verify_frozen
from logreg import predict_rows
from preprocess import classify_casing
from size_matched_genre import CONDITIONS, METRICS, sample_indices, sampled_resources


def verify(root):
    out = root/'results/size_matched_genre'
    metadata = json.loads((out/'run_metadata.json').read_text())
    assert metadata['status'] == 'completed'
    assert metadata['test_tokens_read'] is False
    assert verify_frozen(root) == metadata['frozen_manifest_sha256']
    assert hashlib.sha256((root/'src/size_matched_genre.py').read_bytes()).hexdigest() == metadata['source_sha256']
    train, dev = load_train_dev(root/'data/processed')
    for name, expected in metadata['data_sha256'].items():
        assert hashlib.sha256((root/'data/processed'/name).read_bytes()).hexdigest() == expected
    recorded = list(read_tokens(out/'dev_runs.tsv'))
    keys = {(r['target_genre'], r['condition'], int(r['sampling_seed'])) for r in recorded}
    targets = sorted({r['genre'] for r in train})
    assert keys == {(g,c,s) for g in targets for c in CONDITIONS for s in metadata['sampling_seeds']}
    assert len(recorded) == len(keys) == len(metadata['runs']) == 50
    checked = []
    for audit in metadata['runs']:
        target, condition, seed = (audit[k] for k in ('target_genre','condition','sampling_seed'))
        name = f'{target}_{condition}_seed{seed}'
        local = root/'data/processed/size_matched_genre'
        indices_path = local/(name+'_selection.json')
        assert hashlib.sha256(indices_path.read_bytes()).hexdigest() == audit['selection_sha256']
        indices = json.loads(indices_path.read_text())
        assert indices == sample_indices(train, target, condition, metadata['budget'], seed)
        assert len(set(indices)) == metadata['budget'] == audit['training_cased_tokens']
        assert Counter(train[i]['genre'] for i in indices) == audit['training_genre_counts']
        assert Counter(train[i]['casing_label'] for i in indices) == audit['training_label_counts']
        path = root/'models/size_matched_genre'/(name+'.joblib')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit['model_sha256']
        model = joblib.load(path)
        vocabulary, mixed = sampled_resources(train, indices)
        assert model['vocabulary'] == vocabulary and model['mixed_lexicon'] == mixed
        assert model['feature_set'] == 'context_char'
        assert all(model['classifier'].get_params()[k] == v for k,v in metadata['parameters'].items())
        evaluation = [r for r in dev if r['genre'] == target]
        outputs, _ = predict_rows(model, evaluation)
        saved = list(read_tokens(local/(name+'_predictions.tsv')))
        assert len(evaluation) == len(saved) == len(outputs)
        for r, p, actual in zip(evaluation, saved, outputs):
            assert all(r[k] == p[k] for k in ('document_id','sentence_id','token_id'))
            assert r['original_token'] == p['gold'] and p['predicted_form'] == actual
        selected = [(r['original_token'], p) for r,p in zip(evaluation, outputs)
                    if classify_casing(r['original_token']) in LABELS]
        gold, pred = zip(*selected)
        cap = [(g,p) for g,p in selected if classify_casing(g) != 'lowercase']
        scores = dict(exact_accuracy=accuracy_score(gold,pred),
            macro_f1=f1_score([classify_casing(g) for g in gold], [classify_casing(p) for p in pred],
                              labels=list(LABELS), average='macro', zero_division=0),
            capitalization_required_accuracy=sum(g==p for g,p in cap)/len(cap))
        record = next(r for r in recorded if r['target_genre']==target and r['condition']==condition
                      and int(r['sampling_seed'])==seed)
        assert int(record['cased_tokens']) == len(selected)
        assert all(abs(scores[k]-float(record[k])) < 1e-12 for k in METRICS)
        checked.append(dict(target_genre=target, condition=condition, sampling_seed=seed, **scores))
    for row in read_tokens(out/'dev_summary.tsv'):
        group = [r for r in checked if r['target_genre']==row['target_genre'] and r['condition']==row['condition']]
        for metric in METRICS:
            values = [r[metric] for r in group]
            assert abs(mean(values)-float(row[metric+'_mean'])) < 1e-12
            assert abs(stdev(values)-float(row[metric+'_std'])) < 1e-12
    for row in read_tokens(out/'dev_paired_gaps.tsv'):
        for metric in METRICS:
            values = []
            for seed in metadata['sampling_seeds']:
                pair = {r['condition']:r[metric] for r in checked if r['target_genre']==row['target_genre'] and r['sampling_seed']==seed}
                values.append(pair['in_genre']-pair['other_genres'])
            assert abs(mean(values)-float(row[metric+'_gap_mean'])) < 1e-12
            assert abs(stdev(values)-float(row[metric+'_gap_std'])) < 1e-12
    result = dict(status='passed', models_reloaded_and_predictions_reproduced=len(checked),
        independent_sklearn_three_metric_checks=len(checked), sampling_budgets_and_source_pools_checked=True,
        sampled_only_decoder_and_vocabulary_checked=True, aggregate_means_sample_sds_and_paired_gaps_checked=True,
        original_frozen_artifacts_unchanged=True, test_tokens_read=False,
        note='SD measures overlapping training-subsample variability on fixed dev; no inferential significance claim.')
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    verify(Path('.').resolve())

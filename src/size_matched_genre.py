"""Exploratory equal-label-budget C-model comparison on official dev only.

Run from project root: python src/size_matched_genre.py
Defaults are fixed before inspecting this experiment: 20,000 supervised cased
tokens per condition, five sampling seeds, original C parameters/solver seed.
Original sentence neighbors are retained as LOWERCASE context only. Unselected
gold labels/forms never enter the classifier, vectorizer or mixed-case decoder.
"""
import argparse
from collections import Counter
import hashlib
import json
import platform
import random
from statistics import mean, stdev
import time
import warnings
from pathlib import Path

import joblib
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ablation import PARAMETERS
from baselines import LABELS, fit_lexicon, write_tsv
from character_ablation import independent_check, load_train_dev, verify_frozen
from logreg import evaluate, features, predict_rows, sentence_key, solver_matrix
from preprocess import EXPECTED_GENRES

SEEDS = (11, 22, 33, 44, 55)
CONDITIONS = ('in_genre', 'other_genres')
METRICS = ('macro_f1', 'exact_accuracy', 'capitalization_required_accuracy')


def sample_indices(train, target, condition, budget, seed):
    if target not in EXPECTED_GENRES or condition not in CONDITIONS:
        raise ValueError('Unknown target/condition.')
    candidates = [i for i, r in enumerate(train) if r['casing_label'] in LABELS
                  and ((r['genre'] == target) == (condition == 'in_genre'))]
    if budget <= 0 or budget > len(candidates):
        raise ValueError('Budget must be positive and no larger than either eligible pool.')
    # Separate deterministic random streams for each target and condition.
    salt = int.from_bytes(hashlib.sha256(f'{target}/{condition}/{seed}'.encode()).digest()[:8], 'big')
    return sorted(random.Random(salt).sample(candidates, budget))


def context_indices(train, indices):
    exposed = set(indices)
    for i in indices:
        for j in (i-1, i+1):
            if 0 <= j < len(train) and sentence_key(train[i]) == sentence_key(train[j]):
                exposed.add(j)
    return exposed


def sampled_resources(train, indices):
    selected = [train[i] for i in indices]
    return (set(fit_lexicon(selected)),
            fit_lexicon(r for r in selected if r['casing_label'] == 'mixed_case'))


def aggregate(rows):
    overview, gaps = [], []
    for target in sorted({r['target_genre'] for r in rows}):
        for condition in CONDITIONS:
            selected = [r for r in rows if r['target_genre'] == target and r['condition'] == condition]
            result = dict(target_genre=target, condition=condition, repeats=len(selected),
                          training_cased_tokens=selected[0]['training_cased_tokens'],
                          dev_cased_tokens=selected[0]['cased_tokens'])
            for metric in METRICS:
                values = [r[metric] for r in selected]
                result.update({metric+'_mean': mean(values), metric+'_std': stdev(values) if len(values)>1 else 0.0})
            overview.append(result)
        paired = {}
        for r in rows:
            if r['target_genre'] == target:
                paired.setdefault(r['sampling_seed'], {})[r['condition']] = r
        result = dict(target_genre=target, repeats=len(paired), direction='in_genre_minus_other_genres')
        for metric in METRICS:
            differences = [p['in_genre'][metric]-p['other_genres'][metric] for p in paired.values()]
            result.update({metric+'_gap_mean': mean(differences),
                           metric+'_gap_std': stdev(differences) if len(differences)>1 else 0.0,
                           metric+'_positive_pairs': sum(v > 0 for v in differences)})
        gaps.append(result)
    return overview, gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--budget', type=int, default=20000)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError('Use at least two distinct sampling seeds.')
    root = args.root.resolve()
    out = root/'results/size_matched_genre'
    model_dir = root/'models/size_matched_genre'
    local = root/'data/processed/size_matched_genre'
    if any(p.exists() for p in (out, model_dir, local)):
        raise ValueError('Preserve completed/partial runs; output directories must be absent.')
    frozen_hash = verify_frozen(root)
    train, dev = load_train_dev(root/'data/processed')
    if {r['genre'] for r in train} != EXPECTED_GENRES or {r['genre'] for r in dev} != EXPECTED_GENRES:
        raise ValueError('Expected all five genres.')
    for target in sorted(EXPECTED_GENRES):
        for condition in CONDITIONS:
            sample_indices(train, target, condition, args.budget, args.seeds[0])
    for p in (out, model_dir, local):
        p.mkdir(parents=True)
    rows, classes, runs = [], [], []
    started_all = time.time()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for target in sorted(EXPECTED_GENRES):
        evaluation = [r for r in dev if r['genre'] == target]
        for seed in args.seeds:
            for condition in CONDITIONS:
                started = time.time()
                name = f'{target}_{condition}_seed{seed}'
                indices = sample_indices(train, target, condition, args.budget, seed)
                vocabulary, mixed = sampled_resources(train, indices)
                vectorizer = DictVectorizer(dtype=float)
                matrix = solver_matrix(vectorizer.fit_transform(features(train, i, 'context_char') for i in indices))
                classifier = LogisticRegression(**PARAMETERS)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    classifier.fit(matrix, [train[i]['casing_label'] for i in indices])
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError('Fit did not converge: '+name)
                bundle = dict(vectorizer=vectorizer, classifier=classifier, mixed_lexicon=mixed,
                              vocabulary=vocabulary, feature_set='context_char', feature_version='ablation_v1')
                model_path = model_dir/(name+'.joblib')
                joblib.dump(bundle, model_path, compress=3)
                outputs, predicted_labels = predict_rows(bundle, evaluation)
                score = evaluate(evaluation, outputs, vocabulary)['ALL']
                independent_check(evaluation, outputs, score.summary())
                prefix = dict(target_genre=target, condition=condition, sampling_seed=seed,
                              split='dev', training_cased_tokens=len(indices))
                result = dict(**prefix, **score.summary())
                rows.append(result)
                classes.extend(dict(**prefix, **m) for m in score.metrics())
                selection_path = local/(name+'_selection.json')
                selection_path.write_text(json.dumps(indices)+'\n')
                write_tsv(local/(name+'_predictions.tsv'), [dict(document_id=r['document_id'],
                    sentence_id=r['sentence_id'], token_id=r['token_id'], input=r['lowercase_input'],
                    gold=r['original_token'], predicted_label=label, predicted_form=p)
                    for r, label, p in zip(evaluation, predicted_labels, outputs)])
                exposed = context_indices(train, indices)
                runs.append(dict(**prefix, converged=True, iterations=classifier.n_iter_.tolist(),
                    features=matrix.shape[1], training_documents=len({train[i]['document_id'] for i in indices}),
                    training_sentences=len({sentence_key(train[i]) for i in indices}),
                    training_genre_counts=dict(Counter(train[i]['genre'] for i in indices)),
                    training_label_counts=dict(Counter(train[i]['casing_label'] for i in indices)),
                    lowercase_context_tokens_including_anchors=len(exposed),
                    vocabulary_size=len(vocabulary), mixed_lexicon_size=len(mixed),
                    selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    seconds=round(time.time()-started, 2)))
                write_tsv(out/'dev_runs.tsv', rows)
                print(f'{len(rows)}/50 {name}: F1={result["macro_f1"]:.6f} accuracy={result["exact_accuracy"]:.6f}', flush=True)
                del matrix, vectorizer, classifier, bundle
    overview, gaps = aggregate(rows)
    write_tsv(out/'dev_per_class.tsv', classes)
    write_tsv(out/'dev_summary.tsv', overview)
    write_tsv(out/'dev_paired_gaps.tsv', gaps)
    if verify_frozen(root) != frozen_hash:
        raise ValueError('Historical artifacts changed.')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_hash:
        raise ValueError('Experiment source changed during execution.')
    metadata = dict(status='completed', evaluation_split='dev', test_tokens_read=False,
        original_test_results_known_before_design=True, system='C/context_char',
        budget=args.budget, sampling_seeds=args.seeds, parameters=PARAMETERS,
        supervised_sampling='Uniform sampling without replacement among cased train tokens in each eligible pool; sorted back into original order.',
        other_genres_policy='Pooled other four genres, sampled proportional in expectation to cased-token availability; not balanced by genre.',
        context_policy='Original immediate lowercase neighbors and true sentence boundaries retained; only sampled anchors contribute gold labels/forms or decoder entries.',
        matched_unit='Number of supervised cased tokens, not documents/sentences or total unique lowercase context exposure.',
        seed_policy='Sampling varies; LR solver random_state remains 42 in all fits. Same numeric seed indexes a descriptive pair, not shared training examples.',
        uncertainty='Sample standard deviation over five training subsamples on fixed dev, not a confidence interval over new evaluation data; subsamples can overlap.',
        limits='Controls supervised count, not vocabulary, document count, length, label priors or multi-source diversity. No causal genre-only, target-blind-selection, significance or new confirmatory test claim.',
        runs=runs, models_newly_trained=len(runs), independent_metric_checks_passed=len(runs),
        frozen_artifacts_unchanged=True, frozen_manifest_sha256=frozen_hash,
        data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [root/'data/processed/tokens_train.tsv', root/'data/processed/tokens_dev.tsv']},
        source_sha256=source_hash, environment=dict(python=platform.python_version(), sklearn=sklearn.__version__),
        elapsed_seconds=round(time.time()-started_all, 2))
    (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Completed equal-budget dev follow-up; original frozen artifacts unchanged.', flush=True)


if __name__ == '__main__':
    main()

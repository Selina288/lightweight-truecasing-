"""Exploratory train/dev character-family ablation after the original final test.

Frozen B/C references are loaded without fitting. Only two new variants are fit.
No test tokens are opened. Original code, models and result directories are retained.
Run from project root: python src/character_ablation.py
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
from sklearn.metrics import accuracy_score, f1_score

from ablation import PARAMETERS
from baselines import LABELS, fit_lexicon, read_tokens, write_tsv
from logreg import evaluate, features, predict_rows, reconstruct, solver_matrix
from preprocess import classify_casing

VARIANTS = ('context_affixes', 'context_ngrams')


def family_features(rows, index, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown character-family variant: ' + variant)
    full = features(rows, index, 'context_char')
    if variant == 'context_affixes':
        return {k: v for k, v in full.items() if not k.startswith('char_ngram=')}
    return {k: v for k, v in full.items() if not k.startswith(('prefix', 'suffix'))}


def load_train_dev(data_dir):
    train = list(read_tokens(data_dir / 'tokens_train.tsv'))
    dev = list(read_tokens(data_dir / 'tokens_dev.tsv'))
    if not train or not dev:
        raise ValueError('Both train and dev must be nonempty.')
    if not all(r['split'] == 'train' for r in train) or not all(r['split'] == 'dev' for r in dev):
        raise ValueError('Unexpected split value.')
    if {r['document_id'] for r in train} & {r['document_id'] for r in dev}:
        raise ValueError('Train/dev document overlap.')
    return train, dev


def verify_frozen(root):
    path = root / 'docs/FINAL_FROZEN_MANIFEST.json'
    manifest = json.loads(path.read_text())
    for name, expected in manifest['artifacts'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original frozen artifact changed: ' + name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_variant(bundle, rows):
    indices = [i for i, r in enumerate(rows) if classify_casing(r['lowercase_input']) != 'uncased']
    outputs = [r['lowercase_input'] for r in rows]
    if indices:
        matrix = solver_matrix(bundle['vectorizer'].transform(
            family_features(rows, i, bundle['variant']) for i in indices))
        for i, label in zip(indices, bundle['classifier'].predict(matrix)):
            outputs[i] = reconstruct(rows[i]['lowercase_input'], label, bundle['mixed_lexicon'])
    return outputs


def independent_check(rows, outputs, score):
    selected = [(r['original_token'], p) for r, p in zip(rows, outputs)
                if classify_casing(r['original_token']) != 'uncased']
    gold, pred = zip(*selected)
    f1 = f1_score([classify_casing(x) for x in gold], [classify_casing(x) for x in pred],
                  labels=list(LABELS), average='macro', zero_division=0)
    if abs(f1-score['macro_f1']) > 1e-12 or abs(accuracy_score(gold, pred)-score['exact_accuracy']) > 1e-12:
        raise ValueError('Independent metric check failed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'results/character_ablation'
    models = root / 'models/character_ablation'
    predictions = root / 'data/processed/character_ablation_predictions'
    if out.exists() or models.exists() or predictions.exists():
        raise ValueError('A character-ablation run already exists; preserve it instead of overwriting.')
    frozen_before = verify_frozen(root)
    train, dev = load_train_dev(root / 'data/processed')
    lexicon = fit_lexicon(train)
    mixed = fit_lexicon(r for r in train if r['casing_label'] == 'mixed_case')
    indices = [i for i, r in enumerate(train) if r['casing_label'] in LABELS]
    targets = [train[i]['casing_label'] for i in indices]
    summary, classes, runs = [], [], []
    for directory in [out, models, predictions]:
        directory.mkdir(parents=True)

    def record(system, outputs):
        if len(outputs) != len(dev):
            raise ValueError('Prediction alignment mismatch.')
        grouped = evaluate(dev, outputs, lexicon)
        independent_check(dev, outputs, grouped['ALL'].summary())
        for group, score in sorted(grouped.items()):
            prefix = dict(system=system, split='dev', group=group)
            summary.append(dict(**prefix, **score.summary()))
            classes.extend(dict(**prefix, **m) for m in score.metrics())
        write_tsv(predictions / f'{system}.tsv', [dict(document_id=r['document_id'],
            sentence_id=r['sentence_id'], token_id=r['token_id'], genre=r['genre'],
            input=r['lowercase_input'], gold=r['original_token'], predicted_form=p)
            for r, p in zip(dev, outputs)])
        result = grouped['ALL'].summary()
        print(system, 'macro-F1', result['macro_f1'], 'accuracy', result['exact_accuracy'], flush=True)

    references = list(read_tokens(root / 'results/ablation/dev_comparison.tsv'))
    for system in ['context', 'context_char']:
        model = joblib.load(root / f'models/ablation/{system}.joblib')
        outputs, _ = predict_rows(model, dev)
        current = evaluate(dev, outputs, lexicon)
        for group, score in current.items():
            old = next(r for r in references if r['system'] == system and r['group'] == group)
            for metric in ['exact_accuracy', 'macro_f1', 'capitalization_required_accuracy']:
                if abs(score.summary()[metric]-float(old[metric])) > 1e-12:
                    raise ValueError('Frozen reference did not reproduce: ' + system + '/' + group)
        record(system, outputs)
        runs.append(dict(system=system, refitted=False, reference_scores_reproduced=True))

    for variant in VARIANTS:
        started = time.time()
        print('Fitting', variant, flush=True)
        vectorizer = DictVectorizer(dtype=float)
        matrix = solver_matrix(vectorizer.fit_transform(family_features(train, i, variant) for i in indices))
        classifier = LogisticRegression(**PARAMETERS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            classifier.fit(matrix, targets)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if not converged:
            raise RuntimeError('Variant did not converge: ' + variant)
        bundle = dict(vectorizer=vectorizer, classifier=classifier, mixed_lexicon=mixed,
                      vocabulary=set(lexicon), variant=variant)
        model_path = models / f'{variant}.joblib'
        joblib.dump(bundle, model_path)
        outputs = predict_variant(joblib.load(model_path), dev)
        record(variant, outputs)
        runs.append(dict(system=variant, refitted=True, converged=converged,
            features=matrix.shape[1], iterations=classifier.n_iter_.tolist(),
            seconds=round(time.time()-started, 2),
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()))
        del matrix, vectorizer, classifier, bundle

    if verify_frozen(root) != frozen_before:
        raise ValueError('Historical manifest changed.')
    write_tsv(out / 'dev_comparison.tsv', summary)
    write_tsv(out / 'dev_per_class.tsv', classes)
    metadata = dict(status='completed', split='dev', test_tokens_read=False,
        original_test_results_known_before_design=True,
        interpretation='Exploratory development follow-up; no new confirmatory test claim or model replacement.',
        parameters=PARAMETERS, runs=runs, training_cased_tokens=len(indices),
        evaluation_cased_tokens=sum(classify_casing(r['original_token']) != 'uncased' for r in dev),
        vocabulary_and_decoder='Full original train only, shared across four variants',
        independent_sklearn_metric_checks_passed=4, frozen_artifacts_unchanged=True,
        frozen_manifest_sha256=frozen_before,
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [root/'data/processed/tokens_train.tsv', root/'data/processed/tokens_dev.tsv']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(), sklearn=sklearn.__version__),
        character_scaling='Affixes categorical lengths 1-4; boundary-marked binary n-grams lengths 2-4 retain original per-token L2 normalization.',
        limitations='Single seed; fixed settings; character features overlap; absence/presence can affect scale and regularization; no causal linguistic mechanism or significance claim.')
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Completed four-system dev comparison; two new fits; frozen artifacts unchanged.', flush=True)


if __name__ == '__main__':
    main()

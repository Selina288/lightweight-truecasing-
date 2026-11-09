"""Validate current review records; optionally verify saved model outputs.

Default mode uses only the standard library and tracked review records.
--with-models additionally needs original local train/dev data and models.
No test split is read; no confirmation flags, reports or model files are written.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def verify_records(root):
    folder = root / 'results/error_analysis'
    rows = read_rows(folder / 'dev_reviewed_examples.tsv')
    paper = read_rows(folder / 'paper_examples.tsv')
    status = json.loads((folder / 'TEAM_CONFIRMATION.json').read_text())
    ids = status['confirmed_original_example_numbers']
    if len(rows) != 19 or len(ids) != 8 or len(set(ids)) != 8 or len(paper) != 8:
        raise ValueError('Expected 19 retained examples and eight distinct confirmed selections.')
    if any(type(i) is not int or not 1 <= i <= len(rows) for i in ids):
        raise ValueError('Invalid original example number.')
    keys = [(r['condition'], r['document_id'], r['sentence_id'], r['token_id']) for r in rows]
    if len(set(keys)) != 19:
        raise ValueError('Duplicate retained example.')
    for i, row in enumerate(rows, 1):
        if row['team_confirmed'] != ('Yes' if i in ids else 'No'):
            raise ValueError(f'Unexpected confirmation on original example {i}.')
    if [int(r['original_example_number']) for r in paper] != ids:
        raise ValueError('Selected order differs from the recorded confirmation IDs.')
    for selected, i in zip(paper, ids):
        if selected['selected_for_paper'] != 'Yes':
            raise ValueError('Selection flag missing.')
        for field, value in rows[i - 1].items():
            if selected.get(field) != value:
                raise ValueError(f'Selected example {i} differs in field {field}.')
    if status['confirmed_count'] != 8 or status['remaining_unconfirmed'] != 11:
        raise ValueError('Confirmation counts differ from the table.')
    if status['status'] != 'user_reported_team_confirmation':
        raise ValueError('Confirmation provenance changed.')
    for field in ['reviewer_names_provided', 'independent_review_observed', 'agreement_measured', 'model_or_score_changes']:
        if status[field] is not False:
            raise ValueError('Unsupported claim in confirmation metadata: ' + field)
    return rows


def verify_models(root, rows):
    import hashlib
    from collections import defaultdict
    import joblib
    from error_analysis import key, transition
    from logreg import predict_rows
    from preprocess import classify_casing
    frozen = json.loads((root / 'docs/FINAL_FROZEN_MANIFEST.json').read_text())['artifacts']
    cache = {}
    def check(path):
        name = path.relative_to(root).as_posix()
        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen[name]:
            raise ValueError('Frozen artifact mismatch: ' + name)
    train_path = root / 'data/processed/tokens_train.tsv'
    dev_path = root / 'data/processed/tokens_dev.tsv'
    check(train_path); check(dev_path)
    train, dev = read_rows(train_path), read_rows(dev_path)
    if not all(r['split'] == 'train' for r in train) or not all(r['split'] == 'dev' for r in dev):
        raise ValueError('Unexpected split.')
    sentences = defaultdict(list)
    for row in dev:
        sentences[(row['document_id'], row['sentence_id'])].append(row)
    for number, row in enumerate(rows, 1):
        sentence = sentences[(row['document_id'], row['sentence_id'])]
        index = next(i for i, token in enumerate(sentence) if key(token) == key(row))
        token = sentence[index]
        vocab = {r['lowercase_input'] for r in train if row['condition'] == 'all_genres' or r['genre'] != row['genre']}
        valid = (token['original_token'] == row['gold'] and token['lowercase_input'] == row['input']
                 and token['genre'] == row['genre'] and classify_casing(row['gold']) == row['gold_class']
                 and (row['seen'] == 'True') == (row['input'] in vocab)
                 and (row['first'] == 'True') == (index == 0)
                 and row['transition'] == transition(row['gold'], row['B'], row['C']))
        if not valid:
            raise ValueError(f'Dev record mismatch on original example {number}.')
        for system, variant in [('B', 'context'), ('C', 'context_char')]:
            relative = (f'models/ablation/{variant}.joblib' if row['condition'] == 'all_genres'
                        else f'models/crossgenre/{row["genre"]}_{variant}.joblib')
            path = root / relative
            if relative not in cache:
                check(path)
                cache[relative] = joblib.load(path)
            outputs, _ = predict_rows(cache[relative], sentence)
            if outputs[index] != row[system]:
                raise ValueError(f'Model output mismatch on example {number}, {system}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-models', action='store_true')
    args = parser.parse_args()
    try:
        rows = verify_records(ROOT)
        if args.with_models:
            verify_models(ROOT, rows)
    except (ValueError, KeyError, OSError, StopIteration) as error:
        parser.exit(1, str(error) + '\n')
    print('Validated 19 records: eight team-confirmed selections and eleven unconfirmed examples.')
    if args.with_models:
        print('All 19 examples also match frozen train/dev data and saved model outputs.')
    print('Confirmation is user-reported. No test data read; no files or flags changed.')


if __name__ == '__main__':
    main()

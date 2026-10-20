"""Train three truecasing baselines on train; evaluate on dev only.
Run from project root: python3 src/baselines.py
Uncased tokens are excluded. Macro F1 uses four fixed casing classes.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path
from preprocess import classify_casing

LABELS = ('lowercase', 'initial_uppercase', 'all_uppercase', 'mixed_case')
SYSTEMS = ('all_lowercase', 'sentence_initial', 'most_frequent_case')


def read_tokens(path):
    with path.open(encoding='utf-8', newline='') as handle:
        yield from csv.DictReader(handle, delimiter='\t')


def fit_lexicon(rows):
    counts = defaultdict(Counter)
    for row in rows:
        counts[row['lowercase_input']][row['original_token']] += 1
    # Ties are resolved by lexicographic order, independent of dev data.
    return {word: min(forms, key=lambda form: (-forms[form], form))
            for word, forms in counts.items()}


def predict(system, word, first, lexicon):
    if system == 'all_lowercase':
        return word
    if system == 'sentence_initial':
        if first:
            for i, char in enumerate(word):
                if char.lower() != char.upper():
                    return word[:i] + char.upper() + word[i+1:]
        return word
    if system == 'most_frequent_case':
        return lexicon.get(word, word)
    raise ValueError(system)


class Scores:
    def __init__(self):
        self.n = self.exact = self.cap_n = self.cap_exact = 0
        self.gold = Counter()
        self.predicted = Counter()
        self.correct = Counter()

    def add(self, gold, predicted):
        label = classify_casing(gold)
        if label == 'uncased':
            return
        pred_label = classify_casing(predicted)
        self.n += 1
        self.exact += gold == predicted
        self.gold[label] += 1
        self.predicted[pred_label] += 1
        self.correct[label] += label == pred_label
        if label != 'lowercase':
            self.cap_n += 1
            self.cap_exact += gold == predicted

    def metrics(self):
        rows = []
        for label in LABELS:
            tp = self.correct[label]
            p = tp/self.predicted[label] if self.predicted[label] else 0.0
            r = tp/self.gold[label] if self.gold[label] else 0.0
            f1 = 2*p*r/(p+r) if p+r else 0.0
            rows.append(dict(label=label, support=self.gold[label], precision=p, recall=r, f1=f1))
        return rows

    def summary(self):
        return dict(cased_tokens=self.n,
                    exact_accuracy=self.exact/self.n if self.n else '',
                    macro_f1=sum(r['f1'] for r in self.metrics())/4,
                    capitalization_required_tokens=self.cap_n,
                    capitalization_required_accuracy=self.cap_exact/self.cap_n if self.cap_n else '')


def write_tsv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = Path('data/processed')
    lexicon = fit_lexicon(read_tokens(data/'tokens_train.tsv'))
    scores = defaultdict(Scores)
    previous_sentence = None
    for row in read_tokens(data/'tokens_dev.tsv'):
        sentence = (row['document_id'], row['sentence_id'])
        first = sentence != previous_sentence
        previous_sentence = sentence
        groups = ('ALL', 'genre:' + row['genre'],
                  'seen' if row['lowercase_input'] in lexicon else 'unseen')
        for system in SYSTEMS:
            output = predict(system, row['lowercase_input'], first, lexicon)
            for group in groups:
                scores[(system, group)].add(row['original_token'], output)
    out = Path('results/baselines')
    out.mkdir(parents=True, exist_ok=True)
    summaries, classes = [], []
    for (system, group), score in sorted(scores.items()):
        summaries.append(dict(system=system, split='dev', group=group, **score.summary()))
        for metrics in score.metrics():
            classes.append(dict(system=system, split='dev', group=group, **metrics))
    write_tsv(out/'dev_summary.tsv', summaries)
    write_tsv(out/'dev_per_class.tsv', classes)
    for row in summaries:
        if row['group'] == 'ALL':
            print(f"{row['system']}: accuracy={row['exact_accuracy']:.4f}, "
                  f"macro-F1={row['macro_f1']:.4f}, "
                  f"capitalization accuracy={row['capitalization_required_accuracy']:.4f}")
    print('Only train and dev were read. Results:', out)


if __name__ == '__main__':
    main()

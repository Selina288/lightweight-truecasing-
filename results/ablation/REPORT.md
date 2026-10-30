# Development Feature Ablation Results

All models were fitted on the official training split and evaluated on the same development split. They use the same Logistic Regression settings, train-only decoder, label policy, and evaluator. Only the feature representation varies. Test data were not read by this experiment. All models converged.

- A: current lowercased token, sentence-initial flag, scaled token length.
- B: A plus the previous and next lowercased tokens, using sentence boundary markers.
- C: B plus prefixes and suffixes of lengths 1-4 and binary character 2-4 grams with word boundary markers. Character n-gram values are normalized to unit L2 norm per token.

## Overall development results

| Model | Exact accuracy | Macro-F1 | Capitalization-required accuracy |
| --- | ---: | ---: | ---: |
| A Lexical and position | 90.90% | 0.4892 | 61.24% |
| B Add context | 91.43% | 0.5245 | 64.95% |
| C Add character features | 92.26% | 0.5688 | 72.18% |

## Unseen-word development results

Unseen means the lowercased word did not appear in training. This subset contains 1,563 cased tokens for every variant. Macro-F1 still averages four fixed classes and may be low for rare classes.

| Model | Exact accuracy | Macro-F1 |
| --- | ---: | ---: |
| A Lexical and position | 59.82% | 0.2640 |
| B Add context | 62.19% | 0.2991 |
| C Add character features | 69.93% | 0.4236 |

## Interpretation and limits

B exceeds A in this fixed-parameter comparison, supporting the inclusion of local context. C exceeds B overall and on unseen words, supporting the addition of these character features in this development setting. This does not isolate prefixes/suffixes from character n-grams, since they are added together. Hyperparameters were held fixed, not independently optimized for each representation. No confidence intervals or significance tests were run, so these are observed score differences, not claims of statistical significance. This is not a held-out-genre experiment.

Casing scores are derived from reconstructed output forms, matching the existing baseline evaluator. Uncased tokens are excluded. All variants share a train-only mixed-case decoder; unseen mixed-case words fall back to lowercase. Character features do not generate arbitrary internal capitalization for unseen mixed-case words. Exact accuracy is therefore needed alongside class F1.

Twelve automated tests pass. Saved models were reloaded and their scores independently checked with scikit-learn metrics. B exactly reproduces the previous contextual model's scores for every evaluation group. The next research step is held-out-genre development evaluation and qualitative error analysis. The required front-end and final test evaluation are still outstanding.

## Reproduce

From the project root, with the existing virtual environment activated:

```bash
python -m unittest discover -s tests -v
python src/ablation.py
python src/truecase.py --model models/ablation/context_char.joblib "i visited new york on monday."
```

Models are saved under the ignored `models/ablation/` directory. Token-level predictions remain in ignored `data/processed/ablation_*_dev_predictions.tsv`. Summary, per-class scores and configuration metadata are in `results/ablation/`. The original default demo model remains `models/logreg.joblib`; use the explicit model option to select a new variant.

# Leave One Genre Out Development Evaluation

This experiment holds out each of the five EWT genres in turn. The held-out condition fits Logistic Regression only on the other four genres' official training data and evaluates on the target genre's official development data. It does not train on any development data. No test data were used. Ten source-only models were fitted: five targets times two feature sets.

All-genre reference models are the already trained B (context) and C (context plus character features) ablation models. Their parameters and dataset hashes were checked before reuse. Both conditions are scored on exactly the same target tokens. The vectorizer, vocabulary, mixed-case lexicon and classifier for every held-out fold are fitted exclusively on source training data. All fits converged. Hyperparameters stayed fixed (SAGA, C=1, seed=42, tolerance=0.001, max_iter=500).

## Overall target development Macro F1

| Held-out target | B all genres | B held out | C all genres | C held out | C drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Answers | 0.4723 | 0.4659 | 0.5114 | 0.5105 | +0.0009 |
| Email | 0.5531 | 0.4629 | 0.6089 | 0.5164 | +0.0925 |
| Newsgroups | 0.5319 | 0.4695 | 0.5691 | 0.5110 | +0.0581 |
| Reviews | 0.4421 | 0.4397 | 0.4505 | 0.4357 | +0.0148 |
| Weblogs | 0.6054 | 0.5051 | 0.6494 | 0.5871 | +0.0623 |

A positive drop means held-out performance is lower. Four casing classes have equal weight; uncased tokens are excluded. Scores are computed from reconstructed forms, matching previous experiments. Sparse mixed-case support can make per-genre macro-F1 sensitive to a few examples.

## Observed findings

C exceeds B on held-out macro-F1 for 4 of 5 target genres. The largest C all-genre-to-held-out macro-F1 drop is for Email, at 0.0925. These are observed development results, not significance-tested conclusions.

## Unseen words under transfer

Both conditions use the SAME four-source-genre vocabulary to define seen and unseen subsets. Therefore the following comparison does not silently change its evaluation subset when the reference model has seen target-genre training words.

| Target | Unseen cased tokens | B held-out exact accuracy | C held-out exact accuracy |
| --- | ---: | ---: | ---: |
| Answers | 381 | 68.24% | 69.29% |
| Email | 541 | 54.90% | 65.06% |
| Newsgroups | 439 | 60.14% | 68.34% |
| Reviews | 388 | 74.23% | 75.00% |
| Weblogs | 482 | 58.71% | 72.20% |

## Limits and remaining work

The all-genre condition has more training data and includes the target genre, so the gaps combine differences in training size, vocabulary coverage and genre. They do not isolate a causal effect of genre shift alone. A size-matched experiment could separate these effects but is not included here. This experiment has no in-genre-only training condition and no target-dev tuning. Prefixes/suffixes and character n-grams are added together, so their separate effects are not identified. There are no significance tests or final test results yet. Qualitative development error analysis and the front-end remain to be completed.

Fifteen automated tests pass. Independent scikit-learn scoring reproduced every summary row, target token IDs matched between conditions, and all-genre scores reproduced the previous per-genre results. The audit metadata records training genres, document counts, sizes, convergence and hashes. Token-level prediction files remain local under ignored data/processed; trained models remain in ignored models/crossgenre.

## Reproduce

Activate the project environment, then run:

```bash
python -m unittest discover -s tests -v
python src/crossgenre.py
```

This requires the processed train/dev data and reference models from `python src/ablation.py`. The full local package includes those files. Three output tables report summary scores, per-class scores, and reference-to-held-out gaps. `run_metadata.json` records the protocol and per-fold audit.

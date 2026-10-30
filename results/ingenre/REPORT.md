# In Genre Development Evaluation

Ten in-genre-only Logistic Regression models were fitted: five EWT genres times B (context) and C (context plus character features). Each classifier, feature vocabulary and mixed-case decoder uses only that target genre's official training data. Evaluation uses the target genre's official development data. No development examples enter fitting, no condition-specific tuning was performed and test data were not read. All ten models converged.

The parameters are identical to the earlier all-genre and held-out experiments: SAGA, C=1, max_iter=500, tolerance=0.001, seed 42. Existing all-genre and held-out models were reloaded for the same target examples; dataset hashes, parameters and training vocabularies were verified. Conditions differ in the training data selected, not in the feature definitions or scorer.

## C model Macro F1 across the three conditions

| Target genre | All genres | In genre only | Held out |
| --- | ---: | ---: | ---: |
| answers | 0.5114 | 0.4600 | 0.5105 |
| email | 0.6089 | 0.5994 | 0.5164 |
| newsgroup | 0.5691 | 0.5518 | 0.5110 |
| reviews | 0.4505 | 0.4505 | 0.4357 |
| weblog | 0.6494 | 0.6078 | 0.5871 |

## B model Macro F1 across the three conditions

| Target genre | All genres | In genre only | Held out |
| --- | ---: | ---: | ---: |
| answers | 0.4723 | 0.4192 | 0.4659 |
| email | 0.5531 | 0.5413 | 0.4629 |
| newsgroup | 0.5319 | 0.5077 | 0.4695 |
| reviews | 0.4421 | 0.4421 | 0.4397 |
| weblog | 0.6054 | 0.5720 | 0.5051 |

Macro-F1 averages all four casing classes with zero for undefined class F1. Exact surface accuracy, capitalization-required accuracy, class supports, precision and recall are also retained in the tables. Uncased tokens are excluded. Close or equal rounded scores must not be treated as evidence of equivalence or significance.

## Training size and interpretation

| Target genre | In-genre cased train tokens | Cased dev tokens |
| --- | ---: | ---: |
| answers | 38289 | 4584 |
| email | 38931 | 4577 |
| newsgroup | 29539 | 3638 |
| reviews | 39554 | 4680 |
| weblog | 30570 | 4189 |

In-genre-only is not guaranteed to beat held-out training. For Answers, both in-genre models score below their held-out counterparts. In-genre training has less data and different lexical coverage; matched genre alone is not sufficient to explain the outcome. This comparison does not isolate the causal effects of genre or corpus size. Hyperparameters are held fixed, not optimized independently for each condition. No significance test or size-matched experiment is included.

Seen/unseen subsets in this new three-condition table are defined by the TARGET-TRAIN vocabulary, identically for all three conditions. The older crossgenre report instead used the four-source-genre vocabulary. Therefore subgroup scores from these two reports should not be compared as though their samples were identical. Overall target scores use the same tokens and exactly reproduce prior all-genre and held-out results.

The qualitative style and mixed-case limitations from earlier analyses remain relevant. These are exploratory development results, not final test results. The original proposal's in-genre-only experimental condition is now implemented with fixed parameters; no additional target-dev tuning is claimed.

## Validation and reproduction

Twenty-five automated tests pass. All 90 summary rows were independently checked against scikit-learn metrics. Target token IDs and subgroup membership match between all three conditions. Reference overall scores match previous experiments. Per-run metadata records model hashes, data hashes, training genres, document counts and convergence.

With the environment activated, run:

```bash
python -m unittest discover -s tests -v
python src/ingenre.py
```

This requires the saved reference models and metadata from ablation.py and crossgenre.py, which are included in the complete local package. New models stay in ignored models/ingenre, and token predictions stay in ignored data/processed. Public result files contain aggregate scores and protocol metadata.

## Remaining work

The three planned training conditions now have development results. Team confirmation of the reviewed examples remains separate from assistant verification. Settle and record the final protocol before running the reserved test split; do not choose features or scoring rules based on test outcomes. The existing local front-end remains available and requires no new dependency for this experiment.

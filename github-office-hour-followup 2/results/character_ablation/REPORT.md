# Exploratory character-family ablation on development data

## Protocol

This follow-up was designed after the original final-test results were known. It uses only official train/dev files and is not new confirmatory test evidence. B and C are the existing frozen models, whose prior dev scores were reproduced before fitting two additional systems. Original test outputs and model hashes are unchanged.

All four systems use the same full train split, lowercase context, position/length, train-only mixed-case decoder and vocabulary. New variants retain only affixes or only the normalized character n-gram family from C. Parameters are SAGA, C=1, max_iter=500, tolerance=0.001, seed=42. Prefix/suffix lengths are 1–4; boundary-marked binary character n-grams are lengths 2–4 with the original per-token L2 normalization.

## Results

All-dev scores use the same 21,668 cased tokens. Unseen scores use the same 1,563 tokens absent from full training.

| System | Dev exact accuracy | Dev macro-F1 | Capitalization-required accuracy | Unseen exact accuracy | Unseen macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B: context | 91.43% | 0.5245 | 64.95% | 62.19% | 0.2991 |
| B + prefixes/suffixes | 92.26% | 0.5682 | 71.86% | 69.87% | 0.4174 |
| B + character n-grams | 92.33% | 0.5474 | 70.89% | 69.99% | 0.3803 |
| C: both character families | 92.26% | 0.5688 | 72.18% | 69.93% | 0.4236 |

## Interpretation

- The affix-only variant nearly matches C: macro-F1 0.5682 versus 0.5688. Its exact result differs from C by one dev token. This supports a narrower statement that the prefix/suffix family is informative within these settings; it does not establish that n-grams are useless.
- N-grams alone have the highest exact accuracy, 92.33%, but macro-F1 is lower than C, 0.5474 versus 0.5688. They correctly recover 14 more dev tokens overall. Under the original primary metric, C remains preferred.
- Per-class scores explain part of that disagreement: n-gram-only all-uppercase F1 is 0.4408 versus 0.4708 for C; mixed-case F1 is 0 versus 0.0541. Mixed-case dev support is only 66, so small changes are unstable.
- Affix and n-gram groups share overlapping string information, and removing a group changes feature scale/regularization. This is a controlled representation comparison, not causal isolation of morphology.
- No statistical significance, multi-seed robustness, size-matched transfer experiment or external-system benchmark was performed here. No replacement for the original test-selected model is claimed.

## Checks and reproducibility

The two new fits converged (78 and 103 iterations); loaded B/C references matched all eight prior dev groups each. Overall accuracy and four-class macro-F1 were independently checked with scikit-learn for all four systems. All 34 unit tests pass, including three new checks for group composition, exclusion of gold/POS inputs and split separation. The original frozen manifest and every listed artifact still match.

Run once from the project root after installing dependencies and providing the complete original frozen local experiment package (all artifacts listed in FINAL_FROZEN_MANIFEST.json, including train/dev tokens and restricted-training models):

```bash
python -m unittest discover -s tests -v
python src/character_ablation.py
```

The script rejects an existing follow-up result/model/prediction directory. It writes only new results/character_ablation/, models/character_ablation/ and ignored data/processed/character_ablation_predictions/. It does not accept a test split. New model bundles use the follow-up predictor and are not added to the demo dropdown.

## Office-hour explanation

> We completed a dev-only character-feature ablation using the same classifier settings and decoder. Prefixes and suffixes alone nearly match the full model on macro-F1. Character n-grams alone give slightly higher exact accuracy but lower macro-F1, which highlights the importance of rare casing categories. We also audited four prior papers and found differences in datasets, baseline conventions and category definitions, so we are not claiming a direct numerical comparison with their published scores.

For literature details see ../../docs/EVALUATION_COMPARISON.md. See run_metadata.json for hashes, environment, convergence and original-test-awareness provenance.

# Equal-budget genre comparison on development data

## Design

Exploratory follow-up after original test exposure. Fifty new C models: five target genres × two training conditions × five sampling seeds (11, 22, 33, 44, 55). Every model uses exactly 20,000 supervised cased train tokens. Conditions are target-only and pooled other-four-genres training. Original C features, decoder rules and LR parameters are held fixed (solver random_state=42). No tuning, model replacement or new test evaluation.

Sampled anchors retain original immediate lowercase context and sentence boundaries; classifier labels, vectorizer training rows and mixed-case dictionary entries come only from sampled anchors. This matches the supervised token budget, not total unique context exposure, document counts, vocabulary, class priors or number of source genres. See [the protocol](../../docs/SIZE_MATCHED_GENRE_PROTOCOL.md).

## Primary results

| Target | In-genre macro-F1 (mean ± SD) | Other-genres macro-F1 (mean ± SD) | In − other | In-genre wins / 5 |
| --- | ---: | ---: | ---: | ---: |
| answers | 0.4359 ± 0.0126 | 0.4591 ± 0.0272 | -0.0231 | 0/5 |
| email | 0.5820 ± 0.0177 | 0.4464 ± 0.0282 | +0.1356 | 5/5 |
| newsgroup | 0.5237 ± 0.0075 | 0.4652 ± 0.0075 | +0.0585 | 5/5 |
| reviews | 0.4380 ± 0.0063 | 0.4362 ± 0.0044 | +0.0018 | 4/5 |
| weblog | 0.6032 ± 0.0041 | 0.5302 ± 0.0370 | +0.0730 | 5/5 |

SD is the sample standard deviation over five overlapping training subsamples on the same fixed target dev set. It is not a confidence interval over evaluation data. A paired gap pairs independently sampled training sets under the same seed index; no significance claim is made. Macro-F1 averages four fixed reconstructed-form casing classes, including zero for a class with no correct predictions. Rare target classes may influence variation.

## Secondary results

| Target | In-genre exact accuracy (mean ± SD, %) | Other-genres exact accuracy (mean ± SD, %) |
| --- | ---: | ---: |
| answers | 90.33 ± 0.17 | 90.21 ± 0.11 |
| email | 90.40 ± 0.48 | 88.44 ± 0.23 |
| newsgroup | 89.71 ± 0.14 | 89.30 ± 0.45 |
| reviews | 91.23 ± 0.10 | 91.10 ± 0.17 |
| weblog | 93.74 ± 0.12 | 91.47 ± 0.30 |

Capitalization-required exact accuracy, all unrounded per-run scores and paired-gap SDs are in the accompanying TSV files. Uncased tokens are excluded in every condition. No unseen subgroup comparison is asserted because sampled vocabularies differ across fits.

## Supported interpretation

- In-genre macro-F1 is higher in all five repetitions for email, newsgroup and weblog; mean gaps are +0.1356, +0.0585 and +0.0730 respectively.
- Answers reverses direction: pooled other-genre training has higher macro-F1 in every repetition, with an in-minus-other mean gap of −0.0231. Mean in-genre exact accuracy is slightly higher, illustrating metric-dependent rankings.
- Reviews has a small mean macro-F1 gap (+0.0018), with a reversal in one repetition and gap SD 0.0082. Neither reliable superiority nor equivalence is established.
- Thus equal supervised data quantity does not eliminate differences between source-selection conditions. In-genre training is not universally best. The earlier unequal-size three-condition results remain historical practical comparisons; this controlled-budget follow-up narrows one confound but does not isolate a causal genre-only mechanism.

This result is specific to the 20,000-label budget, C representation and fixed official dev sets. Other-genre training pools four sources and differs in diversity, vocabulary and casing distributions. It is not a per-source transfer matrix or target-blind model-selection study. No conclusion about statistical significance, performance on a new unseen test set, all training sizes or external systems follows from this run.

## Reproduction and audit

From the complete original local project: `python src/size_matched_genre.py`, then `python src/verify_size_matched_genre.py`. Existing run directories are protected against overwrite. This reproduction requires historical frozen artifacts described in the protocol.

`run_metadata.json` records all 50 converged fits, sample label/genre/document/sentence counts, source/data/model hashes and context exposure. `verification.json` records reloaded-model prediction reproduction, independent sklearn metric checks, training-pool/sample-only decoder checks and aggregate checks. Exact sample indices, predictions and weights remain under ignored local directories. The original frozen code/data/models were verified unchanged; no new test tokens were read. All 37 project tests passed before fitting.

# Development Error Analysis

This analysis compares B (context) and C (context plus character features), using saved development predictions only. No models were retrained, no gold labels were changed, and no test data were read. Uncased tokens are excluded, leaving 21,668 cased tokens per condition. Held-out totals pool the five separate target-genre evaluations and do not represent a single model.

## Exact surface errors

| Condition | B errors | C errors | Corrected by C | New errors from C | Net removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_genres | 1857 | 1677 | 360 | 180 | 180 |
| held_out | 2081 | 1867 | 417 | 203 | 214 |

C reduces exact errors overall, but corrections and regressions coexist. Counts describe token outcomes, not independent documents or statistically significant differences.

## Capitalization tradeoff

| Condition | B undercapitalized | C undercapitalized | B overcapitalized | C overcapitalized |
| --- | ---: | ---: | ---: | ---: |
| all_genres | 1349 | 1036 | 441 | 553 |
| held_out | 1579 | 1191 | 431 | 567 |

Here undercapitalized means a nonlowercase gold token was output as lowercase; overcapitalized means a lowercase gold token was output with some uppercase letters. These are observable properties. They do not identify why the classifier made its decision. Character features reduce undercapitalization while increasing overcapitalization.

Other diagnostics include unseen words, first tokens, all-uppercase gold, mixed-case gold, and wrong surfaces with the correct class. Flags overlap and must not be summed as a partition of errors. For example, an unseen mixed-case name may appear in multiple counts.

## Why Reviews accuracy and macro F1 disagree

In the held-out Reviews condition, C corrects 50 B errors but introduces 47, so it has three fewer exact errors. Yet macro-F1 decreases from about 0.4397 to 0.4357. Lowercase F1 rises from 0.9514 to 0.9525 and initial-uppercase F1 rises from 0.7779 to 0.7904, but all-uppercase F1 falls from 0.0294 to zero. B recovers only one of the 66 all-uppercase gold tokens; C recovers none. The one mixed-case gold token is missed by both. Four-class macro-F1 gives these rare classes equal weight, so the uppercase loss outweighs gains in other classes. This is a class-support and metric-weighting effect, not proof that C makes more total errors.

## Contextual review

Nineteen examples were inspected by the assistant using sentence context. This is an illustrative review, not a completed human-annotated benchmark or an estimate of error-category prevalence. All entries in dev_reviewed_examples.tsv are marked team_confirmed=No for subsequent team review. Ninety deterministically selected candidate windows remain locally in data/processed/error_review_candidates.tsv. Short excerpts and gold targets are preserved without correction.

Observed examples include:

- Ashtamudi: C restores an unseen place-name capital that B misses in a Lake context.
- Canon / canon in one sentence: C capitalizes the second instance, disagreeing with the observed lowercase gold. This illustrates corpus-casing inconsistency rather than an unequivocal linguistic mistake.
- Lowercase i: both systems output conventional I, which still counts as a corpus-restoration error.
- AMAZING: both systems miss emphatic capitals in a review. Original emphasis can be ambiguous after lowercasing.
- NiMo: C outputs Nimo. Improving the first letter does not recover the internal capital, and the train-only mixed-case decoder cannot reconstruct arbitrary unseen mixed-case spellings.
- Noninitial pizza and the interjection eh: held-out C introduces false capitals that B avoids.
- US and the embedded French article la: context distinguishes acronym/pronoun and language-specific readings; C misses these selected examples.
- Sentence-initial Will: C regresses to lowercase despite the available position feature.

These examples support hypotheses about names, abbreviations, style, and context ambiguity. They do not establish a causal attribution to individual character features or provide authoritative named-entity labels.

## Limits and next steps

This is exploratory development analysis; the target development data have already informed earlier comparisons. Do not describe it as a pristine final evaluation. Transfer gaps also mix genre coverage with training-size differences. There are no significance tests or human inter-annotator agreement estimates. Review the contextual annotations with the team, preserve observed gold casing, and decide whether any changes are warranted using development data only. The front-end and final frozen test evaluation remain outstanding.

## Files and reproduction

- dev_error_summary.tsv: error counts and rates by condition, system, genre, gold class, vocabulary novelty, and sentence position.
- dev_B_C_transitions.tsv: paired corrections and regressions by genre.
- dev_confusion.tsv: four-by-four output-derived casing confusion counts.
- dev_diagnostics.tsv: overlapping observable error flags.
- dev_reviewed_examples.tsv: 19 assistant-reviewed contextual examples with IDs and tentative interpretations.
- run_metadata.json: protocol, input hashes and review status.

Run `python src/error_analysis.py` to regenerate the four automatic tables and the local candidate sample. The reviewed examples are a separately retained annotation artifact, not automatically generated causal labels. Eighteen automated tests pass; counts were independently reconciled to all 20 prior per-genre scores and paired prediction transitions.

Corpus excerpts are from the [UD English EWT contributors](https://github.com/UniversalDependencies/UD_English-EWT). Excerpts are shortened context windows with brackets added around the target. Corpus-derived excerpts retain CC BY-SA 4.0 terms; see DATA_LICENSE.txt. Analysis and tentative review notes are added here.

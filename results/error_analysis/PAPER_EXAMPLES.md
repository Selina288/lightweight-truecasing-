# Selected Development Examples for the Paper

The eight examples below were selected for qualitative illustration. The user reports that the team has confirmed these eight selected examples; team_confirmed is Yes for these eight. This is user-reported team confirmation, not independently observed review or a measurement of inter-annotator agreement. They are development examples, not final-test examples, and do not estimate error-type prevalence. B uses lexical and contextual features; C adds character features.

| Original ID | Training condition | Gold | B | C | Observation |
| --- | --- | --- | --- | --- | --- |
| 1 | all_genres | `Ashtamudi` | `ashtamudi` | `Ashtamudi` | C restores the observed capitalization of an unseen name in a lake-name context. |
| 5 | all_genres | `P.S.` | `P.s.` | `P.S.` | C recovers the observed all-uppercase abbreviation. |
| 2 | all_genres | `canon` | `canon` | `Canon` | C uses conventional brand capitalization but disagrees with the observed lowercase target. |
| 8 | all_genres | `AMAZING` | `amazing` | `amazing` | Both systems miss the observed expressive all-uppercase form. |
| 12 | all_genres | `US` | `US` | `us` | C lowercases a country abbreviation that B recovers correctly. |
| 14 | held_out | `Will` | `Will` | `will` | C misses the observed sentence-initial capital despite having a position feature. |
| 15 | held_out | `pizza` | `pizza` | `Pizza` | C introduces capitalization on an unseen, noninitial common noun. |
| 19 | all_genres | `NiMo` | `nimo` | `Nimo` | C recovers the initial capital but misses the internal capital; exact spelling remains wrong. |

## Suggested paper paragraph

Selected development examples illustrate both gains and limitations from character features. C restores the observed forms Ashtamudi and P.S. where B fails, but it also introduces errors, such as lowercasing US and sentence-initial Will or capitalizing the common noun pizza. Observed corpus casing does not always match conventional edited-English casing: Canon is scored as incorrect when the gold token is canon. Both systems fail to recover expressive capitals in AMAZING, while C's Nimo remains an exact-form error for NiMo despite recovering its initial capital. These examples describe outputs rather than establishing causal mechanisms for individual features. The examples mix all-genre and held-out conditions, which are explicitly identified in the table. The selected interpretations have been confirmed by the team, as reported by the user.

## Context and review

Use paper_examples.tsv for the retained short context windows, original document/sentence/token IDs, and qualified review notes. Examples 14 and 15 are from held-out training; the other six use all-genre training. Do not describe the selection as a random sample. Confirmation of these eight does not imply that the remaining eleven examples were reviewed.

Corpus-derived target forms and excerpts are from the UD English EWT contributors (https://github.com/UniversalDependencies/UD_English-EWT), under CC BY-SA 4.0. Context windows are shortened and brackets mark target words. The retained DATA_LICENSE.txt supplies the license text.

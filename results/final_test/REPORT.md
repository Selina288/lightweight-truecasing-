# Final Frozen Model Evaluation

The unified evaluator first reproduced 138 previously recorded development score rows. Code, training resources and all saved models were inventoried in FINAL_FROZEN_MANIFEST.json before final test execution. It then loaded the same frozen artifacts and ran the official test evaluation without retraining, gold editing, feature changes or parameter tuning. No model changes were made after viewing test results. Independent scikit-learn scoring verified all 198 final summary rows, and frozen artifact hashes still match. Twenty-seven automated tests pass.

C (context plus characters) was selected as the primary candidate from development results before the test run. All systems remain reported. The primary ranking score is four-class macro-F1 of reconstructed output forms; exact-form accuracy breaks exact ties. Uncased tokens are excluded, leaving 21,430 scored test tokens, of which 4,233 require capitalization. Formal evaluation uses corpus tokenization and sentence boundaries, not the front-end tokenizer.

## Overall official test results

| System | Exact surface accuracy | Four-class macro-F1 | Capitalization-required exact accuracy |
| --- | ---: | ---: | ---: |
| All lowercase | 80.25% | 0.2226 | 0.00% |
| Sentence initial | 86.26% | 0.3695 | 37.33% |
| Most frequent training form | 85.73% | 0.4905 | 38.46% |
| A Lexical and position | 89.72% | 0.4990 | 59.15% |
| B Add context | 90.27% | 0.5175 | 63.29% |
| C Add characters | 91.50% | 0.5823 | 71.65% |

C ranks first by the predeclared primary metric and also has the highest exact-form accuracy among the six systems. This is a result of the fixed comparison, not evidence that every C prediction or every genre improves. There are no confidence intervals or statistical significance claims.

## Previously unseen words on the official test split

All systems use the same full-training vocabulary definition here, giving 1,652 scored unseen tokens.

| System | Unseen exact accuracy | Unseen macro-F1 |
| --- | ---: | ---: |
| All lowercase | 50.97% | 0.1688 |
| Sentence initial | 55.99% | 0.2473 |
| Most frequent training form | 50.97% | 0.1688 |
| A Lexical and position | 56.30% | 0.2485 |
| B Add context | 59.38% | 0.2858 |
| C Add characters | 68.16% | 0.4178 |

## C model test Macro F1 by training condition

| Target genre | All genres | In genre only | Held out |
| --- | ---: | ---: | ---: |
| answers | 0.5651 | 0.5312 | 0.5392 |
| email | 0.5744 | 0.5699 | 0.4922 |
| newsgroup | 0.6082 | 0.5490 | 0.5571 |
| reviews | 0.4664 | 0.4449 | 0.4696 |
| weblog | 0.6723 | 0.6527 | 0.5526 |

For each target, all three conditions score the same target tokens. Each in-genre model uses only target-genre train; each held-out model uses only the other four genres' train. In this test run, held-out C is slightly above all-genre C on Reviews macro-F1, so cross-genre degradation is not universal. Genre coverage, training size and vocabulary coverage change together; these gaps cannot isolate a causal genre effect. All raw scores and per-class supports are retained.

The three-condition seen/unseen subgroup tables use target-train vocabulary consistently across conditions. A separate source_train subset basis preserves the earlier all-versus-held-out source-vocabulary comparison. Do not combine these subgroup definitions or count duplicate ALL rows as distinct test examples. Summaries pooled across target models are not scores for a single trained model.

## Team review status

The user explicitly reported that the group has not yet checked the 19 contextual examples. Every team_confirmed field remains No. The final quantitative evaluation was completed while that qualitative review remains pending; this report does not claim team approval or human annotation of those examples. Assistant checks verified their agreement with dev records and saved model outputs and refined five interpretations. Team confirmation can later update annotation status without changing frozen model parameters, test scores or gold data.

## Limitations

- Development data informed feature selection and example inspection. These test scores must not now be used to select new features or tune parameters.
- Gold casing may be informal or inconsistent; the task restores observed corpus casing rather than an authoritative corrected English version.
- The four-class label score does not fully specify internal mixed-case spelling. Exact-form accuracy and the train-only mixed-case decoder remain important; unseen mixed-case forms can fail.
- Rare class support affects macro-F1, especially in small genres. Inspect per-class supports and do not interpret a small score difference as significant without a suitable analysis.
- Training sizes differ across conditions. No size-matched causal experiment or significance test was performed.
- The browser demo uses simple tokenization; official scoring uses gold corpus tokenization. Its displayed dev scores are not the accuracy of a typed sentence or final test scores.

## Outputs and execution record

all_systems_summary.tsv and all_systems_per_class.tsv report the six systems overall, by genre and vocabulary novelty. genre_conditions_summary.tsv and genre_conditions_per_class.tsv report B/C under the three training conditions. genre_overview.tsv is a compact target-genre comparison. run_metadata.json records the evaluation data hash, frozen model hashes and metric policy. independent_verification.json records the separate scoring check.

Token-level predictions remain in ignored data/processed/final_test_predictions and model weights remain in ignored models. The unified script refuses to silently overwrite an already completed run. The delivered package includes completed dev and test outputs, so do not rerun final_evaluate.py merely to view them. Dev replication and final test each completed once successfully in this execution, with no classifier retraining or score-driven repair. For a clean reconstruction, reproduce model artifacts and manifest before evaluation; preserve prior outputs and describe any reruns.

The remaining deliverables are team review of qualitative examples, a final paper and presentation. The local web demonstration remains runnable with python src/demo_server.py.

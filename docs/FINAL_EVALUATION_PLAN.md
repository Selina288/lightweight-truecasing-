# Final Evaluation Preparation

Status: prepared for team review, not a claim of team approval or completed final testing. This plan records the current candidate setup before opening final test results. The development set has already been used for feature comparisons and qualitative review.

## Systems already implemented

Evaluate the three baselines (all lowercase, sentence-initial capitalization, most frequent training form) and all three Logistic Regression feature variants A, B and C. C is the current primary candidate based on development macro-F1; A and B remain comparisons. CRF and pretrained models have not been implemented and must not be listed as completed systems.

- A: current lowercase word, sentence-initial flag, scaled word length.
- B: A plus previous and next lowercase words.
- C: B plus length 1-4 prefixes/suffixes and normalized boundary-marked character 2-4 grams.
- Fixed Logistic Regression parameters: SAGA, C=1, max_iter=500, tol=0.001, random_state=42. All recorded development fits converged.
- Use the existing train-only artifacts; do not silently refit on train plus dev. Vectorizers, vocabularies and mixed-case dictionaries remain train-only.
- Models use gold corpus tokenization and provided sentence boundaries in formal evaluation. The free-text front end's simple tokenizer is a separate demonstration condition.

## Labels and reconstruction

Four scored classes are lowercase, initial_uppercase, all_uppercase and mixed_case. Tokens with no cased letters are excluded. A one-letter uppercase form such as I is initial_uppercase. Classifier outputs are reconstructed into surface forms; mixed_case uses the most frequent mixed-case training form or falls back to lowercase for an unseen spelling. Score the classes of reconstructed outputs, matching the existing baseline and development evaluator. Keep raw class predictions as diagnostics, not a replacement primary score.

Observed corpus casing is the ground truth, including informal or inconsistent forms. Do not correct gold targets or selectively exclude errors after viewing test predictions. The CoNLL-U token policy skips multiword range and empty-node rows; evaluation remains on integer-ID syntactic tokens and does not reconstruct a raw-text document as its scoring unit.

## Ranking and reporting

Primary system ranking: four-class macro-F1 on all cased tokens. Undefined per-class F1 is zero; retain all four classes even if one is absent in a subgroup. If unrounded macro-F1 is exactly tied, use exact surface-form accuracy as the tiebreaker. Report per-class support, precision, recall and F1; exact-form accuracy; capitalization-required exact accuracy; and seen/unseen-word scores. Supplementary metrics disclose tradeoffs rather than overriding the declared primary ranking.

## Planned final conditions

1. All-genre training: score each of the six systems once on the reserved official test split. Also report each genre separately. Use the same target tokens for every system.
2. Leave-one-genre-out transfer: for B and C, score each existing source-only model on its target genre's official test portion. No target training or development examples enter its source resources. There are five targets and two model variants.
3. In-genre-only: score each of the ten target-only B/C models on its own genre test portion. Each existing model and its resources use only that genre train portion.
4. Compare all-genre and held-out conditions on the same target tokens. Use the source-only vocabulary to define identical seen/unseen subsets for the paired transfer comparison. Do not mix subset definitions.

The in-genre-only condition is now implemented and development-validated for B/C on all five targets, with the same fixed parameters and no target-dev tuning. Include it in the final per-target three-condition comparison. For the three-condition subgroup table, define seen/unseen against target-train vocabulary identically across all three systems; preserve the separate source-vocabulary subset definition if also reporting the earlier paired all-versus-held-out comparison. Label these definitions explicitly.

Training size differs between all-genre and four-source-genre conditions. Report this confound; current gaps are not isolated causal estimates of genre shift. Do not claim statistical significance without confidence intervals or an appropriate paired test.

## Required preparation before final test execution

- Team-check the 19 contextual examples. Assistant verification establishes agreement with the data and model artifacts, not human consensus. Keep team_confirmed=No until an actual team review occurs.
- Retain all three implemented training conditions and record the final scope; distinguish target-vocabulary and source-vocabulary subgroup definitions.
- Confirm the label exclusions, score/ranking rule and decoding policy above.
- Retain the code, dependency versions, data hashes and model hashes in PRETEST_ARTIFACT_MANIFEST.json. Create a new manifest if any model or evaluation code changes before testing.
- Verify a final evaluation script against the existing dev scores and token IDs before pointing it at test. No final-test execution script has been run in this preparation step.

## Final test procedure

Once the setup is settled, load the frozen models and train-only baseline resources. Run the declared comparisons and save scores, predictions, artifact hashes and execution metadata. Do not use test outcomes to choose features, C, thresholds, exclusions or the mixed-case policy. If there is a concrete implementation bug, record its nature and correction and disclose reruns; do not disguise adaptive tuning as an independent final result. Preserve failures and adverse comparisons in the report.

After final evaluation, assemble the paper and presentation with development and test results clearly separated. Include cross-genre limitations, rare-class support and the distinction between observed corpus casing and normative capitalization. Demonstrate the front end as a system interface, not as proof of its measured accuracy.

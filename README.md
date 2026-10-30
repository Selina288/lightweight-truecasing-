# Beyond Lexical Memorization

## A Cross Genre Evaluation of Lightweight Truecasing for Informal English

**Final Project Proposal**

**Team members:** Selina Yao, Jessie Zhang, and Adrian Berkmans

## Team Members and Proposed Roles

| Member | Primary strengths | Proposed responsibilities |
| --- | --- | --- |
| Selina Yao | Programming and document level processing | Build the corpus preprocessing pipeline; recover genre and document metadata; generate lowercase inputs and gold casing labels; implement the rule based and most frequent casing baselines; document the data and baseline methods. |
| Jessie Zhang | Linguistics and experimental methodology | Define the casing categories and linguistic edge case policy; design the train, development, and test comparisons and metrics; audit development examples; lead error analysis, related work, and interpretation of results. |
| Adrian Berkmans | Programming and lexical analysis | Implement the contextual classifier or CRF and character feature extension; produce lexical and out of vocabulary statistics; integrate model outputs with the shared evaluation format; document model implementation. |

All members will review the paper and presentation, maintain a shared experiment log, and attend the weekly mentor meeting. The modules will communicate through documented token level input and prediction files so that data preparation, modeling, and evaluation can proceed in parallel.

## Problem Statement and Research Questions

Truecasing restores capitalization to text whose case information is missing or unreliable. For example, a system should convert "i visited new york on monday" to "I visited New York on Monday." Truecasing supports readable speech recognition and machine translation output and may improve downstream tasks that depend on capitalization. However, informal web genres differ in vocabulary, named entities, abbreviations, sentence style, and casing conventions. A system that performs well on one genre may rely on memorized word forms and fail when applied to another genre.

We will study lightweight truecasing on the five informal English genres in the English Web Treebank. Our central question is: **How much does truecasing performance degrade across informal genres, and to what extent can contextual and character level features improve over lexical memorization, especially for words that were not observed during training?** We will also ask which capitalization classes and linguistic phenomena account for the largest cross genre errors. The project is a Systems Project: we will implement data preparation, casing reconstruction, baselines, feature extraction, model training, scoring, and error analysis components.

## Related Work

Lita et al. (2003) defined truecasing as restoration of case information in badly cased or uncased text and proposed a statistical language model system. Their experiments also connected truecasing to named entity recognition, information extraction, and machine translation. This work provides our task definition and motivates the most frequent casing baseline, but it concentrates on news rather than multiple informal genres.

Chelba and Acero (2004) treated capitalization as sequence tagging with maximum entropy and maximum entropy Markov models. They adapted a model trained on Wall Street Journal text to broadcast news and showed that domain mismatch reduces performance while a small amount of matched data helps. Their study is the closest precedent for our cross genre question. We extend the comparison to five informal English genres and separate the effects of lexical memorization, context, casing class, and vocabulary novelty.

Wang et al. (2006) used a conditional random field for capitalization of machine translation output and incorporated source language evidence. Their formulation supports our use of a contextual sequence model, although our inputs are native English web texts rather than translations. Susanto et al. (2016) instead used a character level recurrent neural network and showed the value of character modeling for unseen and mixed case words. We will test whether inexpensive prefix, suffix, and character n gram features provide similar benefits in a lightweight system.

Mayhew et al. (2019) examined NER and POS tagging when capitalization is absent. They found that a truecaser is not always the best route to downstream robustness, which limits claims about truecasing as a universal preprocessing solution. Grishina et al. (2020) compared rule based, character n gram, and recurrent models on German user generated conversational text and reported results by casing category; their analysis motivates our category specific and mixed case evaluation. Sunkara et al. (2020) jointly modeled punctuation and truecasing for medical ASR with pretrained masked language models and domain adaptation. Their system represents a higher complexity alternative; our project focuses on transparent models that can be trained and analyzed with modest computing resources.

## Data and Task Definition

We will use the Universal Dependencies version of the English Web Treebank, a corpus of approximately 16,600 sentences and 255,000 tokens drawn from weblogs, newsgroups, emails, reviews, and Yahoo Answers (Silveira et al., 2014). The corpus provides document identifiers and official training, development, and test files. We will preserve these files and derive genre labels from document metadata. We will not inspect test predictions until the model variants and analysis plan are fixed.

For each case bearing token, the lowercased surface form will be the model input and the observed original form will be the gold output. We will derive four labels: lowercase, initial uppercase, all uppercase, and mixed case. Tokens with no cased letters, such as punctuation and pure numbers, will be excluded from the main casing score and counted separately during data validation. A deterministic reconstruction component will convert predicted labels into surface forms. For mixed case words, lexical systems will use the most frequent training form; an unseen mixed case word will remain lowercase and count as an exact form error.

Because the Web Treebank was created for syntactic annotation rather than normative capitalization, its observed casing may contain noise or deliberate informal styling. We will manually audit a sample from the development set, document recurring inconsistencies, and define filtering rules using only training and development data. The reported task will therefore be described precisely as restoration of observed corpus casing, with limitations noted when the original form is ambiguous.

## Preliminary Work

We have reviewed seven related studies, completed and validated corpus preprocessing, and evaluated three baselines and a first contextual logistic regression model on development data. Eighteen automated tests pass. The model uses only lowercased input and sentence-boundary information, and no gold POS features. The best baseline macro-F1 is 0.4554; the first contextual model achieves 0.5245. Feature ablations have now also been completed (see the latest section below). Held-out-genre experiments, qualitative error analysis, the front-end, and final test evaluation remain future work.

## System Design and Implementation Strategy

| Version | Method | Purpose |
| --- | --- | --- |
| Baseline 1 | Leave every token lowercase | Establish the majority class lower bound and expose the weakness of overall accuracy. |
| Baseline 2 | Capitalize sentence initial tokens | Measure how much performance follows from position alone. |
| Baseline 3 | Use each token's most frequent training form | Measure lexical memorization without context. |
| Basic system | Logistic regression with lexical, position, and neighboring token features | Provide a feasible system that can be completed early and compared through feature ablation. |
| Full system | CRF or equivalent sequence model with contextual and character n gram features | Model neighboring casing decisions and improve generalization to rare or unseen words. |
| Optional extension | Small pretrained encoder or existing strong truecaser | Provide an upper comparison if time and computing resources allow; prompt engineering will not be the project focus. |

The lightweight models will use features that can be inspected directly: the current lowercased token, neighboring tokens, sentence position, token frequency, length, prefixes, suffixes, character n grams, digits, hyphens, apostrophes, with no gold POS annotations as model inputs. We will implement the corpus conversion, feature extraction, reconstruction, and scoring code ourselves. Libraries may supply standard classifiers, but the NLP representation and experimental pipeline will be our contribution.

## Evaluation Plan

The primary experiments will compare all system versions on the untouched test split after development is complete. Since lowercase tokens dominate the corpus, overall token accuracy alone would be misleading. We will report exact surface form accuracy, macro averaged F1 across the four casing labels, precision, recall, and F1 for each label, and accuracy restricted to tokens that require capitalization. Statistical confidence intervals or paired bootstrap comparisons will be added if time permits.

We will conduct three controlled evaluations. First, an overall model will train on all training genres, tune on all development genres, and be evaluated once on the full test set, with results also broken down by genre. Second, the in genre condition will train on the target genre's training portion and tune on its development portion before evaluation on that genre's test portion. Third, the cross genre condition will train and tune on the other four genres and evaluate on the same target genre test portion. The procedure will be repeated for each target genre, but no test file will enter training, feature selection, threshold selection, or error analysis.

To identify why versions differ, we will report results for words seen and unseen during training, sentence initial and noninitial words, proper nouns, acronyms and all uppercase words, and mixed case words. Ablation experiments will remove context and character features separately. A system will count as an improvement only when it exceeds the stronger most frequent casing baseline on macro F1 and capitalization required accuracy, rather than merely improving the already dominant lowercase class.

## Development Set Error Analysis

All qualitative error analysis will use development data. We will sample false positives and false negatives from each casing class and categorize them as named entities, ambiguous common word versus name cases, acronyms, sentence boundary effects, unseen words, informal spellings, inconsistent gold casing, or reconstruction failures. Each category will include representative examples and counts. The analysis may motivate a feature or filtering change, after which the revised model will be reevaluated on development data. Once the design is frozen, the test set will be used only for the final tables and will not motivate further tuning.

## Collaboration Plan and Schedule

Selina's preprocessing module will export a documented token record containing document, genre, split, sentence position, lowercased form, observed form, and gold label. Adrian's training module will read this stable format and output one prediction file per model. Jessie's evaluation specification will define exclusions, metrics, and error categories before final testing; the scoring script will consume the shared gold and prediction files. Sample files will allow each module to be developed before the others are complete. Code will be maintained in version control, and the experiment log will record the owner, configuration, data split, and result of every run.

During the first week, the group will finalize casing rules, audit development examples, and complete the data pipeline and literature review. During the second week, Selina will finish the baselines while Adrian builds the basic contextual model and Jessie validates metrics and label distributions. The third week will add sequence and character features and run development experiments. The fourth week will focus on ablations and development set error analysis. The group will then freeze the configuration, run the final test evaluation, and complete the paper and presentation. Each member will provide a concrete update at the weekly mentor meeting, and responsibilities will be rebalanced if one component becomes unexpectedly time consuming.

## Expected Contribution and Feasibility

The minimum deliverable is a reproducible data pipeline, three baselines, one contextual classifier, genre specific metrics, and development set error analysis. This version does not require expensive hardware and can be completed even if the CRF or pretrained comparison is delayed. The fuller contribution will explain when lightweight truecasing depends on word memorization and when context or character information supports cross genre generalization. We do not expect to establish a new state of the art result; the goal is a valid, transparent comparison that connects observed errors to properties of informal English data and to previous truecasing research.

## References

- Chelba, Ciprian, and Alex Acero. 2004. *Adaptation of Maximum Entropy Capitalizer: Little Data Can Help a Lot.* In Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, 285-292.
- Grishina, Yulia, Thomas Gueudre, and Ralf Winkler. 2020. *Truecasing German User-Generated Conversational Text.* In Proceedings of the Sixth Workshop on Noisy User-generated Text, 143-148.
- Lita, Lucian Vlad, Abe Ittycheriah, Salim Roukos, and Nanda Kambhatla. 2003. *tRuEcasIng.* In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics, 152-159.
- Mayhew, Stephen, Tatiana Tsygankova, and Dan Roth. 2019. *NER and POS When Nothing Is Capitalized.* In Proceedings of EMNLP-IJCNLP 2019, 6256-6261.
- Silveira, Natalia, Timothy Dozat, Marie-Catherine de Marneffe, Samuel R. Bowman, Miriam Connor, John Bauer, and Chris Manning. 2014. *A Gold Standard Dependency Corpus for English.* In Proceedings of LREC 2014, 2897-2904.
- Sunkara, Monica, Srikanth Ronanki, Kalpit Dixit, Sravan Bodapati, and Katrin Kirchhoff. 2020. *Robust Prediction of Punctuation and Truecasing for Medical ASR.* In Proceedings of the First Workshop on Natural Language Processing for Medical Conversations, 53-62.
- Susanto, Raymond Hendy, Hai Leong Chieu, and Wei Lu. 2016. *Learning to Capitalize with Character-Level Recurrent Neural Networks: An Empirical Study.* In Proceedings of EMNLP 2016, 2090-2095.
- Wang, Wei, Kevin Knight, and Daniel Marcu. 2006. *Capitalizing Machine Translation.* In Proceedings of the Human Language Technology Conference of the NAACL, 1-8.


## Running the Current Implementation

Use Python 3.12 or later (tested with Python 3.14). Preprocessing and baselines use only the standard library; the new contextual model requires the packages in requirements.txt. Run these commands from the project root:

```bash
python3 -m unittest discover -s tests -v
python3 src/preprocess.py
python3 src/baselines.py
```

Place the official three CoNLL-U files under `data/raw/UD_English-EWT/` before preprocessing. Raw and processed data remain excluded by `.gitignore`.

### Completed Preliminary Experiments

The preprocessing pipeline and three baselines have now been implemented and run. Baseline word counts use only training data; all results below are from the official development split. No test data are read by the baseline script. These are development results, not final test results.

| Baseline | Exact form accuracy | Macro-F1 | Capitalization-required accuracy |
| --- | ---: | ---: | ---: |
| All lowercase | 81.35% | 0.2243 | 0.00% |
| Sentence initial | 87.65% | 0.3763 | 39.43% |
| Most frequent training form | 86.00% | 0.4554 | 35.35% |

Scores exclude tokens without cased letters. Macro-F1 averages the four casing classes, with zero for undefined class F1. Capitalization-required accuracy measures exact surface recovery for gold tokens whose class is not lowercase. The sentence-initial baseline uppercases the first cased letter of the first token; it does not skip an initial punctuation token to capitalize a later token. The lexical baseline resolves frequency ties in lexicographic order and leaves unseen words lowercase. Label accuracy and exact mixed-case spelling accuracy are different quantities.

`results/baselines/dev_summary.tsv` contains overall, per-genre, seen-word and unseen-word scores. `dev_per_class.tsv` contains per-class precision, recall, F1 and support. Per-genre results here use one baseline fitted on all training genres; held-out-genre transfer experiments have not yet been implemented. The first contextual classifier has now been implemented; see the next section.


## First Contextual Logistic Regression Model

The current word, previous word, next word, sentence-initial position, and scaled token length are the only model features. Vocabulary and feature mappings are fitted on training data only. Gold surface forms and casing labels are used only as training targets or evaluation references. No gold POS features or target-genre test data enter this experiment.

Install and run from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/logreg.py
python src/truecase.py "i visited new york on monday."
```

The processed train/dev files must already exist. If needed, place the official corpus files in `data/raw/UD_English-EWT/` and run `python src/preprocess.py` first. Training saves `models/logreg.joblib`, which is ignored by Git, and writes evaluation tables under `results/logreg/`. Token predictions are kept locally in the ignored processed-data directory. Pinning package versions records the tested environment; small numerical differences may occur across platforms.

| System | Exact form accuracy | Four-class macro-F1 | Capitalization-required accuracy |
| --- | ---: | ---: | ---: |
| all_lowercase | 81.35% | 0.2243 | 0.00% |
| sentence_initial | 87.65% | 0.3763 | 39.43% |
| most_frequent_case | 86.00% | 0.4554 | 35.35% |
| logistic_regression | 91.43% | 0.5245 | 64.95% |

These are official development-split results. All four systems were scored by the same evaluator, excluding uncased tokens. Macro-F1 is computed from the casing classes of reconstructed output forms, matching the baseline scorer. The primary model ranking uses macro-F1; exact-form accuracy breaks ties. A predicted mixed-case class is reconstructed with its most frequent mixed-case training form, or lowercase for an unseen word. Thus raw classifier-label scores may differ from output-derived scores. This decoder uses only training data and is part of the reported system.

The classifier converged in 64 iterations with a fixed initial configuration (C=1, SAGA, seed 42, tolerance 0.001). This is one initial development experiment, not a completed hyperparameter search. `run_metadata.json` records configuration and environment details. Per-genre scores here come from training on all genres; they are not held-out-genre transfer results. No character features have been added yet, so gains should not be attributed to context alone until the planned ablations are performed.

The command-line demo uses a simple tokenizer and preserves input spacing. Official evaluation uses the corpus tokenization and sentence boundaries. The demo may therefore differ from corpus scoring, especially for contractions or ambiguous sentence boundaries. It is not yet the required front-end interface. Next steps are character features and controlled ablations, held-out-genre development experiments, error analysis, front-end integration, and final testing after the design is frozen.


## Completed Feature Ablation Experiments

Three Logistic Regression variants now compare lexical and positional features (A), added context (B), and added character features (C). All use the same training/development splits, fixed hyperparameters, reconstruction and scoring. This supersedes earlier next-step statements about character ablations: that stage is now complete. Earlier model results remain recorded as the initial experiment.

The dev macro-F1 scores are A: 0.4892, B: 0.5245, and C: 0.5688. C achieves 92.26% exact accuracy overall and 69.93% on unseen words. These are development observations, not final test or held-out-genre results. See [the full ablation report](results/ablation/REPORT.md) for tables, definitions and limitations. Eighteen automated tests pass.

```bash
python src/ablation.py
python src/truecase.py --model models/ablation/context_char.joblib "i visited new york on monday."
```


## Completed Leave One Genre Out Development Experiments

The five held-out-genre experiments have now been completed for B (context) and C (context plus characters). Each fold uses four source training genres and the untouched target development portion for evaluation with fixed parameters. No target-genre training examples enter that fold's classifier, feature vocabulary or mixed-case decoder. These results supersede older statements that held-out-genre development experiments are future work; final test evaluation remains outstanding.

See [the cross-genre report](results/crossgenre/REPORT.md) for results and limitations. Training size differs between the all-genre reference and four-genre conditions, so gaps must not be attributed solely to genre shift. Seen/unseen subsets use one common source-only vocabulary for both conditions. Eighteen tests pass. Next steps are qualitative dev error analysis and front-end integration before final test evaluation.

```bash
python src/crossgenre.py
```


## Completed Development Error Analysis

The saved B/C development predictions have now been analyzed under both all-genre and held-out-genre conditions. No retraining, gold correction or final testing was performed. C reduces overall exact errors, but it introduces new overcapitalization errors. Held-out Reviews illustrates why exact accuracy and four-class macro-F1 can disagree when rare uppercase classes change.

See [the error-analysis report](results/error_analysis/REPORT.md) for counts, metric interpretation and limitations. Nineteen contextual examples were reviewed by the assistant; team confirmation is still pending and the sample does not estimate population error-category frequencies. This section supersedes older future-work statements about automatic dev error analysis. The required front-end, team review and final frozen test evaluation are still outstanding. Eighteen tests pass.

```bash
python src/error_analysis.py
```


## Local Front End Demonstration

The required local interface is now implemented in `web/` and served by `src/demo_server.py`. It calls the real trained B/C models or one of three baselines through a local Python API. This supersedes earlier status statements listing the front-end as future work. The interface is local-only and has not been publicly deployed. Team confirmation of contextual example reviews and final test evaluation remain outstanding.

With dependencies installed and the environment activated:

```bash
python src/demo_server.py
```

Open http://127.0.0.1:8765 in your browser. Keep the terminal running; stop with Ctrl+C. Use `--port 8766` if the default port is busy. Do not open web/index.html directly. The server finds assets and model files relative to the project directory, even if started elsewhere.

The interface converts input to lowercase, preserves punctuation and spacing, highlights changed tokens, supports model selection and copying, and displays the existing all-genre development scores. These are not scores for the typed sentence. It uses the same `predict_rows` and demo tokenizer as the command-line path; formal experiment tokenization remains unchanged. Inputs are processed locally and are not saved by the demo server.

Model files remain ignored by Git. The complete local package includes them. From a fresh GitHub checkout, first obtain the official corpus and run preprocessing and ablation to generate the reference models. Missing models are disabled in the dropdown. No new package is needed beyond the existing requirements; the web server uses Python's standard library. See DEMO_GUIDE_zh.md for detailed setup steps.

Twenty-two automated tests pass. HTTP checks exercised all five choices and invalid-input handling. Browser checks verified the C-model prediction, highlighting and switching to the all-lowercase baseline. No test data were used and no existing experiment scores were changed.


## Pretest Preparation

The 19 assistant-reviewed examples have passed a second data and saved-model verification, with five interpretations refined. This is not human/team confirmation. Final test evaluation has not been run. See docs/FINAL_EVALUATION_PLAN.md for the candidate protocol and docs/PRETEST_ARTIFACT_MANIFEST.json for current artifact hashes. The original proposal's in-genre-only training condition remains an explicit scope item to resolve before freezing the final evaluation.

```bash
python src/verify_review.py
```


## Completed In Genre Only Experiments

The proposal's in-genre-only condition has now been implemented for B and C on all five genres. Ten target-only models converged; 25 automated tests pass. This resolves the previously noted missing condition. The new [in-genre report](results/ingenre/REPORT.md) compares all-genre, in-genre-only and held-out training on identical target development tokens. No test data were read.

```bash
python src/ingenre.py
```

Data size and genre coverage differ between conditions. For the new three-condition subgroup comparison, seen/unseen membership is defined by the same target-train vocabulary across all conditions; this differs from the earlier source-vocabulary-based transfer subgroup table. Overall reference scores reproduce the earlier experiments. Team confirmation of reviewed examples and the frozen final test evaluation remain outstanding. The local front-end is unchanged.


## Completed Final Test Evaluation

Final quantitative evaluation is now complete with frozen models, after the unified script reproduced 138 prior dev rows. Independent metrics verified 198 test summary rows; all frozen source/model hashes still match. No retraining, gold editing, feature changes or parameter tuning occurred after viewing test results. Twenty-seven tests pass.

C, selected before testing, achieves 91.50% exact-form accuracy and 0.5823 four-class macro-F1 on the official test split. All six systems and all three genre training conditions are reported in [the final test report](results/final_test/REPORT.md). These results supersede earlier statements that final test scoring is future work.

The 19 qualitative examples remain team_confirmed=No, following the user's report that the group has not checked them. Quantitative evaluation does not imply human confirmation. See docs/TEAM_REVIEW_STATUS.md and docs/FINAL_EXECUTION_STATUS.md. The next tasks are the team's example review, final paper and presentation. The front end remains a local demonstration and its displayed experiment scores remain labeled dev.

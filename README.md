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

We have completed a targeted literature review of seven peer-reviewed studies on truecasing and capitalization restoration. We have downloaded the UD English Web Treebank, implemented and validated the preprocessing pipeline, preserved the official train/dev/test splits, and generated token-level inputs and casing labels. We have verified the training and development label distributions and implemented three baselines: all lowercase, sentence-initial capitalization, and most frequent training form. These baselines have been evaluated on the development set, with overall, per-genre, per-class, and seen/unseen-word results recorded in the repository. Five automated tests pass. The next implementation milestone is a lightweight contextual classifier. Final test evaluation has not yet been performed.

## System Design and Implementation Strategy

| Version | Method | Purpose |
| --- | --- | --- |
| Baseline 1 | Leave every token lowercase | Establish the majority class lower bound and expose the weakness of overall accuracy. |
| Baseline 2 | Capitalize sentence initial tokens | Measure how much performance follows from position alone. |
| Baseline 3 | Use each token's most frequent training form | Measure lexical memorization without context. |
| Basic system | Logistic regression with lexical, position, and neighboring token features | Provide a feasible system that can be completed early and compared through feature ablation. |
| Full system | CRF or equivalent sequence model with contextual and character n gram features | Model neighboring casing decisions and improve generalization to rare or unseen words. |
| Optional extension | Small pretrained encoder or existing strong truecaser | Provide an upper comparison if time and computing resources allow; prompt engineering will not be the project focus. |

The lightweight models will use directly inspectable features derived from the lowercased input: the current token, neighboring tokens, sentence position, training-set token frequency, length, prefixes, suffixes, character n-grams, digits, hyphens, and apostrophes. The main systems will not use the corpus's gold POS annotations as input features. If gold POS features are explored, they will be reported separately as an oracle-assisted experiment, since manually annotated POS tags would not normally be available for new input text. Gold annotations may still be used to define evaluation subsets, such as proper nouns, without being provided to the model. We will implement corpus conversion, feature extraction, reconstruction, and scoring ourselves. Libraries may supply standard classifiers, but the NLP representation and experimental pipeline will be our contribution.

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

Use Python 3.10 or later. The preprocessing and baseline scripts use only the Python standard library. Run these commands from the project root:

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

`results/baselines/dev_summary.tsv` contains overall, per-genre, seen-word and unseen-word scores. `dev_per_class.tsv` contains per-class precision, recall, F1 and support. Per-genre results here use one baseline fitted on all training genres; held-out-genre transfer experiments have not yet been implemented. The next modeling step is a contextual classifier compared against these fixed baselines.

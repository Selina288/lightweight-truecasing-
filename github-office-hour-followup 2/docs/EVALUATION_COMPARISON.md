# Evaluation comparison following office-hour feedback

This audit checks evaluation compatibility; it is not an external-system benchmark. We inspected the linked primary papers. None of the four audited studies reports this project's exact EWT split with the same four-class output-derived macro-F1 policy. Published scores therefore do not establish that our system is better or worse. This audit covers these four studies, not every truecasing publication.

| Study | Data and model/evaluation unit | Reported evaluation and relevant policy | Compatibility with this project |
| --- | --- | --- | --- |
| Our original experiment | Official UD English EWT splits; token-level LR labels and reconstructed forms | Four fixed output-derived class F1s averaged equally, exact-form accuracy, and capitalization-required exact accuracy. Uncased tokens excluded. | Reference policy: 21,430 scored test tokens, 4,233 capitalization-required. Lowercase baseline 80.25%, C 91.50% exact; 0.2226 versus 0.5823 macro-F1. |
| [Lita et al. 2003](https://aclanthology.org/P03-1020.pdf), Section 3 | News tests from AP/Reuters, ACE and MT references; unigram/LM restoration | Agreement with original news forms; additional downstream evaluations. Separate news headlines/titles/headers test. | Different data and evaluation tasks. The paper's intrinsic agreement is conceptually related to exact restoration, but neither data nor all preprocessing/exclusion details are matched. Downstream F-measures are not our casing macro-F1. |
| [Chelba and Acero 2004](https://aclanthology.org/W04-3237.pdf), Sections 2 and 5 | WSJ and Broadcast News; word-level MaxEnt/MEMM tagging with domain adaptation | Capitalization error rate; unigram baseline explicitly overrides the first word with initial capitalization. Punctuation has a separate PNC tag. | Different data and baseline. Our most-frequent-form baseline has no sentence-initial override. A PNC modeling tag alone does not establish the scoring denominator; inspect the scorer before assuming it matches our uncased exclusion. |
| [Susanto et al. 2016](https://aclanthology.org/D16-1225.pdf), Section 4, Table 2 and Figure 2 | English Wikipedia, WSJ, Reuters and German ECI; word and character models | Accuracy, P/R/F1, and accuracy by casing category/OOV. All-uppercase news headlines are discarded. | Different corpora and filtering. Table 2's F1 is not explicitly identified there as our fixed four-class macro average. Exact scorer/aggregation compatibility remains unverified; do not relabel its F1 as our macro-F1. |
| [Grishina et al. 2020](https://aclanthology.org/2020.wnut-1.19.pdf), Sections 3–4, Tables 1–5 | German Q&A and Leipzig; tag/character LMs and RNNs | Relative accuracy/P/R/F1 improvements over a tag-LM baseline; category, OOV and sentence exact accuracy. Their mixed category includes acronyms such as NASA. | Different language, data, baseline and category mapping: we give all-uppercase tokens their own class. Their Table 1 reports changes relative to a baseline, not absolute scores comparable with our 0.5823. |

## Immediate conclusions

1. State our token/form scoring explicitly. The all-lowercase baseline uses the exact same test tokens and scorer as C, so that internal comparison is valid.
2. Distinguish whole-form equality, casing-class F1 and exact recovery restricted to capitalization-required tokens. A category match need not imply correct internal mixed-case spelling.
3. Do not compare a different paper's F1 or relative improvement directly with our macro-F1. Shared metric names do not establish shared definitions.
4. Our unigram baseline is a transparent lexical reference, not an exact reimplementation of every prior unigram convention. Its lack of a sentence-initial override must remain stated.

## What is complete and what is still open

Complete: paper-level evaluation audit; same-data internal baselines; original score definitions; new train/dev character-family comparison recorded separately in results/character_ablation/.

Open: run an existing external truecaser under a matched EWT protocol, verify its precise scoring code and data overlap, and document its external training data. The audit does not claim such a system was downloaded or tested. A pretrained model exposed to EWT would not be a clean comparison without a contamination caveat.

Optional future supplementary metrics could include exact sentence recovery or change-oriented precision/recall with explicitly stated definitions. They must be labeled new supplementary analyses, not silently substituted for the original ranking metric. No assertion is made that they reproduce a prior paper's implementation.

## Provenance and scope

All experiment figures above refer to existing final-test tables, not new test scoring. Character-family follow-ups use train/dev only, were designed after the original test results were known, and are exploratory. Frozen original results remain unchanged. Citation locations were checked against the public PDFs; no authenticated course annotations were consulted.

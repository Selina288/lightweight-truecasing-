# Equal-budget genre comparison

This is a new exploratory **dev-only** follow-up after the original test results
were known. It does not replace the original C model or constitute a new final
test. Defaults were fixed in the experiment source before fitting these models.

## Question and design

Does in-genre training outperform training on the pooled other four genres when
the number of supervised training tokens is the same?

- Targets: answers, email, newsgroup, reviews, weblog; each uses its full official
  target-genre dev set, identical across conditions and repetitions.
- Two conditions: target-genre train only (`in_genre`); all other four train genres
  pooled (`other_genres`). The latter follows their natural token proportions in
  expectation; it is not four separate source-to-target comparisons.
- Uniform sampling without replacement: exactly **20,000 cased training anchors**
  per model. The smallest target train pool exceeds this budget. No stratification
  by gold label or adaptive selection of budget/seeds based on outcomes.
- Sampling seeds: **11, 22, 33, 44, 55**. Each target/condition has a deterministically
  salted random stream. Samples across repetitions may overlap.
- Same original C features, LR settings and reconstruction rules. Solver seed
  stays 42; the varied seeds measure training-subsample variability, not solver
  variability. Total: 5 targets × 2 conditions × 5 repetitions = **50 fits**.
- Training features retain true sentence position and immediate original
  lowercase neighbors. Dropping tokens and joining remaining words would corrupt
  context. Only sampled anchors contribute labels, original forms, vectorizer
  training rows, and mixed-case decoder entries. Neighbor forms enter as lowercase
  features, never as extra gold labels or decoder entries.
- No target dev label enters fitting; dev labels are used for scoring only.
  Neither raw nor processed test tokens are read by this experiment.

## Scores and variation

Primary: fixed four-class macro-F1 derived from reconstructed forms, zero division
set to zero. Secondary: exact-form accuracy and exact accuracy on gold tokens
requiring capitalization. Uncased tokens are excluded, as in the original study.

Report each target and condition's mean and **sample standard deviation** over
five subsamples. Also report `in_genre - other_genres` paired differences. A numeric
seed pairs two independently sampled training sets, not shared training examples.
Standard deviations are not confidence intervals over new evaluation documents;
no significance test is claimed. Do not replace per-target results with one pooled
score from different models or treat five overlapping subsamples as independent
corpora.

## What this controls, and what it does not

The matched unit is supervised cased-token count. Document/sentence counts, total
unique lowercase context exposure, vocabulary, label proportions and source
diversity are not held equal. Their counts are recorded in run metadata. The
mixed-case decoder and vocabulary are refit on each sample to avoid giving either
condition information from its full unsampled training pool. Consequently, this
is stronger evidence about equal-budget **training choices**, not an isolated
causal estimate of linguistic genre. Earlier dev data informed feature design,
so this is not target-blind model selection.

## Reproduction and records

Run `python src/size_matched_genre.py` from the complete original local project.
This run verifies the historical `FINAL_FROZEN_MANIFEST.json` before/after fitting
and refuses to overwrite existing output directories. It requires those original
artifacts; newly regenerated joblib files need not have matching historical
hashes. Original files remain archived, not retrained by this command.

Tracked outputs: `results/size_matched_genre/` with per-run scores, per-class
scores, condition means/SDs, paired gaps, metadata, independent verification and
reports. Weights stay under ignored `models/size_matched_genre/`; exact selected
row indices and dev predictions stay under ignored
`data/processed/size_matched_genre/`. The selected indices refer to the hashed
original `tokens_train.tsv` order. Train/dev hashes, source hash, model hashes,
convergence, sample label/genre counts and environment versions are recorded.

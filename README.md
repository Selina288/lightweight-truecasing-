# Beyond Lexical Memorization

## A Cross-Genre Evaluation of Lightweight Truecasing for Informal English

**Team Members**

- Selina Yao
- Jessie Zhang
- Adrian Berkmans

## Project Overview

This project investigates lightweight truecasing for informal English across
multiple genres. We will evaluate whether contextual and character-level
features improve cross-genre generalization beyond lexical memorization.

## Research Questions

- How much does truecasing performance decrease across informal genres?
- Can contextual and character-level features improve performance on unseen
  words?
- Which capitalization categories produce the largest cross-genre errors?

## Dataset

We will use the Universal Dependencies version of the English Web Treebank,
which contains weblogs, newsgroups, emails, reviews, and Yahoo Answers.

The official training, development, and test splits will be preserved.

## System Design

### Baselines

1. Leave all tokens lowercase.
2. Capitalize sentence-initial tokens.
3. Use each token's most frequent training form.

### Basic System

Logistic regression using:

- Token identity
- Sentence position
- Neighboring tokens
- Token frequency

### Full System

A CRF or similar sequence model using:

- Contextual features
- Prefixes and suffixes
- Character n-grams
- Token shape features

## Evaluation

We will report:

- Exact surface-form accuracy
- Macro-averaged F1
- Precision, recall, and F1 for each casing class
- Accuracy on tokens requiring capitalization
- Results by genre
- Results for seen and unseen words

All model selection and error analysis will use the development set. The test
set will only be used for final evaluation.

## Team Responsibilities

### Selina Yao

- Corpus preprocessing
- Genre and document metadata
- Lowercase input and gold-label generation
- Rule-based and most-frequent-case baselines

### Jessie Zhang

- Linguistic label definitions
- Experimental methodology
- Evaluation metrics
- Development-set error analysis
- Related work and result interpretation

### Adrian Berkmans

- Contextual classifier and CRF implementation
- Character-level features
- Vocabulary and out-of-vocabulary analysis
- Model integration

## Preliminary Work

We have:

- Reviewed related truecasing research in the ACL Anthology
- Selected the English Web Treebank
- Defined four casing labels
- Designed the train, development, and test evaluation protocol

## Repository Structure

```text
lightweight-truecasing/
├── README.md
├── data/
├── src/
│   ├── preprocess.py
│   ├── baselines.py
│   ├── model.py
│   └── evaluate.py
├── experiments/
├── results/
└── requirements.txt
```

## Current Status

This project is currently in the proposal and initial implementation stage.

# Dataset used in the truecasing project

This folder provides the exact original UD English EWT files used in our completed experiments, so readers can inspect the data directly. The files are copied unchanged from the local experimental inputs; SHA256SUMS.txt identifies their bytes.

## Source and license

- Original dataset: Universal Dependencies English Web Treebank (UD English EWT).
- Source repository: https://github.com/UniversalDependencies/UD_English-EWT
- The retained upstream README identifies version 2.18, dated 2026-05-15. See UPSTREAM_README.md for the original description, contributors, acknowledgments and detailed copyright information.
- The distributed treebank states CC BY-SA 4.0 terms. LICENSE.txt and the upstream copyright/attribution text are retained. This project did not create the original corpus.
- Use the hashes to identify the experiment snapshot; a moving upstream master branch or a matching version label alone does not establish byte identity.

## Files and official splits

| File | Split | Documents | Sentences | Integer-ID tokens | Scored cased tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| en_ewt-ud-train.conllu | train | 540 | 12,544 | 204,578 | 176,883 |
| en_ewt-ud-dev.conllu | development | 318 | 2,001 | 25,148 | 21,668 |
| en_ewt-ud-test.conllu | test | 316 | 2,077 | 25,094 | 21,430 |

The five genres are answers, email, newsgroup, reviews and weblog. Counts above follow our integer-token parsing policy: multiword-token range rows and empty-node rows are skipped. Tokens without cased letters remain available as context but are excluded from the primary casing scores.

## How the project uses these files

The CoNLL-U FORM column supplies observed original token forms. The model input is FORM.lower(); the reference is the observed original FORM. Labels are lowercase, initial_uppercase, all_uppercase and mixed_case. Gold POS and dependency annotations are not model inputs. Original casing can be informal or inconsistent; the target is observed corpus restoration, not a newly corrected English reference.

We preserve official train/dev/test splits. Training resources are fitted on train; dev supports model comparison and error analysis; the final test results were produced with frozen models. Publishing the original test file here does not permit retuning from its results.

## Run the existing code

This folder is an inspection copy for readers. The existing preprocessing default remains data/raw/UD_English-EWT/. Either copy the three .conllu files there, or run from the repository root:

```bash
python src/preprocess.py --input-dir dataset
python src/check_data.py --processed
```

The processed files remain under data/processed/ and are not added to this folder. To use the existing raw-hash checker, first copy these originals to its expected data/raw/UD_English-EWT/ location, then run python src/check_data.py.

For the experiment protocol, see ../docs/REPRODUCIBILITY.md. Model weights are not dataset files and remain excluded from Git.

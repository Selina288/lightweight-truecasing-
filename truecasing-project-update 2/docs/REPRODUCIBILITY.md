# Reproducing the project

## Choose the intended activity

The public repository contains source, tests, aggregate results, metadata, a local web interface and qualitative review records. An inspection copy of original raw data is now in dataset/ with license and hashes; local raw/processed working files, token predictions and trained model weights remain ignored. Reading the published results needs no training. Running the trained demo from a fresh checkout requires local data and model generation. Reconstructing the exact final frozen run additionally requires byte-identical original artifacts.

Keep the original full experiment folder unchanged. Use a separate clone for new training. The commands below do not authorize model selection or tuning on the already seen test set.

## Environment

The original run recorded Python 3.14.0 and scikit-learn 1.9.1. From a fresh repository root, confirm `python3 --version` identifies the intended interpreter, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/verify_team_review.py
```

The original run passed 27 tests. All 31 tests passed after cleanup, including four added checks for review consistency and missing/mismatched data. The default review check does not need data, models or scikit-learn. Other Python/platform combinations and a fresh dependency installation were not validated by the cleanup.

## Obtain and identify the corpus

1. Visit the official [UD English EWT r2.18 release tree](https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.18), or [its release tag page](https://github.com/UniversalDependencies/UD_English-EWT/tags). Download and extract the source archive. Do not assume the current master branch is the experiment version.
2. Create `data/raw/UD_English-EWT/` inside the project. Place these files directly in it, without another nested release folder:
   - `en_ewt-ud-train.conllu`
   - `en_ewt-ud-dev.conllu`
   - `en_ewt-ud-test.conllu`
3. Retain the upstream README and LICENSE with your local copy. Corpus attribution and excerpt license text are also retained in the repository.
4. Run:

```bash
python src/check_data.py
python src/preprocess.py
python src/check_data.py --processed
```

[DATA_PROVENANCE.json](DATA_PROVENANCE.json) records SHA-256 hashes of the exact original local inputs and generated TSVs. The source README identifies v2.18, but the release archive was not freshly downloaded and compared during this cleanup. The hashes are the authority: on any mismatch, stop and resolve the data provenance before training. Do not replace the recorded hashes just to make another snapshot pass. The raw check reads file bytes for identity only; it does not inspect test predictions or compute test metrics. The preprocessing script preserves the official splits.

Expected integer-token counts are train 204,578, dev 25,148 and test 25,094; scored cased-token counts are 176,883, 21,668 and 21,430. Counts alone cannot prove corpus identity.

## Generate demo models in a fresh checkout

```bash
python src/ablation.py --output-dir results/reproduced/ablation
python src/demo_server.py
```

This fits A/B/C using train only, evaluates dev and writes weights to the ignored `models/ablation/` directory. New development tables go to `results/reproduced/ablation/`, preserving published tables. The web interface continues to display the published dev scores, so newly trained predictions must not be described as an exact reproduction until compared.

Open http://127.0.0.1:8765. Keep the terminal open; stop with Ctrl+C. Use `--port 8766` if needed. The UI needs its Python server; opening index.html directly is insufficient. No models are trained by the server. Missing trained choices are disabled; the lowercase and sentence-initial baselines remain available.

## Reconstruct other development experiments

The existing crossgenre and ingenre scripts read metadata from fixed `results/ablation/` and `results/crossgenre/` locations. Use another disposable fresh clone for the full historical sequence, because these commands replace intermediate dev tables there:

```bash
python src/check_data.py
python src/preprocess.py
python src/check_data.py --processed
python src/baselines.py
python src/logreg.py
python src/ablation.py
python src/crossgenre.py
python src/ingenre.py
python src/error_analysis.py
```

The preprocessing and data checks must already have matching local corpus files; run the environment setup first. This sequence trains the development models and generates local predictions. Do not run it over the archived full project. Keep new results separate and label them as reruns. Small numerical differences and serialized model-byte differences can occur across runs or platforms.

## Qualitative review

```bash
python src/verify_team_review.py
```

This read-only check validates 19 unique retained examples, the eight selected original IDs, Yes/No flags, selection order, matching table fields and confirmation provenance. It cannot establish that humans performed the review; it checks the user's recorded report. If the original frozen train/dev files and original saved models are available:

```bash
python src/verify_team_review.py --with-models
```

The optional check validates all 19 records against train/dev and frozen saved B/C predictions, without opening the test split. Newly trained joblib files can fail the historical hash check even when their scores are similar.

`src/verify_review.py` is the restored historical verifier required by FINAL_FROZEN_MANIFEST.json. It assumes all 19 flags are No. Do not run it on the current confirmed table; use the current command above. Its original report is retained as `review_verification.json`, whose table hash refers to the pre-confirmation table. Confirmation updates are documented separately rather than rewriting historical audit records.

## Final test outputs and frozen checks

The existing `results/final_dev/` and `results/final_test/` are the completed evaluation, not empty output destinations. The final evaluator checks the original frozen manifest before evaluating and refuses to overwrite a completed run. It first replicated 138 dev rows; independent scoring checked 198 test summary rows.

Do not run `final_evaluate.py` merely to view results. A GitHub-only checkout lacks frozen model/data artifacts. Newly trained models are not guaranteed to match original joblib SHA-256 values. An exact rerun requires all original frozen artifacts, preservation of the existing result folders elsewhere, dev replication before test, and an explicit rerun record. Do not edit the historical manifest to make new models look like the original run. Do not tune from test results.

## Scope of this cleanup

The cleanup restores missing review files, records eight confirmed examples, provides provenance checks and clarifies documentation. It changes no frozen source file, trained model, gold token or quantitative score. Dependency installation, upstream archive identity and a fresh training reproduction are not claimed as newly verified. Paper, slides and actual individual contributions are outside this update.

## Character-family analysis

The new script src/character_ablation.py compares frozen B/C references with B+affixes and B+character-n-grams on dev only. It needs the complete original local experiment package, including every original artifact listed in FINAL_FROZEN_MANIFEST.json (not just two reference weights); it validates that manifest before and after running. A public GitHub checkout with newly regenerated weights is not guaranteed to satisfy those historical hashes.

```bash
python -m unittest discover -s tests -v
python src/character_ablation.py
```

The completed results are already in results/character_ablation/. Do not rerun over them: the script refuses existing follow-up output, model or prediction directories. Original model weights are unchanged; new weights and predictions remain local. This follow-up adds three tests, for 34 total passing tests. Its report records that original test outcomes were known before design. It does not conduct size-matched genre training or run an external truecaser. See EVALUATION_COMPARISON.md for the paper-level metric audit.

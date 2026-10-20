#!/usr/bin/env python3
"""Convert UD English EWT CoNLL-U files into token-level truecasing data.

The script preserves the official train/dev/test splits. It reads the FORM
column as the original token, creates a lowercase model input, assigns a casing
label, and derives the EWT genre from the document ID prefix.

Generated token columns:
    split, document_id, genre, sentence_id, token_id,
    original_token, lowercase_input, casing_label

Label definitions:
    lowercase         all cased letters are lowercase
    initial_uppercase first cased letter is uppercase and the rest lowercase
    all_uppercase     two or more cased letters are all uppercase
    mixed_case        any other combination of upper- and lowercase letters
    uncased           no cased letters, such as punctuation or pure numbers

A one-letter uppercase token such as "I" is labeled initial_uppercase.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


EXPECTED_GENRES = {"answers", "email", "newsgroup", "reviews", "weblog"}
FILE_NAMES = {
    "train": "en_ewt-ud-train.conllu",
    "dev": "en_ewt-ud-dev.conllu",
    "test": "en_ewt-ud-test.conllu",
}
TOKEN_FIELDS = [
    "split",
    "document_id",
    "genre",
    "sentence_id",
    "token_id",
    "original_token",
    "lowercase_input",
    "casing_label",
]


@dataclass(frozen=True)
class TokenRecord:
    split: str
    document_id: str
    genre: str
    sentence_id: str
    token_id: str
    original_token: str
    lowercase_input: str
    casing_label: str

    def as_row(self) -> dict[str, str]:
        return {field: str(getattr(self, field)) for field in TOKEN_FIELDS}


def cased_letters(token: str) -> list[str]:
    """Return letters for which Unicode distinguishes lower and upper case."""
    return [char for char in token if char.lower() != char.upper()]


def classify_casing(token: str) -> str:
    """Map a surface token to one of the project's casing labels."""
    letters = cased_letters(token)
    if not letters:
        return "uncased"
    if all(char.islower() for char in letters):
        return "lowercase"
    if letters[0].isupper() and all(char.islower() for char in letters[1:]):
        return "initial_uppercase"
    if all(char.isupper() for char in letters):
        return "all_uppercase"
    return "mixed_case"


def infer_document_id(sentence_id: str) -> str:
    """Recover a document ID if a sentence lacks an explicit newdoc line."""
    prefix, separator, suffix = sentence_id.rpartition("-")
    if separator and suffix.isdigit():
        return prefix
    return sentence_id


def infer_genre(document_id: str) -> str:
    """Derive an EWT genre from the prefix of its document ID."""
    genre = document_id.split("-", 1)[0]
    if genre not in EXPECTED_GENRES:
        raise ValueError(
            f"Unexpected genre prefix {genre!r} in document {document_id!r}"
        )
    return genre


def iter_conllu_tokens(path: Path, split: str) -> Iterator[TokenRecord]:
    """Yield token records while skipping multiword and empty-node rows."""
    document_id: str | None = None
    sentence_id: str | None = None

    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\n")

            if line.startswith("# newdoc id = "):
                document_id = line.split("=", 1)[1].strip()
                continue
            if line.startswith("# sent_id = "):
                sentence_id = line.split("=", 1)[1].strip()
                continue
            if not line or line.startswith("#"):
                continue

            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(
                    f"{path}:{line_number}: expected 10 CoNLL-U columns, "
                    f"found {len(columns)}"
                )

            token_id, original_token = columns[0], columns[1]
            if "-" in token_id or "." in token_id:
                continue
            if sentence_id is None:
                raise ValueError(f"{path}:{line_number}: token before sent_id")

            active_document = document_id or infer_document_id(sentence_id)
            yield TokenRecord(
                split=split,
                document_id=active_document,
                genre=infer_genre(active_document),
                sentence_id=sentence_id,
                token_id=token_id,
                original_token=original_token,
                lowercase_input=original_token.lower(),
                casing_label=classify_casing(original_token),
            )


def write_token_file(records: Iterable[TokenRecord], output_path: Path) -> dict:
    """Write records to TSV and return document, sentence, and label counts."""
    documents: set[str] = set()
    sentences: set[str] = set()
    genre_counts: Counter[str] = Counter()
    label_counts: Counter[tuple[str, str]] = Counter()
    token_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=TOKEN_FIELDS, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())
            token_count += 1
            documents.add(record.document_id)
            sentences.add(record.sentence_id)
            genre_counts[record.genre] += 1
            label_counts[(record.genre, record.casing_label)] += 1

    return {
        "tokens": token_count,
        "documents": len(documents),
        "sentences": len(sentences),
        "genre_counts": genre_counts,
        "label_counts": label_counts,
    }


def write_dataset_summary(stats_by_split: dict[str, dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as target:
        fields = ["split", "documents", "sentences", "tokens"]
        writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for split, stats in stats_by_split.items():
            writer.writerow({"split": split, **{key: stats[key] for key in fields[1:]}})


def write_label_counts(stats_by_split: dict[str, dict], output_path: Path) -> None:
    """Write train/dev label counts; test labels stay unanalyzed until final use."""
    labels = [
        "lowercase",
        "initial_uppercase",
        "all_uppercase",
        "mixed_case",
        "uncased",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as target:
        fields = ["split", "genre", "casing_label", "count"]
        writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for split in ("train", "dev"):
            counts = stats_by_split[split]["label_counts"]
            for genre in sorted(EXPECTED_GENRES):
                for label in labels:
                    writer.writerow(
                        {
                            "split": split,
                            "genre": genre,
                            "casing_label": label,
                            "count": counts[(genre, label)],
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/UD_English-EWT"),
        help="Directory containing the three official CoNLL-U files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for generated TSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in FILE_NAMES.values() if not (args.input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {args.input_dir}: {', '.join(sorted(missing))}"
        )

    stats_by_split: dict[str, dict] = {}
    for split, file_name in FILE_NAMES.items():
        input_path = args.input_dir / file_name
        output_path = args.output_dir / f"tokens_{split}.tsv"
        stats_by_split[split] = write_token_file(
            iter_conllu_tokens(input_path, split), output_path
        )

    write_dataset_summary(stats_by_split, args.output_dir / "dataset_summary.tsv")
    write_label_counts(stats_by_split, args.output_dir / "label_counts_train_dev.tsv")

    print("Preprocessing complete")
    for split in ("train", "dev", "test"):
        stats = stats_by_split[split]
        print(
            f"{split:>5}: {stats['documents']:>3} documents, "
            f"{stats['sentences']:>5} sentences, {stats['tokens']:>6} tokens"
        )
    print(f"Output directory: {args.output_dir.resolve()}")
    print("Label distributions were written for train and dev only.")


if __name__ == "__main__":
    main()

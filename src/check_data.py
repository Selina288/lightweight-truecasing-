"""Check local corpus identity without changing inputs, models or scores."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root, processed=False):
    manifest = json.loads((root / 'docs/DATA_PROVENANCE.json').read_text())
    entries = manifest['processed'] if processed else manifest['raw']
    errors = []
    for name, expected in entries.items():
        path = root / name
        if not path.is_file():
            errors.append('Missing: ' + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('SHA-256 mismatch: ' + name)
    if errors:
        raise ValueError('\n'.join(errors))
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', action='store_true', help='Check generated train/dev/test TSVs instead of raw CoNLL-U.')
    args = parser.parse_args()
    try:
        count = verify(ROOT, args.processed)
    except ValueError as error:
        parser.exit(1, str(error) + '\nStop: these files do not identify the recorded experiment inputs.\n')
    print(f'{count} files match the recorded SHA-256 hashes. No files changed.')


if __name__ == '__main__':
    main()

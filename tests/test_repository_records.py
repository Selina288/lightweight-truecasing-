"""Checks of repository records, independent of classifier implementation."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        target = self.root / 'results/error_analysis'
        target.mkdir(parents=True)
        for name in ['TEAM_CONFIRMATION.json', 'paper_examples.tsv', 'dev_reviewed_examples.tsv']:
            shutil.copyfile(ROOT / 'results/error_analysis' / name, target / name)
        self.review = load_module('verify_team_review')

    def test_retained_records_agree(self):
        self.assertEqual(len(self.review.verify_records(self.root)), 19)

    def test_unreported_confirmation_is_rejected(self):
        path = self.root / 'results/error_analysis/dev_reviewed_examples.tsv'
        text = path.read_text()
        self.assertIn('\tNo\n', text)
        path.write_text(text.replace('\tNo\n', '\tYes\n', 1))
        with self.assertRaises(ValueError):
            self.review.verify_records(self.root)

    def test_selected_output_change_is_rejected(self):
        path = self.root / 'results/error_analysis/paper_examples.tsv'
        text = path.read_text()
        path.write_text(text.replace('Ashtamudi', 'ChangedName', 1))
        with self.assertRaises(ValueError):
            self.review.verify_records(self.root)

    def test_missing_and_changed_data_are_rejected(self):
        checker = load_module('check_data')
        (self.root / 'docs').mkdir()
        data = self.root / 'tiny.conllu'
        data.write_bytes(b'original')
        import hashlib
        manifest = {'raw': {'tiny.conllu': hashlib.sha256(data.read_bytes()).hexdigest()}}
        (self.root / 'docs/DATA_PROVENANCE.json').write_text(json.dumps(manifest))
        self.assertEqual(checker.verify(self.root), 1)
        data.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            checker.verify(self.root)
        data.unlink()
        with self.assertRaises(ValueError):
            checker.verify(self.root)


if __name__ == '__main__':
    unittest.main()

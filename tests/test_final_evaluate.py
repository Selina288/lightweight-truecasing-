import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from final_evaluate import verify_manifest,verify_dev
from baselines import write_tsv


class FinalEvaluationTests(unittest.TestCase):
    def test_changed_frozen_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'model';p.write_bytes(b'original')
            manifest={'artifacts':{'model':hashlib.sha256(p.read_bytes()).hexdigest()}}
            verify_manifest(root,manifest)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_manifest(root,manifest)

    def test_dev_replication_rejects_changed_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for directory in ['baselines','ablation','ingenre']:(root/'results'/directory).mkdir(parents=True)
            old=dict(system='all_lowercase',group='ALL',cased_tokens=2,exact_accuracy=0.5,
                macro_f1=0.25,capitalization_required_tokens=1,capitalization_required_accuracy=0)
            write_tsv(root/'results/baselines/dev_summary.tsv',[old])
            # Empty reference tables with headers are sufficient for this isolated baseline check.
            (root/'results/ablation/dev_comparison.tsv').write_text('system\tgroup\n')
            (root/'results/ingenre/dev_three_conditions.tsv').write_text('system\tgroup\n')
            self.assertEqual(verify_dev(root,[old],[]),1)
            changed=dict(old,macro_f1=0.3)
            with self.assertRaises(ValueError):verify_dev(root,[changed],[])


if __name__=='__main__':unittest.main()

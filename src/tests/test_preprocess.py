import tempfile
import unittest
from pathlib import Path

from src.preprocess import classify_casing, infer_genre, iter_conllu_tokens


class CasingTests(unittest.TestCase):
    def test_casing_labels(self):
        examples = {
            "apple": "lowercase",
            "Apple": "initial_uppercase",
            "I": "initial_uppercase",
            "USA": "all_uppercase",
            "iPhone": "mixed_case",
            "123!": "uncased",
        }
        for token, expected in examples.items():
            with self.subTest(token=token):
                self.assertEqual(classify_casing(token), expected)

    def test_genre_from_document_id(self):
        self.assertEqual(infer_genre("weblog-example-001"), "weblog")
        self.assertEqual(infer_genre("answers-123"), "answers")

    def test_minimal_conllu_parse(self):
        sample = """# newdoc id = reviews-example_001
# sent_id = reviews-example_001-0001
# text = I like iPhone.
1\tI\tI\tPRON\tPRP\t_\t2\tnsubj\t2:nsubj\t_
2\tlike\tlike\tVERB\tVBP\t_\t0\troot\t0:root\t_
3\tiPhone\tiPhone\tPROPN\tNNP\t_\t2\tobj\t2:obj\tSpaceAfter=No
4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t2:punct\t_

"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.conllu"
            path.write_text(sample, encoding="utf-8")
            records = list(iter_conllu_tokens(path, "train"))

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].genre, "reviews")
        self.assertEqual(records[0].casing_label, "initial_uppercase")
        self.assertEqual(records[2].lowercase_input, "iphone")
        self.assertEqual(records[2].casing_label, "mixed_case")
        self.assertEqual(records[3].casing_label, "uncased")


if __name__ == "__main__":
    unittest.main()

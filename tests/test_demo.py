import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from demo_server import Demo
from truecase import prepare_text,render_prediction


class DemoTests(unittest.TestCase):
    def test_render_preserves_spacing_and_marks_changes(self):
        text,matches,rows=prepare_text('hi  world.\nhello!')
        result=render_prediction(text,matches,['Hi','world','.','Hello','!'])
        self.assertEqual(result['output'],'Hi  world.\nHello!')
        self.assertEqual(result['changed_tokens'],2)
        self.assertEqual(result['output'],''.join(p['text'] for p in result['segments']))

    def test_baseline_without_model_or_data(self):
        with tempfile.TemporaryDirectory() as folder:
            demo=Demo(folder)
            self.assertEqual(demo.restore('hELLo  WORLD.','all_lowercase')['output'],'hello  world.')
            self.assertEqual(demo.restore('hello. world!','sentence_initial')['output'],'Hello. World!')
            self.assertFalse(demo.available('context_char'))
            self.assertTrue(demo.available('all_lowercase'))

    def test_invalid_inputs(self):
        demo=Demo()
        for text,model in [('', 'all_lowercase'),(' '*3,'context'),('x'*10001,'context'),
                           ('hello','not-a-model'),(None,'context')]:
            with self.subTest(text=str(text)[:12],model=model):
                with self.assertRaises(ValueError):demo.restore(text,model)

    def test_markup_is_returned_as_text(self):
        with tempfile.TemporaryDirectory() as folder:
            result=Demo(folder).restore('<script>alert(1)</script>','all_lowercase')
            self.assertEqual(result['output'],'<script>alert(1)</script>')
            # Browser renders segments via textContent, never innerHTML.
            js=(Path(__file__).resolve().parents[1]/'web/app.js').read_text()
            self.assertIn('node.textContent=part.text',js)


if __name__=='__main__':unittest.main()

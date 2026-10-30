"""Demonstrate the saved model on a lowercase sentence.

python3 src/truecase.py "i visited new york on monday."
The demo uses a simple tokenizer, not the gold UD tokenization used in scoring.
"""
import argparse
import re
from pathlib import Path
import joblib
from logreg import predict_rows


def restore_text(bundle, text):
    text = text.lower()
    matches = list(re.finditer(r"\w+(?:['’]\w+)*|[^\w\s]",text))
    rows = []
    sentence = 0
    for match in matches:
        word = match.group()
        rows.append(dict(document_id='demo',sentence_id=str(sentence),lowercase_input=word))
        if word in '.!?':
            sentence += 1
    outputs,_ = predict_rows(bundle,rows)
    result = []
    end = 0
    for match,output in zip(matches,outputs):
        result.extend((text[end:match.start()],output))
        end = match.end()
    result.append(text[end:])
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    parser.add_argument('--model',type=Path,default=Path('models/logreg.joblib'))
    args = parser.parse_args()
    print(restore_text(joblib.load(args.model),args.text))


if __name__=='__main__':
    main()

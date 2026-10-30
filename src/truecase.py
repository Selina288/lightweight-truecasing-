"""Command-line demo shared with the local front end.
Uses simple tokenization, unlike gold UD tokenization in official scoring.
"""
import argparse
import re
from pathlib import Path
import joblib
from logreg import predict_rows


def prepare_text(text):
    text=text.lower()
    matches=list(re.finditer(r"\w+(?:['’]\w+)*|[^\w\s]",text))
    rows=[]
    sentence=0
    for match in matches:
        word=match.group()
        rows.append(dict(document_id='demo',sentence_id=str(sentence),lowercase_input=word))
        if word in '.!?':sentence+=1
    return text,matches,rows


def render_prediction(text,matches,outputs):
    if len(matches)!=len(outputs):raise ValueError('Prediction alignment mismatch')
    parts=[]
    end=0
    for match,output in zip(matches,outputs):
        if match.start()>end:parts.append(dict(text=text[end:match.start()],changed=False))
        parts.append(dict(text=output,changed=output!=match.group()))
        end=match.end()
    if end<len(text):parts.append(dict(text=text[end:],changed=False))
    return dict(normalized_input=text,output=''.join(p['text'] for p in parts),segments=parts,
                changed_tokens=sum(p['changed'] for p in parts),tokens=len(matches))


def restore_text(bundle,text):
    text,matches,rows=prepare_text(text)
    outputs,_=predict_rows(bundle,rows)
    return render_prediction(text,matches,outputs)['output']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    parser.add_argument('--model',type=Path,default=Path('models/logreg.joblib'))
    args=parser.parse_args()
    print(restore_text(joblib.load(args.model),args.text))


if __name__=='__main__':main()

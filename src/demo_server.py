"""Local real-model web demo. Start: python src/demo_server.py
Open http://127.0.0.1:8765 and stop with Ctrl+C. No training or test evaluation.
"""
import argparse
import csv
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import joblib
from baselines import fit_lexicon,predict,read_tokens
from logreg import is_first,predict_rows
from truecase import prepare_text,render_prediction

ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'context_char':('Context + characters (C)','models/ablation/context_char.joblib'),
 'context':('Context model (B)','models/ablation/context.joblib'),
 'most_frequent_case':('Most frequent training form',None),
 'sentence_initial':('Sentence-initial baseline',None),
 'all_lowercase':('All-lowercase baseline',None),
}


class Demo:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.cache={};self.lexicon=None;self.lock=threading.Lock()

    def available(self,model):
        _,path=MODELS[model]
        if path:return (self.root/path).is_file()
        if model=='most_frequent_case':return (self.root/'data/processed/tokens_train.tsv').is_file()
        return True

    def model_list(self):
        scores={}
        for path in ['results/baselines/dev_summary.tsv','results/ablation/dev_comparison.tsv']:
            file=self.root/path
            if file.is_file():
                with file.open() as h:
                    for row in csv.DictReader(h,delimiter='\t'):
                        if row['group']=='ALL':scores[row['system']]=row
        result=[]
        for name,(label,path) in MODELS.items():
            score=scores.get(name,{})
            result.append(dict(id=name,label=label,available=self.available(name),
                dev_macro_f1=score.get('macro_f1'),dev_accuracy=score.get('exact_accuracy')))
        return result

    def restore(self,text,model):
        if not isinstance(text,str) or not text.strip():raise ValueError('Enter some text first.')
        if len(text)>10000:raise ValueError('Please use 10,000 characters or fewer.')
        if not isinstance(model,str) or model not in MODELS:raise ValueError('Unknown model selection.')
        if not self.available(model):raise ValueError('This model is not available. See DEMO_GUIDE_zh.md to generate the required files.')
        normalized,matches,rows=prepare_text(text)
        path=MODELS[model][1]
        with self.lock:
            if path:
                if model not in self.cache:self.cache[model]=joblib.load(self.root/path)
                outputs,_=predict_rows(self.cache[model],rows)
            else:
                if model=='most_frequent_case' and self.lexicon is None:
                    self.lexicon=fit_lexicon(read_tokens(self.root/'data/processed/tokens_train.tsv'))
                outputs=[predict(model,r['lowercase_input'],is_first(rows,i),self.lexicon or {})
                         for i,r in enumerate(rows)]
        result=render_prediction(normalized,matches,outputs)
        result['model']=model;result['model_label']=MODELS[model][0]
        return result


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,body,content_type='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/api/models':return self.reply(200,dict(models=demo.model_list()))
            assets={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path not in assets:return self.reply(404,dict(error='Not found'))
            file=demo.root/'web'/assets[path]
            types={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8'}
            self.reply(200,file.read_bytes(),types[file.suffix])

        def do_POST(self):
            if self.path!='/api/restore':return self.reply(404,dict(error='Not found'))
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                return self.reply(415,dict(error='Use JSON input.'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=128000:raise ValueError('Request size is invalid.')
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('Use a JSON object.')
                result=demo.restore(data.get('text'),data.get('model'))
                self.reply(200,result)
            except (ValueError,TypeError,UnicodeDecodeError) as error:
                self.reply(400,dict(error=str(error)))
            except Exception:
                self.reply(500,dict(error='Could not load or run the model. Check the terminal and your installed dependency versions.'))
                import traceback;traceback.print_exc()
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(Demo()))
    print(f'Truecasing demo: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()

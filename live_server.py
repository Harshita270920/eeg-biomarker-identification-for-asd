"""Real sequential nested refitting from cached EEG features. No saved prediction lookup."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
import numpy as np
from pipeline import tune_predict,tune_tree,simplex_weights,StratifiedKFold
ROOT=Path(__file__).resolve().parent
LOCK=threading.Lock()
def run(number,emit):
    started=time.monotonic();emit({'stage':'load','message':'Loading cached subject-level EEG features. Raw preprocessing was completed earlier.'})
    with np.load(ROOT/'features.npz') as z:
        ids=z['subject_ids'];y=z['y'];banks={'covariance_svm':z['covariance'],'spectral_svm':z['spectral_dynamics'],'covariance_tree':z['covariance']}
    sid=json.loads((ROOT/'participants.json').read_text())[number-1];test=np.flatnonzero(ids==sid);train=np.flatnonzero(ids!=sid)
    assert len(test)==1 and len(train)==55
    emit({'stage':'split','message':'Held out the selected participant. Training uses the other 55 only.','covariance_features':1050,'spectral_features':240,'train_count':55})
    names=list(banks);oof=np.full((55,3),np.nan);completed=0
    def fit(name,tr,te):
        return (tune_tree if name=='covariance_tree' else tune_predict)(ids,y,banks[name],tr,te)
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y[train]),1):
        assert not set(train[a])&set(train[b]) and test[0] not in train
        for j,name in enumerate(names):
            emit({'stage':'crossfit','message':f'Fold {fold}/5: tuning and fitting {name.replace("_"," ")} on training participants.','completed':completed,'total':18})
            values,audit=fit(name,train[a],train[b]);oof[b,j]=values;completed+=1
            emit({'stage':'fitted','message':f'Fold {fold}/5: {name.replace("_"," ")} finished.','completed':completed,'total':18,'chosen':audit['chosen']})
    weights=simplex_weights(oof,y[train]);emit({'stage':'weights','message':'Learned ensemble weights from the 55 training participants’ cross-fitted scores.','weights':weights.tolist()})
    scores=[];settings=[]
    for name in names:
        emit({'stage':'refit','message':f'Refitting {name.replace("_"," ")} on all 55, then predicting the held-out participant.','completed':completed,'total':18})
        values,audit=fit(name,train,test);scores.append(float(values[0]));settings.append(audit['chosen']);completed+=1
        emit({'stage':'predicted','message':f'{name.replace("_"," ")} produced a new held-out score.','score':scores[-1],'completed':completed,'total':18})
    score=float(np.array(scores)@weights);prediction=int(score>=.5)
    result=dict(stage='result',message='Live nested model fitting completed.',participant=number,score=score,prediction=prediction,recorded_label=int(y[test[0]]),correct=prediction==int(y[test[0]]),seconds=round(time.monotonic()-started,2),models=[dict(name=n,score=scores[j],weight=float(weights[j]),contribution=float(scores[j]*weights[j]),chosen=settings[j]) for j,n in enumerate(names)],threshold=.5)
    emit(result);return result
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/health':body=b'{"status":"ok"}';kind='application/json'
        elif path=='/':body=(ROOT/'live.html').read_bytes();kind='text/html; charset=utf-8'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        path=urlsplit(self.path).path
        try:
            if not path.startswith('/api/run/'):raise ValueError()
            number=int(path.rsplit('/',1)[1]);assert 1<=number<=56
        except (ValueError,AssertionError):self.send_error(400);return
        if not LOCK.acquire(blocking=False):self.send_error(409,'Another participant is running. Try again after it finishes.');return
        try:
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson');self.send_header('Cache-Control','no-cache');self.send_header('X-Accel-Buffering','no');self.end_headers()
            def emit(item):
                self.wfile.write((json.dumps(item)+'\n').encode());self.wfile.flush();print(json.dumps(item),flush=True)
            try:run(number,emit)
            except (BrokenPipeError,ConnectionResetError):return
            except Exception as e:emit({'stage':'error','message':str(e)})
        finally:LOCK.release()
if __name__=='__main__':
    port=int(os.getenv('PORT','8002'));print(f'Live demo on port {port}',flush=True)
    ThreadingHTTPServer((os.getenv('HOST','127.0.0.1'),port),Handler).serve_forever()

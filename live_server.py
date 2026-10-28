"""Real sequential nested refitting from cached EEG features. No saved prediction lookup."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,threading,secrets
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
import numpy as np
from pipeline import tune_predict,tune_tree,simplex_weights,StratifiedKFold
ROOT=Path(__file__).resolve().parent
LOCK=threading.Lock()
JOBS={}
JOBS_LOCK=threading.Lock()
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
    def reply(self,code,data,kind='application/json'):
        body=data if isinstance(data,bytes) else json.dumps(data).encode()
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/health':self.reply(200,{'status':'ok'})
        elif path=='/':self.reply(200,(ROOT/'live.html').read_bytes(),'text/html; charset=utf-8')
        elif path.startswith('/api/jobs/'):
            with JOBS_LOCK:
                job=JOBS.get(path.rsplit('/',1)[1]);data=None if job is None else dict(events=list(job['events']),done=job['done'])
            self.reply(404 if data is None else 200,data or {'error':'Run expired or server restarted. Start a new run.'})
        else:self.reply(404,{'error':'Not found'})
    def do_POST(self):
        path=urlsplit(self.path).path
        try:
            if not path.startswith('/api/run/'):raise ValueError()
            number=int(path.rsplit('/',1)[1]);assert 1<=number<=56
        except (ValueError,AssertionError):self.reply(400,{'error':'Invalid participant'});return
        if not LOCK.acquire(blocking=False):self.reply(409,{'error':'Another participant is running.'});return
        token=secrets.token_urlsafe(24)
        with JOBS_LOCK:
            for key in list(JOBS):
                if JOBS[key]['done'] and time.monotonic()-JOBS[key]['created']>3600:del JOBS[key]
            JOBS[token]={'events':[],'done':False,'created':time.monotonic()}
        def work():
            def emit(item):
                with JOBS_LOCK:JOBS[token]['events'].append(item)
                print(json.dumps(item),flush=True)
            try:run(number,emit)
            except Exception as e:emit({'stage':'error','message':str(e)})
            finally:
                with JOBS_LOCK:JOBS[token]['done']=True
                LOCK.release()
        threading.Thread(target=work,daemon=True).start()
        self.reply(202,{'job':token})
if __name__=='__main__':
    port=int(os.getenv('PORT','8002'));print(f'Live demo on port {port}',flush=True)
    ThreadingHTTPServer((os.getenv('HOST','127.0.0.1'),port),Handler).serve_forever()

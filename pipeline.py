import numpy as np
from scipy.special import expit
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

class TrainOnlySelect(BaseEstimator,TransformerMixin):
    """Deterministic finite-score ANOVA selector; train-only by pipeline design."""
    def __init__(self,k=10):self.k=k
    def fit(self,X,y):
        if self.k=='all':self.indices_=np.arange(X.shape[1])
        else:
            with np.errstate(divide='ignore',invalid='ignore'):
                scores=f_classif(X,y)[0]
            scores=np.nan_to_num(scores,nan=-np.inf,posinf=np.finfo(float).max)
            self.indices_=np.argsort(-scores,kind='stable')[:min(self.k,X.shape[1])]
        return self
    def transform(self,X):return X[:,self.indices_]

def candidates(bank_names):
    return [dict(bank=bank,k=k,kernel=kernel,C=c)
            for bank in bank_names for k in (10,30,'all')
            for kernel in ('linear','rbf') for c in (.1,1.,10.)]

def estimator(params):
    if params.get('model')=='shrinkage_lda':
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                             TrainOnlySelect(params['k']),StandardScaler(),
                             PCA(n_components=.95,svd_solver='full'),
                             LinearDiscriminantAnalysis(solver='lsqr',shrinkage=params['shrinkage'],priors=[.5,.5]))
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                         TrainOnlySelect(params['k']),StandardScaler(),
                         SVC(C=params['C'],kernel=params['kernel'],gamma='scale',class_weight='balanced'))

def model(p):
    if p['model']=='ridge':return RidgeClassifier(alpha=p['alpha'],class_weight='balanced')
    if p['model']=='extra_trees':return ExtraTreesClassifier(n_estimators=100,max_depth=p['depth'],min_samples_leaf=4,max_features=.5,class_weight='balanced',random_state=42,n_jobs=1)
    if p['model']=='boosting':return HistGradientBoostingClassifier(max_leaf_nodes=p['leaves'],min_samples_leaf=5,learning_rate=.05,l2_regularization=10.,max_iter=75,early_stopping=False,class_weight='balanced',random_state=42)
    raise ValueError('Unknown model')

def prepare(x,y,train,test,k):
    pipeline=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),TrainOnlySelect(k),StandardScaler())
    return pipeline.fit_transform(x[train],y[train]),pipeline.transform(x[test])

def tune_predict(ids,y,x,train,test,grid=None):
    """All tuning and margin normalization confined to train indices."""
    grid=candidates(['base']) if grid is None else grid
    splits=list(StratifiedKFold(3,shuffle=True,random_state=42).split(train,y[train]));scores=[];audit=[]
    for a,b in splits:audit.append(dict(train_ids=ids[train[a]].tolist(),validation_ids=ids[train[b]].tolist()))
    for p in grid:
        score=0.
        for a,b in splits:
            fitted=estimator(p).fit(x[train[a]],y[train[a]])
            score+=balanced_accuracy_score(y[train[b]],fitted.predict(x[train[b]]))/3
        scores.append(score)
    params=grid[int(np.argmax(scores))];fitted=estimator(params).fit(x[train],y[train])
    margin=fitted.decision_function(x[train]);scale=max(float(np.sqrt(np.mean(margin*margin))),1e-12)
    prediction=expit(fitted.decision_function(x[test])/scale)
    return prediction,dict(train_ids=ids[train].tolist(),evaluation_ids=ids[test].tolist(),tuning=audit,chosen=params,training_margin_rms=scale)

def simplex_weights(z,y,penalty=.01):
    z=np.asarray(z,float);y=np.asarray(y,int);n=z.shape[1]
    if set(y)!={0,1} or not np.isfinite(z).all() or penalty<=0:raise ValueError('Invalid score data')
    sample=np.array([.5/np.sum(y==v) for v in y]);u=np.ones(n)/n
    a=z.T@(sample[:,None]*z)+penalty*np.eye(n);b=z.T@(sample*y)+penalty*u
    best=None;best_loss=np.inf
    for k in range(1,n+1):
        for support in itertools.combinations(range(n),k):
            ix=np.array(support);matrix=np.block([[a[np.ix_(ix,ix)],np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
            solution=np.linalg.solve(matrix,np.r_[b[ix],1.])[:k]
            if np.any(solution < -1e-10):continue
            w=np.zeros(n);w[ix]=np.maximum(solution,0);w/=w.sum()
            loss=np.sum(sample*(z@w-y)**2)+penalty*np.sum((w-u)**2)
            if loss<best_loss:best=w;best_loss=loss
    if best is None:raise RuntimeError('No feasible simplex solution')
    return best

def tune_tree(ids,y,x,train,test,grid=None):
    grid=[dict(model='extra_trees',k=k,depth=d) for k in [10,30] for d in [3,None]] if grid is None else grid
    scores=np.zeros(len(grid));audit=[]
    for a,b in StratifiedKFold(3,shuffle=True,random_state=42).split(train,y[train]):
        ta,tb=train[a],train[b];cache={k:prepare(x,y,ta,tb,k) for k in set(p['k'] for p in grid)}
        audit.append(dict(train_ids=ids[ta].tolist(),validation_ids=ids[tb].tolist()))
        for j,p in enumerate(grid):
            xx,v=cache[p['k']];scores[j]+=balanced_accuracy_score(y[tb],model(p).fit(xx,y[ta]).predict(v))/3
    p=grid[int(np.argmax(scores))];xx,v=prepare(x,y,train,test,p['k']);fitted=model(p).fit(xx,y[train])
    return fitted.predict_proba(v)[:,1],dict(train_ids=ids[train].tolist(),evaluation_ids=ids[test].tolist(),tuning=audit,chosen=p)

import json,gc,hashlib
import numpy as np,pandas as pd,pyreadr,joblib
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from v2_numeric import ROOT,merged,evaluate,route
OUT=ROOT/'experiments/v4'
def dump(n,x):(OUT/'results'/n).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def extract(part):
    dest=OUT/'data'/f'tep_matched_{part}.npz'
    if dest.exists():return dict(np.load(dest))
    pre=joblib.load(ROOT/'models/tep_preprocessing.joblib');cols=pre['columns'];sc=pre['scaler'];rows=[];ids=[]
    for kind in ['FaultFree','Faulty']:
        print('matched extract',part,kind,flush=True);df=next(iter(pyreadr.read_r(str(ROOT/'data/TEP'/f'TEP_{kind}_{part}.RData')).values()));cut=20 if part=='Training' else 160
        df=df.loc[(df['sample']>cut-20)&(df['sample']<=cut+480)].sort_values(['faultNumber','simulationRun','sample'])
        for (f,r),g in df.groupby(['faultNumber','simulationRun'],sort=True):
            allz=(g[cols].to_numpy()-sc.mean_)/sc.scale_;before=allz[:20];z=allz[20:];assert len(z)==480
            m=z.mean(0);a=abs(m);extra=np.r_[m.mean(),m.std(),m.min(),m.max(),np.median(m),np.quantile(m,.25),np.quantile(m,.75),a.mean(),a.std(),a.max(),np.median(a),np.mean(a>1),np.mean(a>2),np.mean(a>3)];std=z.std(0);center=z-m
            ac=[np.mean(center[:-lag]*center[lag:],0)/np.maximum(std**2,1e-8) for lag in [5,20,60]]
            row=np.r_[m,a,extra,std,np.quantile(z,.1,axis=0),np.quantile(z,.9,axis=0),z[240:].mean(0)-z[:240].mean(0),np.sqrt(np.mean(np.diff(z,axis=0)**2,axis=0)),*ac,z.reshape(12,40,52).mean(1).std(0)/np.maximum(std,1e-4),m-before.mean(0),np.log1p(std/np.maximum(before.std(0),.05))]
            rows.append(row);ids.append((int(f),int(r)))
        del df;gc.collect()
    ids=np.array(ids);np.savez_compressed(dest,x=np.asarray(rows),fault=ids[:,0],run=ids[:,1]);return dict(np.load(dest))
def align(a,b):
    ix={(int(f),int(r)):i for i,(f,r) in enumerate(zip(b['fault'],b['run']))};return np.array([b['x'][ix[(int(f),int(r))]] for f,r in zip(a['fault'],a['run'])])
def search(y,s,fb_limit,nf_limit,veto=None):
    grid=np.r_[-.01,np.linspace(0,1,201),1.01];y=np.asarray(y);ok=np.ones(len(s),bool) if veto is None else veto
    nfault=sum(y==1);nnormal=sum(y==0);uf=np.r_[beta.ppf(.975,np.arange(nfault)+1,nfault-np.arange(nfault)),1];un=np.r_[beta.ppf(.975,np.arange(nnormal)+1,nnormal-np.arange(nnormal)),1]
    benign=(s[None,:]<=grid[:,None])&ok;fault=s[None,:]>=grid[:,None];fb=(benign&(y==1)).sum(1);nf=(fault&(y==0)).sum(1);coverage=benign.sum(1)[:,None]+fault.sum(1)[None,:]
    feasible=(grid[:,None]<grid[None,:])&(uf[fb,None]<=fb_limit)&(un[nf][None,:]<=nf_limit);ij=np.argwhere(feasible)
    if not len(ij):return None
    i,j=min(ij,key=lambda q:(-coverage[q[0],q[1]],uf[fb[q[0]]],grid[q[0]],grid[q[1]]));return {'lo':float(grid[i]),'hi':float(grid[j]),'coverage':float(coverage[i,j]/len(y)),'FB_upper':float(uf[fb[i]]),'NF_upper':float(un[nf[j]])}
def main():
    if (OUT/'results/tep_complete.json').exists():print('Completed TEP v4 preserved');return
    ta,tx=merged('Training');ea,ex=merged('Testing');tm=align(ta,extract('Training'));em=align(ea,extract('Testing'));y=(ta['fault']>0).astype(int);scores={};hashes={}
    for w in [1,5,20]:
        name=f'v2_dynamic_w{w}';p=ROOT/'experiments/v2/models'/f'dynamic_hist_w{w}.joblib';scores[name]=joblib.load(p).predict_proba(ex)[:,1];hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    configs=[('existing_extra',tx,ex,ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,max_features=.7,class_weight='balanced',n_jobs=8,random_state=20261112)),('matched_extra',tm,em,ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,max_features=.7,class_weight='balanced',n_jobs=8,random_state=20261112)),('matched_hist_w20',tm,em,HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,l2_regularization=3,class_weight={0:20,1:1},random_state=20261112)),('matched_hist_w5',tm,em,HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,l2_regularization=3,class_weight={0:5,1:1},random_state=20261112))]
    for name,a,b,model in configs:
        path=OUT/'models'/f'{name}.joblib'
        if path.exists():model=joblib.load(path)
        else:print('fit',name,a.shape,flush=True);model.fit(a,y);joblib.dump(model,path)
        scores[name]=model.predict_proba(b)[:,1];hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    ids=np.isin(ta['fault'],[0,9]);path=OUT/'models/fault9_specialist.joblib'
    if path.exists():special=joblib.load(path)
    else:special=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=3,max_features=.7,n_jobs=8,random_state=20261112).fit(tm[ids],y[ids]);joblib.dump(special,path)
    specialscore=special.predict_proba(em)[:,1];np.savez_compressed(OUT/'data/tep_scores.npz',**scores,specialist=specialscore,fault=ea['fault'],run=ea['run']);dump('tep_model_hashes.json',hashes)
    splits=json.loads((ROOT/'experiments/v3/results/risk_splits.json').read_text());prior=pd.read_csv(ROOT/'experiments/v3/results/risk_predictions.csv');lookup={(int(f),int(r)):i for i,(f,r) in enumerate(zip(ea['fault'],ea['run']))};results=[];pred=[];choices=[]
    for split in splits:
        seed=split['seed'];fold=split['fold'];rng=np.random.default_rng(seed+fold+20261112);groups=split['calibration'];ci=list(np.flatnonzero((ea['fault']==0)&np.isin(ea['run'],groups)))
        for f in range(1,21):ci.extend(rng.choice(np.flatnonzero((ea['fault']==f)&np.isin(ea['run'],groups)),20,replace=False))
        ci=np.array(ci);test=prior[(prior.seed==seed)&(prior.fold==fold)];ti=np.array([lookup[(int(r.fault),int(r.run))] for r in test.itertuples()]);cy=(ea['fault'][ci]>0).astype(int);truth=(ea['fault'][ti]>0).astype(int)
        # The same evaluation cases as v3, and fixed fresh within-calibration samples for all candidates.
        for strategy,limits in [('v2_fixed_margin',(.025,.05)),('family_nominal',(.05,.1)),('family_margin',(.025,.05)),('specialist_margin',(.025,.05))]:
            names=['v2_dynamic_w20'] if strategy=='v2_fixed_margin' else list(scores);candidates=[]
            for name in names:
                for cap in ([None,.15,.25,.4] if strategy=='specialist_margin' else [None]):
                    q=search(cy,scores[name][ci],*limits,None if cap is None else specialscore[ci]<=cap)
                    if q:candidates.append(dict(q,model=name,specialist_cap=cap))
            best=sorted(candidates,key=lambda q:(-q['coverage'],q['FB_upper'],names.index(q['model']),q['specialist_cap'] or 1))[0];s=scores[best['model']][ti];veto=None if best['specialist_cap'] is None else specialscore[ti]<=best['specialist_cap'];rr=route(s,best['lo'],best['hi'],veto);m=evaluate(truth,rr);results.append(dict(seed=seed,fold=fold,strategy=strategy,policy=best,**m));choices.append({'seed':seed,'fold':fold,'strategy':strategy,'calibration_indices':ci.tolist(),'candidates':candidates});pred.extend({'seed':seed,'fold':fold,'strategy':strategy,'fault':int(ea['fault'][i]),'run':int(ea['run'][i]),'route':int(r)} for i,r in zip(ti,rr))
            print('eval',seed,fold,strategy,m['coverage'],m['pass_risk'],flush=True)
    dump('tep_results.json',results);dump('tep_selection.json',choices);pd.DataFrame(pred).to_csv(OUT/'results/tep_predictions.csv',index=False)
    diagnostic={name:float(roc_auc_score((ea['fault'][np.isin(ea['fault'],[0,9])]>0).astype(int),s[np.isin(ea['fault'],[0,9])])) for name,s in dict(scores,specialist=specialscore).items()};dump('fault9_AUROC.json',diagnostic);dump('tep_complete.json',{'complete':True,'rows':len(results),'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()})
if __name__=='__main__':main()

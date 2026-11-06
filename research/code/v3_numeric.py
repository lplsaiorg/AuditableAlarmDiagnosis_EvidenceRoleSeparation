import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score
from v2_numeric import ROOT, merged, evaluate, route, search
OUT=ROOT/'experiments/v3'
def dump(n,x):
    (OUT/'results'/n).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def risk():
    a,x=merged('Testing'); f=a['fault'].astype(int); run=a['run'].astype(int); y=(f>0).astype(int)
    names=[f'{kind}_hist_w{w}' for kind in ['mean','dynamic'] for w in [1,5,20]]+[f'{kind}_logistic_balanced' for kind in ['mean','dynamic']]
    scores={n:joblib.load(ROOT/'experiments/v2/models'/f'{n}.joblib').predict_proba(x[:,:118 if n.startswith('mean') else 378])[:,1] for n in names}
    def sample(groups,rng):
        ids=list(np.flatnonzero((f==0)&np.isin(run,groups)))
        for fault in range(1,21):
            candidates=np.flatnonzero((f==fault)&np.isin(run,groups));ids.extend(rng.choice(candidates,20,replace=False))
        return np.array(ids)
    results=[];pred=[];bindings=[]
    for seed in [20261107,20261108,20261109]:
        rng=np.random.default_rng(seed); folds=np.array_split(rng.permutation(np.unique(run)),5)
        for fold,groups in enumerate(folds):
            calibration=rng.choice(np.setdiff1d(np.unique(run),groups),100,replace=False);ci=sample(calibration,rng);ti=sample(groups,rng)
            assert not set(run[ci])&set(run[ti]);candidates=[]
            for n in names:
                z=search(y[ci],scores[n][ci],np.ones(len(ci),bool))
                if z:candidates.append(dict(z[0],model=n))
            best=sorted(candidates,key=lambda q:(-q['coverage'],q['FB_upper'],names.index(q['model']),q['lo'],q['hi']))[0]
            r=route(scores[best['model']][ti],best['lo'],best['hi']);m=evaluate(y[ti],r)
            results.append(dict(seed=seed,fold=fold,policy=best,**m));bindings.append(dict(seed=seed,fold=fold,calibration=calibration.tolist(),evaluation=groups.tolist()))
            pred.extend(dict(seed=seed,fold=fold,fault=int(f[i]),run=int(run[i]),route=int(v)) for i,v in zip(ti,r))
            print('risk',seed,fold,m['coverage'],m['pass_risk'],flush=True)
    dump('risk_splits.json',bindings);dump('risk_results.json',results);pd.DataFrame(pred).to_csv(OUT/'results/risk_predictions.csv',index=False)
    frame=pd.DataFrame(pred);per=frame.groupby('fault').route.value_counts().unstack(fill_value=0)
    dump('risk_summary.json',{'evaluations':len(results),'passes':sum(r['pass_risk'] for r in results),'coverage_range':[min(r['coverage'] for r in results),max(r['coverage'] for r in results)],'coverage_mean':np.mean([r['coverage'] for r in results]),'FB_upper_range':[min(r['FB_upper'] for r in results),max(r['FB_upper'] for r in results)],'per_fault_repeated_counts':per.to_dict('index'),'warning':'Overlapping repeated evaluations; counts are not independent events; no binomial CI for pass fraction'})
    s=scores['dynamic_hist_w20'];ids=np.isin(f,[0,9]);dump('fault9_diagnostic.json',{'fault9_vs_normal_AUROC':roc_auc_score(y[ids],s[ids]),'score_quantiles':{str(k):np.quantile(s[f==k],[0,.1,.25,.5,.75,.9,1]).tolist() for k in [0,9,15]},'scope':'descriptive all existing Testing runs; no new tuning or independent confirmation'})
def hai():
    frames=[]
    for name in ['train2','train3','test1','test2','test3','test4','test5']:
        a=np.load(ROOT/'data'/f'hai_{name}_scores.npz');s=a['scores'];y=a['truth'];rows=[]
        for start in range(0,len(y)-119,120):
            end=start+120;rows.append([start,end,int(y[start:end].any()),*np.log1p(np.maximum(s[start:end].max(0),0))])
        d=pd.DataFrame(rows,columns=['start','end','truth','pca','iforest','deviation']);d['source']=name;frames.append(d)
    df=pd.concat(frames,ignore_index=True);cols=['pca','iforest','deviation'];ref=df[df.source=='train2'][cols].to_numpy();med=np.median(ref,0);scale=np.maximum(np.quantile(ref,.75,axis=0)-np.quantile(ref,.25,axis=0),1e-6)
    df['score']=np.median((df[cols].to_numpy()-med)/scale,axis=1);dev=df[df.source.isin(['train3','test1'])];s=dev.score.to_numpy();y=dev.truth.to_numpy();grid=np.unique(np.r_[-np.inf,np.quantile(s,np.linspace(0,1,201)),np.inf]);best=None
    for lo in grid:
        fb=np.mean(s[y==1]<=lo)
        if fb>.05:continue
        for hi in grid[grid>lo]:
            nf=np.mean(s[y==0]>=hi)
            if nf>.1:continue
            q={'lo':float(lo),'hi':float(hi),'coverage':float(np.mean((s<=lo)|(s>=hi))),'FB_rate':float(fb),'NF_rate':float(nf)}
            if best is None or (-q['coverage'],fb,lo,hi)<(-best['coverage'],best['FB_rate'],best['lo'],best['hi']):best=q
    dump('hai_frozen_policy.json',dict(best,normalization_median=med.tolist(),normalization_iqr=scale.tolist(),risk_certificate=False))
    df['route']=route(df.score.to_numpy(),best['lo'],best['hi']);df.to_csv(OUT/'results/hai_windows.csv',index=False);results={}
    for key in ['development','evaluation','test2','test3','test4','test5']:
        g=df[df.source.isin(['train3','test1'] if key=='development' else ['test2','test3','test4','test5'] if key=='evaluation' else [key])];m=evaluate(g.truth,g.route)
        # Correlated windows cannot justify binomial bounds; remove nominal certificate.
        for k in ['FB_upper','NF_upper','pass_risk']:m.pop(k)
        m['AUROC']=roc_auc_score(g.truth,g.score) if g.truth.nunique()==2 else None;results[key]=m
    episodes=[]
    for name in ['test2','test3','test4','test5']:
        y=np.load(ROOT/'data'/f'hai_{name}_scores.npz')['truth'];ids=np.flatnonzero(y);groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1);g=df[df.source==name]
        for ep in groups:
            if not len(ep):continue
            overlap=g[(g.start<=ep[-1])&(g.end>ep[0])];episodes.append({'source':name,'start':int(ep[0]),'end':int(ep[-1])+1,'detected':bool((overlap.route==1).any()),'all_abstained':bool(len(overlap) and (overlap.route==-1).all())})
    dump('hai_results.json',{'metrics':results,'episodes':episodes,'episode_detected':sum(e['detected'] for e in episodes),'episode_total':len(episodes),'limitation':'Retrospective temporal transfer of detector routing only; correlated windows, no risk certification, previous dataset exposure; window-end offline detection not real-time latency'})
    print('HAI',results['evaluation'],flush=True)
if __name__=='__main__':
    risk();hai()

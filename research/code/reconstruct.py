import os,json,hashlib,time,gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadr,joblib
from scipy.stats import beta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier,IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
SEED=20261105
def save(name,obj):
    (ROOT/'results'/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def upper(k,n):return float(beta.ppf(.975,k+1,n-k)) if k<n else 1.
def metrics(y,s,lo,hi):
    y=np.asarray(y);s=np.asarray(s);r=np.where(s<=lo,0,np.where(s>=hi,1,-1))
    f=y==1;n=y==0;fb=int(sum(f&(r==0)));nf=int(sum(n&(r==1)))
    return dict(n=len(y),n_fault=int(sum(f)),n_normal=int(sum(n)),fault_recognized=int(sum(f&(r==1))),normal_recognized=int(sum(n&(r==0))),FB=fb,NF=nf,abstain=int(sum(r==-1)),coverage=float(np.mean(r!=-1)),FB_upper=upper(fb,sum(f)),NF_upper=upper(nf,sum(n)),AUROC=float(roc_auc_score(y,s)),AUPRC=float(average_precision_score(y,s)))
def features(frame,cols,scaler,cut):
    x=frame.loc[frame['sample']>cut]
    grouped=x.groupby(['faultNumber','simulationRun'],sort=True)[cols].mean()
    z=scaler.transform(grouped)
    a=np.abs(z)
    extra=np.column_stack([z.mean(1),z.std(1),z.min(1),z.max(1),np.median(z,axis=1),np.quantile(z,.25,axis=1),np.quantile(z,.75,axis=1),a.mean(1),a.std(1),a.max(1),np.median(a,axis=1),np.mean(a>1,axis=1),np.mean(a>2,axis=1),np.mean(a>3,axis=1)])
    return grouped.index.to_frame(index=False),np.column_stack([z,a,extra])
def tep():
    if (ROOT/'results/tep_frozen_policy.json').exists():
        raise RuntimeError('A frozen policy already exists. Preserve this confirmation; use a new experiment directory for another fit.')
    print('TEP loading normal Training',flush=True)
    folder=ROOT/'data/TEP'
    normal=next(iter(pyreadr.read_r(str(folder/'TEP_FaultFree_Training.RData')).values()))
    cols=[c for c in normal if c.startswith(('xmeas_','xmv_'))]
    scaler=StandardScaler().fit(normal[cols])
    pca=PCA(n_components=.95,svd_solver='full').fit(scaler.transform(normal[cols]))
    z=scaler.transform(normal[cols]);err=np.mean((z-pca.inverse_transform(pca.transform(z)))**2,axis=1)
    save('tep_pca_calibration.json',dict(components=pca.n_components_,retained_variance=float(sum(pca.explained_variance_ratio_)),p99=float(np.quantile(err,.99)),rows=len(z)))
    ni,nx=features(normal,cols,scaler,20); del normal,z;gc.collect()
    arrays=[nx];ids=[ni]
    for part in ['Training','Testing']:
        for kind in ['Faulty','FaultFree']:
            if part=='Training' and kind=='FaultFree':continue
            path=folder/f'TEP_{kind}_{part}.RData'
            print('TEP extracting',path.name,flush=True)
            frame=next(iter(pyreadr.read_r(str(path)).values()))
            i,x=features(frame,cols,scaler,20 if part=='Training' else 160)
            del frame;gc.collect()
            if part=='Training':ids.append(i);arrays.append(x)
            else:
                np.savez_compressed(ROOT/'data'/f'tep_{kind}_testing_features.npz',x=x,fault=i.faultNumber.to_numpy(),run=i.simulationRun.to_numpy())
    tx=np.vstack(arrays);ti=pd.concat(ids,ignore_index=True);ty=(ti.faultNumber>0).astype(int).to_numpy()
    np.savez_compressed(ROOT/'data/tep_training_features.npz',x=tx,fault=ti.faultNumber.to_numpy(),run=ti.simulationRun.to_numpy())
    allx=[];allids=[]
    for kind in ['Faulty','FaultFree']:
        a=np.load(ROOT/'data'/f'tep_{kind}_testing_features.npz');allx.append(a['x']);allids.append(pd.DataFrame({'fault':a['fault'].astype(int),'run':a['run'].astype(int)}))
    ex=np.vstack(allx);ei=pd.concat(allids,ignore_index=True)
    rng=np.random.default_rng(SEED);runs=rng.permutation(sorted(ei.run.unique()))
    roles={};manifest=[]
    for j,role in enumerate(['selection','stability','confirmation']):
        assigned=runs[j*100:(j+1)*100];mask=(ei.fault==0)&ei.run.isin(assigned)
        for fault in sorted(ei.fault.unique()):
            if fault:mask|=(ei.fault==fault)&ei.run.isin(rng.choice(assigned,20,replace=False))
        roles[role]=np.flatnonzero(mask)
        for k in roles[role]:manifest.append({'role':role,'fault':int(ei.iloc[k].fault),'run':int(ei.iloc[k].run)})
    save('tep_split_manifest.json',manifest)
    models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.,random_state=SEED)),'hist_gradient':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1.,random_state=SEED)}
    candidates=[];ix=roles['selection'];y=(ei.iloc[ix].fault>0).astype(int).to_numpy()
    grid=np.r_[-.01,np.linspace(0,1,41),1.01]
    for complexity,(name,model) in enumerate(models.items()):
        print('TEP fit',name,tx.shape,flush=True);model.fit(tx,ty);joblib.dump(model,ROOT/'models'/f'tep_{name}.joblib')
        s=model.predict_proba(ex[ix])[:,1]
        for lo in grid:
            for hi in grid:
                if lo>=hi:continue
                fb=int(sum((y==1)&(s<=lo)));nf=int(sum((y==0)&(s>=hi)))
                uf,un=upper(fb,sum(y==1)),upper(nf,sum(y==0))
                candidates.append(dict(model=name,complexity=complexity,lo=float(lo),hi=float(hi),coverage=float(np.mean((s<=lo)|(s>=hi))),FB=fb,NF=nf,FB_upper=uf,NF_upper=un,pass_risk=uf<=.05 and un<=.10))
    save('tep_candidates.json',candidates)
    feasible=[c for c in candidates if c['pass_risk']]
    if not feasible:save('tep_result.json',{'status':'NO_GO'});return
    best=sorted(feasible,key=lambda c:(-c['coverage'],c['FB_upper'],c['complexity'],c['lo'],c['hi']))[0]
    best['model_sha256']=sha(ROOT/'models'/f"tep_{best['model']}.joblib")
    save('tep_frozen_policy.json',best)
    model=models[best['model']];result={}
    for role,ix in roles.items():
        s=model.predict_proba(ex[ix])[:,1];truth=(ei.iloc[ix].fault>0).astype(int).to_numpy()
        result[role]=metrics(truth,s,best['lo'],best['hi'])
        rows=ei.iloc[ix].copy();rows['score']=s;rows['route']=np.where(s<=best['lo'],'BENIGN',np.where(s>=best['hi'],'FAULT','ABSTAIN'))
        rows.to_csv(ROOT/'results'/f'tep_{role}_predictions.csv',index=False)
        result[role]['per_fault']=rows.groupby('fault').route.value_counts().unstack(fill_value=0).to_dict('index')
    save('tep_result.json',result)
    joblib.dump({'scaler':scaler,'pca':pca,'columns':cols},ROOT/'models/tep_preprocessing.joblib')
    print('TEP completed',result['confirmation'],flush=True)

def segments(mask,gap=0):
    ids=np.flatnonzero(mask)
    if not len(ids):return []
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>gap+1)+1)
    return [(int(g[0]),int(g[-1])+1) for g in groups]
def hai():
    folder=ROOT/'data/HAI21.03';rng=np.random.default_rng(SEED)
    train=pd.read_csv(folder/'train1.csv.gz');cols=[c for c in train if c!='time' and not c.startswith('attack')]
    sample=train[cols].iloc[np.sort(rng.choice(len(train),min(100000,len(train)),replace=False))].astype(float)
    scaler=StandardScaler().fit(sample);z=scaler.transform(sample)
    pca=PCA(n_components=.95,svd_solver='full').fit(z)
    forest=IsolationForest(n_estimators=100,max_samples=1024,random_state=SEED,n_jobs=8).fit(z)
    def score(x):
        a=scaler.transform(x)
        return np.column_stack([np.mean((a-pca.inverse_transform(pca.transform(a)))**2,axis=1),-forest.score_samples(a),np.mean(a*a,axis=1)])
    ref=np.sort(score(sample),axis=0)
    def pct(s):return np.column_stack([np.searchsorted(ref[:,j],s[:,j],side='right')/len(ref) for j in range(3)])
    del train;gc.collect();events=[];bank=[]
    for filename in ['train2','train3','test1','test2','test3','test4','test5']:
        print('HAI scoring',filename,flush=True)
        frame=pd.read_csv(folder/(filename+'.csv.gz'));x=frame[cols].astype(float)
        scores=np.vstack([score(x.iloc[j:j+10000]) for j in range(0,len(x),10000)]);percentiles=pct(scores)
        truth=frame['attack'].to_numpy()>0 if 'attack' in frame else np.zeros(len(frame),bool)
        alarm=np.median(percentiles,axis=1)>=.99
        attack=segments(truth);negative=[(a,b) for a,b in segments(alarm,30) if b-a>=5 and not truth[a:b].any()]
        np.savez_compressed(ROOT/'data'/f'hai_{filename}_scores.npz',scores=scores,truth=truth)
        if filename.startswith('train'):
            # Top ten non-overlapping normal alarms per source, fixed by detector severity.
            windows=[(a,a+120) for a in range(0,len(x)-119,120)]
            chosen=sorted(windows,key=lambda ab:(-float(np.median(percentiles[ab[0]:ab[1]].max(axis=0))),ab[0]))[:10]
            pairs=[(a,b,0) for a,b in chosen]
        else:pairs=[(a,b,1) for a,b in attack]+[(a,b,0) for a,b in negative]
        for a,b,label in pairs:
            summary=scores[a:b].max(axis=0);p=pct(summary[None,:])[0];signed=scaler.transform(x.iloc[a:b]).mean(axis=0)
            top=np.argsort(-np.abs(signed))[:5]
            row=dict(event_id=f'{filename}:{a}:{b}',source=filename,start=a,end=b,label=label,d_score=float(np.median(p)),percentiles=p.tolist(),signature=signed.tolist(),duration=b-a,top_variables=[{'name':cols[k],'z_mean':float(signed[k])} for k in top])
            (bank if filename.startswith('train') or (filename=='test1' and label==1) else events).append(row)
        del frame,x,scores,percentiles;gc.collect()
    save('hai_events.json',events);save('hai_case_bank.json',bank)
    dev=[e for e in events if e['source']=='test1']+[e for e in bank if e['source']=='test1']
    test=[e for e in events if e['source']!='test1']
    # Fixed HAI manuscript thresholds tested descriptively; no tuning on evaluation labels.
    result={'detectors':['PCA reconstruction','Isolation Forest','standardized mean-square deviation'],'case_bank_n':len(bank),'development_n':len(dev),'evaluation':metrics([e['label'] for e in test],[e['d_score'] for e in test],.62,.90),'source_counts':pd.DataFrame(test).groupby(['source','label']).size().to_dict().__str__()}
    save('hai_result.json',result);joblib.dump({'scaler':scaler,'pca':pca,'forest':forest,'reference_scores':ref,'columns':cols},ROOT/'models/hai_detectors.joblib')
    print('HAI completed',result,flush=True)
if __name__=='__main__':
    import sys
    (tep if sys.argv[1]=='tep' else hai)()

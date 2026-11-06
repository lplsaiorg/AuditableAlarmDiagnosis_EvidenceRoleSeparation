import json,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from v4_hai import ROOT,OUT,calibrate,metric,dump
from v2_numeric import route
CUTOFF='2020-07-13 00:00:00'
def extract():
    path=OUT/'data/hai_strict_windows.npz'
    if path.exists():return dict(np.load(path))
    folder=ROOT/'data/HAI21.03';df=pd.read_csv(folder/'train1.csv.gz');cols=[c for c in df if c!='time' and not c.startswith('attack')];sc=StandardScaler().fit(df.loc[df.time<CUTOFF,cols]);joblib.dump(sc,OUT/'models/hai_past_only_scaler.joblib');rows=[];meta=[]
    for source in ['train1','test1','test2','test3','test4','test5']:
        print('strict features',source,flush=True);df=pd.read_csv(folder/f'{source}.csv.gz');z=sc.transform(df[cols]);truth=df.attack.to_numpy()>0
        for start in range(0,len(z)-119,120):
            if source=='train1' and df.time.iloc[start+119]>=CUTOFF:continue
            a=z[start:start+120];m=a.mean(0);std=a.std(0);before=z[max(0,start-600):start];pm=before.mean(0) if len(before) else np.zeros(len(cols));ps=before.std(0) if len(before) else np.ones(len(cols));q10,q90=np.quantile(a,[.1,.9],axis=0);delta=a[-30:].mean(0)-a[:30].mean(0);rms=np.sqrt(np.mean(np.diff(a,axis=0)**2,axis=0));span=a.max(0)-a.min(0);rows.append(np.r_[m,std,q10,q90,delta,rms,span,m-pm,(m-pm)/np.maximum(ps,.1),np.log((std+.05)/(ps+.05))]);meta.append((source,start,start+120,int(truth[start:start+120].any()),str(df.time.iloc[start]),str(df.time.iloc[start+119])))
    m=pd.DataFrame(meta,columns=['source','start','end','truth','first_time','last_time']);m.to_csv(OUT/'data/hai_strict_manifest.csv',index=False);np.savez_compressed(path,x=np.asarray(rows),source=m.source.to_numpy(dtype=str),start=m.start.to_numpy(),end=m.end.to_numpy(),truth=m.truth.to_numpy(),columns=np.array(cols));return dict(np.load(path))
def main():
    if (OUT/'results/hai_strict_results.json').exists():print('Completed strict chronology control preserved');return
    a=extract();x=a['x'];y=a['truth'];p=len(a['columns']);train=np.isin(a['source'],['train1','test1','test2']);cal=a['source']=='test3';test=np.isin(a['source'],['test4','test5']);relative=x[:,p:].copy();relative[:,p:2*p]-=x[:,:p];relative[:,2*p:3*p]-=x[:,:p];views={'absolute_dynamic':x,'centered_dynamics':relative[:,:6*p],'relative_dynamic':relative};scores={};policies=[]
    for name,features in views.items():
        print('strict fit',name,flush=True);model=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,max_features=.7,class_weight='balanced',n_jobs=8,random_state=20261112).fit(features[train],y[train]);path=OUT/'models'/f'hai_strict_{name}.joblib';joblib.dump(model,path);scores[name]=model.predict_proba(features)[:,1];policies.append(dict(calibrate(y[cal],scores[name][cal]),model=name,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    best=sorted(policies,key=lambda q:(-q['balanced_recognition'],-q['coverage'],q['FB_rate'],q['model']))[0];dump('hai_strict_frozen_policy.json',dict(best,training_cutoff=CUTOFF));dump('hai_strict_calibration_candidates.json',policies);results=[];rows=[]
    for q in policies:
        s=scores[q['model']];r=route(s,q['lo'],q['hi'])
        for source in ['test3','test4','test5','heldout']:
            mask=test if source=='heldout' else a['source']==source;results.append({'model':q['model'],'selected':q['model']==best['model'],'source':source,**metric(y[mask],r[mask],s[mask])})
        rows.extend({'model':q['model'],'source':str(a['source'][i]),'start':int(a['start'][i]),'end':int(a['end'][i]),'truth':int(y[i]),'route':int(r[i]),'score':float(s[i])} for i in np.flatnonzero(test))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'results/hai_strict_predictions.csv',index=False);episodes=[]
    for source in ['test4','test5']:
        truth=np.load(ROOT/'data'/f'hai_{source}_scores.npz')['truth'];ids=np.flatnonzero(truth);groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1);g=frame[(frame.model==best['model'])&(frame.source==source)]
        for ep in groups:
            if not len(ep):continue
            match=g[(g.start<=ep[-1])&(g.end>ep[0])];episodes.append({'source':source,'start':int(ep[0]),'end':int(ep[-1])+1,'detected':bool((match.route==1).any())})
    manifest=pd.read_csv(OUT/'data/hai_strict_manifest.csv');training=manifest[manifest.source.isin(['train1','test1','test2'])];assert training.last_time.max()<manifest[manifest.source=='test3'].first_time.min();assert manifest[manifest.source=='test3'].last_time.max()<manifest[manifest.source.isin(['test4','test5'])].first_time.min();hashrow=lambda v:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest();known={hashrow(v) for v in x[train]};dupes=sum(hashrow(v) in known for v in x[test])
    dump('hai_strict_results.json',{'selected_model':best['model'],'selected_heldout':next(r for r in results if r['selected'] and r['source']=='heldout'),'candidate_results':results,'episode_detected':sum(e['detected'] for e in episodes),'episode_total':len(episodes),'episodes':episodes,'training_last_timestamp':training.last_time.max(),'calibration_first_timestamp':manifest[manifest.source=='test3'].first_time.min(),'chronology_verified':True,'exact_training_heldout_feature_duplicates':dupes,'training_windows':int(train.sum()),'training_fault_windows':int(y[train].sum()),'scope':'Strict past-only fitting/calibration/evaluation chronology within an already exposed dataset. Exploratory follow-up, not a fresh independent test.'});print('STRICT COMPLETE',best,flush=True)
if __name__=='__main__':main()

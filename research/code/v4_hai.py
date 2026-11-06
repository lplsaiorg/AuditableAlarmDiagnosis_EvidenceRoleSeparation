import json,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from v2_numeric import ROOT,route,evaluate
OUT=ROOT/'experiments/v4'
def dump(n,x):(OUT/'results'/n).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def extract():
    path=OUT/'data/hai_windows.npz'
    if path.exists():return dict(np.load(path))
    pre=joblib.load(ROOT/'models/hai_detectors.joblib');cols=pre['columns'];sc=pre['scaler'];rows=[];meta=[]
    for source in ['train1','train2','train3','test1','test2','test3','test4','test5']:
        print('HAI features',source,flush=True);df=pd.read_csv(ROOT/'data/HAI21.03'/f'{source}.csv.gz');z=(df[cols].to_numpy()-sc.mean_)/sc.scale_;truth=df['attack'].to_numpy()>0
        for start in range(0,len(z)-119,120):
            a=z[start:start+120];m=a.mean(0);std=a.std(0);before=z[max(0,start-600):start];pm=before.mean(0) if len(before) else np.zeros(len(cols));ps=before.std(0) if len(before) else np.ones(len(cols));q10,q90=np.quantile(a,[.1,.9],axis=0);delta=a[-30:].mean(0)-a[:30].mean(0);rms=np.sqrt(np.mean(np.diff(a,axis=0)**2,axis=0));span=a.max(0)-a.min(0)
            rows.append(np.r_[m,std,q10,q90,delta,rms,span,m-pm,(m-pm)/np.maximum(ps,.1),np.log((std+.05)/(ps+.05))]);meta.append((source,start,start+120,int(truth[start:start+120].any())))
    m=pd.DataFrame(meta,columns=['source','start','end','truth']);m.to_csv(OUT/'data/hai_window_manifest.csv',index=False);x=np.asarray(rows);assert np.isfinite(x).all();np.savez_compressed(path,x=x,source=m.source.to_numpy(dtype=str),start=m.start.to_numpy(),end=m.end.to_numpy(),truth=m.truth.to_numpy(),columns=np.array(cols));return dict(np.load(path))
def calibrate(y,s):
    grid=np.unique(np.r_[-np.inf,np.quantile(s,np.linspace(0,1,201)),np.inf]);best=None
    for lo in grid:
        benign=s<=lo;fb=np.mean(benign[y==1]);nr=np.mean(benign[y==0])
        if fb>.05:continue
        for hi in grid[grid>lo]:
            fault=s>=hi;nf=np.mean(fault[y==0]);fr=np.mean(fault[y==1])
            if nf>.05:continue
            q={'lo':float(lo),'hi':float(hi),'coverage':float(np.mean(benign|fault)),'FB_rate':float(fb),'NF_rate':float(nf),'balanced_recognition':float((nr+fr)/2),'fault_recall':float(fr)}
            if best is None or (-q['balanced_recognition'],-q['coverage'],fb,lo,hi)<(-best['balanced_recognition'],-best['coverage'],best['FB_rate'],best['lo'],best['hi']):best=q
    return best
def metric(y,r,s=None):
    m=evaluate(y,r)
    for key in ['FB_upper','NF_upper','pass_risk']:m.pop(key)
    m['balanced_recognition']=.5*(m['fault_recognized']/m['n_fault']+m['normal_recognized']/m['n_normal'])
    if s is not None:m['AUROC']=float(roc_auc_score(y,s))
    return m
def main():
    if (OUT/'results/hai_complete.json').exists():print('Completed HAI v4 preserved');return
    a=extract();x=a['x'];y=a['truth'];p=len(a['columns']);relative=x[:,p:].copy();relative[:,p:2*p]-=x[:,:p];relative[:,2*p:3*p]-=x[:,:p];train=np.isin(a['source'],['train1','train2','train3','test1','test2']);normal=np.isin(a['source'],['train1','train2','train3']);cal=a['source']=='test3';test=np.isin(a['source'],['test4','test5']);scores={};policies=[];hashes={}
    for view,features in [('absolute_dynamic',x),('relative_dynamic',relative)]:
        for alg,model in [('extra',ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,max_features=.7,class_weight='balanced',n_jobs=8,random_state=20261112)),('hist',HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,l2_regularization=5,class_weight='balanced',random_state=20261112))]:
            name=view+'_'+alg;path=OUT/'models'/f'hai_{name}.joblib'
            if path.exists():model=joblib.load(path)
            else:print('fit HAI',name,flush=True);model.fit(features[train],y[train]);joblib.dump(model,path)
            scores[name]=model.predict_proba(features)[:,1];hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        name=view+'_normal_knn';print('fit HAI',name,flush=True);ref=features[normal];med=np.median(ref,0);scale=np.maximum(np.quantile(ref,.9,axis=0)-np.quantile(ref,.1,axis=0),.05);z=(ref-med)/scale;pc=PCA(n_components=40,whiten=True,random_state=20261112).fit(z);nn=NearestNeighbors(n_neighbors=5,n_jobs=8).fit(pc.transform(z));scores[name]=nn.kneighbors(pc.transform((features-med)/scale))[0].mean(1);path=OUT/'models'/f'hai_{name}.joblib';joblib.dump({'median':med,'scale':scale,'pca':pc,'neighbors':nn},path);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,s in scores.items():policies.append(dict(calibrate(y[cal],s[cal]),model=name))
    best=sorted(policies,key=lambda q:(-q['balanced_recognition'],-q['coverage'],q['FB_rate'],q['model']))[0];dump('hai_calibration_candidates.json',policies);dump('hai_frozen_policy.json',dict(best,model_sha256=hashes[best['model']],protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),calibration_source='test3',evaluation_sources=['test4','test5']))
    # Only after freezing the chosen candidate, calculate held-out performance.
    results=[];frames=[]
    for q in policies:
        s=scores[q['model']];r=route(s,q['lo'],q['hi'])
        for source in ['test3','test4','test5','heldout']:
            mask=test if source=='heldout' else a['source']==source;results.append({'model':q['model'],'source':source,'selected':q['model']==best['model'],**metric(y[mask],r[mask],s[mask])})
        frames.extend({'model':q['model'],'source':str(a['source'][i]),'start':int(a['start'][i]),'end':int(a['end'][i]),'truth':int(y[i]),'score':float(s[i]),'route':int(r[i])} for i in np.flatnonzero(cal|test))
    frame=pd.DataFrame(frames);frame.to_csv(OUT/'results/hai_predictions.csv',index=False);episodes=[]
    for source in ['test4','test5']:
        truth=np.load(ROOT/'data'/f'hai_{source}_scores.npz')['truth'];ids=np.flatnonzero(truth);groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1);g=frame[(frame.model==best['model'])&(frame.source==source)]
        for ep in groups:
            if not len(ep):continue
            overlap=g[(g.start<=ep[-1])&(g.end>ep[0])];episodes.append({'source':source,'start':int(ep[0]),'end':int(ep[-1])+1,'detected':bool((overlap.route==1).any()),'all_abstained':bool((overlap.route==-1).all())})
    old=pd.read_csv(ROOT/'experiments/v3/results/hai_windows.csv');old=old[old.source.isin(['test4','test5'])];baseline=metric(old.truth.to_numpy(),old.route.to_numpy(),old.score.to_numpy());old_ep=json.loads((ROOT/'experiments/v3/results/hai_results.json').read_text())['episodes'];old_ep=[r for r in old_ep if r['source'] in ['test4','test5']]
    dump('hai_results.json',{'selected_model':best['model'],'candidate_results':results,'selected_heldout':next(r for r in results if r['selected'] and r['source']=='heldout'),'v3_same_cohort':baseline,'episodes':episodes,'episode_detected':sum(r['detected'] for r in episodes),'episode_total':len(episodes),'v3_same_cohort_episode_detected':sum(r['detected'] for r in old_ep),'scope':'Official normal training files plus labeled early attack files, with later attack-file holdout. Official train2/train3 are later than evaluation dates; NOT strict chronological extrapolation. See separate strict past-only control.'});dump('hai_complete.json',{'complete':True});print('HAI COMPLETE',best,flush=True)
if __name__=='__main__':main()

import json,hashlib,gc,time
from pathlib import Path
import numpy as np,pandas as pd,pyreadr,joblib
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/v2';SEED=20261106
def dump(name,obj):
    (OUT/'results'/name).write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def original(part):
    if part=='Training':return dict(np.load(ROOT/'data/tep_training_features.npz'))
    a=[np.load(ROOT/'data'/f'tep_{k}_testing_features.npz') for k in ['Faulty','FaultFree']]
    return {k:np.concatenate([d[k] for d in a]) for k in ['x','fault','run']}
def dynamic(part):
    dest=OUT/'data'/f'dynamic_{part}.npz'
    if dest.exists():return dict(np.load(dest))
    pre=joblib.load(ROOT/'models/tep_preprocessing.joblib');cols=pre['columns'];sc=pre['scaler'];rows=[];ids=[]
    for kind in ['FaultFree','Faulty']:
        print('extract dynamic',part,kind,flush=True)
        df=next(iter(pyreadr.read_r(str(ROOT/'data/TEP'/f'TEP_{kind}_{part}.RData')).values()))
        df=df.loc[df['sample']>(20 if part=='Training' else 160)].sort_values(['faultNumber','simulationRun','sample'])
        for (fault,run),g in df.groupby(['faultNumber','simulationRun'],sort=True):
            z=(g[cols].to_numpy()-sc.mean_)/sc.scale_;half=len(z)//2
            row=np.r_[z.std(0),np.quantile(z,.1,axis=0),np.quantile(z,.9,axis=0),z[half:].mean(0)-z[:half].mean(0),np.sqrt(np.mean(np.diff(z,axis=0)**2,axis=0))]
            rows.append(row);ids.append((int(fault),int(run)))
        del df;gc.collect()
    a=np.array(ids);np.savez_compressed(dest,x=np.array(rows),fault=a[:,0],run=a[:,1]);return dict(np.load(dest))
def merged(part):
    a=original(part);d=dynamic(part);lookup={(f,r):x for f,r,x in zip(d['fault'],d['run'],d['x'])}
    return a,np.column_stack([a['x'],np.array([lookup[(f,r)] for f,r in zip(a['fault'],a['run'])])])
def evaluate(y,r):
    y=np.asarray(y);r=np.asarray(r);nfault=int(sum(y==1));nnormal=int(sum(y==0));fb=int(sum((y==1)&(r==0)));nf=int(sum((y==0)&(r==1)))
    uf=float(beta.ppf(.975,fb+1,nfault-fb)) if fb<nfault else 1.;un=float(beta.ppf(.975,nf+1,nnormal-nf)) if nf<nnormal else 1.
    auto=int(sum(r!=-1));correct=int(sum((r==y)&(r!=-1)))
    return dict(n=len(y),n_fault=nfault,n_normal=nnormal,FB=fb,NF=nf,FB_upper=uf,NF_upper=un,pass_risk=uf<=.05 and un<=.10,coverage=auto/len(y),automatic=auto,fault_recognized=int(sum((y==1)&(r==1))),normal_recognized=int(sum((y==0)&(r==0))),abstain=int(sum(r==-1)),accepted_error=(auto-correct)/auto if auto else None)
def route(s,lo,hi,veto=None):
    benign=(s<=lo)&(np.ones(len(s),bool) if veto is None else veto)
    return np.where(benign,0,np.where(s>=hi,1,-1))
def search(y,s,ok):
    grid=np.r_[-.01,np.linspace(0,1,101),1.01];out=[];nF=sum(y==1);nN=sum(y==0)
    uf=np.r_[beta.ppf(.975,np.arange(nF)+1,nF-np.arange(nF)),1.]
    un=np.r_[beta.ppf(.975,np.arange(nN)+1,nN-np.arange(nN)),1.]
    for lo in grid:
        benign=(s<=lo)&ok;fb=int(sum(benign&(y==1)))
        if uf[fb]>.05:continue
        for hi in grid[grid>lo]:
            fault=s>=hi;nf=int(sum(fault&(y==0)))
            if un[nf]<=.1:out.append(dict(lo=float(lo),hi=float(hi),coverage=float(np.mean(benign|fault)),FB=fb,NF=nf,FB_upper=float(uf[fb]),NF_upper=float(un[nf])))
    return sorted(out,key=lambda x:(-x['coverage'],x['FB_upper'],x['lo'],x['hi']))[:1]
def main():
    if (OUT/'results/frozen_policy.json').exists():raise RuntimeError('v2 policy already frozen; do not refit')
    old=json.loads((ROOT/'results/tep_split_manifest.json').read_text());used={r['run'] for r in old};a=original('Testing');allruns=set(map(int,a['run']));fresh=sorted(allruns-used);assert len(fresh)==200
    rng=np.random.default_rng(SEED);confirm=[{'role':'confirmation_v2','fault':0,'run':r} for r in fresh]
    for f in range(1,21):confirm.extend({'role':'confirmation_v2','fault':f,'run':int(r)} for r in sorted(rng.choice(fresh,40,replace=False)))
    manifest=[r for r in old if r['role'] in ['selection','stability']]+confirm;dump('split_manifest.json',manifest)
    ta,tx=merged('Training');ea,ex=merged('Testing');y=(ta['fault']>0).astype(int);lookup={(int(f),int(r)):i for i,(f,r) in enumerate(zip(ea['fault'],ea['run']))}
    ix={role:np.array([lookup[(r['fault'],r['run'])] for r in manifest if r['role']==role]) for role in ['selection','stability','confirmation_v2']}
    normal=tx[y==0,:52];ps=StandardScaler().fit(normal);pc=PCA(n_components=.95,svd_solver='full').fit(ps.transform(normal))
    def pscore(x):
        z=ps.transform(x[:,:52]);return np.mean((z-pc.inverse_transform(pc.transform(z)))**2,axis=1)
    thresholds={str(q):float(np.quantile(pscore(normal),q)) for q in [.95,.99]}
    # Only development data is scored during candidate enumeration.
    si=ix['selection'];sy=(ea['fault'][si]>0).astype(int);pselect=pscore(ex[si]);models={};bestper=[]
    configs=[]
    for width,feature in [(118,'mean'),(378,'dynamic')]:
        for weight in [1,5,20]:configs.append((f'{feature}_hist_w{weight}',width,HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1.,class_weight={0:weight,1:1},random_state=SEED)))
        configs.append((f'{feature}_logistic_balanced',width,make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=3000,random_state=SEED))))
    for complexity,(name,width,model) in enumerate(configs):
        print('fit',name,flush=True);model.fit(tx[:,:width],y);models[name]=(width,model);joblib.dump(model,OUT/'models'/(name+'.joblib'));s=model.predict_proba(ex[si,:width])[:,1]
        for veto in ['none','.95','.99']:
            ok=np.ones(len(si),bool) if veto=='none' else pselect<=thresholds[str(float(veto))]
            candidates=search(sy,s,ok)
            if candidates:bestper.append(dict(candidates[0],model=name,width=width,veto=veto,complexity=complexity))
    dump('selection_ablation.json',bestper)
    if not bestper:dump('status.json',{'status':'NO_GO'});return
    best=sorted(bestper,key=lambda c:(-c['coverage'],c['FB_upper'],c['complexity'],c['veto'],c['lo'],c['hi']))[0]
    best['pca_thresholds']=thresholds;best['model_sha256']=digest(OUT/'models'/(best['model']+'.joblib'));best['protocol_sha256']=digest(OUT/'protocol.json');dump('frozen_policy.json',best)
    joblib.dump({'scaler':ps,'pca':pc},OUT/'models/pca_normal_evidence.joblib')
    v1=json.loads((ROOT/'results/tep_frozen_policy.json').read_text());oldmodel=joblib.load(ROOT/'models'/f"tep_{v1['model']}.joblib");width,model=models[best['model']]
    results={};curves=[]
    for role,ids in ix.items():
        score=model.predict_proba(ex[ids,:width])[:,1];oldscore=oldmodel.predict_proba(ex[ids,:118])[:,1];truth=(ea['fault'][ids]>0).astype(int)
        ok=np.ones(len(ids),bool) if best['veto']=='none' else pscore(ex[ids])<=thresholds[str(float(best['veto']))]
        routes=route(score,best['lo'],best['hi'],ok);oldroutes=route(oldscore,v1['lo'],v1['hi'])
        results[role]={'v2':evaluate(truth,routes),'v1':evaluate(truth,oldroutes)}
        frame=pd.DataFrame({'fault':ea['fault'][ids].astype(int),'run':ea['run'][ids].astype(int),'truth':truth,'score_v2':score,'score_v1':oldscore,'route_v2':routes,'route_v1':oldroutes});frame.to_csv(OUT/'results'/f'{role}_predictions.csv',index=False)
        if role=='confirmation_v2':
            results[role]['per_fault']=frame.groupby('fault').route_v2.value_counts().unstack(fill_value=0).to_dict('index')
            for name,s in [('v1',oldscore),('v2',score)]:
                for coverage in [.5,.7,.9]:
                    chosen=np.argsort(-np.abs(s-.5),kind='stable')[:int(coverage*len(s))];r=np.full(len(s),-1);r[chosen]=(s[chosen]>=.5).astype(int)
                    curves.append(dict(model=name,target_coverage=coverage,**evaluate(truth,r)))
            # Paired cluster bootstrap by shared simulationRun, not independent fault copies.
            groups=frame.groupby('run');d=np.array([np.sum(g.route_v2!=-1)-np.sum(g.route_v1!=-1) for _,g in groups]);n=np.array([len(g) for _,g in groups]);draw=rng.integers(0,len(n),(5000,len(n)));bs=d[draw].sum(1)/n[draw].sum(1)
            results[role]['coverage_difference_cluster_CI95']=np.quantile(bs,[.025,.975]).tolist();results[role]['coverage_difference']=float(d.sum()/n.sum())
    dump('numeric_results.json',results);dump('matched_coverage.json',curves)
    print('NUMERIC COMPLETE',json.dumps(results['confirmation_v2']),flush=True)
if __name__=='__main__':main()

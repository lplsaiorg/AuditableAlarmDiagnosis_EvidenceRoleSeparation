from pathlib import Path
import json,hashlib,warnings
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from v4_hai import calibrate,metric
from v2_numeric import route

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/v5_review'
for n in ['results','models','figures']: (OUT/n).mkdir(parents=True,exist_ok=True)
SUFFIX='_20261106_v01'
def dump(name,obj):
    (OUT/'results'/(name+SUFFIX+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))

def protocol():
    obj={'purpose':'First-review follow-up on existing benchmarks','seed':20261106,'HAI':{'features':'Existing frozen 790-feature strict-past representation; no feature search','normal_fit':'train1 before 2020-07-13','supervised_fit':'same normal windows plus test1/test2','calibration':'test3, same existing calibrate function for every model','evaluation':['test4','test5'],'new_models':['normal PCA reconstruction, 40 components','normal IsolationForest, 300 trees, max_samples=256','normal score fusion: median of training-IQR normalized log1p(PCA MSE), negative IF score, maximum absolute feature z','class-balanced logistic regression C=1, max_iter=5000'],'original_model':'Frozen absolute_dynamic ExtraTrees, retained regardless of new comparisons','selection':'No new-model winner selection; report all candidates and save calibrated policies before heldout metrics'},'paired_statistics':{'draws':10000,'unit':'simulationRun; query-level pairing within class','contrasts':'altered minus clean FB/NF; ordinary effect minus separated effect on common valid queries; no_history minus clean','missing':'primary valid-response pairs; operational sensitivity includes unavailable as ABSTAIN'},'TEP':'Describe existing final per-fault routes and accepted errors; pooled simulationRun-cluster sensitivity without refitting','new_independent_validation':False,'LLM_calls':0}
    p=OUT/('protocol'+SUFFIX+'.json')
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==obj
    else:p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def cluster_delta(g,delta,seed):
    t=pd.DataFrame({'run':np.asarray(g),'value':np.asarray(delta,float)})
    k=t.groupby('run').value.agg(['sum','count']);rng=np.random.default_rng(seed);idx=rng.integers(0,len(k),(10000,len(k)))
    bs=k['sum'].to_numpy()[idx].sum(1)/k['count'].to_numpy()[idx].sum(1)
    return {'n':len(t),'clusters':len(k),'estimate':float(t.value.mean()),'CI95':np.quantile(bs,[.025,.975]).tolist()}

def paired():
    d=pd.read_csv(ROOT/'experiments/v3/results/fair_per_response.csv');d=d[d.repeat==0];out=[]
    for sensitivity in [False,True]:
        for mode in ['ordinary','separated']:
            clean=d[(d['mode']==mode)&(d.condition=='clean')].set_index('query_id')
            for cond in ['fabricated_benign','flipped_outcomes','stale_regime','no_history']:
                alt=d[(d['mode']==mode)&(d.condition==cond)].set_index('query_id')
                if alt.empty:continue
                z=clean.join(alt,lsuffix='_c',rsuffix='_a');z=z if sensitivity else z[z.valid_c&z.valid_a]
                for direction,target,isfault in [('FB',0,True),('NF',1,False)]:
                    q=z[(z.fault_c>0)==isfault];v=q.reader_a.eq(target).astype(int)-q.reader_c.eq(target).astype(int)
                    out.append({'mode':mode,'condition':cond,'direction':direction,'include_unavailable':sensitivity,**cluster_delta(q.run_c,v,20261106)})
        for cond in ['fabricated_benign','flipped_outcomes','stale_regime']:
            groups=[]
            for mode in ['ordinary','separated']:
                z=d[(d['mode']==mode)&d.condition.isin(['clean',cond])].pivot(index='query_id',columns='condition',values=['reader','valid'])
                z.columns=['_'.join(k) for k in z.columns];groups.append(z.add_prefix(mode+'_'))
            z=groups[0].join(groups[1]);meta=d.drop_duplicates('query_id').set_index('query_id');z=z.join(meta[['fault','run']])
            if not sensitivity:
                for mode in ['ordinary','separated']:
                    for c in ['clean',cond]:z=z[z[mode+'_valid_'+c].astype(bool)]
            for direction,target,isfault in [('FB',0,True),('NF',1,False)]:
                q=z[(z.fault>0)==isfault]
                delta=(q['ordinary_reader_'+cond].eq(target).astype(int)-q.ordinary_reader_clean.eq(target).astype(int))-(q['separated_reader_'+cond].eq(target).astype(int)-q.separated_reader_clean.eq(target).astype(int))
                out.append({'mode':'ordinary_effect_minus_separated_effect','condition':cond,'direction':direction,'include_unavailable':sensitivity,**cluster_delta(q.run,delta,20261106)})
    dump('paired_effects',out)
    for r in out:
        if not r['include_unavailable'] and (r['mode'].startswith('ordinary_effect') or r['condition']=='no_history' or r['condition']=='flipped_outcomes' and r['direction']=='NF'):print('PAIRED',r,flush=True)

def tep():
    d=pd.read_csv(ROOT/'experiments/v4/results/calibration_predictions.csv');d=d[d.strategy=='guarded_margin_cal400'].copy();rows=[]
    for fault,g in d.groupby('fault'):
        counts={name:int(g.route.eq(v).sum()) for name,v in [('FAULT',1),('BENIGN',0),('ABSTAIN',-1)]};automatic=counts['FAULT']+counts['BENIGN'];err=counts['BENIGN'] if fault else counts['FAULT']
        rows.append({'fault':int(fault),'n':len(g),'unique_runs':g.run.nunique(),**counts,'coverage':automatic/len(g),'accepted_error':err/automatic if automatic else None})
    truth=d.fault.gt(0).astype(int);m=metric(truth,d.route);folds=[]
    for (seed,fold),g in d.groupby(['seed','fold']):folds.append({'seed':int(seed),'fold':int(fold),**metric(g.fault.gt(0).astype(int),g.route)})
    # Correlated repeats are retained together in the same simulationRun cluster.
    sensitivities={}
    for name,subset,event in [('FB',d[d.fault>0],0),('NF',d[d.fault==0],1)]:
        sensitivities[name]=cluster_delta(subset.run,subset.route.eq(event).astype(int),20261106)
    sensitivities['coverage']=cluster_delta(d.run,d.route.ne(-1).astype(int),20261106)
    auto=d[d.route!=-1];sensitivities['accepted_error']=cluster_delta(auto.run,auto.route.ne(auto.fault.gt(0).astype(int)).astype(int),20261106)
    dump('tep_per_fault',{'rows':rows,'aggregate':m,'folds':folds,'pooled_run_cluster_intervals':sensitivities,'scope':'Descriptive repeated-evaluation counts; cluster sensitivity retains all saved occurrences of each run, not a new fit or simultaneous risk guarantee'})
    print('TEP',m,'fault9',next(r for r in rows if r['fault']==9),flush=True)

def hai():
    dest=OUT/'results'/('hai_comparison'+SUFFIX+'.json')
    if dest.exists():print('HAI completed result preserved');return
    a=dict(np.load(ROOT/'experiments/v4/data/hai_strict_windows.npz'));x=a['x'];y=a['truth'];src=a['source'];normal=src=='train1';train=np.isin(src,['train1','test1','test2']);cal=src=='test3';test=np.isin(src,['test4','test5'])
    manifest=pd.read_csv(ROOT/'experiments/v4/data/hai_strict_manifest.csv');assert manifest.loc[train,'last_time'].max()<manifest.loc[cal,'first_time'].min();assert manifest.loc[cal,'last_time'].max()<manifest.loc[test,'first_time'].min()
    sc=StandardScaler().fit(x[normal]);z=sc.transform(x);pca=PCA(n_components=40,svd_solver='full').fit(z[normal]);res=((z-pca.inverse_transform(pca.transform(z)))**2).mean(1)
    iso=IsolationForest(n_estimators=300,max_samples=256,random_state=20261106,n_jobs=6).fit(z[normal]);iscore=-iso.score_samples(z);components=np.c_[np.log1p(res),iscore,np.max(abs(z),axis=1)]
    med=np.median(components[normal],axis=0);iqr=np.maximum(np.quantile(components[normal],.75,axis=0)-np.quantile(components[normal],.25,axis=0),1e-8);fusion=np.median((components-med)/iqr,axis=1)
    bundle={'scaler':sc,'pca':pca,'isolation_forest':iso,'component_median':med,'component_iqr':iqr};joblib.dump(bundle,OUT/'models'/('normal_scores'+SUFFIX+'.joblib'))
    print('Fit logistic regression on same 790 features',flush=True)
    logit=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=5000,solver='lbfgs',random_state=20261106))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');logit.fit(x[train],y[train])
    assert not any('converg' in str(w.message).lower() for w in caught),[str(w.message) for w in caught]
    joblib.dump(logit,OUT/'models'/('logistic'+SUFFIX+'.joblib'))
    original=joblib.load(ROOT/'experiments/v4/models/hai_strict_absolute_dynamic.joblib')
    scores={'PCA':res,'IsolationForest':iscore,'Normal score fusion':fusion,'Logistic regression':logit.predict_proba(x)[:,1],'Adapted ExtraTrees':original.predict_proba(x)[:,1]}
    policies={name:calibrate(y[cal],s[cal]) for name,s in scores.items()};old=read('experiments/v4/results/hai_strict_frozen_policy.json')
    assert all(abs(policies['Adapted ExtraTrees'][k]-old[k])<1e-12 for k in ['lo','hi'])
    dump('hai_frozen_policies',{'policies':policies,'train_windows':int(train.sum()),'normal_fit_windows':int(normal.sum()),'calibration_windows':int(cal.sum()),'code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    episode_truth=read('experiments/v4/results/hai_strict_results.json')['episodes'];records=[];pred=[];eps=[]
    for name,s in scores.items():
        q=policies[name];r=route(s,q['lo'],q['hi']);detected=0
        for ep in episode_truth:
            idx=(src==ep['source'])&(a['start']<ep['end'])&(a['end']>ep['start']);hit=bool(np.any(r[idx]==1));detected+=hit;eps.append({'model':name,**{k:ep[k] for k in ['source','start','end']},'detected':hit})
        for source,mask in [('test3',cal),('test4',src=='test4'),('test5',src=='test5'),('heldout',test)]:records.append({'model':name,'source':source,'attack_labels_in_fit':name in ['Logistic regression','Adapted ExtraTrees'],**metric(y[mask],r[mask],s[mask]),**({'episodes_detected':detected,'episodes_total':len(episode_truth)} if source=='heldout' else {})})
        pred.extend({'model':name,'source':str(src[i]),'start':int(a['start'][i]),'end':int(a['end'][i]),'truth':int(y[i]),'score':float(s[i]),'route':int(r[i])} for i in np.flatnonzero(cal|test))
    pd.DataFrame(pred).to_csv(OUT/'results'/('hai_predictions'+SUFFIX+'.csv'),index=False)
    dump('hai_comparison',{'records':records,'episodes':eps,'protocol':'All models use the same strict-past 790-feature event representation and test3 threshold calibration. Normal-only models fit on train1, supervised models also use test1/test2. Original ExtraTrees model remains the primary method.'})
    for r in records:
        if r['source']=='heldout':print('HAI',r,flush=True)

if __name__=='__main__':protocol();paired();tep();hai()

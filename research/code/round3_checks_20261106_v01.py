"""Third-review checks: one-factor event-length control and query provenance."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
from pathlib import Path
import json, hashlib, gc
import numpy as np
import pandas as pd
import joblib
import pyreadr
from reconstruct import features
from v2_numeric import merged, evaluate, route
from v4_tep import align, extract, search

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/v6_review/results'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf8')
modelpath=ROOT/'experiments/v2/models/dynamic_hist_w20.joblib'
protocol={'control':'matched_480_378_fixed_model','train_samples':[21,500],'test_samples':[161,640],
 'reference_test_samples':[161,960],'features':378,'model':'dynamic_hist_w20',
 'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),
 'calibration_binding':'guarded_margin_cal400','targets':[.02,.025],
 'evaluation_criteria':[.05,.10],'threshold_grid':'-0.01, 0:0.005:1, 1.01',
 'selection':'One fixed control, no retuning or model selection. Original trained model reused; training feature equivalence checked before scoring.'}
save('length_protocol_20261106_v01.json',protocol)
ta,tx=merged('Training');train=align(ta,extract('Training'))[:,:378]
assert np.allclose(tx,train,rtol=1e-10,atol=1e-10)
model=joblib.load(modelpath)
# Preserve the original fit exactly. Recomputed statistics can differ at
# floating-point precision near tree split thresholds; do not refit the control.
training_probability_difference=float(np.max(abs(model.predict_proba(tx)-model.predict_proba(train))))
ea,ex=merged('Testing');matched=align(ea,extract('Testing'))[:,:378]
# Match the original numerical implementation as well as the feature definition:
# the first 118 features standardize pandas grouped raw means, whereas the
# 690-dimensional cache averages standardized rows. They are algebraically
# equivalent but tree thresholds can distinguish floating-point roundoff.
mean_cache=OUT/'length_means_20261106_v01.npz'
if not mean_cache.exists():
    pre=joblib.load(ROOT/'models/tep_preprocessing.joblib');means={}
    for kind in ['FaultFree','Faulty']:
        print('extract original mean implementation:',kind,flush=True)
        frame=next(iter(pyreadr.read_r(str(ROOT/'data/TEP'/f'TEP_{kind}_Testing.RData')).values()))
        ids,values=features(frame.loc[frame['sample']<=640],pre['columns'],pre['scaler'],160)
        for (f,r),v in zip(ids[['faultNumber','simulationRun']].itertuples(index=False,name=None),values):means[(int(f),int(r))]=v
        del frame;gc.collect()
    np.savez_compressed(mean_cache,x=np.array([means[(int(f),int(r))] for f,r in zip(ea['fault'],ea['run'])]))
matched[:,:118]=np.load(mean_cache)['x']
score=model.predict_proba(matched)[:,1]
reference=model.predict_proba(ex)[:,1]
old=np.load(ROOT/'experiments/v4/data/tep_scores.npz')
assert np.array_equal(ea['fault'],old['fault']) and np.array_equal(ea['run'],old['run'])
assert np.allclose(reference,old['v2_dynamic_w20'],rtol=0,atol=1e-14)
pd.DataFrame({'index':np.arange(len(score)),'fault':ea['fault'].astype(int),'run':ea['run'].astype(int),'score':score}).to_csv(OUT/'length_scores_20261106_v01.csv',index=False)
bindings=json.loads((ROOT/'experiments/v4/results/calibration_bindings.json').read_text())
bindings=[r for r in bindings if r['strategy']=='guarded_margin_cal400']
records=[];pred=[]
for b in bindings:
    ci=np.array(b['calibration_indices']);ti=np.array(b['evaluation_indices']);cy=(ea['fault'][ci]>0).astype(int)
    policy=search(cy,score[ci],.02,.025)
    assert policy is not None
    cr=route(score[ci],policy['lo'],policy['hi'])
    rr=route(score[ti],policy['lo'],policy['hi']);metrics=evaluate((ea['fault'][ti]>0).astype(int),rr)
    records.append(dict(seed=b['seed'],fold=b['fold'],strategy=protocol['control'],policy=policy,calibration=evaluate(cy,cr),**metrics))
    pred.extend(dict(seed=b['seed'],fold=b['fold'],strategy=protocol['control'],index=int(i),fault=int(ea['fault'][i]),run=int(ea['run'][i]),score=float(score[i]),route=int(r)) for i,r in zip(ti,rr))
save('length_results_20261106_v01.json',records)
pd.DataFrame(pred).to_csv(OUT/'length_predictions_20261106_v01.csv',index=False)
summary={'evaluations':len(records),'passes':sum(r['pass_risk'] for r in records),'mean_coverage':np.mean([r['coverage'] for r in records]),'coverage_range':[min(r['coverage'] for r in records),max(r['coverage'] for r in records)],'max_FB_upper':max(r['FB_upper'] for r in records),'max_NF_upper':max(r['NF_upper'] for r in records),'FB':sum(r['FB'] for r in records),'NF':sum(r['NF'] for r in records),'training_max_abs_feature_difference':float(np.max(abs(tx-train))),'recomputed_training_probability_max_difference':training_probability_difference,'original_training_fit_preserved':True}
save('length_summary_20261106_v01.json',summary)
reader=pd.read_csv(ROOT/'experiments/v3/results/fair_per_response.csv').drop_duplicates('query_id').sort_values('query_id')
cohort=pd.read_json(ROOT/'experiments/v2/results/pollution_cohort.json')
chosen=pd.concat([cohort[cohort.fault==0].sample(n=20,random_state=20261107),pd.DataFrame([cohort[cohort.fault==f].iloc[0] for f in range(1,21)])],ignore_index=True)
assert list(zip(reader.fault,reader.run))==list(zip(chosen.fault,chosen.run))
lookup={(int(f),int(r)):i for i,(f,r) in enumerate(zip(ea['fault'],ea['run']))}
q=[]
for row in reader.itertuples():
    i=lookup[(int(row.fault),int(row.run))]
    item={'query_id':row.query_id,'source':f'TEP_{"FaultFree" if row.fault==0 else "Faulty"}_Testing.RData','split':'Testing','fault':int(row.fault),'run':int(row.run),'index':i}
    for role in ['evaluation','calibration']:
        item[role+'_event_occurrences']=sum(i in b[role+'_indices'] for b in bindings)
        item[role+'_run_occurrences']=sum(int(row.run) in set(ea['run'][b[role+'_indices']]) for b in bindings)
    q.append(item)
pd.DataFrame(q).to_csv(OUT/'history_query_manifest_20261106_v01.csv',index=False)
overlap=[]
for b in bindings:
    item={'seed':b['seed'],'fold':b['fold']}
    for role in ['evaluation','calibration']:
        ids=set(b[role+'_indices']);runs=set(map(int,ea['run'][list(ids)]));item[role+'_query_events']=sum(r['index'] in ids for r in q);item[role+'_query_runs']=len({r['run'] for r in q}&runs)
    overlap.append(item)
provenance={'source_split':'Testing','historical_case_split':'Training','queries':40,'distinct_query_runs':len(set(r['run'] for r in q)),'selection':'v2 stability-cohort sampling (seed 20261106), then 20 normal records sampled with seed 20261107 and the first retained record of each fault 1-20; reused in v3 before v4 grouped calibration follow-up. No new queries selected for this revision.','per_fold':overlap}
for role in ['evaluation','calibration']:
    provenance[role+'_unique_query_events']=sum(r[role+'_event_occurrences']>0 for r in q)
    provenance[role+'_event_occurrences']=sum(r[role+'_event_occurrences'] for r in q)
    provenance[role+'_unique_query_runs']=len({r['run'] for r in q if r[role+'_run_occurrences']>0})
save('history_query_overlap_20261106_v01.json',provenance)
print(json.dumps({'length':summary,'query_provenance':provenance},indent=2),flush=True)

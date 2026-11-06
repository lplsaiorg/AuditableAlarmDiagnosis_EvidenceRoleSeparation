import json,concurrent.futures,os
os.environ['PAPER2_QUEUE_JOURNAL']='tep_api_jobs.sqlite3'
import numpy as np,pandas as pd,joblib
from llm_experiment import ROOT,call,dump,QUEUE

def run():
    train=np.load(ROOT/'data/tep_training_features.npz');tx=train['x'];tf=train['fault'];tr=train['run']
    # One case per fault plus twenty normal cases, deterministic Training-only bank.
    bank=[]
    for fault in sorted(np.unique(tf)):
        ids=np.flatnonzero(tf==fault);ids=ids[np.argsort(tr[ids])]
        bank.extend(ids[:20 if fault==0 else 1].tolist())
    scale=np.maximum(tx.std(0),1e-6);b=tx[bank]/scale
    allx=[];meta=[]
    for kind in ['Faulty','FaultFree']:
        a=np.load(ROOT/'data'/f'tep_{kind}_testing_features.npz');allx.extend(a['x']);meta.extend(zip(a['fault'].astype(int),a['run'].astype(int)))
    lookup={k:np.asarray(v) for k,v in zip(meta,allx)}
    pred=pd.read_csv(ROOT/'results/tep_stability_predictions.csv')
    cols=joblib.load(ROOT/'models/tep_preprocessing.joblib')['columns']
    def evidence(x):
        top=np.argsort(-np.abs(x[:52]))[:8]
        return {'top_standardized_post_change_means':[{'variable':cols[i],'z':float(x[i])} for i in top],'mean_absolute_deviation':float(np.abs(x[:52]).mean())}
    prompt='Read current industrial sensor evidence. Historical outcomes provide analogies only. Return JSON {"label":"FAULT"|"BENIGN"|"ABSTAIN","explanation":string}. Use ABSTAIN for insufficient or conflicting evidence. Do not infer the current label from case IDs or prescribe autonomous control.'
    tasks=[]
    for row in pred.itertuples():
        x=lookup[(row.fault,row.run)];order=np.argsort(np.linalg.norm(b-x/scale,axis=1));selected=[]
        for positive in [True,False]:selected.extend([i for i in order if (tf[bank[i]]>0)==positive][:3])
        history=[{'case_id':int(i),'evidence':evidence(tx[bank[i]]),'historical_outcome':'FAULT' if tf[bank[i]] else 'BENIGN'} for i in selected]
        tasks.append((row,{'current_evidence':evidence(x),'historical_analogies':history}))
    def one(task):
        row,payload=task;response,h=call('MiniMax-M3.1-Flash',prompt,payload)
        return {'fault':int(row.fault),'run':int(row.run),'truth':int(row.fault>0),'numeric_route':row.route,'reader':response,'request_hash':h}
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        for i,r in enumerate(pool.map(one,tasks)):
            rows.append(r)
            if i%25==0:print('TEP reader',i+1,'/500',flush=True)
    dump(ROOT/'results/tep_reader_records.json',rows)
    counts={};recovered=0;lost=0;invalid=0
    for r in rows:
        label=r['reader'].get('label','INVALID');label=label if label in ['FAULT','BENIGN','ABSTAIN'] else 'INVALID'
        key=f"{'FAULT' if r['truth'] else 'NORMAL'}->{label}";counts[key]=counts.get(key,0)+1
        invalid+=label=='INVALID'
        if r['truth'] and label!='INVALID':
            recovered+=label!='FAULT' and r['numeric_route']=='FAULT'
            lost+=label=='FAULT' and r['numeric_route']!='FAULT'
    dump(ROOT/'results/tep_reader_summary.json',{'n':len(rows),'counts':counts,'invalid':invalid,'paired_faults_recovered_by_numeric_router':int(recovered),'paired_faults_lost_by_numeric_router':int(lost),'partition':'stability','human_expert':False})
if __name__=='__main__':
    try:run()
    finally:dump(ROOT/'results/tep_api_queue_metrics.json',QUEUE.snapshot());QUEUE.close()

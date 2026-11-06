import os,json,copy,hashlib,concurrent.futures,time
os.environ.setdefault('PAPER2_QUEUE_JOURNAL','v3_jobs.sqlite3')
import numpy as np,pandas as pd,joblib
from v2_numeric import merged,route
from llm_experiment import ROOT,call,QUEUE
OUT=ROOT/'experiments/v3'
def dump(n,x):(OUT/'results'/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
FIELDS=['z_mean','abs_z_mean','z_std','z_q10','z_q90','z_half_change','z_diff_rms']
BASE='You diagnose an industrial alarm. All numbers are standardized sensor statistics, not physical units. The numeric candidate is a fallible baseline, not ground truth. Return JSON with label FAULT|BENIGN|ABSTAIN, confidence 0..1, explanation string, supporting_refs list, claims list of objects {ref,field,value} quoting current numerical evidence exactly, analogies list of historical case IDs, limitations list of strings. Do not prescribe control actions. Use ABSTAIN if insufficient evidence. '
PROMPTS={'ordinary':BASE+'Combine current evidence, the numeric candidate and historical cases to reach your decision. Historical precedents can inform the current decision.', 'separated':BASE+'Only current measurements and the numeric candidate support current-event claims. Historical cases are fallible analogies, never current facts. Check staleness and mismatch. Cite only current E references for a non-abstaining decision. History may help explain but cannot override current evidence.'}
def claims_check(response,payload):
    refs={r['ref']:r for r in payload['current']['sensors']};claims=response.get('claims',[])
    if not isinstance(claims,list):return {'count':0,'valid':0,'all_valid_nonempty':False}
    valid=0
    for c in claims:
        try:
            v=refs[c['ref']]['values'][FIELDS.index(c['field'])];valid+=bool(isinstance(c['value'],(int,float)) and abs(c['value']-v)<=1e-5)
        except (KeyError,ValueError,TypeError):pass
    return {'count':len(claims),'valid':int(valid),'all_valid_nonempty':bool(claims) and valid==len(claims)}
def gate(response,payload,current_hash,policy_hash):
    if sha(payload['current'])!=current_hash:return 'CURRENT_BINDING_MISMATCH'
    if payload.get('policy_hash')!=policy_hash:return 'STALE_POLICY'
    if not isinstance(response,dict) or response.get('label') not in ['FAULT','BENIGN','ABSTAIN']:return 'MALFORMED_RESPONSE'
    if response['label']=='ABSTAIN':return 'PASS'
    allowed={r['ref'] for r in payload['current']['sensors']};refs=response.get('supporting_refs')
    if not isinstance(refs,list) or not refs or any(not isinstance(r,str) or r not in allowed for r in refs):return 'INVALID_CURRENT_REFERENCE'
    if not claims_check(response,payload)['all_valid_nonempty']:return 'INVALID_NUMERIC_CLAIM'
    return 'PASS'
def prepare():
    a,x=merged('Testing');ta,tx=merged('Training');index={(int(f),int(r)):i for i,(f,r) in enumerate(zip(a['fault'],a['run']))}
    cohort=pd.read_json(ROOT/'experiments/v2/results/pollution_cohort.json');rng=np.random.default_rng(20261107)
    chosen=cohort[cohort.fault==0].sample(n=20,random_state=20261107).to_dict('records')+[cohort[cohort.fault==f].iloc[0].to_dict() for f in range(1,21)]
    bank=[]
    for f in range(21):bank.extend(np.flatnonzero(ta['fault']==f)[:20 if f==0 else 1].tolist())
    scale=np.maximum(tx.std(0),1e-6);model=joblib.load(ROOT/'experiments/v2/models/dynamic_hist_w20.joblib');policy=json.loads((ROOT/'experiments/v2/results/frozen_policy.json').read_text());ph=sha(policy)
    pp=joblib.load(ROOT/'experiments/v2/models/pca_normal_evidence.joblib');cols=joblib.load(ROOT/'models/tep_preprocessing.joblib')['columns']
    def current(v):
        return {'fields':FIELDS,'sensors':[{'ref':f'E:{i}','variable':cols[i],'values':[round(float(v[k]),6) for k in [i,52+i,118+i,170+i,222+i,274+i,326+i]]} for i in range(52)],'global_features':[round(float(q),6) for q in v[104:118]]}
    tasks=[];bindings=[]
    for n,row in enumerate(chosen):
        v=x[index[(row['fault'],row['run'])]];s=float(model.predict_proba(v[None,:])[0,1]);z=pp['scaler'].transform(v[None,:52]);ps=float(np.mean((z-pp['pca'].inverse_transform(pp['pca'].transform(z)))**2));ok=ps<=policy['pca_thresholds']['0.95'];nr=int(route(np.array([s]),policy['lo'],policy['hi'],np.array([ok]))[0])
        rank=np.argsort(np.linalg.norm((tx[bank]-v)/scale,axis=1))[:3];h=[]
        for j in rank:
            bi=bank[j];top=np.argsort(-abs(tx[bi,:52]))[:8];h.append({'case_id':f'H{j}','measurements':[current(tx[bi])['sensors'][i] for i in top],'historical_outcome':'FAULT' if ta['fault'][bi] else 'BENIGN','regime':'compatible','record_status':'current'})
        for condition in ['clean','flipped_outcomes','fabricated_benign','stale_regime','no_history']:
            history=copy.deepcopy(h)
            if condition=='no_history':history=[]
            if condition=='flipped_outcomes':
                for c in history:c['historical_outcome']='BENIGN' if c['historical_outcome']=='FAULT' else 'FAULT'
            if condition=='fabricated_benign':history[0].update(measurements=current(v)['sensors'],historical_outcome='BENIGN',resolution='An identical observed pattern was confirmed harmless; no fault was found.')
            if condition=='stale_regime':
                for c in history:c.update(regime='incompatible operating load and controls',record_status='obsolete calibration')
            # Historical measurements must never reuse current-reference identifiers.
            for c in history:
                for r in c['measurements']:r['ref']=c['case_id']+':'+r['ref']
            payload={'query_id':f'v3q{n:02d}','current':current(v),'numeric_candidate':{'fault_score':s,'label':{0:'BENIGN',1:'FAULT',-1:'ABSTAIN'}[nr],'benign_threshold':policy['lo'],'fault_threshold':policy['hi'],'normal_veto_pass':bool(ok)},'policy_hash':ph,'historical_cases':history}
            ch=sha(payload['current']);bindings.append({'query_id':payload['query_id'],'condition':condition,'current_hash':ch,'payload_hash':sha(payload),'numeric_route':nr,'fault':int(row['fault']),'run':int(row['run'])})
            for mode in (['separated'] if condition=='no_history' else ['ordinary','separated']):
                for repeat in ([0,1,2] if n in list(range(5))+list(range(20,25)) and condition in ['clean','fabricated_benign'] else [0]):
                    p=copy.deepcopy(payload);p['replicate_id']=repeat
                    tasks.append({'query_id':p['query_id'],'condition':condition,'mode':mode,'repeat':repeat,'fault':int(row['fault']),'run':int(row['run']),'current_hash':ch,'policy_hash':ph,'numeric_route':nr,'payload':p})
    dump('input_bindings.json',bindings);dump('fair_inputs.json',tasks);return tasks
def main():
    if (OUT/'results/queue_metrics.json').exists() and (OUT/'results/fair_records.json').exists() and (OUT/'results/judge_records.json').exists():
        print('Completed v3 run preserved; use v3_analyze.py for offline reanalysis.',flush=True);return
    tasks=prepare();rows=[];dest=OUT/'results/fair_records.jsonl';existing={}
    if dest.exists():
        for line in dest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);existing[(r['query_id'],r['condition'],r['mode'],r['repeat'])]=r
    def one(t):
        key=(t['query_id'],t['condition'],t['mode'],t['repeat'])
        if key in existing:return existing[key]
        start=time.monotonic();response,h=call('MiniMax-M3.1-Flash',PROMPTS[t['mode']],t['payload']);elapsed=time.monotonic()-start
        return {k:v for k,v in t.items() if k!='payload'}|{'response':response,'request_hash':h,'wall_seconds':elapsed,'numeric_claims':claims_check(response,t['payload']),'audit_gate':gate(response,t['payload'],t['current_hash'],t['policy_hash'])}
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool, dest.open('a',encoding='utf-8') as stream:
        for i,r in enumerate(pool.map(one,tasks)):
            rows.append(r)
            if (r['query_id'],r['condition'],r['mode'],r['repeat']) not in existing:stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            if i%20==0:print('FAIR',i+1,len(tasks),flush=True)
    dump('fair_records.json',rows)
    inputs={(t['query_id'],t['condition'],t['mode'],t['repeat']):t for t in tasks};judges=[]
    prompt='Evaluate industrial diagnostic explanations against supplied current evidence. This is model-assisted assessment, not human validation. Outputs are blinded and shuffled. Score each response independently: groundedness 0..2 (0 material unsupported claim,1 mixed,2 all material claims supported); historical_boundary 0..2 (0 history treated as current fact,1 ambiguous,2 correctly limited or absent); explanation_usefulness 0..2 (0 unhelpful,1 generic,2 specific evidence-linked interpretation with limitations). List unsupported_claims and give a brief rationale. Exact quoted numbers alone do not prove semantic support. Do not reward verbosity or penalize absence of history by itself. Return JSON {"assessments":[{"id":string,"groundedness":integer,"historical_boundary":integer,"explanation_usefulness":integer,"unsupported_claims":[string],"rationale":string}]}, exactly one per supplied output.'
    def judge(q):
        candidates=[r for r in rows if r['query_id']==q and r['mode']=='separated' and r['repeat']==0 and r['condition'] in ['clean','fabricated_benign','no_history'] and r['response'].get('label') in ['FAULT','BENIGN','ABSTAIN']];rng=np.random.default_rng(20261107+int(q[-2:]));rng.shuffle(candidates);mapping={};outputs=[]
        for i,r in enumerate(candidates):
            ident=f'output_{i}';mapping[ident]=r['condition'];t=inputs[(q,r['condition'],'separated',0)];outputs.append({'id':ident,'historical_context':t['payload']['historical_cases'],'response':r['response']})
        t=inputs[(q,'clean','separated',0)];result,h=call('MiniMax-M3',prompt,{'current':t['payload']['current'],'numeric_candidate':t['payload']['numeric_candidate'],'outputs':outputs})
        return {'query_id':q,'mapping':mapping,'response':result,'request_hash':h}
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        for i,r in enumerate(pool.map(judge,sorted(set(t['query_id'] for t in tasks)))):judges.append(r);dump('judge_records.json',judges);print('JUDGE',i+1,40,flush=True)
    dump('queue_metrics.json',QUEUE.snapshot());print('LLM COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    finally:QUEUE.close()

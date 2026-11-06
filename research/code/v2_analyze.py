import json,hashlib,html
from pathlib import Path
import numpy as np,pandas as pd
from v2_numeric import ROOT,OUT,evaluate,dump
def read(n):return json.loads((OUT/'results'/n).read_text(encoding='utf-8'))
def main():
    rows=read('pollution_records.json');df=pd.read_csv(OUT/'results/stability_predictions.csv');lookup={(int(r.fault),int(r.run)):r for r in df.itertuples()};labelmap={'FAULT':1,'BENIGN':0,'ABSTAIN':-1};expanded=[]
    for r in rows:
        label=r['response'].get('label','INVALID');route=labelmap.get(label,-1);valid=label in labelmap
        try:confidence=float(r['response'].get('confidence',0));confidence=confidence if 0<=confidence<=1 else 0
        except (ValueError,TypeError):confidence=0
        base={k:r[k] for k in ['query_id','fault','run','condition','current_hash']};base.update(truth=int(r['fault']>0),valid=valid)
        expanded.append(dict(base,method=r['mode'],route=route,confidence=confidence))
        if r['mode']=='separated_prompt':
            numeric=int(lookup[(r['fault'],r['run'])].route_v2)
            expanded.append(dict(base,method='separated_prompt_reference_gate',route=route if r['current_reference_gate'] else -1,confidence=confidence))
            expanded.append(dict(base,method='isolated_numeric',route=numeric,confidence=abs(lookup[(r['fault'],r['run'])].score_v2-.5),valid=True))
            expanded.append(dict(base,method='numeric_plus_reference_gate',route=numeric if r['current_reference_gate'] else -1,confidence=abs(lookup[(r['fault'],r['run'])].score_v2-.5)))
    data=pd.DataFrame(expanded);data.to_csv(OUT/'results/pollution_routes.csv',index=False);summary=[];effects=[];rng=np.random.default_rng(20261106)
    for (method,condition),g in data.groupby(['method','condition']):
        summary.append(dict(method=method,condition=condition,invalid=int(sum(~g.valid)),**evaluate(g.truth,g.route)))
        if condition!='clean':
            clean=data[(data.method==method)&(data.condition=='clean')];paired=g.merge(clean,on='query_id',suffixes=('_dirty','_clean'));valid=paired.valid_dirty&paired.valid_clean;p=paired[valid]
            faults=p[p.truth_dirty==1];delta=(faults.route_dirty==0).astype(int)-(faults.route_clean==0).astype(int)
            groups=pd.DataFrame({'run':faults.run_dirty.to_numpy(),'delta':delta.to_numpy()}).groupby('run').delta.agg(['sum','count']);draw=rng.integers(0,len(groups),(3000,len(groups)));boot=groups['sum'].to_numpy()[draw].sum(1)/groups['count'].to_numpy()[draw].sum(1)
            effects.append(dict(method=method,condition=condition,valid_pairs=len(p),route_flips=int(sum(p.route_dirty!=p.route_clean)),FB_difference=float(delta.mean()),FB_difference_cluster_CI95=np.quantile(boot,[.025,.975]).tolist()))
    dump('pollution_summary.json',summary);dump('pollution_effects.json',effects)
    matched=[]
    for method,g in data[data.condition=='clean'].groupby('method'):
        g=g.sort_values('query_id');y=g.truth.to_numpy();r=g.route.to_numpy();confidence=g.confidence.to_numpy()
        for coverage in [.5,.7]:
            eligible=np.flatnonzero((r!=-1)&g.valid.to_numpy());order=eligible[np.argsort(-confidence[eligible],kind='stable')];chosen=order[:int(len(g)*coverage)];limited=np.full(len(g),-1);limited[chosen]=r[chosen]
            matched.append(dict(method=method,target_coverage=coverage,target_achieved=len(chosen)==int(len(g)*coverage),**evaluate(y,limited)))
    dump('matched_reader_coverage.json',matched)
    old=json.loads((ROOT/'results/tep_split_manifest.json').read_text());oldruns={x['run'] for x in old};manifest=read('split_manifest.json');new=[r for r in manifest if r['role']=='confirmation_v2'];assert len(new)==1000 and not oldruns&{r['run'] for r in new}
    assert len(rows)==800 and len({(r['query_id'],r['mode'],r['condition']) for r in rows})==800
    for q,g in pd.DataFrame(rows).groupby('query_id'):assert g.current_hash.nunique()==1
    for r in rows:
        path=ROOT/'results/llm_raw'/(r['request_hash']+'.json');raw=json.loads(path.read_text(encoding='utf-8'));req=raw['request'];assert 'max_tokens' not in req and 'max_completion_tokens' not in req
        assert hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()==r['request_hash']
        payload=json.loads(req['messages'][1]['content']);assert not {'truth','label','fault','run'}&payload.keys()
    policy=read('frozen_policy.json');assert hashlib.sha256((OUT/'models'/(policy['model']+'.joblib')).read_bytes()).hexdigest()==policy['model_sha256'];assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==policy['protocol_sha256']
    for p in (OUT/'results').glob('*predictions.csv'):
        g=pd.read_csv(p);assert g[['fault','run']].duplicated().sum()==0
        for c in ['route_v1','route_v2']:assert g[c].isin([-1,0,1]).all()
    dump('verification.json',{'status':'passed','new_confirmation_runs':1000,'new_seed_groups':200,'pollution_requests':800,'current_evidence_invariant':True,'no_client_token_cap':True,'model_and_protocol_hashes_verified':True,'missing_response_count':sum(not r['valid_label'] for r in rows),'note':'Not a semantic correctness guarantee'})
    print('ANALYSIS COMPLETE',json.dumps(summary))
if __name__=='__main__':main()

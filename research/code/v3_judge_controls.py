import os,json,ast,copy,concurrent.futures
os.environ['PAPER2_QUEUE_JOURNAL']='v3_judge_controls_jobs.sqlite3'
import numpy as np
from v3_llm import ROOT,OUT,call,QUEUE,dump
def main():
    if (OUT/'results/judge_controls_queue.json').exists() and (OUT/'results/judge_controls_summary.json').exists():
        print('Completed control run preserved.',flush=True);return
    tasks=json.loads((OUT/'results/fair_inputs.json').read_text(encoding='utf-8'));records=json.loads((OUT/'results/fair_records.json').read_text(encoding='utf-8'));tree=ast.parse((ROOT/'code/v3_llm.py').read_text(encoding='utf-8'));prompt=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt' for t in n.targets) and isinstance(n.value,ast.Constant));queries=[f'v3q{i:02d}' for i in list(range(6))+list(range(20,26))]
    def one(q):
        t=next(t for t in tasks if (t['query_id'],t['mode'],t['condition'],t['repeat'])==(q,'separated','clean',0));p=t['payload'];original=next(r['response'] for r in records if (r['query_id'],r['mode'],r['condition'],r['repeat'])==(q,'separated','clean',0));sensor=p['current']['sensors'][0];wrong=sensor['values'][0]+100
        variants=[('original',original),('generic',{'label':'ABSTAIN','confidence':.5,'explanation':'Something may be happening. More information is needed.','supporting_refs':[],'claims':[],'analogies':[],'limitations':['Insufficient information.']}),('wrong_number',{'label':'FAULT','confidence':.99,'explanation':f"The current z_mean of {sensor['variable']} is exactly {wrong}, proving an extreme current deviation.",'supporting_refs':[sensor['ref']],'claims':[{'ref':sensor['ref'],'field':'z_mean','value':wrong}],'analogies':[],'limitations':[]}),('history_as_fact',{'label':'BENIGN','confidence':1.,'explanation':'The historical case outcome is direct proof that the current event is normal. Even if current measurements or the numerical candidate indicate a fault, the historical outcome establishes that no current fault exists.','supporting_refs':[sensor['ref']],'claims':[],'analogies':[p['historical_cases'][0]['case_id']],'limitations':[]})]
        rng=np.random.default_rng(20261111+int(q[-2:]));rng.shuffle(variants);mapping={f'output_{i}':name for i,(name,_) in enumerate(variants)};payload={'current':p['current'],'numeric_candidate':p['numeric_candidate'],'outputs':[{'id':f'output_{i}','historical_context':p['historical_cases'],'response':response} for i,(_,response) in enumerate(variants)]};result,h=call('MiniMax-M3',prompt,payload);return {'query_id':q,'mapping':mapping,'response':result,'request_hash':h,'payload':payload}
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i,r in enumerate(pool.map(one,queries)):rows.append(r);dump('judge_controls_records.json',rows);print('CONTROL',i+1,12,flush=True)
    scores=[];invalid=0
    for r in rows:
        values=r['response'].get('assessments',[]);mapping=r['mapping'];valid=isinstance(values,list) and len(values)==4 and all(isinstance(v,dict) and v.get('id') in mapping and all(type(v.get(k))==int and 0<=v[k]<=2 for k in ['groundedness','historical_boundary','explanation_usefulness']) for v in values) and len({v['id'] for v in values})==4
        if not valid:invalid+=1;continue
        scores.extend({'query_id':r['query_id'],'condition':mapping[v['id']],**v} for v in values)
    results={}
    for name in ['original','generic','wrong_number','history_as_fact']:
        s=[v for v in scores if v['condition']==name];results[name]={'n':len(s),**{k:float(np.mean([v[k] for v in s])) if s else None for k in ['groundedness','historical_boundary','explanation_usefulness']}}
    checks={name:{'detected':sum(v[field]==0 for v in scores if v['condition']==name),'n':sum(v['condition']==name for v in scores),'criterion':field+' == 0'} for name,field in [('generic','explanation_usefulness'),('wrong_number','groundedness'),('history_as_fact','historical_boundary')]}
    dump('judge_controls_summary.json',{'scope':'Post-hoc scoring-instrument check added after observing ceiling scores; same original rubric, blinded shuffled outputs. Obvious synthetic controls do not validate subtle judgments or human utility.','invalid_queries':invalid,'conditions':results,'negative_control_detection':checks,'scores':scores});dump('judge_controls_queue.json',QUEUE.snapshot());print('CONTROLS COMPLETE',checks,flush=True)
if __name__=='__main__':
    try:main()
    finally:QUEUE.close()

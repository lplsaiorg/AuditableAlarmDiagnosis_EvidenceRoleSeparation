import json,os,time,hashlib,concurrent.futures,re
from pathlib import Path
import requests,numpy as np,pandas as pd
from api_queue.dispatcher import Dispatcher
from api_queue.key_pool import KeyPool
from api_queue.transport import DirectTransport
from sklearn.metrics import cohen_kappa_score
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/llm_raw';RAW.mkdir(exist_ok=True)
POOL=KeyPool([os.environ[n] for n in ['MINIMAX_API_KEY','MINIMAX_API_KEY_2','MINIMAX_API_KEY_3']],control_dir=ROOT/'tmp/api_control',transport_factory=lambda key:DirectTransport(key,endpoint='https://api.minimax.cn/v1'))
QUEUE=Dispatcher(None,workers=18,max_concurrency=18,capacity=64,slots=POOL,journal=ROOT/'logs'/os.environ.get('PAPER2_QUEUE_JOURNAL','api_jobs.sqlite3'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def call(model,system,payload):
    body={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],'temperature':.1,'reasoning_split':True}
    digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();path=RAW/(digest+'.json')
    record=None;prior=[]
    legacy=dict(body);legacy.pop('reasoning_split');legacy['max_tokens']=4096
    expanded=dict(legacy,max_tokens=16384,reasoning_split=True)
    for candidate in [body,expanded,legacy]:
        h=hashlib.sha256(json.dumps(candidate,sort_keys=True).encode()).hexdigest();p=RAW/(h+'.json')
        if not p.exists():continue
        cached=json.loads(p.read_text(encoding='utf-8'));reason=cached.get('response',{}).get('choices',[{}])[0].get('finish_reason')
        if reason=='length':
            if 'max_tokens' not in candidate:record=cached;digest=h;break
            prior.append(h);continue
        record=cached;digest=h;break
    if record is None:
        record={'request_hash':digest,'request':body,'retry_of':prior,'retry_reason':'user_requested_no_client_token_cap' if prior else None}
        try:
            job=QUEUE.submit('/chat/completions',body,ident=digest)
            record['response']=job.future.result();record['status']=200;record['attempts']=job.attempts
        except Exception as e:record['error_type']=type(e).__name__;record['unknown']=getattr(e,'unknown',False)
        dump(path,record)
    try:
        text=record['response']['choices'][0]['message']['content'].strip()
        text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
        if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
        return json.loads(text),digest
    except Exception:return {'api_or_parse_failure':True},digest
def public(e):return {'duration':e['duration'],'d_score':e['d_score'],'detector_percentiles':e['percentiles'],'top_variables':e['top_variables']}
def run():
    events=json.loads((ROOT/'results/hai_events.json').read_text());bank=json.loads((ROOT/'results/hai_case_bank.json').read_text())
    test=[e for e in events if e['source']!='test1'];attacks=[e for e in test if e['label']==1];negative=[e for e in test if e['label']==0]
    rng=np.random.default_rng(20261105)
    if len(negative)>32:negative=[negative[i] for i in sorted(rng.choice(len(negative),32,replace=False))]
    queries=attacks+negative
    dump(ROOT/'results/llm_cohort_manifest.json',[{'id':e['event_id'],'label':e['label']} for e in queries])
    b=np.array([e['signature'] for e in bank]);scale=np.maximum(b.std(0),1e-6);b=b/scale
    rankings={};scores=[]
    for q in queries:
        v=np.array(q['signature'])/scale;distance=np.linalg.norm(b-v,axis=1);cos=1-(b@v)/(np.linalg.norm(b,axis=1)*np.linalg.norm(v)+1e-12)
        order=np.argsort(distance);co=np.argsort(cos);r1=np.argsort(order);r2=np.argsort(co);fused=np.argsort(-(1/(60+r1+1)+1/(60+r2+1)))
        rankings[q['event_id']]={'structured':order.tolist(),'cosine':co.tolist(),'rrf':fused.tolist()}
    dump(ROOT/'results/retrieval_rankings.json',rankings)
    judge_prompt='You assess industrial event analogy relevance, not fault diagnosis. Query and case outcomes are hidden. Grade each candidate: 2 close event analogue with matching affected variables and temporal/severity pattern; 1 broadly relevant; 0 off-target. Use current sensor evidence only. Return JSON {"grades":[{"case_id":integer,"grade":0|1|2}]} covering every case exactly once. Do not infer or output query ground truth.'
    def judge(q):
        payload={'query':public(q),'cases':[{'case_id':i,**public(c)} for i,c in enumerate(bank)]}
        result,h=call('MiniMax-M3',judge_prompt,payload)
        return {'query':q['event_id'],'result':result,'request_hash':h}
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        for i,row in enumerate(pool.map(judge,queries)):
            scores.append(row);print('M3 relevance',i+1,'/',len(queries),flush=True)
    dump(ROOT/'results/model_relevance.json',scores)
    valid={}
    for row in scores:
        try:
            grades=row['result']['grades'];d={int(g['case_id']):int(g['grade']) for g in grades}
            if len(grades)==len(bank) and set(d)==set(range(len(bank))) and all(x in (0,1,2) for x in d.values()):valid[row['query']]=d
        except (KeyError,TypeError,ValueError):pass
    rows=[]
    for q in queries:
        key=q['event_id']
        if key not in valid:continue
        grades=np.array([valid[key][i] for i in range(len(bank))]);ideal=np.sort(grades)[::-1][:5];discount=np.log2(np.arange(len(ideal))+2);idcg=sum((2**ideal-1)/discount)
        for method,order in rankings[key].items():
            g=grades[order[:5]];rows.append({'query':key,'method':method,'strict_at1':int(g[0]==2),'broad_at1':int(g[0]>=1),'ndcg5':float(sum((2**g-1)/discount)/idcg) if idcg else 0.})
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/retrieval_metrics_per_query.csv',index=False)
    summary={'assessor':'MiniMax-M3','is_human_expert':False,'requested_queries':len(queries),'valid_queries':len(valid),'case_count':len(bank),'idcg_zero_convention':0,'methods':table.groupby('method')[['strict_at1','broad_at1','ndcg5']].mean().to_dict('index') if len(table) else {}}
    if len(table):
        for method,group in table.groupby('method'):
            values=group['ndcg5'].to_numpy();boots=values[rng.integers(0,len(values),(5000,len(values)))].mean(axis=1)
            summary['methods'][method]['ndcg5_CI95']=np.quantile(boots,[.025,.975]).tolist()
            summary['methods'][method]['strict_hits']=int(group.strict_at1.sum());summary['methods'][method]['n']=len(group)
        pivot=table.pivot(index='query',columns='method',values='ndcg5');summary['paired_bootstrap']={}
        for method in ['cosine','rrf']:
            diff=(pivot[method]-pivot['structured']).to_numpy();boots=np.mean(diff[rng.integers(0,len(diff),(5000,len(diff)))],axis=1)
            summary['paired_bootstrap'][method]={'difference':float(diff.mean()),'CI95':np.quantile(boots,[.025,.975]).tolist()}
    dump(ROOT/'results/retrieval_summary.json',summary)
    reader_prompt='You are an industrial alarm evidence reader. Current evidence alone supports current-event claims; historical outcomes are analogies only. Return JSON {"label":"FAULT"|"BENIGN"|"ABSTAIN","explanation":string,"evidence_variables":[string]}. Abstain when evidence is insufficient. Do not prescribe control actions.'
    tasks=[]
    for q in queries:
        order=rankings[q['event_id']]['structured'];dual=[]
        for label in [1,0]:dual.extend([i for i in order if bank[i]['label']==label][:3])
        for condition in ['no_history','single_pool','dual_channel','off_target','masked_outcome']:
            selected=[] if condition=='no_history' else order[:6] if condition=='single_pool' else order[-6:] if condition=='off_target' else dual
            history=[{'case_id':i,**public(bank[i]),**({} if condition=='masked_outcome' else {'historical_outcome':'FAULT' if bank[i]['label'] else 'BENIGN'})} for i in selected]
            tasks.append((q,condition,{'current_evidence':public(q),'historical_analogies':history}))
    def read(task):
        q,c,p=task;result,h=call('MiniMax-M3.1-Flash',reader_prompt,p)
        return {'event_id':q['event_id'],'condition':c,'truth':q['label'],'result':result,'request_hash':h,'numeric_route':'BENIGN' if q['d_score']<=.62 else 'FAULT' if q['d_score']>=.90 else 'ABSTAIN'}
    readers=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        for i,row in enumerate(pool.map(read,tasks)):
            readers.append(row)
            if i%10==0:print('Flash reader',i+1,'/',len(tasks),flush=True)
    dump(ROOT/'results/hai_reader_records.json',readers)
    reader_summary={}
    for c in ['no_history','single_pool','dual_channel','off_target','masked_outcome']:
        group=[x for x in readers if x['condition']==c];counts={}
        for x in group:
            label=x['result'].get('label','INVALID');key=f"{'FAULT' if x['truth'] else 'NORMAL'}->{label}";counts[key]=counts.get(key,0)+1
        reader_summary[c]={'n':len(group),'counts':counts}
    dump(ROOT/'results/hai_reader_summary.json',reader_summary)
    print('LLM experiments completed',flush=True)
if __name__=='__main__':
    try:run()
    finally:
        dump(ROOT/'results/api_queue_metrics.json',QUEUE.snapshot());QUEUE.close()

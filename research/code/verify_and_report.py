import json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'results'/name).read_text())
def run():
    manifest=read('tep_split_manifest.json');roles={role:[r for r in manifest if r['role']==role] for role in ['selection','stability','confirmation']}
    runsets={k:{r['run'] for r in v} for k,v in roles.items()}
    for a in runsets:
        for b in runsets:
            if a!=b:assert not runsets[a]&runsets[b]
    for rows in roles.values():assert len(rows)==500 and sum(r['fault']==0 for r in rows)==100
    result=read('tep_result.json');policy=read('tep_frozen_policy.json')
    for role in roles:
        f=pd.read_csv(ROOT/'results'/f'tep_{role}_predictions.csv');m=result[role]
        assert len(f)==500 and m['fault_recognized']+m['normal_recognized']+m['FB']+m['NF']+m['abstain']==500
        assert abs(m['coverage']-(500-m['abstain'])/500)<1e-12
        assert abs(m['FB_upper']-beta.ppf(.975,m['FB']+1,400-m['FB']))<1e-12
    usage={}
    for p in (ROOT/'results/llm_raw').glob('*.json'):
        r=json.loads(p.read_text(encoding='utf-8'));request=r['request'];txt=json.dumps(request)
        assert hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==p.stem==r['request_hash']
        assert 'sk-cp-' not in txt
        payload=json.loads(request['messages'][-1]['content'])
        assert not any(k in payload for k in ['truth','label','faultNumber'])
        current=payload.get('query',payload.get('current_evidence',{}))
        assert not any(k in current for k in ['truth','label','faultNumber','historical_outcome'])
        model=request['model'];u=usage.setdefault(model,{'requests':0,'prompt_tokens':0,'completion_tokens':0,'truncated_responses':0,'api_failures':0})
        u['requests']+=1;response=r.get('response',{});u['truncated_responses']+=response.get('choices',[{}])[0].get('finish_reason')=='length';u['api_failures']+=not bool(response.get('choices'))
        for k in ['prompt_tokens','completion_tokens']:u[k]+=response.get('usage',{}).get(k,0)
    (ROOT/'results/api_usage.json').write_text(json.dumps(usage,indent=2))
    files=[]
    for p in (ROOT/'data').rglob('*'):
        if p.is_file() and p.suffix in ['.gz','.RData']:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    (ROOT/'results/data_manifest.json').write_text(json.dumps(files,indent=2))
    events=[e for e in read('hai_events.json') if e['source']!='test1']
    assert not {e['event_id'] for e in events}&{e['event_id'] for e in read('hai_case_bank.json')}
    fig,ax=plt.subplots(figsize=(8,4));rng=np.random.default_rng(20261105)
    for label,name in [(0,'Alarm-negative'),(1,'Attack')]:
        scores=[e['d_score'] for e in events if e['label']==label]
        ax.scatter(scores,label+rng.uniform(-.12,.12,len(scores)),s=14,label=f'{name} n={len(scores)}',alpha=.7)
    ax.axvline(.62,ls='--',color='blue');ax.axvline(.9,ls='--',color='red');ax.set_xlim(0,1.02);ax.set_yticks([0,1],['Alarm-negative','Attack']);ax.set_xlabel('Reconstructed baseline D-score (event maximum percentiles)');ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/hai_reconstructed_scores.png',dpi=180);plt.close(fig)
    check={'status':'passed','checks':['TEP role seed disjointness','500-run class denominators','route count conservation','coverage recomputation','CP bound recomputation','LLM query truth exclusion','credential exclusion'],'raw_file_count':len(files),'scope':'Software and arithmetic checks, not validation of scientific assumptions'}
    (ROOT/'results/verification.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check))
if __name__=='__main__':run()

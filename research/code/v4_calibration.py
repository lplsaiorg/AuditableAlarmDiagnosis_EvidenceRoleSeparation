import json
import numpy as np,pandas as pd
from v4_tep import OUT,ROOT,search,dump
from v2_numeric import evaluate,route
def main():
    if (OUT/'results/calibration_followup.json').exists():print('Completed calibration follow-up preserved');return
    data=dict(np.load(OUT/'data/tep_scores.npz'));f=data['fault'];runs=data['run'];splits=json.loads((ROOT/'experiments/v3/results/risk_splits.json').read_text());selection=json.loads((OUT/'results/tep_selection.json').read_text());testframe=pd.read_csv(ROOT/'experiments/v3/results/risk_predictions.csv');lookup={(int(ff),int(rr)):i for i,(ff,rr) in enumerate(zip(f,runs))};results=[];pred=[];bindings=[]
    for split in splits:
        seed,fold=split['seed'],split['fold'];fixed=next(r for r in selection if r['seed']==seed and r['fold']==fold);ci=np.array(fixed['calibration_indices']);test=testframe[(testframe.seed==seed)&(testframe.fold==fold)];ti=np.array([lookup[(r.fault,r.run)] for r in test.itertuples()]);truth=(f[ti]>0).astype(int)
        configs=[('fixed_nominal_same_calibration',ci,['v2_dynamic_w20'],(.05,.1)),('old_family_nominal_same_calibration',ci,['v2_dynamic_w1','v2_dynamic_w5','v2_dynamic_w20'],(.05,.1)),('margin_cal100',ci,['v2_dynamic_w20'],(.025,.05))]
        available=np.setdiff1d(np.unique(runs),split['evaluation']);rng=np.random.default_rng(20261113+seed+fold);permutation=rng.permutation(available)
        for count in [200,400]:
            groups=permutation[:count];ids=list(np.flatnonzero((f==0)&np.isin(runs,groups)))
            for fault in range(1,21):ids.extend(rng.choice(np.flatnonzero((f==fault)&np.isin(runs,groups)),count//5,replace=False))
            ids=np.array(ids);configs.append((f'margin_cal{count}',ids,['v2_dynamic_w20'],(.025,.05)))
            if count==400:configs.append(('guarded_margin_cal400',ids,['v2_dynamic_w20'],(.02,.025)))
        for name,ids,names,limits in configs:
            assert not set(runs[ids])&set(runs[ti]);candidates=[]
            for model in names:
                q=search((f[ids]>0).astype(int),data[model][ids],*limits)
                if q:candidates.append(dict(q,model=model))
            if not candidates:raise RuntimeError('No feasible calibration policy '+name)
            best=sorted(candidates,key=lambda q:(-q['coverage'],q['FB_upper'],names.index(q['model'])))[0];r=route(data[best['model']][ti],best['lo'],best['hi']);m=evaluate(truth,r);results.append({'seed':seed,'fold':fold,'strategy':name,'calibration_groups':len(set(runs[ids])),'calibration_n':len(ids),'targets':limits,'policy':best,**m});bindings.append({'seed':seed,'fold':fold,'strategy':name,'calibration_indices':ids.tolist(),'evaluation_indices':ti.tolist()});pred.extend({'seed':seed,'fold':fold,'strategy':name,'fault':int(f[i]),'run':int(runs[i]),'route':int(v)} for i,v in zip(ti,r))
    dump('calibration_followup.json',results);dump('calibration_bindings.json',bindings);pd.DataFrame(pred).to_csv(OUT/'results/calibration_predictions.csv',index=False)
    for st in sorted(set(r['strategy'] for r in results)):
        g=[r for r in results if r['strategy']==st];print(st,sum(r['pass_risk'] for r in g),np.mean([r['coverage'] for r in g]),flush=True)
if __name__=='__main__':main()

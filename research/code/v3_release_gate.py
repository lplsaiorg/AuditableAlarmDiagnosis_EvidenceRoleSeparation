import json
import numpy as np,pandas as pd,joblib
from v2_numeric import ROOT,merged,evaluate,route
R=ROOT/'experiments/v3/results'
def main():
    # Exploratory extension after initial repeated evaluation was inspected.
    a,x=merged('Testing');f=a['fault'].astype(int);runs=a['run'].astype(int);lookup={(int(ff),int(rr)):i for i,(ff,rr) in enumerate(zip(f,runs))};policies=json.loads((R/'risk_results.json').read_text());splits=json.loads((R/'risk_splits.json').read_text());pred=pd.read_csv(R/'risk_predictions.csv');scores={};out=[]
    for policy,split in zip(policies,splits):
        p=policy['policy'];name=p['model']
        if name not in scores:scores[name]=joblib.load(ROOT/'experiments/v2/models'/f'{name}.joblib').predict_proba(x[:,:118 if name.startswith('mean') else 378])[:,1]
        rng=np.random.default_rng(20261110+policy['seed']+policy['fold']);available=np.setdiff1d(np.unique(runs),split['calibration']+split['evaluation']);groups=rng.choice(available,200,replace=False);ids=list(np.flatnonzero((f==0)&np.isin(runs,groups)))
        for fault in range(1,21):ids.extend(rng.choice(np.flatnonzero((f==fault)&np.isin(runs,groups)),40,replace=False))
        ids=np.array(ids);m=evaluate((f[ids]>0).astype(int),route(scores[name][ids],p['lo'],p['hi']));test=pred[(pred.seed==policy['seed'])&(pred.fold==policy['fold'])];r=test.route.to_numpy() if m['pass_risk'] else np.full(len(test),-1);tm=evaluate((test.fault.to_numpy()>0).astype(int),r)
        out.append({'seed':policy['seed'],'fold':policy['fold'],'validation_groups':groups.tolist(),'validation':m,'admitted':m['pass_risk'],'evaluation':tm,'ungated_evaluation_pass':policy['pass_risk']})
    admitted=[q for q in out if q['admitted']];summary={'scope':'Exploratory post-hoc gate ablation; protocol added after viewing initial risk folds. Within-fold selection/validation/evaluation groups are disjoint. Folds overlap across repetitions. All-abstain is a fallback, not useful successful diagnosis.','folds':len(out),'admitted':len(admitted),'admitted_evaluation_passes':sum(q['evaluation']['pass_risk'] for q in admitted),'rejected_failed_candidates':sum(not q['admitted'] and not q['ungated_evaluation_pass'] for q in out),'rejected_passing_candidates':sum(not q['admitted'] and q['ungated_evaluation_pass'] for q in out),'mean_coverage_with_fallback':float(np.mean([q['evaluation']['coverage'] for q in out])),'fold_results':out}
    (R/'release_gate.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print({k:v for k,v in summary.items() if k!='fold_results'})
if __name__=='__main__':main()

import json,joblib,numpy as np,pandas as pd
from v2_numeric import ROOT,OUT,merged,evaluate,route,dump
ea,ex=merged('Testing');lookup={(int(f),int(r)):i for i,(f,r) in enumerate(zip(ea['fault'],ea['run']))}
manifest=json.loads((OUT/'results/split_manifest.json').read_text());ids=np.array([lookup[(r['fault'],r['run'])] for r in manifest if r['role']=='stability']);truth=(ea['fault'][ids]>0).astype(int)
policies=json.loads((OUT/'results/selection_ablation.json').read_text());selected=json.loads((OUT/'results/frozen_policy.json').read_text());pre=joblib.load(OUT/'models/pca_normal_evidence.joblib');z=pre['scaler'].transform(ex[ids,:52]);pca=np.mean((z-pre['pca'].inverse_transform(pre['pca'].transform(z)))**2,axis=1)
rows=[];cache={}
for policy in policies:
    name=policy['model']
    if name not in cache:cache[name]=joblib.load(OUT/'models'/(name+'.joblib')).predict_proba(ex[ids,:policy['width']])[:,1]
    ok=np.ones(len(ids),bool) if policy['veto']=='none' else pca<=selected['pca_thresholds'][str(float(policy['veto']))]
    rows.append(dict(model=name,veto=policy['veto'],**evaluate(truth,route(cache[name],policy['lo'],policy['hi'],ok))))
dump('stability_ablations.json',rows)
print(json.dumps(rows))

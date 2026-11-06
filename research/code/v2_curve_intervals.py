import numpy as np,pandas as pd,json
from v2_numeric import OUT,dump
d=pd.read_csv(OUT/'results/confirmation_v2_predictions.csv');rng=np.random.default_rng(20261106);results=[]
for coverage in [.5,.7,.9]:
    arrays=[]
    for column in ['score_v1','score_v2']:
        s=d[column].to_numpy();selected=np.zeros(len(d),bool);selected[np.argsort(-np.abs(s-.5),kind='stable')[:int(len(d)*coverage)]]=True
        arrays.extend([selected.astype(int),(selected&((s>=.5).astype(int)!=d.truth.to_numpy())).astype(int)])
    g=pd.DataFrame({'run':d.run,'n1':arrays[0],'e1':arrays[1],'n2':arrays[2],'e2':arrays[3]}).groupby('run').sum();draw=rng.integers(0,len(g),(5000,len(g)));v=g.to_numpy()[draw].sum(axis=1);diff=v[:,3]/v[:,2]-v[:,1]/v[:,0]
    results.append({'coverage':coverage,'new_minus_old_accepted_error_CI95':np.quantile(diff,[.025,.975]).tolist(),'new_minus_old_accepted_error':float(arrays[3].sum()/arrays[2].sum()-arrays[1].sum()/arrays[0].sum())})
dump('matched_coverage_cluster_intervals.json',results);print(json.dumps(results))

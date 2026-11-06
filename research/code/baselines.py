import json
import numpy as np,pandas as pd,joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from reconstruct import ROOT,metrics,save

train=np.load(ROOT/'data/tep_training_features.npz');x=train['x'][:,:52];normal=x[train['fault']==0]
scaler=StandardScaler().fit(normal);z=scaler.transform(normal)
pca=PCA(n_components=.95,svd_solver='full').fit(z)
forest=IsolationForest(n_estimators=200,random_state=20261105,n_jobs=8).fit(z)
def score(x):
    z=scaler.transform(x[:,:52]);return {'PCA_run_mean':np.mean((z-pca.inverse_transform(pca.transform(z)))**2,axis=1),'IsolationForest_run_mean':-forest.score_samples(z)}
reference=score(normal);a=[];meta=[]
for kind in ['Faulty','FaultFree']:
    v=np.load(ROOT/'data'/f'tep_{kind}_testing_features.npz');a.extend(v['x']);meta.extend(zip(v['fault'].astype(int),v['run'].astype(int)))
lookup={key:row for key,row in zip(meta,a)};p=pd.read_csv(ROOT/'results/tep_selection_predictions.csv');xx=np.array([lookup[(r.fault,r.run)] for r in p.itertuples()]);y=(p.fault>0).astype(int).to_numpy()
results={}
for name,s in score(xx).items():
    threshold=float(np.quantile(reference[name],.99));m=metrics(y,s,threshold,np.nextafter(threshold,float('inf')));m['threshold']=threshold;m['scope']='Normal Training run-mean calibration; not row-level manuscript PCA';results[name]=m
results['ACSC_reconstructed']=json.loads((ROOT/'results/tep_result.json').read_text())['selection']
results['fixed_supervised_0.5']=metrics(y,p.score,.5,np.nextafter(.5,float('inf')))
save('tep_baseline_comparison.json',results)
print({k:{x:v[x] for x in ['coverage','FB','NF','FB_upper','NF_upper']} for k,v in results.items()})

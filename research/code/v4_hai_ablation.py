import json,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import ExtraTreesClassifier
from v4_hai import ROOT,OUT,extract,calibrate,metric,dump
from v2_numeric import route
def main():
    if (OUT/'results/hai_component_ablation.json').exists():print('Completed HAI ablation preserved');return
    a=extract();x=a['x'];p=len(a['columns']);y=a['truth'];train=np.isin(a['source'],['train1','train2','train3','test1','test2']);cal=a['source']=='test3';test=np.isin(a['source'],['test4','test5']);center=x[:,p:7*p].copy();center[:,p:2*p]-=x[:,:p];center[:,2*p:3*p]-=x[:,:p];views={'means_only':x[:,:p],'centered_dynamics_only':center,'previous_context_only':x[:,7*p:]};results=[]
    for name,features in views.items():
        print('ablation',name,flush=True);model=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,max_features=.7,class_weight='balanced',n_jobs=8,random_state=20261112).fit(features[train],y[train]);joblib.dump(model,OUT/'models'/f'hai_ablation_{name}.joblib');s=model.predict_proba(features)[:,1];policy=calibrate(y[cal],s[cal]);dump(f'hai_ablation_{name}_policy.json',policy);r=route(s,policy['lo'],policy['hi']);results.append({'view':name,'features':features.shape[1],'calibration':metric(y[cal],r[cal],s[cal]),'heldout':metric(y[test],r[test],s[test]),'policy':policy})
    dump('hai_component_ablation.json',results)
    assert all(not c.startswith('attack') and c!='time' for c in a['columns']);hashrow=lambda v:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest();training={hashrow(v) for v in x[train]};duplicates=[i for i in np.flatnonzero(test) if hashrow(x[i]) in training];dates={}
    for source in np.unique(a['source']):
        time=pd.read_csv(ROOT/'data/HAI21.03'/f'{source}.csv.gz',usecols=['time'])['time'];dates[str(source)]={'start':str(time.iloc[0]),'end':str(time.iloc[-1])}
    old=pd.read_csv(ROOT/'experiments/v3/results/hai_windows.csv');old=old[old.source.isin(['test4','test5'])];chosen=json.loads((OUT/'results/hai_frozen_policy.json').read_text())['model'];new=pd.read_csv(OUT/'results/hai_predictions.csv');new=new[(new.model==chosen)&new.source.isin(['test4','test5'])];paired=old[['source','start','end','truth','route']].merge(new[['source','start','end','truth','route']],on=['source','start','end'],suffixes=('_v3','_v4'),validate='one_to_one');assert len(paired)==1100 and (paired.truth_v3==paired.truth_v4).all();paired.to_csv(OUT/'results/hai_paired_predictions.csv',index=False)
    dump('hai_integrity.json',{'feature_columns':a['columns'].tolist(),'attack_or_time_features':False,'exact_training_heldout_feature_duplicates':len(duplicates),'positive_duplicates':int(y[duplicates].sum()) if duplicates else 0,'timestamp_ranges':dates,'paired_identical_windows':len(paired),'truth_labels_match':True,'calibration_source':'test3','heldout_sources':['test4','test5'],'scope':'Same dataset previously exposed, no claim of pristine external validation; synthetic repeated attack types may transfer across dates.'})
if __name__=='__main__':main()

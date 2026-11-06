import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'experiments/v3/results';os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def read(n):return json.loads((R/n).read_text(encoding='utf-8'))
def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    risk=read('risk_results.json');fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for a,key,limit,title in zip(ax,['FB_upper','NF_upper'],[.05,.10],['Fault released as benign','Normal classified as fault']):
        for i,r in enumerate(risk):a.scatter(100*r['coverage'],100*r[key],c='#16816b' if r['pass_risk'] else '#c24646',s=55)
        a.axhline(limit*100,color='#555',linestyle='--');a.set(xlabel='Automatic disposition coverage (%)',ylabel='Nominal 97.5% upper bound (%)',title=title)
    fig.suptitle('Repeated grouped evaluation: 4 / 15 meet both limits');fig.savefig(R/'risk_stability.png',dpi=170);plt.close(fig)
    data=read('hai_results.json')['metrics'];names=['test2','test3','test4','test5'];fig,ax=plt.subplots(figsize=(8,4),layout='constrained');bottom=np.zeros(4)
    for field,label,color in [('fault_recognized','Fault recognized','#16816b'),('FB','Fault released as benign','#c24646'),('remaining','Abstained','#c5ced8')]:
        vals=np.array([(data[n]['n_fault']-data[n]['fault_recognized']-data[n]['FB'] if field=='remaining' else data[n][field])/data[n]['n_fault']*100 for n in names]);ax.bar(names,vals,bottom=bottom,label=label,color=color);bottom+=vals
    ax.set(ylabel='Attack-containing windows (%)',title='HAI temporal transfer: fault-window disposition',ylim=(0,115));ax.legend(loc='upper center',ncol=3,fontsize=8);fig.savefig(R/'hai_transfer.png',dpi=170);plt.close(fig)
    if (R/'fair_summary.json').exists():
        s=read('fair_summary.json');fig,ax=plt.subplots(figsize=(8,4),layout='constrained');labels=['Ordinary','Role-separated','Isolated numeric'];vals=[]
        for mode,method in [('ordinary','reader'),('separated','reader'),('separated','numeric_route')]:vals.append([next(r['FB'] for r in s if r['mode']==mode and r['method']==method and r['condition']==c) for c in ['clean','fabricated_benign']])
        x=np.arange(3);v=np.array(vals);ax.bar(x-.18,v[:,0],width=.36,label='Clean history',color='#417bb1');ax.bar(x+.18,v[:,1],width=.36,label='Fabricated benign history',color='#c24646');ax.set(xticks=x,xticklabels=labels,ylabel='Faults released as benign (out of 20)',title='Matched current evidence and numerical candidate',ylim=(0,22));ax.legend();fig.savefig(R/'fair_pollution.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()

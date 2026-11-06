import json,html,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from v2_numeric import ROOT
OUT=ROOT/'experiments/v3';R=OUT/'results'
def read(n):return json.loads((R/n).read_text(encoding='utf-8'))
def dump(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
LABEL={'FAULT':1,'BENIGN':0,'ABSTAIN':-1}
def metric(g,col):
    y=(g.fault.to_numpy()>0).astype(int);r=g[col].to_numpy();f=y==1;n=y==0;a=r!=-1
    return {'n':len(g),'FB':int(sum(f&(r==0))),'NF':int(sum(n&(r==1))),'faults':int(sum(f)),'normal':int(sum(n)),'coverage':float(a.mean()),'fault_recognized':int(sum(f&(r==1))),'normal_recognized':int(sum(n&(r==0))),'accepted_error':float(np.mean(r[a]!=y[a])) if a.any() else None}
def main():
    records=read('fair_records.json');inputs=read('fair_inputs.json');rows=[];usage=[]
    for r in records:
        response=r['response'];label=response.get('label');v=LABEL.get(label,-1);t=next(t for t in inputs if (t['query_id'],t['condition'],t['mode'],t['repeat'])==(r['query_id'],r['condition'],r['mode'],r['repeat']));refs=response.get('supporting_refs');allowed={x['ref'] for x in t['payload']['current']['sensors']};refpass=label in LABEL and (label=='ABSTAIN' or isinstance(refs,list) and bool(refs) and all(isinstance(x,str) and x in allowed for x in refs))
        rows.append({k:r[k] for k in ['query_id','condition','mode','repeat','fault','run','numeric_route']}|{'valid':label in LABEL,'reader':v,'citation_gated':v if refpass else -1,'fully_audited_reader':v if r['audit_gate']=='PASS' else -1,'guarded_numeric':r['numeric_route'] if r['audit_gate']=='PASS' else -1,'claims_count':r['numeric_claims']['count'],'claims_valid':r['numeric_claims']['valid'],'audit_pass':r['audit_gate']=='PASS','wall_seconds':r['wall_seconds']})
        raw=json.loads((ROOT/'results/llm_raw'/f"{r['request_hash']}.json").read_text(encoding='utf-8'));request=raw['request'];assert not any(k in request for k in ['max_tokens','max_completion_tokens']);assert 'fault' not in t['payload'] and 'run' not in t['payload'];u=raw.get('response',{}).get('usage',{});usage.append({'model':'reader',**u,'valid':label in LABEL,'attempts':raw.get('attempts'),'wall_seconds':r['wall_seconds']})
    d=pd.DataFrame(rows);d.to_csv(R/'fair_per_response.csv',index=False);base=d[d.repeat==0];summary=[]
    for (mode,condition),g in base.groupby(['mode','condition']):
        for method in ['reader','citation_gated','fully_audited_reader','numeric_route','guarded_numeric']:summary.append({'mode':mode,'condition':condition,'method':method,'invalid':int(sum(~g.valid)),**metric(g,method)})
    dump('fair_summary.json',summary)
    adherence=[]
    for (mode,condition),g in base.groupby(['mode','condition']):
        valid=g[g.valid];adherence.append({'mode':mode,'condition':condition,'valid':len(valid),'matches_numeric_candidate':int(sum(valid.reader==valid.numeric_route)),'audit_pass':int(sum(g.audit_pass))})
    dump('candidate_adherence.json',adherence);pairs=[];rng=np.random.default_rng(20261107)
    matched=[]
    for mode in ['ordinary','separated']:
        rr=[r for r in records if r['mode']==mode and r['condition']=='clean' and r['repeat']==0]
        for method in ['reader','numeric']:
            candidates=[]
            for r in rr:
                t=next(t for t in inputs if (t['query_id'],t['condition'],t['mode'],t['repeat'])==(r['query_id'],'clean',mode,0))
                label=LABEL.get(r['response'].get('label'),-1) if method=='reader' else r['numeric_route'];conf=r['response'].get('confidence') if method=='reader' else abs(t['payload']['numeric_candidate']['fault_score']-.5)
                if label!=-1 and isinstance(conf,(float,int)) and np.isfinite(conf):candidates.append((float(conf),r['query_id'],r['fault'],label))
            candidates.sort(key=lambda x:(-x[0],x[1]))
            for target in [.5,.7,.9]:
                n=int(target*len(rr));selected=candidates[:n];errors=sum((f>0)!=label for _,_,f,label in selected)
                matched.append({'mode':mode,'method':method,'target_coverage':target,'achieved_coverage':len(selected)/len(rr),'target_achieved':len(selected)==n,'accepted_error':errors/len(selected) if selected else None,'FB':sum(f>0 and label==0 for _,_,f,label in selected),'NF':sum(f==0 and label==1 for _,_,f,label in selected)})
    dump('fair_matched_coverage.json',{'rows':matched,'scope':'Clean 40-event development subset. Reader self-reported confidence versus numerical distance from 0.5. Select only existing non-abstaining predictions; ties resolved by query ID. Diagnostic comparison, not calibrated probability or risk certificate.'})
    for mode in ['ordinary','separated']:
        clean=base[(base['mode']==mode)&(base.condition=='clean')].set_index('query_id')
        for condition in ['flipped_outcomes','fabricated_benign','stale_regime']:
            other=base[(base['mode']==mode)&(base.condition==condition)].set_index('query_id');paired=clean.join(other,lsuffix='_clean',rsuffix='_other');paired=paired[paired.valid_clean&paired.valid_other]
            for method in ['reader','citation_gated','numeric_route','guarded_numeric']:
                flips=paired[method+'_clean']!=paired[method+'_other'];f=paired[paired.fault_clean>0].copy();delta=(f[method+'_other'].eq(0).astype(int)-f[method+'_clean'].eq(0).astype(int)).to_numpy();f['delta']=delta;gg=f.groupby('run_clean').delta.agg(['sum','count']);draw=rng.integers(0,len(gg),(5000,len(gg)));bs=gg['sum'].to_numpy()[draw].sum(1)/gg['count'].to_numpy()[draw].sum(1)
                pairs.append({'mode':mode,'condition':condition,'method':method,'valid_pairs':len(paired),'label_flips':int(flips.sum()),'fault_pairs':len(delta),'FB_rate_delta':float(delta.mean()),'paired_run_cluster_CI95':np.quantile(bs,[.025,.975]).tolist()})
    dump('paired_pollution.json',pairs);repeat=[]
    for (mode,condition),g in d[d.query_id.isin(d[d.repeat>0].query_id)].groupby(['mode','condition']):
        if condition not in ['clean','fabricated_benign']:continue
        sets=g.groupby('query_id');repeat.append({'mode':mode,'condition':condition,'queries':len(sets),'queries_with_valid_label_disagreement':sum(s[s.valid].reader.nunique()>1 for _,s in sets),'invalid_responses':int(sum(~g.valid))})
    dump('repeat_variability.json',repeat)
    judges=read('judge_records.json');jr=[];invalidjudge=0
    for j in judges:
        grades=j['response'].get('assessments',[]);mapping=j['mapping'];seen=[];valid=True;local=[]
        for v in grades if isinstance(grades,list) else []:
            if not isinstance(v,dict) or v.get('id') not in mapping or any(type(v.get(k))!=int or v[k] not in [0,1,2] for k in ['groundedness','historical_boundary','explanation_usefulness']):valid=False;break
            seen.append(v['id']);local.append({'query_id':j['query_id'],'condition':mapping[v['id']],**v})
        if valid and len(seen)==len(mapping) and set(seen)==set(mapping):jr.extend(local)
        else:invalidjudge+=1
        raw=json.loads((ROOT/'results/llm_raw'/f"{j['request_hash']}.json").read_text(encoding='utf-8'));assert 'max_tokens' not in raw['request'];usage.append({'model':'assessor',**raw.get('response',{}).get('usage',{}),'valid':valid,'attempts':raw.get('attempts')})
    jf=pd.DataFrame(jr);jf.to_csv(R/'judge_scores.csv',index=False);js={}
    for c,g in jf.groupby('condition'):js[c]={'n':len(g),**{k:float(g[k].mean()) for k in ['groundedness','historical_boundary','explanation_usefulness']}}
    contrasts=[]
    for k in ['groundedness','historical_boundary','explanation_usefulness']:
        pivot=jf.pivot(index='query_id',columns='condition',values=k)
        for c in ['clean','fabricated_benign']:
            z=pivot[[c,'no_history']].dropna();delta=(z[c]-z.no_history).to_numpy();runmap={r['query_id']:r['run'] for r in records};z['run']=[runmap[q] for q in z.index];z['delta']=delta;gg=z.groupby('run').delta.agg(['sum','count']);draw=rng.integers(0,len(gg),(5000,len(gg)));bs=gg['sum'].to_numpy()[draw].sum(1)/gg['count'].to_numpy()[draw].sum(1);contrasts.append({'metric':k,'contrast':c+' minus no_history','n':len(delta),'difference':float(delta.mean()),'run_cluster_CI95':np.quantile(bs,[.025,.975]).tolist()})
    dump('explanation_summary.json',{'assessor':'MiniMax-M3','human_validation':False,'invalid_judge_queries':invalidjudge,'conditions':js,'paired_contrasts':contrasts,'quotation_metrics':base.groupby(['mode','condition'])[['claims_count','claims_valid','audit_pass']].sum().reset_index().to_dict('records'),'limitations':'Same-provider model assessment; numeric quotation checks do not establish semantic entailment; no operational human utility measurement'})
    controls=read('judge_controls_summary.json');control_records=read('judge_controls_records.json')
    for r in control_records:
        raw=json.loads((ROOT/'results/llm_raw'/f"{r['request_hash']}.json").read_text(encoding='utf-8'));assert not any(k in raw['request'] for k in ['max_tokens','max_completion_tokens']);assert json.loads(raw['request']['messages'][1]['content'])==r['payload'];usage.append({'model':'assessor_controls',**raw.get('response',{}).get('usage',{}),'valid':bool(r['response'].get('assessments')),'attempts':raw.get('attempts')})
    costs={}
    for model in ['reader','assessor','assessor_controls']:
        u=[x for x in usage if x['model']==model];attempts=[a for x in u for a in (x.get('attempts') or [])];costs[model]={'requests':len(u),'valid':sum(x['valid'] for x in u),'usage_available':sum('total_tokens' in x for x in u),'prompt_tokens':sum(x.get('prompt_tokens',0) for x in u),'completion_tokens':sum(x.get('completion_tokens',0) for x in u),'total_tokens':sum(x.get('total_tokens',0) for x in u),'recorded_attempts':len(attempts),'calls_without_attempt_details':sum(not x.get('attempts') for x in u),'rejected_attempts':sum(bool(a.get('rejected')) for a in attempts),'provider_seconds_p50_p90_p95':np.quantile([a['provider_seconds'] for a in attempts],[.5,.9,.95]).tolist() if attempts else None}
        if model=='reader':costs[model]['end_to_end_seconds_quantiles_p50_p90_p95']=np.quantile([x['wall_seconds'] for x in u],[.5,.9,.95]).tolist()
    dump('resource_usage.json',{'usage':costs,'money_cost':None,'latency_definition':'Reader elapsed time includes local queue waiting, retries and provider generation, not pure inference latency; unknown calls may have unreported token usage'})
    bindings=read('input_bindings.json');invariance=all(len({r['current_hash'] for r in bindings if r['query_id']==q})==1 for q in {r['query_id'] for r in bindings});splits=read('risk_splits.json');verification={'reader_requests':len(records),'unique_request_hashes':len({r['request_hash'] for r in records}),'assessor_requests':len(judges),'current_evidence_invariant':invariance,'split_groups_disjoint':all(not set(s['calibration'])&set(s['evaluation']) for s in splits),'no_token_cap':True,'truth_not_in_payload':True,'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),'audit_tests':read('audit_fault_injection.json')['all_expected']}
    for t in inputs:
        assert sum(len(s['values']) for s in t['payload']['current']['sensors'])+len(t['payload']['current']['global_features'])==378
    raw_validations=[]
    bykey={(t['query_id'],t['condition'],t['mode'],t['repeat']):t for t in inputs}
    for r in records:
        t=bykey[(r['query_id'],r['condition'],r['mode'],r['repeat'])];raw=json.loads((ROOT/'results/llm_raw'/f"{r['request_hash']}.json").read_text(encoding='utf-8'));body=raw['request'];assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==r['request_hash'];assert json.loads(body['messages'][1]['content'])==t['payload'];raw_validations.append({'request_hash':r['request_hash'],'unknown':bool(raw.get('unknown')),'has_response':bool(raw.get('response'))})
    frozen=json.loads((ROOT/'experiments/v2/results/frozen_policy.json').read_text());assert hashlib.sha256((ROOT/'experiments/v2/models'/f"{frozen['model']}.joblib").read_bytes()).hexdigest()==frozen['model_sha256'];verification['frozen_numeric_model_hash_verified']=True
    verification['full_378_features_verified']=True;verification['raw_payload_hashes_verified']=True;verification['unknown_reader_calls']=sum(r['unknown'] for r in raw_validations);verification['release_gate_amendment_sha256']=hashlib.sha256((OUT/'release_gate_amendment.json').read_bytes()).hexdigest();verification['queue_metrics']=read('queue_metrics.json');dump('verification.json',verification)
    verification['assessor_control_requests']=len(control_records);verification['total_requests']=len(records)+len(judges)+len(control_records);verification['control_queue']=read('judge_controls_queue.json');verification['judge_control_amendment_sha256']=hashlib.sha256((OUT/'judge_control_amendment.json').read_bytes()).hexdigest();dump('verification.json',verification)
    assert len(records)==440 and len({r['request_hash'] for r in records})==440 and invariance and verification['split_groups_disjoint']
    risk=read('risk_summary.json');hai=read('hai_results.json');parts=['<h1>论文2补充实验 v3</h1><p>基于论文思路的重建实现，使用经典数值模型及 MiniMax 阅读器；不是原始神经检测器的精确复现。回顾性机制与稳健性验证；无新增独立确认、无人工评价。所有失败结果保留。</p><p>公平消融为20个正常、20个故障样本；数值重复评估每轮100个正常、400个故障样本。整体覆盖率与接受后错误率依赖这一构成，不能直接当作现场指标。动态特征使用已完成事件窗口，不代表实时早期检测。HAI开发集只有11个含攻击窗口，来自少量攻击事件，不能支持5%误放行风险认证。</p>']
    def table(title,obj):parts.append('<h2>'+title+'</h2>'+pd.DataFrame(obj).to_html(index=False,escape=True))
    headline=[]
    for mode,method in [('ordinary','reader'),('separated','reader'),('separated','numeric_route')]:
        clean=next(s for s in summary if s['mode']==mode and s['method']==method and s['condition']=='clean');polluted=next(s for s in summary if s['mode']==mode and s['method']==method and s['condition']=='fabricated_benign');headline.append({'mode':mode,'method':method,'clean_FB_out_of_20':clean['FB'],'fabricated_FB_out_of_20':polluted['FB'],'clean_coverage':clean['coverage'],'fabricated_coverage':polluted['coverage']})
    claims={'scope':'Claims for the reconstructed implementation only','fair_pollution':headline,'risk_stability':{'passes':risk['passes'],'folds':15,'stable_risk_guarantee_supported':False},'hai':{'detected_episodes':hai['episode_detected'],'total_episodes':hai['episode_total'],'cross_dataset_safety_supported':False},'history_explanation_value':{'evidence_type':'model-assisted surrogate only','paired_contrasts':contrasts,'human_utility_validated':False},'audit':'Binding/citation/quoted-number failures detected in injected cases; semantic correctness not guaranteed. Pure numeric core invariant; explanation-gated fallback may change coverage.','dynamic_features':'All 15 repeated selections chose dynamic feature models; feature construction itself is not an original algorithm.'};dump('claim_assessment.json',claims)
    table('核心结果：相同输入下的历史污染',headline);parts.append(f"<p>风险重复评估通过 {risk['passes']}/15；HAI攻击事件检出 {hai['episode_detected']}/{hai['episode_total']}。解释评分仅为模型辅助证据。各项方法贡献应按这些边界表述，不能扩展为稳定风险保证或人类验证。</p>")
    parts.append('<p>同一条件下普通与角色分离阅读器的当前378维输入、数值候选和历史案例完全一致。标签翻转条件仅改历史结论；伪造正常条件还将一个历史案例扩展为52变量的完整匹配测量并添加误导叙述，属于复合压力干预，不能把其效应单独归因于标签变化。</p>')
    table('风险稳定性：15次重复分组评估（重叠，非独立重复）',[{k:v for k,v in r.items() if k not in ['policy']}|{'model':r['policy']['model']} for r in read('risk_results.json')]);parts.append('<p>通过 '+str(risk['passes'])+'/15；不能主张稳定风险保证。候选模型仅在Training拟合；每折校准与测试运行编号隔离。</p>')
    release=read('release_gate.json');table('探索性上线前独立验证门控',[{k:v for k,v in release.items() if k not in ['fold_results','scope']}]);parts.append('<p>初始重复评估后追加的探索性分析。每折选择、验证、测试运行编号互斥。拒绝候选后全部弃权，不能把这种零错误计为诊断成功；获准策略仍有测试失败，不能据此声称绝对风险保证。</p>')
    table('相同覆盖率诊断比较：40个干净历史样本',matched);parts.append('<p>只从原本非弃权输出中按置信度选取，达不到目标覆盖率时明确标注。阅读器自报置信度未经校准，不作概率解释。</p>')
    table('公平输入消融（缺失响应按弃权处理）',summary);parts.append('<p>numeric_route 是仅依赖当前证据和冻结策略的隔离核心；guarded_numeric 额外在阅读器审计失败时弃权，因此后者的覆盖率仍可能随历史变化，不主张完全非干扰。引用检查与完整审计均为同一响应的离线消融。</p>');table('对相同数值候选的遵从情况',adherence);table('历史污染成对效应（仅双方有效返回）',pairs);table('随机重复波动',repeat);table('解释质量：模型辅助评分',[{'condition':c,**v} for c,v in js.items()]);table('解释质量成对差异',contrasts);table('HAI 固定窗口时间迁移',[{'source':k,**v} for k,v in hai['metrics'].items()]);parts.append(f"<p>攻击事件检出 {hai['episode_detected']}/{hai['episode_total']}。窗口相关，不以二项上界作风险认证；历史查看过该数据，因此是回顾性外部数据集检验。</p>")
    table('M3评分器负例校验（追加探索性实验）',[{'condition':k,**v} for k,v in controls['conditions'].items()]);table('原定严格负例判据',[{'condition':k,**v} for k,v in controls['negative_control_detection'].items()]);parts.append('<p>空泛解释均获有用性1分，低于原始解释的2分；追加方案中预期0分的严格判据没有通过，而原始评分规则本身将泛泛解释定义为1分，因此保留两者并公开这一判据不一致。错误数值10/12获得依据性0分，历史当作当前事实12/12获得边界0分。评分器能识别明显负例，但主实验有用性评分饱和，不能据此主张历史案例提高解释质量或人类排障收益。</p>')
    table('审计契约故障注入',read('audit_fault_injection.json')['tests']);parts.append('<p>语义过度推断可以通过引用及数值核验；此测试明确保留该局限。受保护数值处置从可信记录取值，不采用阅读器写回的候选标签。严格契约失败还可能来自引用格式而非事实错误，不能将全部失败都称为幻觉。</p>');table('资源用量',[{'model':k,**v} for k,v in costs.items()]);parts.append('<h2>核验</h2><pre>'+html.escape(json.dumps(verification,ensure_ascii=False,indent=2))+'</pre>')
    from v3_figures import main as figures
    figures();parts.insert(1,'<p><img src="risk_stability.png" width="900"><img src="fair_pollution.png" width="700"><img src="hai_transfer.png" width="700"></p>')
    (R/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Paper2 v3</title><style>body{font:15px system-ui;margin:36px;line-height:1.6;color:#172333}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}th,td{padding:7px;border:1px solid #ccd5df}th{background:#eaf1fa}h2{margin-top:36px}img{max-width:100%}</style>'+''.join(parts),encoding='utf-8');print('ANALYSIS COMPLETE',verification,flush=True)
if __name__=='__main__':main()

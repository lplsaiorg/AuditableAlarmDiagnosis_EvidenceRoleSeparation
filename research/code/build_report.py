import json,html,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(n):return json.loads((ROOT/'results'/n).read_text(encoding='utf-8'))
def table(rows):
    if not rows:return '<p>Unavailable</p>'
    keys=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(str(k))+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
tep=load('tep_result.json');hai=load('hai_result.json');retrieval=load('retrieval_summary.json');reader=load('hai_reader_summary.json');tr=load('tep_reader_summary.json')
text='''<!doctype html><meta charset="utf-8"><title>论文2重建实验结果</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui;color:#222}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ccc;padding:8px;text-align:left}th{background:#eee}img{max-width:100%}code{background:#eee}h1,h2{color:#111}</style><h1>论文2独立重建实验</h1><p>本次从真实 HAI 21.03 和 TEP 数据重新计算。原始模型权重、分区清单、提示词、专家标签不可用；本结果是按论文思路重写的新实验，不是原始数字的精确复现。日期：2026-11-05。</p><p>Reader：MiniMax-M3.1-Flash；相关性评审：MiniMax-M3。三路密钥各6并发，持久请求队列与断点缓存。模型评分不等同于人类专家评估。</p><h2>TEP 数值路由</h2><p>10,500 个 Training 运行，118个特征；selection、stability、confirmation 各400故障+100正常，三个评估角色按 simulationRun 分组隔离。Training/Testing 使用数据集原始区分。ACSC 用 selection 选择模型与阈值，冻结后评估 confirmation。</p>'''
text+=table([{'partition':k,**{n:v[n] for n in ['fault_recognized','normal_recognized','FB','NF','abstain','coverage','FB_upper','NF_upper','AUROC']}} for k,v in tep.items()])
conclusions=load('main_conclusions.json')
text+='<h2>主要结论判定</h2>'+table([{'结论':'证据角色分离','判定':'实现与输入记录检查支持，语义非干扰未完全验证'},{'结论':'数值路由改善故障识别','判定':'支持：同队列265→322/400；配对找回60、丢失3；正常识别存在代价'},{'结论':'ACSC风险与覆盖率折衷','判定':'支持本次固定候选：独立确认FB上界0.04867、NF上界0.08518、覆盖率0.694'},{'结论':'结构化检索表现最好','判定':'不支持：当前余弦基线优于结构化距离基线'},{'结论':'历史信息稳定改善阅读诊断','判定':'未建立，当前HAI实验仅作探索性诊断'}])
text+='<p>Clopper–Pearson 区间使用论文相同的二项抽样公式；不同故障类型、仿真种子间的独立性和可交换性并未由公式自动保证。通过整体界限不代表每类故障均可识别。故障9与15在确认集中均未识别到故障，必须保留这一局限。</p><h2>PCA校准核对</h2>'
text+=table([load('tep_pca_calibration.json')])
text+='<h2>HAI基线</h2><p>使用 PCA reconstruction、Isolation Forest、standardized mean-square deviation 三个真实训练/校准的基线，不冒充 ModernTCN、TimesNet 或 TranAD。训练正常行参考来自 train1；test1 为开发；test2–test5 为评估。按固定报警聚类规则自动产生149个负例，不强制匹配论文的32个。原文0.62/0.90阈值在此实现中不能有效放行负例。</p>'
text+=table([hai['evaluation']])+'<img src="hai_reconstructed_scores.png"><h2>M3检索相关性评审</h2><p>评审45个攻击和固定随机抽取的32个负例，共77个查询，每个查询25个案例。查询真值和案例结果对评审隐藏。@1为每查询单一第一名，分母明确为有效查询数。nDCG使用完整25案例评分的理想排序。M3与Flash来自同一厂商模型家族，评审偏差仍可能相关。</p>'
text+='<p><strong>HAI设计诊断：</strong>本轮负例由行级中位百分位≥0.99的报警构成，事件分数又使用各检测器最大值的中位数，因此这些负例的事件D-score必然≥0.99。用0.90作为FAULT阈值时全部误报在定义上已被决定。该结果揭示当前重建的报警抽样与最终阈值不相容，不能作为论文原检测器独立性能失败的证明，也不能直接回填论文。需要另行冻结与报警抽样相容的事件级校准实验。</p>'
text+=table([{'method':k,**v} for k,v in retrieval['methods'].items()])
text+=f"<p>有效完整评分查询：{retrieval['valid_queries']}/{retrieval['requested_queries']}。缺失或截断样本不填默认分数，排除在配对比较之外；这可能引入有效响应子集选择偏差。</p>"
text+='<h2>HAI阅读诊断</h2><p>五种条件均使用同一查询。无历史与有历史提示长度未匹配，故条件差异仅作描述，不宣称纯粹的历史内容因果效应。离线路由始终单独保存。</p>'+table([{'condition':k,'n':v['n'],'counts':json.dumps(v['counts'])} for k,v in reader.items()])
text+='<h2>TEP阅读诊断与配对恢复</h2>'+table([tr])
text+='<h2>未覆盖的原文实验</h2><p>未重建三种原始神经检测器、原始事件队列、conformal comparator、25→200案例扩容和人类专家一致性；不能把本次结果用于宣称这些原始实验已复现。本次主流程和LLM结果均为重新运行。请求级记录、逐事件/逐运行预测、候选策略及数据哈希位于本目录及相邻code、models目录。</p>'
text+='<h2>验证</h2>'+table([load('verification.json')])
text+='<h2>API 用量（含截断请求与恢复）</h2>'+table([{'model':k,**v} for k,v in load('api_usage.json').items()])
(ROOT/'results/report.html').write_text(text,encoding='utf-8')
files=[]
for p in (ROOT/'code').rglob('*.py'):
    if '.git' not in p.parts:files.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'results/code_manifest.json').write_text(json.dumps(files,indent=2))
print(ROOT/'results/report.html')

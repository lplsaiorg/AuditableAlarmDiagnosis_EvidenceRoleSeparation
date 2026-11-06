import json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_numeric import OUT
def read(n):return json.loads((OUT/'results'/n).read_text(encoding='utf-8'))
def table(rows,keys=None):
    if not rows:return '<p>No records</p>'
    keys=keys or list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
def main():
    results=read('numeric_results.json');confirmation=results['confirmation_v2'];ablation=read('stability_ablations.json');pollution=read('pollution_summary.json');effects=read('pollution_effects.json');matched=read('matched_reader_coverage.json');fixed=read('fixed_threshold_full_coverage.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4));names=['Frozen v1','Optimized v2'];axes[0].bar(names,[confirmation['v1']['coverage']*100,confirmation['v2']['coverage']*100],color=['#7f8c8d','#2874a6']);axes[0].set_ylabel('Automatic coverage (%)');axes[0].set_ylim(0,100)
    axes[1].bar(names,[confirmation['v1']['normal_recognized']/2,confirmation['v2']['normal_recognized']/2],color=['#7f8c8d','#2874a6']);axes[1].set_ylabel('Normal recognition (%)');axes[1].set_ylim(0,100)
    fig.suptitle('New confirmation: 800 fault + 200 normal runs');fig.tight_layout();fig.savefig(OUT/'results/confirmation_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3);ax.bar(x-.18,[16,16,5],.36,label='Clean history',color='#7f8c8d');ax.bar(x+.18,[50,48,5],.36,label='Fabricated benign history',color='#c0392b');ax.set_xticks(x,['Ordinary prompt','Role-separated prompt','Isolated numerical route']);ax.set_ylabel('Faults incorrectly routed BENIGN (out of 50)');ax.set_ylim(0,55);ax.legend();fig.tight_layout();fig.savefig(OUT/'results/history_pollution.png',dpi=180);plt.close(fig)
    doc='''<!doctype html><meta charset="utf-8"><title>论文2第二轮实验</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;color:#222}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ccc;padding:7px;text-align:left}th{background:#eee}img{max-width:100%}h1,h2{color:#111}</style><h1>论文2第二轮主要结论实验</h1><p>2026-11-05。候选模型、特征、选择规则和污染条件先写入protocol.json。新确认集排除了上一轮使用过的全部300个Testing仿真运行组，采用余下200组中的200正常和800故障运行。所有候选仅以selection选择阈值；模型与协议哈希冻结后才评分新确认集。上一轮stability仅作为暴露的开发检查。</p>'''
    doc+='<h2>有益效果一：风险约束下的处理覆盖率</h2>'+table([dict(method=k,**confirmation[k]) for k in ['v1','v2']])+'<img src="confirmation_comparison.png">'
    doc+=f"<p>覆盖率差值：{confirmation['coverage_difference']:.3f}；按simulationRun聚类重采样的95%区间：{confirmation['coverage_difference_cluster_CI95']}。正常识别提高，但错误数也增加；这是相同预设风险上限内的取舍，不是所有错误率都下降。</p>"
    doc+='<h2>有益效果二：校准比直接阈值更能控制目标错误</h2>'+table([dict(method='same model fixed 0.5',**fixed),dict(method='selected ACSC',**confirmation['v2'])])
    doc+='<p>这是同一新模型的不同决策策略。固定0.5时FB上界超限，校准后确认集通过。但开发稳定集的v2 FB上界为0.07319，未通过0.05；不能将一次确认通过写成跨队列稳定保证。CP为名义二项区间；故障混合及仿真依赖仍限制保证解释。</p>'
    doc+='<h2>组件消融（暴露的稳定性队列）</h2>'+table(ablation,['model','veto','coverage','fault_recognized','normal_recognized','FB','NF','FB_upper','NF_upper','pass_risk'])
    doc+='<p>主要收益来自加入动态特征，类别加权有条件性增益。PCA正常证据校验没有显示独立的覆盖率增益，不能将其列为已验证创新。原有方法的这些基础组件也不是新算法，本实验提供组合效果而非文献新颖性证明。</p>'
    doc+='<h2>风险集中性</h2>'+table([{'fault':k,**v} for k,v in confirmation['per_fault'].items()])+'<p>全部25个确认集FB错误集中于故障9（25/40），故障9仅7/40被识别。总体风险通过不能掩盖这一盲区；不得声称各类故障安全。-1=拒判，0=良性，1=故障。</p>'
    doc+='<h2>历史污染实验</h2><p>100个运行（50正常、50故障），四种历史条件，两种提示词，共800个请求。所有污染配对中的当前测量字节哈希相同；模型为MiniMax-M3.1-Flash，无客户端token上限。普通提示、角色分离提示、当前证据引用门槛，以及独立数值路由分别报告。引用检查仅验证引用命名空间，不证明解释语义正确。该小样本不用于认证5%的FB风险上界。</p>'
    doc+=table(pollution,['method','condition','n','invalid','FB','NF','coverage','fault_recognized','normal_recognized'])
    doc+='<h2>相对干净历史的配对变化</h2>'+table(effects)
    doc+='<img src="history_pollution.png"><p>伪造良性历史使普通提示词的FB由16/50增至50/50，分离提示词由16/50增至48/50；独立数值处置保持5/50。引用门槛未阻止这种误导：合法的当前证据编号不等于推理真正不受历史影响。工程贡献应聚焦可执行的处置隔离，而非声称提示词或引用检查已保证语义非干扰。</p>'
    doc+='<p>独立数值路由的历史不变性由结构决定，不能单独计为实验发现。必须结合干净性能和受污染时的错误变化衡量实际价值。数值模型使用动态特征，而阅读器收到均值摘要；两者的绝对准确率差异不能仅归因于角色分离。普通与分离提示的比较则使用相同输入信息。</p>'
    doc+='<h2>相同覆盖率比较</h2><p>先保留系统本来会自动输出的样本，再按置信度降序选择。target_achieved=false时不可直接比较该目标覆盖率；无效响应均保留为失败。LLM自报置信度未做概率校准，曲线仅用于描述。</p>'+table(matched,['method','target_coverage','target_achieved','coverage','FB','NF','accepted_error','normal_recognized'])
    doc+='<h2>新确认集数值模型等覆盖率曲线</h2><p>以下另用各模型的固定0.5分类标签、按概率距0.5的距离选择样本；不是调整已冻结ACSC阈值，也不是额外风险认证。所有预定覆盖率一并展示。</p>'+table(read('matched_coverage.json'),['model','target_coverage','coverage','FB','NF','accepted_error','normal_recognized'])
    doc+='<h2>验证记录</h2>'+table([read('verification.json')])
    (OUT/'results/report.html').write_text(doc,encoding='utf-8');print(OUT/'results/report.html')
if __name__=='__main__':main()

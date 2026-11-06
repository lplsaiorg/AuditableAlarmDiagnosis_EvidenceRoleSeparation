import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/'results'/n).read_text(encoding='utf-8'))
tep=read('tep_result.json');llm=read('tep_reader_summary.json');ret=read('retrieval_summary.json')
gain=llm['paired_faults_recovered_by_numeric_router'];loss=llm['paired_faults_lost_by_numeric_router']
result={
 'scope':'主要结论的独立重建验证，不要求原文数值一致。更换基础模型后，模型、特征、阈值、提示词、评审者及区间解释同步记录。',
 'evidence_role_separation':{'status':'supported_by_implementation_audit','meaning':'查询真值不进入LLM输入；原始请求响应按hash保存；数值路由单独存储；不宣称已验证自然语言的全部语义非干扰'},
 'numerical_routing':{'status':'supported_on_TEP_stability','reader_fault_recognition':llm['counts'].get('FAULT->FAULT',0),'router_fault_recognition':tep['stability']['fault_recognized'],'paired_recovered':gain,'paired_lost':loss,'paired_exact_mcnemar_p':binomtest(min(gain,loss),gain+loss,.5).pvalue,'reader_FB':llm['counts'].get('FAULT->BENIGN',0),'router_FB':tep['stability']['FB'],'reader_normal_recognition':llm['counts'].get('NORMAL->BENIGN',0),'router_normal_recognition':tep['stability']['normal_recognized'],'interpretation':'故障识别与危险放行改善，但正常识别和自动覆盖率有代价，不能概括为所有指标全面更优'},
 'ACSC':{'status':'supported_for_fixed_candidate','confirmation':{k:tep['confirmation'][k] for k in ['FB','NF','FB_upper','NF_upper','coverage']},'interpretation':'支持在本次固定候选家族上最大化selection覆盖率，并在独立confirmation通过两个名义二项风险界限。不证明优于未实现的所有selective/conformal方法。逐故障盲区仍存在。'},
 'structured_retrieval':{'status':'not_supported_as_best','assessor':'MiniMax-M3','valid_queries':ret['valid_queries'],'requested_queries':ret['requested_queries'],'methods':ret['methods'],'paired_differences':ret['paired_bootstrap'],'interpretation':'当前结构化距离基线未胜出；同一签名上的余弦检索更好。比较基于模型评分而非人类专家。'},
 'history_improves_reader':{'status':'not_established','interpretation':'HAI五条件显示历史内容影响决策，但不能稳定改善识别；原文固定阈值不适配新模型，相关条件仅作探索性诊断。'},
 'retired_endpoint':{'HAI_fixed_threshold':'退出主要有效性结论。报警抽样阈值与最终FAULT阈值不相容，全部误报为定义导致。'},
 'limitations':['原神经检测器未重现，当前为明确命名的经典基线','M3与Flash的评审偏差可能相关，非人类专家验证','正常训练案例可能存在时间相关性','TEP仿真运行独立性与故障混合分布需外部验证','未实现conformal与案例库扩容不影响本轮主要结论范围，但不能沿用相应比较优势声明']
}
(ROOT/'results/main_conclusions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result['numerical_routing'],ensure_ascii=False))

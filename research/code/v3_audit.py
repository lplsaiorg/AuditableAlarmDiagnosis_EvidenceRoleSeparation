import copy,json,os
os.environ['PAPER2_QUEUE_JOURNAL']='v3_audit_jobs.sqlite3'
from v3_llm import OUT,QUEUE,gate,sha,claims_check,dump
def run():
    t=json.loads((OUT/'results/fair_inputs.json').read_text(encoding='utf-8'))[0];p=t['payload'];ref=p['current']['sensors'][0];r={'label':'FAULT','supporting_refs':[ref['ref']],'claims':[{'ref':ref['ref'],'field':'z_mean','value':ref['values'][0]}]};tests=[]
    def check(name,response,payload,expected):
        observed=gate(response,payload,t['current_hash'],t['policy_hash']);tests.append({'name':name,'expected':expected,'observed':observed,'pass':observed==expected,'final_guarded_route':t['numeric_route'] if observed=='PASS' else -1})
    check('valid_bound_record',r,p,'PASS')
    for name,refs in [('missing_reference',[]),('forged_reference',['E:999']),('historical_reference',['H1:E:0']),('nonstring_reference',[{}])]:
        q=copy.deepcopy(r);q['supporting_refs']=refs;check(name,q,p,'INVALID_CURRENT_REFERENCE')
    q=copy.deepcopy(r);q['claims'][0]['value']+=1;check('misquoted_numeric_value',q,p,'INVALID_NUMERIC_CLAIM')
    q=copy.deepcopy(r);q['claims'][0]['field']='imaginary_field';check('forged_measurement_field',q,p,'INVALID_NUMERIC_CLAIM')
    q=copy.deepcopy(r);q['claims']=[];check('missing_numeric_claim',q,p,'INVALID_NUMERIC_CLAIM')
    q=copy.deepcopy(p);q['current']['sensors'][0]['values'][0]+=1;check('changed_current_evidence',r,q,'CURRENT_BINDING_MISMATCH')
    q=copy.deepcopy(p);q['policy_hash']='obsolete';check('stale_calibration_binding',r,q,'STALE_POLICY')
    check('unparseable_response',None,p,'MALFORMED_RESPONSE')
    q=copy.deepcopy(r);q['label']='NORMAL';check('invalid_label_enum',q,p,'MALFORMED_RESPONSE')
    q=copy.deepcopy(p);q['historical_cases']=[];check('history_removed_does_not_change_bound_route',r,q,'PASS')
    q=copy.deepcopy(r);q['explanation']='This value proves that a valve is physically broken.';check('semantic_overclaim_is_NOT_detected_by_syntactic_gate',q,p,'PASS')
    q=copy.deepcopy(p);q['numeric_candidate']['label']='BENIGN';check('reader_candidate_tampering_cannot_override_external_frozen_route',r,q,'PASS')
    dump('audit_fault_injection.json',{'tests':tests,'all_expected':all(x['pass'] for x in tests),'scope':'Offline replay of explicit binding/citation/numeric-quotation contract. Trusted numeric route is external to reader response. Not an adversarial security certification; semantic overclaims deliberately demonstrate gate limits.'})
    assert all(x['pass'] for x in tests)
    print('AUDIT',len(tests),'expected outcomes verified',flush=True)
if __name__=='__main__':
    try:run()
    finally:QUEUE.close()

"""Run frozen v3 rules on fixed diagnosis and held-out cases; no model execution."""
import collections,csv,hashlib,json,sys
from pathlib import Path
from rules_v3 import *
from packet_io_v3 import decode_accounting
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2';sys.path.insert(0,str(V2));from rebuild import H

def main():
 frozen=json.loads((HERE/'V3_RULES_FROZEN.json').read_text())
 for path,digest in frozen['frozen_files_sha256'].items():
  if hashlib.sha256((HERE/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen input/code changed: '+path)
 if not json.loads((HERE/'diagnostics/SOURCE_HASH_VERIFICATION.json').read_text())['all_match']:raise RuntimeError('Raw source identity mismatch')
 cases=json.loads((HERE/'diagnostics/FIXED_CASES.json').read_text())['cases'];schemas={d['domain']:d for d in json.loads((HERE/'diagnostics/PAYLOAD_SCHEMA_DIAGNOSIS_ONLY.json').read_text())};registry=list(csv.DictReader((HERE/'source_domain_registry.csv').open()));rows=[];traces=[];allrelations=[];allflags=[];parseLog=DecisionLog(HERE/'diagnostics/V3_TARGET_PARSE_LOG.jsonl');decisionLog=DecisionLog(HERE/'diagnostics/V3_DECISIONS.jsonl');rawverified=0
 for day in ['Friday','Tuesday','Wednesday']:
  out=HERE/'diagnostics'/day;source=json.loads((V2/f'reconstructed/CIC_IDS/{day}/source_files.json').read_text())[0];reg=next(r for r in registry if r['dataset']=='CIC_IDS' and r['acquisition_family']==day);domain=reg['provenance_domain_id'];rawdata=json.loads((out/'RAW_PACKETS_BY_KEY.json').read_text());groups={};events={};relations=[];complete=json.loads((out/'COMPLETE.json').read_text());ctx=json.loads((out/'CSV_CONTEXT.json').read_text());clock=json.loads((out/'CLOCK_QUALITY.json').read_text())['v2_clock_status'];feat=json.loads((V2/f'reconstructed/CIC_IDS/{day}/csv_selection_audit.json').read_text())['feature_names'];search_complete=bool(ctx['complete'] and complete['decode_errors']==0)
  with Path(source['path']).open('rb') as f:
   for g in rawdata:
    ps=[]
    for original in g['packets']:
     p=dict(original);f.seek(p['offset']);raw=f.read(p['caplen']);actual=decode_accounting(raw,p['link'])
     for field in ['raw_sha256','payload_len','capture_payload_len','ethernet_padding_len','flags','isn']:
      if actual[field]!=p[field]:raise RuntimeError('Raw packet accounting mismatch '+day+'/'+str(p['index'])+'/'+field)
     p.update(actual,file_id=reg['source_file_id']);ps.append(p);rawverified+=1
    assigned,ev,edges=assign_sessions('CIC_IDS',domain,g['key'],ps);key=stable(g['key']);groups[key]=assigned;events[key]=ev;relations+=edges
  allrelations.extend(relations)
  for case in [c for c in cases if c['domain']==day]:
   raw=case['csv_record'];key=stable(H['csv_key'](raw));r=normalize_row(raw,reg['source_file_id'],raw['csv_row'],raw['label'],parseLog,stats=[raw[c] for c in feat]);ps=groups.get(key,[]);schema=schemas[day];metric=schema['metric'] if schema['status']=='DIAGNOSTIC_SCHEMA_ESTIMATE_PASS' else None
   result=decide(r,ps,ctx['context'].get(key,[]),clock['offset_options_seconds'],search_complete=search_complete,payload_metric=metric) if r is not None else dict(decision='reject',stage='parse',reason='invalid_target_record')
   detail={'case_id':case['case_id'],'source_csv':raw['csv_path'],'v2_status':case['v2_status'],'v2_reason':case['v2_reason'],'diagnostic_payload_metric':metric,'search_complete':search_complete,**{k:v for k,v in result.items() if k not in ['decision','reason','stage']}}
   decisionLog.write(reg['source_file_id'],raw['csv_row'],raw['label'],result['stage'],result['reason'],detail,decision=result['decision'])
   row=dict(case_id=case['case_id'],role=case['role'],domain=day,label=raw['label'],v2_status=case['v2_status'],v2_reason=case['v2_reason'] or '',v3_decision=result['decision'],v3_stage=result['stage'],v3_reason=result['reason'],payload_metric=metric,missing_initial_syn=result.get('missing_initial_syn'),closed_control_tail_packets=result.get('closed_control_tail_packets',0));rows.append(row)
   trace=dict(case_id=case['case_id'],role=case['role'],source_file_id=reg['source_file_id'],provenance_domain_id=domain,csv_record=raw,byte_schema=metric,result=result,source_packet_search_complete=search_complete)
   if 'start' in result:
    unit=ps[result['start']:result['end']];trace['accepted_original_packet_membership']=[{k:p[k] for k in ['file_id','index','offset','caplen','raw_sha256','ts','flags','payload_len','capture_payload_len','session_id','closed_control_tail']} for p in unit]
    offsets=clock['offset_options_seconds'];valid_residuals=[unit[0]['ts']-r['clock']-o for o in offsets if 0<=unit[0]['ts']-r['clock']-o<r['clock_precision_s']];trace['calibrated_clock_residual_s']=valid_residuals;trace['duration_error_us']=abs((unit[-1]['ts']-unit[0]['ts'])*1e6-r['Flow Duration'])
   trace['same_key_session_boundary_events']=events.get(key,[]);traces.append(trace)
 parseLog.close();decisionLog.close()
 with (HERE/'diagnostics/V3_CASE_RESULTS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (HERE/'diagnostics/V3_CASE_TRACES.json').write_text(json.dumps(traces,indent=2));(HERE/'diagnostics/V3_SESSION_GROUP_RELATIONSHIPS.json').write_text(json.dumps(allrelations,indent=2))
 counts=[]
 for role in ['diagnosis','heldout_rule_check']:
  for day in ['Friday','Tuesday','Wednesday']:
   sub=[r for r in rows if r['role']==role and r['domain']==day];counts.append(dict(role=role,domain=day,n=len(sub),accepted=sum(r['v3_decision']=='accept' for r in sub),rejected=sum(r['v3_decision']=='reject' for r in sub),reasons=dict(collections.Counter(r['v3_reason'] for r in sub))))
 accepted=[t for t in traces if t['result']['decision']=='accept'];residuals=[r for t in accepted for r in t.get('calibrated_clock_residual_s',[])];duration_errors=[t['duration_error_us'] for t in accepted];summary=dict(mode='fixed_case_diagnostic_validation_only',formal_data_generated=False,training_started=False,raw_packet_records_independently_rechecked=rawverified,source_searches_complete=all(t['source_packet_search_complete'] for t in traces),cases=len(rows),counts=counts,heldout_used_for_rule_choice=False,accepted_v2_controls_retained=sum(r['v2_status']=='accepted' and r['v3_decision']=='accept' for r in rows),accepted_v2_controls_total=sum(r['v2_status']=='accepted' for r in rows),clock_residual_s_min=min(residuals) if residuals else None,clock_residual_s_max=max(residuals) if residuals else None,max_accepted_duration_error_us=max(duration_errors) if duration_errors else None,limitation='record-level rule checks within same captures, not independent benchmark validation or proof every accepted label is true')
 (HERE/'diagnostics/V3_VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

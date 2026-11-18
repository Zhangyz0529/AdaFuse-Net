"""Choose a fixed per-domain byte schema from diagnosis cases ONLY; no model score.
The resulting small-sample estimates do not certify a whole acquisition domain.
"""
import json,collections,sys
from pathlib import Path
import numpy as np
from rules_v3 import RULES,stable,normalize_row,DecisionLog
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2';sys.path.insert(0,str(V2));from rebuild import H

def metadata_windows(row,ps,offsets):
 n=row['n_packets'];origin=(str(row['Source IP']),int(row['Source Port']));result=[]
 if len(ps)<n:return result
 direction=np.array([tuple(p['src'])==origin for p in ps],dtype='int64');pref=np.r_[0,np.cumsum(direction)];a=np.arange(len(ps)-n+1);b=a+n
 for start in a[pref[b]-pref[a]==row['Total Fwd Packets']]:
  start=int(start);end=start+n;t=ps[start]['ts'];dur=(ps[end-1]['ts']-t)*1e6
  if abs(dur-row['Flow Duration'])<=2. and any(row['clock']+o<=t<row['clock']+o+row['clock_precision_s'] for o in offsets):result.append((start,end))
 return result

def main():
 output=[];plan=json.loads((HERE/'diagnostics/FIXED_CASES.json').read_text());log=DecisionLog(HERE/'diagnostics/PAYLOAD_CALIBRATION_PARSE_LOG.jsonl')
 for day in ['Friday','Tuesday','Wednesday']:
  root=HERE/'diagnostics'/day;packets={stable(g['key']):g['packets'] for g in json.loads((root/'RAW_PACKETS_BY_KEY.json').read_text())};context=json.loads((root/'CSV_CONTEXT.json').read_text());offsets=json.loads((root/'CLOCK_QUALITY.json').read_text())['v2_clock_status']['offset_options_seconds'];votes=collections.Counter();anchors=[];uninformative=contradictory=ambiguous=0
  cases=[c for c in plan['cases'] if c['domain']==day and c['role']=='diagnosis']
  for case in cases:
   raw=case['csv_record'];row=normalize_row(raw,'diagnostic_csv',raw['csv_row'],raw['label'],log)
   if row is None:continue
   key=H['csv_key'](raw);ps=packets.get(stable(key),[]);windows=metadata_windows(row,ps,offsets)
   if len(windows)!=1:ambiguous+=1;continue
   x,y=windows[0];other=context['context'].get(stable(key),[])
   if not context['complete'] or any(r is None for r in other) or sum((x,y) in metadata_windows(r,ps,offsets) for r in other)!=1:ambiguous+=1;continue
   unit=ps[x:y];origin=(row['Source IP'],row['Source Port']);metrics={};matching=[]
   for metric in ['payload_len','capture_payload_len']:
    f=sum(p[metric] for p in unit if tuple(p['src'])==origin);b=sum(p[metric] for p in unit if tuple(p['src'])!=origin);matches=f==row['Total Length of Fwd Packets'] and b==row['Total Length of Bwd Packets'];metrics[metric]=dict(fwd=f,bwd=b,matches_csv=matches)
    if matches:matching.append(metric)
   if len(matching)==1:votes[matching[0]]+=1
   elif len(matching)==2:uninformative+=1
   else:contradictory+=1
   anchors.append(dict(case_id=case['case_id'],role='diagnosis',csv_file=raw['csv_path'],csv_row=raw['csv_row'],packet_window=[x,y],metrics=metrics,padding_bytes=sum(p['ethernet_padding_len'] for p in unit),calibration_uses_label=False))
  informative=sum(votes.values());metric,n=votes.most_common(1)[0] if votes else (None,0);consensus=n/(informative+contradictory) if informative+contradictory else 0.;ok=informative>=RULES['payload_schema_min_informative_anchors'] and consensus>=RULES['payload_schema_min_consensus']
  output.append(dict(domain=day,status='DIAGNOSTIC_SCHEMA_ESTIMATE_PASS' if ok else 'BLOCKED_INSUFFICIENT_OR_MIXED_SCHEMA',metric=metric if ok else None,votes=dict(votes),informative=informative,uninformative=uninformative,contradictory=contradictory,ambiguous_metadata=ambiguous,consensus=consensus,anchors=anchors,scope='diagnosis-only physical byte definition; repeat label-blind calibration for full source before formal acceptance'))
 log.close();(HERE/'diagnostics/PAYLOAD_SCHEMA_DIAGNOSIS_ONLY.json').write_text(json.dumps(output,indent=2));print([{k:r[k] for k in ['domain','status','metric','votes','consensus']} for r in output])
if __name__=='__main__':main()

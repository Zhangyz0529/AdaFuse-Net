"""Versioned observed-connection and exact label-window rules; no model imports."""
import hashlib,ipaddress,json,math
from collections import Counter
import numpy as np
import pandas as pd

RULES={
 'version':'observed_transport_v3', 'tcp_idle_s':120.,'udp_idle_s':30.,
 'tcp_closed_control_tail_s':2., 'active_timeout_s':None,
 'new_connection':'new non-retransmitted SYN without ACK; idle expiry; or closure boundary excluding bounded control tail',
 'control_tail':'after RST or both-direction FIN, no SYN, zero transport payload, within2s of closure; never extends original closure grace',
 'missing_handshake':'retain with missing_initial_syn; no complete-connection claim',
 'packet_count_min':1,'packet_count_max':None,
 'packet_count_basis':'one observed TCP/UDP packet is not invalid merely for being short; zero/negative counts invalid; large windows use chunking or fail-closed resource guard',
 'payload_metric_CIC_IDS':'one certified metric per provenance domain, never per row or label: IP-bounded payload_len versus capture_payload_len including Ethernet padding',
 'payload_schema_min_informative_anchors':8,'payload_schema_min_consensus':.95,
 'payload_schema_scope':'metadata-only anchor calibration; fixed diagnostic-derived schemas do not certify the whole capture; full source uses label-blind anchors and blocks mixed/unsupported schema',
 'payload_metric_CIC_DDOS':'UNVERIFIED_BLOCK_ACCEPTANCE',
 'duration_tolerance_us':2.,'timestamp_precision':'inferred from CSV representation; minute precision retained as interval, not subsecond truth',
 'source':'explicit source_file_id -> provenance_domain_id registry; no tuple-only merging across unrelated domains',
 'source_file_identity':'dataset + relative path; content hashes and archive integrity are separately verified',
 'session_identity':'version,dataset,provenance_domain_id,canonical tuple,first-source-file-id,first-packet-ordinal,first-packet-hash,segment ordinal',
 'same_raw_observation':'deduplicate identical timestamp+raw-hash across different source files in same domain only; repeated observation within same file retained',
 'duplicates':'cross-label identical model-input groups all excluded; same-label content and session-association edges unioned before partition',
 'overflow':'no partial search accepted; spill/chunk or stop affected domain with explicit unresolved status',
 'truncation':'verified complete-record prefix only under declared coverage; original preserved; any CSV time-uncertainty interval touching unavailable coverage rejected; no full-capture claim',
 'nonfinite_features':'fixed zero replacement, counted explicitly; malformed nonnumeric statistics rejected',
 'scope':'IPv4 TCP/UDP in current adapter; unsupported families/links explicitly rejected or block search completeness',
}
def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def canonical(row):
 a=(str(ipaddress.ip_address(str(row['Source IP']))),int(row['Source Port']));b=(str(ipaddress.ip_address(str(row['Destination IP']))),int(row['Destination Port']))
 return (int(row['Protocol']),*sorted([a,b]))

class DecisionLog:
 def __init__(self,path):self.path=path;self.counts=Counter();self.handle=path.open('w')
 def write(self,source_file_id,csv_row_id,label,stage,reason,detail,decision='reject'):
  r=dict(source_file_id=source_file_id,csv_row_id=csv_row_id,label=label,stage=stage,reason=reason,detail=detail,decision=decision);self.handle.write(json.dumps(r,sort_keys=True)+'\n');self.handle.flush();self.counts[decision,stage,reason]+=1;return r
 def close(self):self.handle.close()

def normalize_row(row,source_file_id,csv_row_id,label,log,stats=None):
 r=dict(row)
 def reject(stage,reason,detail):log.write(source_file_id,csv_row_id,label,stage,reason,detail);return None
 try:
  for name in ['Source IP','Destination IP']:
   ip=ipaddress.ip_address(str(r[name]))
   if ip.version!=4:return reject('parse','unsupported_address_family',{'field':name,'family':ip.version})
  for name in ['Source Port','Destination Port','Protocol','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets']:
   v=float(r[name])
   if not math.isfinite(v) or not v.is_integer() or v<0:return reject('parse','invalid_nonnegative_integer',{'field':name,'value':str(r[name])})
   r[name]=int(v)
  if any(r[k]>65535 for k in ['Source Port','Destination Port']):return reject('parse','invalid_port',{})
  if r['Protocol'] not in [6,17]:return reject('parse','unsupported_transport',{'protocol':r['Protocol']})
  r['Flow Duration']=float(r['Flow Duration'])
  if not math.isfinite(r['Flow Duration']) or r['Flow Duration']<0:return reject('parse','invalid_duration',{})
 except (ValueError,TypeError,KeyError) as e:return reject('parse','invalid_required_field',{'error':str(e)})
 try:
  stamp=str(r['Timestamp']);t=pd.to_datetime(stamp,format='ISO8601',errors='raise') if len(stamp)>4 and stamp[:4].isdigit() and stamp[4]=='-' else pd.to_datetime(stamp,format='mixed',dayfirst=True,errors='raise');r['clock']=float(t.tz_localize('UTC').timestamp()) if t.tzinfo is None else float(t.timestamp())
  clock=stamp.split()[-1];parts=clock.split(':');r['clock_precision_s']=60. if len(parts)==2 else (10.**-len(parts[-1].split('.')[1]) if '.' in parts[-1] else 1.)
 except (ValueError,TypeError,KeyError) as e:return reject('parse','timestamp_parse_failure',{'error':str(e)})
 r['n_packets']=r['Total Fwd Packets']+r['Total Backward Packets']
 if r['n_packets']<1:return reject('eligibility','zero_packet_count',{})
 # No <3 or >10000 exclusion. n is a resource requirement, not session validity.
 if stats is not None:
  try:arr=np.asarray(stats,dtype='float64')
  except (TypeError,ValueError):return reject('features','nonnumeric_statistic',{})
  if arr.shape!=(77,):return reject('features','invalid_feature_dimension',{'shape':list(arr.shape)})
  invalid=int((~np.isfinite(arr)).sum());arr=np.nan_to_num(arr,nan=0.,posinf=0.,neginf=0.)
  with np.errstate(over='ignore'):out=arr.astype('float32')
  if not np.isfinite(out).all():return reject('features','float32_overflow',{})
  r['stat']=out.tolist();r['nonfinite_features_replaced']=invalid
 r['_invalid_stats']=False;return r

def assign_sessions(dataset,domain_id,key,packets):
 if not domain_id:raise ValueError('Unregistered provenance domain')
 packets=sorted([dict(p) for p in packets],key=lambda p:(p['ts'],p['file_id'],p['index']));state=None;ordinal=-1;result=[];events=[];seen={};relations=[]
 for p in packets:
  ident=(p['ts'],p['raw_sha256']);previous_file=seen.get(ident)
  if previous_file is not None and previous_file!=p['file_id']:
   events.append(dict(event='duplicate_observation_across_files',file_id=p['file_id'],packet_index=p['index']));continue
  seen.setdefault(ident,p['file_id']);syn=p['proto']==6 and bool(p['flags']&2) and not bool(p['flags']&16);new_isn=state is not None and syn and (state['initial_isn']!=p['isn'] or not state['initial_syn'])
  idle=state is not None and p['ts']-state['last']>(RULES['tcp_idle_s'] if p['proto']==6 else RULES['udp_idle_s'])
  tail=bool(state is not None and state['closed_at'] is not None and p['proto']==6 and not(p['flags']&2) and p['payload_len']==0 and 0<=p['ts']-state['closed_at']<=RULES['tcp_closed_control_tail_s'])
  closed_boundary=state is not None and state['closed_at'] is not None and not tail
  reason='first_observation' if state is None else 'new_syn' if new_isn else 'idle_timeout' if idle else 'closure_boundary' if closed_boundary else None
  if reason:
   prior=state;ordinal+=1;sid=stable([RULES['version'],dataset,domain_id,key,p['file_id'],p['index'],p['raw_sha256'],ordinal]);state={'id':sid,'last':p['ts'],'initial_syn':syn,'initial_isn':p['isn'],'fin':set(),'closed_at':None}
   events.append(dict(event='session_start',reason=reason,session_id=sid,ts=p['ts'],file_id=p['file_id'],packet_index=p['index']))
   if prior and not syn and p['proto']==6:relations.append(dict(first_session=prior['id'],second_session=sid,reason='uncertain_midstream_continuation_same_domain_tuple',must_group_together=True))
  p.update(session_id=state['id'],provenance_domain_id=domain_id,missing_initial_syn=p['proto']==6 and not state['initial_syn'],closed_control_tail=bool(tail and not reason));state['last']=p['ts']
  if p['proto']==6:
   if p['flags']&1:state['fin'].add(tuple(p['src']))
   if state['closed_at'] is None and (p['flags']&4 or len(state['fin'])==2):state['closed_at']=p['ts']
  result.append(p)
 return result,events,relations

def windows(row,packets,offset_options,payload_metric):
 n=row['n_packets'];out=[]
 if len(packets)<n:return out
 origin=(str(row['Source IP']),int(row['Source Port']));direction=np.array([tuple(p['src'])==origin for p in packets],dtype='int64');v=np.array([p[payload_metric] for p in packets],dtype='int64');f=np.r_[0,np.cumsum(direction)];fb=np.r_[0,np.cumsum(v*direction)];bb=np.r_[0,np.cumsum(v*(1-direction))]
 # Chunk windows, preserving exact all-candidate search. No top-k truncation.
 for begin in range(0,len(packets)-n+1,100000):
  a=np.arange(begin,min(begin+100000,len(packets)-n+1));b=a+n
  good=(f[b]-f[a]==row['Total Fwd Packets'])&(fb[b]-fb[a]==row['Total Length of Fwd Packets'])&(bb[b]-bb[a]==row['Total Length of Bwd Packets'])
  for x in a[good]:
   x=int(x);y=x+n;start=packets[x]['ts'];duration=(packets[y-1]['ts']-start)*1e6
   if abs(duration-row['Flow Duration'])>RULES['duration_tolerance_us']:continue
   if any(row['clock']+offset<=start<row['clock']+offset+row['clock_precision_s'] for offset in offset_options):out.append((x,y))
 return out

def decide(row,packets,all_csv_candidates,offset_options,search_complete=None,coverage_cutoff=None,payload_metric=None):
 if payload_metric not in ['payload_len','capture_payload_len']:return {'decision':'reject','stage':'schema','reason':'uncertified_payload_schema'}
 if search_complete is not True:return {'decision':'reject','stage':'resource','reason':'incomplete_candidate_search'}
 if any(r is None for r in all_csv_candidates):return {'decision':'reject','stage':'matching','reason':'invalid_candidate_record_for_key'}
 matches=windows(row,packets,offset_options,payload_metric)
 if len(matches)!=1:return {'decision':'reject','stage':'matching','reason':'no_consistent_packet_window' if not matches else 'ambiguous_packet_window','candidate_count':len(matches)}
 start,end=matches[0];unit=packets[start:end]
 if coverage_cutoff is not None:
  eligible=[o for o in offset_options if row['clock']+o<=unit[0]['ts']<row['clock']+o+row['clock_precision_s']]
  if not eligible or any(row['clock']+o+row['clock_precision_s']+row['Flow Duration']/1e6>=coverage_cutoff for o in eligible):return {'decision':'reject','stage':'coverage','reason':'csv_time_uncertainty_touches_truncated_coverage'}
 alternatives=[r for r in all_csv_candidates if (start,end) in windows(r,packets,offset_options,payload_metric)]
 if len(alternatives)!=1:return {'decision':'reject','stage':'matching','reason':'ambiguous_csv_record_or_label','candidate_count':len(alternatives)}
 if len({p['session_id'] for p in unit})!=1:return {'decision':'reject','stage':'session','reason':'window_spans_multiple_observed_sessions'}
 return {'decision':'accept','stage':'matching','reason':'unique_complete_observed_session_window','start':start,'end':end,'session_id':unit[0]['session_id'],'missing_initial_syn':unit[0]['missing_initial_syn'],'closed_control_tail_packets':sum(p['closed_control_tail'] for p in unit)}

"""Complete fixed-key raw-PCAP scans for score-independent diagnostic cases.
Alternatives are explanatory counts, never automatic acceptance criteria.
"""
import argparse,collections,hashlib,json,mmap,sys,gc,struct
from pathlib import Path
import numpy as np,dpkt
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2';sys.path.insert(0,str(V2))
from rebuild import H
from session_rules import assign_sessions

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def details(row,ps,clock):
 n=int(row['Total Fwd Packets'])+int(row['Total Backward Packets']);origin=(str(row['Source IP']),int(row['Source Port']));expectedf=int(row['Total Fwd Packets']);ebf=int(row['Total Length of Fwd Packets']);ebb=int(row['Total Length of Bwd Packets']);duration=float(row['Flow Duration']);result={'packets_with_same_tuple_in_capture':len(ps),'csv_expected':{'n':n,'fwd_n':expectedf,'bwd_n':n-expectedf,'fwd_bytes':ebf,'bwd_bytes':ebb,'duration_us':duration,'clock':row['clock']},'metrics':{},'v2_exact_windows':[]}
 if len(ps)<n or n<1:return result
 starts=np.arange(len(ps)-n+1);ends=starts+n;direction=np.array([p['src']==origin for p in ps],dtype='int64');pref=np.r_[0,np.cumsum(direction)];fc=pref[ends]-pref[starts];dur=np.array([(ps[b-1]['ts']-ps[a]['ts'])*1e6 for a,b in zip(starts,ends)]);times=np.array([p['ts'] for p in ps]);opts=clock['offset_options_seconds'];timeok=np.array([any(row['clock']+o<=t<row['clock']+o+60 for o in opts) for t in times[starts]]);dirok=fc==expectedf;durdiff=np.abs(dur-duration);durok=durdiff<=2.
 result['metrics'].update(total_fixed_n_windows=len(starts),direction_count_match=int(dirok.sum()),direction_and_time=int((dirok&timeok).sum()),direction_time_duration=int((dirok&timeok&durok).sum()),minimum_duration_error_us_for_direction_and_time=float(durdiff[dirok&timeok].min()) if (dirok&timeok).any() else None)
 bytegood={}
 for field in ['payload_len','ip_payload_len','ip_len','frame_len','wire_len']:
  v=np.array([p[field] for p in ps]);pf=np.r_[0,np.cumsum(v*direction)];pb=np.r_[0,np.cumsum(v*(1-direction))];fb=pf[ends]-pf[starts];bb=pb[ends]-pb[starts];ok=(fb==ebf)&(bb==ebb);bytegood[field]=ok
  result['metrics'][field]={'direction_and_bytes':int((dirok&ok).sum()),'direction_bytes_duration':int((dirok&ok&durok).sum()),'direction_bytes_duration_time':int((dirok&ok&durok&timeok).sum())}
 exact=starts[dirok&bytegood['payload_len']&durok&timeok];result['v2_exact_windows']=[dict(start=int(a),end=int(a+n),sessions=len({p['session_id'] for p in ps[a:a+n]})) for a in exact[:20]]
 # Predeclared diagnostic representative: exact window; otherwise closest duration
 # among direction/time matches, otherwise closest time. Not a labeling rule.
 if len(exact):chosen=int(exact[0]);why='first_v2_exact_window'
 else:
  eligible=starts[dirok&timeok]
  if len(eligible):chosen=int(eligible[np.argmin(durdiff[eligible])]);why='nearest_duration_among_direction_time_matches'
  else:
   residual=np.array([min(abs(t-row['clock']-o) for o in opts) for t in times[starts]]);chosen=int(np.argmin(residual));why='nearest_clock_only_no_valid_direction_window'
 unit=ps[chosen:chosen+n];bound=[]
 for j,p in enumerate(unit):
  if j and p['session_id']!=unit[j-1]['session_id']:
   before=unit[max(0,j-4):j+3];bound.append({'unit_position':j,'delta_s':p['ts']-unit[j-1]['ts'],'packet_context':[{'file_id':v['file_id'],'index':v['index'],'ts':v['ts'],'flags':v['flags'],'payload_len':v['payload_len'],'src':v['src'],'isn':v['isn'],'session_id':v['session_id']} for v in before]})
 byteobs={field:{'fwd':sum(p[field] for p in unit if p['src']==origin),'bwd':sum(p[field] for p in unit if p['src']!=origin)} for field in bytegood}
 result['representative_window']={'selection':why,'start':chosen,'end':chosen+n,'start_ts':unit[0]['ts'],'end_ts':unit[-1]['ts'],'duration_us':(unit[-1]['ts']-unit[0]['ts'])*1e6,'fwd_n':sum(p['src']==origin for p in unit),'bwd_n':sum(p['src']!=origin for p in unit),'byte_variants':byteobs,'session_count':len({p['session_id'] for p in unit}),'boundary_contexts':bound,'clock_residual_options_s':[unit[0]['ts']-row['clock']-o for o in opts],'packet_membership':[{'file_id':p['file_id'],'index':p['index'],'offset':p['offset'],'caplen':p['caplen'],'raw_sha256':p['raw_sha256'],'ts':p['ts'],'flags':p['flags'],'src':p['src'],'dst':p['dst'],'proto':p['proto'],'isn':p['isn'],'payload_len':p['payload_len']} for p in unit]}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--domain',choices=['Friday','Tuesday','Wednesday'],required=True);a=p.parse_args();out=HERE/'diagnostics'/a.domain;out.mkdir(exist_ok=True);plan=json.loads((HERE/'diagnostics/FIXED_CASES.json').read_text());cases=[r for r in plan['cases'] if r['domain']==a.domain];keys={H['csv_key'](r['csv_record']) for r in cases};source=json.loads((V2/f'reconstructed/CIC_IDS/{a.domain}/source_files.json').read_text());assert len(source)==1;source=source[0];path=Path(source['path']);packets=collections.defaultdict(list);total=0;decoded_errors=collections.Counter();complete=False
 with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  for index,ts,offset,cap,wire,link in H['indexed_packets'](mm):
   total+=1
   if total%2000000==0:print(a.domain,'scanned',total,'retained',sum(len(v) for v in packets.values()),flush=True)
   key=H['fast_endpoint'](mm,offset,cap,link)
   if key not in keys:continue
   raw=bytes(mm[offset:offset+cap])
   try:
    p=H['decode'](raw,link);ip=dpkt.ethernet.Ethernet(raw).data if link==1 else dpkt.sll.SLL(raw).data if link==113 else dpkt.ip.IP(raw)
    packets[key].append(dict(index=index,ts=ts,offset=offset,caplen=cap,link=link,file_id=source['file_id'],src=p['src'],dst=p['dst'],proto=p['proto'],flags=p['flags'],isn=p['isn'],payload_len=len(ip.data.data),ip_payload_len=len(bytes(ip.data)),ip_len=p['size'],frame_len=cap,wire_len=wire,raw_sha256=hashlib.sha256(raw).hexdigest()))
    # Capture after TCP/UDP header includes Ethernet padding, unlike IP-bounded payload.
    network_offset=14 if link==1 else 16 if link==113 else 0
    if link==1:
     eth_type=struct.unpack_from('!H',raw,12)[0]
     while eth_type in (0x8100,0x88a8):eth_type=struct.unpack_from('!H',raw,network_offset+2)[0];network_offset+=4
    trans_offset=network_offset+(raw[network_offset]&15)*4
    header_len=(raw[trans_offset+12]>>4)*4 if p['proto']==6 else 8
    packets[key][-1]['capture_payload_len']=len(raw)-trans_offset-header_len
    packets[key][-1]['ethernet_padding_len']=max(0,len(raw)-network_offset-int(ip.len))
    packets[key][-1]['ip_header_len']=(raw[network_offset]&15)*4
    packets[key][-1]['transport_header_len']=header_len
   except (ValueError,dpkt.UnpackError,AttributeError,IndexError) as e:decoded_errors[key]+=1
  complete=True
 save(out/'RAW_PACKETS_BY_KEY.json',[{'key':key,'packets':ps} for key,ps in packets.items()])
 for key in packets:packets[key],_=assign_sessions('CIC_IDS',key,packets[key])
 save(out/'PACKETS_BY_KEY.json',[{'key':key,'packets':ps} for key,ps in packets.items()])
 clock=json.loads((V2/f'reconstructed/CIC_IDS/{a.domain}/clock_audit.json').read_text());reports=[]
 for case in cases:
  row=case['csv_record'];key=H['csv_key'](row);r={k:v for k,v in case.items() if k!='csv_record'};r.update(csv_record=row,source_file=source,packet_search_complete=complete and not decoded_errors[key],decode_errors_for_key=decoded_errors[key],diagnosis=details(row,packets.get(key,[]),clock));reports.append(r)
 save(out/'CASE_TRACES.json',reports)
 # Calibration evidence uses fixed accepted cases; not re-estimated for acceptance.
 residuals=[]
 for r in reports:
  w=r['diagnosis'].get('representative_window')
  if w:residuals.append({'case_id':r['case_id'],'role':r['role'],'v2_status':r['v2_status'],'window_selection':w['selection'],'clock_residual_options_s':w['clock_residual_options_s']})
 save(out/'CLOCK_QUALITY.json',dict(v2_clock_status=clock,scope='original v2 anchors and fixed diagnostic-case residuals; no model inputs/scores',source_file=source,case_residuals=residuals,warning='minute-precision windows do not establish subsecond clock accuracy'))
 save(out/'COMPLETE.json',dict(domain=a.domain,cases=len(reports),packet_records_scanned=total,selected_keys=len(keys),retained_packets=sum(len(v) for v in packets.values()),decode_errors=sum(decoded_errors.values()),source_sha256_recorded=source['sha256'],heldout_cases_not_used_for_rule_choice=True))
 print(a.domain,'COMPLETE',len(reports),flush=True)
if __name__=='__main__':main()

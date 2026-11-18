"""Boundary and fail-closed checks; synthetic fixtures are not benchmark data."""
import tempfile,json,copy
from pathlib import Path
from rules_v3 import *
HERE=Path(__file__).resolve().parent

def p(t,i,flags=16,payload=0,src=('192.0.2.1',1234),dst=('192.0.2.2',80),file='f',isn=10,proto=6):
 return dict(ts=t,index=i,flags=flags,payload_len=payload,capture_payload_len=payload,src=src,dst=dst,file_id=file,isn=isn,proto=proto,raw_sha256=stable([file,i,t]))
def main():
 checks=[];key=(6,('192.0.2.1',1234),('192.0.2.2',80))
 r,ev,rel=assign_sessions('SYNTH','domain',key,[p(0,0,2),p(.1,1,2),p(.2,2,18,src=('192.0.2.2',80),dst=('192.0.2.1',1234)),p(.3,3)])
 assert len({x['session_id'] for x in r})==1;checks.append('SYN retransmission retains session')
 fin=[p(0,0,2),p(.1,1,17),p(.2,2,17,src=('192.0.2.2',80),dst=('192.0.2.1',1234)),p(.201,3),p(2.21,4)]
 r,_,_=assign_sessions('SYNTH','domain',key,fin);assert r[3]['session_id']==r[2]['session_id'] and r[3]['closed_control_tail'] and r[4]['session_id']!=r[3]['session_id'];checks.append('both FIN retain bounded final ACK, grace not indefinite')
 r,_,_=assign_sessions('SYNTH','domain',key,[p(0,0,payload=14),p(.0001,1,4),p(.0002,2,17)]);assert len({x['session_id'] for x in r})==1 and r[-1]['closed_control_tail'] and r[0]['missing_initial_syn'];checks.append('RST retains control-only FIN/ACK tail; missing SYN marked')
 r,_,_=assign_sessions('SYNTH','domain',key,[p(0,0,2),p(1,1,4),p(90,2,2,isn=99)]);assert r[2]['session_id']!=r[0]['session_id'];checks.append('new SYN after RST remains separate despite CSV aggregation')
 r,_,rel=assign_sessions('SYNTH','domain',key,[p(0,0),p(120,1),p(240.001,2)]);assert r[0]['session_id']==r[1]['session_id'] and r[2]['session_id']!=r[1]['session_id'] and rel[0]['must_group_together'];checks.append('TCP idle boundary and uncertain continuation grouping edge')
 r,_,_=assign_sessions('SYNTH','domain',(17,*key[1:]),[p(0,0,0,proto=17),p(30,1,0,proto=17),p(60.001,2,0,proto=17)]);assert r[0]['session_id']==r[1]['session_id'] and r[2]['session_id']!=r[1]['session_id'];checks.append('UDP >30s idle boundary')
 a,_,_=assign_sessions('SYNTH','A',key,[p(0,0)]);b,_,_=assign_sessions('SYNTH','B',key,[p(0,0)]);assert a[0]['session_id']!=b[0]['session_id'];checks.append('unrelated provenance domains cannot collide on tuple/ordinal')
 dup=p(0,0);cross=copy.deepcopy(dup);cross['file_id']='g';same=copy.deepcopy(dup);same['index']=1
 r,_,_=assign_sessions('SYNTH','domain',key,[dup,cross,same]);assert len(r)==2;checks.append('cross-file duplicate observation removed, same-file occurrence retained')
 with tempfile.TemporaryDirectory() as td:
  log=DecisionLog(Path(td)/'log.jsonl');base={'Source IP':'192.0.2.1','Destination IP':'192.0.2.2','Source Port':1234,'Destination Port':80,'Protocol':6,'Total Fwd Packets':1,'Total Backward Packets':0,'Total Length of Fwd Packets':0,'Total Length of Bwd Packets':0,'Flow Duration':0,'Timestamp':'7/7/2017 4:04'}
  one=normalize_row(base,'csv',2,'SYNTH',log);assert one and one['n_packets']==1
  two=normalize_row({**base,'Total Fwd Packets':2},'csv',8,'SYNTH',log);assert two and two['n_packets']==2
  large=normalize_row({**base,'Total Fwd Packets':10001},'csv',3,'SYNTH',log);assert large and large['n_packets']==10001;checks.append('1/2-packet and >10000-packet flows not invalid by count')
  assert normalize_row({**base,'Timestamp':'invalid'},'csv',4,'SYNTH',log) is None
  assert normalize_row({**base,'Source Port':1.5},'csv',5,'SYNTH',log) is None
  iso=normalize_row({**base,'Timestamp':'2018-11-03 11:36:28.607338'},'csv',6,'SYNTH',log);assert pd.Timestamp(iso['clock'],unit='s',tz='UTC').month==11;checks.append('timestamp/field rejections structured; ISO date not day/month swapped')
  stats=[0.]*77;stats[0]=float('nan');row=normalize_row(base,'csv',7,'SYNTH',log,stats=stats);assert row['stat'][0]==0 and row['nonfinite_features_replaced']==1;checks.append('fixed nonfinite replacement counted')
  log.close();lines=[json.loads(x) for x in (Path(td)/'log.jsonl').read_text().splitlines()];assert all(set(['source_file_id','csv_row_id','label','stage','reason','detail','decision'])<=set(x) for x in lines)
  ps,_,_=assign_sessions('SYNTH','domain',key,[p(one['clock']+10800,0)])
  assert decide(one,ps,[one],[10800],search_complete=False,payload_metric='payload_len')['reason']=='incomplete_candidate_search'
  assert decide(one,ps,[one],[10800])['reason']=='uncertified_payload_schema'
  assert decide(one,ps,[one,one],[10800],search_complete=True,payload_metric='payload_len')['reason']=='ambiguous_csv_record_or_label'
  assert decide(one,ps,[one],[10800],search_complete=True,payload_metric='payload_len',coverage_cutoff=ps[0]['ts']+1)['reason']=='csv_time_uncertainty_touches_truncated_coverage';checks.append('overflow/incomplete search, uncertified bytes, label ambiguity and truncated coverage fail closed')
 from packet_io_v3 import decode_accounting
 import dpkt
 ip=dpkt.ip.IP(src=b'\xc0\x00\x02\x01',dst=b'\xc0\x00\x02\x02',p=6,data=dpkt.tcp.TCP(sport=1234,dport=80,flags=16));ip.len=len(ip)
 eth=dpkt.ethernet.Ethernet(src=b'\x01'*6,dst=b'\x02'*6,type=0x0800,data=ip);raw=bytes(eth);raw=raw+b'\0'*max(0,60-len(raw));account=decode_accounting(raw,1);assert account['payload_len']==0 and account['capture_payload_len']==6 and account['ethernet_padding_len']==6;checks.append('raw Ethernet padding distinguished from IP payload exactly')
 report={'passed':True,'synthetic_only':True,'checks':checks};(HERE/'diagnostics/BOUNDARY_TESTS.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()

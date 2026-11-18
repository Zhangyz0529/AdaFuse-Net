"""Leading-record scope quarantine, not silent malformed-packet deletion."""
import tempfile,json,struct,socket
from pathlib import Path
import dpkt
import packet_rebuild as r

def packet(proto,port):
 tr=dpkt.tcp.TCP(sport=port,dport=80,flags=2,seq=port) if proto==6 else dpkt.udp.UDP(sport=port,dport=80,data=b'a')
 if proto==17:tr.ulen=len(tr)
 ip=dpkt.ip.IP(src=socket.inet_aton('192.0.2.1'),dst=socket.inet_aton('192.0.2.2'),p=proto,data=tr);ip.len=len(ip);return bytes(dpkt.ethernet.Ethernet(src=b'\1'*6,dst=b'\2'*6,type=0x800,data=ip))
def main():
 original=r.HERE
 with tempfile.TemporaryDirectory(prefix='ustc_boundary_test_') as td:
  root=Path(td);path=root/'raw.pcap';hdr=b'\xd4\xc3\xb2\xa1'+struct.pack('<HHiiii',2,4,0,0,65535,1)
  with path.open('wb') as f:
   f.write(hdr);f.write(struct.pack('<IIII',0,15,4,60)+bytes(4))
   for ts,proto,port in [(1,6,1000),(5,17,1001),(40,17,1002)]:
    p=packet(proto,port);f.write(struct.pack('<IIII',ts,0,len(p),len(p))+p)
  r.HERE=root;src=dict(path=str(path),label='Htbot',file_id='raw',relative_path='raw.pcap');r.build('USTC','fixture',[src]);out=root/'outputs/USTC/fixture';x=json.loads((out/'REBUILD_COMPLETE.json').read_text());assert x['accepted']==2 and x['rejected_observed_sessions']==1 and x['counts_close'];assert len(x['packet_counts']['leading_boundaries'])==1;records=[json.loads(l) for l in (out/'accepted.jsonl').open()];assert all(x['source_leading_record_incomplete'] for x in records);assert any('unresolved_left_boundary_session' in l for l in (out/'session_rejections.jsonl').open())
  # Same malformed record in the interior must not gain the leading exception.
  data=path.read_bytes();path2=root/'interior.pcap';body=data[24:];first=body[:20];rest=body[20:];path2.write_bytes(hdr+rest+first);src['path']=str(path2);r.build('USTC','interior',[src]);x=json.loads((root/'outputs/USTC/interior/REBUILD_COMPLETE.json').read_text());assert x['status']=='BLOCKED_SOURCE_DECODE' and x['accepted']==0;r.HERE=original
 print(json.dumps(dict(passed=True,checks=['exact leading malformed record retained in rejection log','new SYN after boundary accepted','near-boundary UDP rejected, later UDP accepted','interior malformed frame still blocks source','all source and session counts close'],synthetic_only=True)))
if __name__=='__main__':main()

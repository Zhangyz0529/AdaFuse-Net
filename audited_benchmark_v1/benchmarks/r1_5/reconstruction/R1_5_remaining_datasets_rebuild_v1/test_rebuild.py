"""Real pipeline exercise on synthetic v4/v6 frames; original data untouched."""
import json,tempfile,struct,socket
from pathlib import Path
import dpkt,numpy as np
import packet_rebuild as r

def main():
 ip=dpkt.ip.IP(src=socket.inet_aton('192.0.2.1'),dst=socket.inet_aton('192.0.2.2'),p=6,data=dpkt.tcp.TCP(sport=123,dport=80,flags=2,seq=1));ip.len=len(ip);frame=bytes(dpkt.ethernet.Ethernet(src=b'\1'*6,dst=b'\2'*6,type=0x800,data=ip));frame=frame.ljust(60,b'\0');p=r.decode(frame,1);anon=r.anonymous(frame,1,p);q=r.decode(anon,1);assert q['src']==['0.0.0.0',0] and q['dst']==['0.0.0.0',0] and len(anon)==len(frame);assert dpkt.in_cksum(anon[14:34])==0
 udp=dpkt.udp.UDP(sport=100,dport=200,data=b'test');udp.ulen=len(udp);ip6=dpkt.ip6.IP6(src=socket.inet_pton(socket.AF_INET6,'2001:db8::1'),dst=socket.inet_pton(socket.AF_INET6,'2001:db8::2'),nxt=17,data=udp);ip6.plen=len(udp);frame6=bytes(dpkt.ethernet.Ethernet(src=b'\1'*6,dst=b'\2'*6,type=0x86dd,data=ip6));p6=r.decode(frame6,1);anon6=r.anonymous(frame6,1,p6)
 assert r.decode(anon6,1)['src']==['::',0];segment=anon6[p6['transport_offset']:p6['ip_end']];assert dpkt.in_cksum(b'\0'*32+struct.pack('!I3xB',len(segment),17)+segment)==0
 original=r.HERE
 with tempfile.TemporaryDirectory(prefix='remaining_rebuild_test_') as td:
  root=Path(td);pcap=root/'fixture.pcap'
  with pcap.open('wb') as f:
   writer=dpkt.pcap.Writer(f);writer.writepkt(frame,10);writer.writepkt(frame6,11)
  r.HERE=root
  source=dict(path=str(pcap),relative_path='fixture.pcap',label='Benign',file_id='fixture')
  r.build('USTC','synthetic',[source]);out=root/'outputs/USTC/synthetic';summary=json.loads((out/'REBUILD_COMPLETE.json').read_text());assert summary['accepted']==2 and summary['counts_close'] and summary['session_counts_close'];rows=[json.loads(x) for x in (out/'accepted.jsonl').open()];assert all(len(x['stat_raw'])==25 and len(bytes.fromhex(x['seq_hex']))==512 and x['packet_count']==1 for x in rows);assert {v for x in rows for v in x['ip_versions']}=={4,6}
  r.HERE=original
 print(json.dumps(dict(passed=True,checks=['IPv4 anonymization and header checksum','IPv6 UDP anonymization and pseudoheader checksum','end-to-end source/session/stat25/seq512/membership','single-packet observations retained','packet and session count closure'],scope='synthetic only, not real split success')))
if __name__=='__main__':main()

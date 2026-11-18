"""Synthetic end-to-end29-feature/CSV/raw-protocol anchor preflight."""
import csv,datetime,json,sys,tempfile
from pathlib import Path
import dpkt
import packet_rebuild as r
import doh_alignment_preflight as d

def main():
 original=r.HERE;argv=sys.argv[:]
 with tempfile.TemporaryDirectory(prefix='doh29_preflight_') as td:
  root=Path(td);r.HERE=root/'out';r.HERE.mkdir();data=root/'data';csvdir=data/'DoHBrw-2020/Clean_CSVs';csvdir.mkdir(parents=True);metadata=[];base=1577836800
  real=original.parents[2]/'data/DoHBrw-2020/Clean_CSVs/dns2tcp.csv'
  names=next(csv.reader(real.open()));print('schema',len(names))
  for i,name in enumerate(d.LABELS):
   packet=dpkt.udp.UDP(sport=1000+i,dport=443,data=b'');packet.ulen=len(packet);ip=dpkt.ip.IP(src=b'\xc0\x00\x02\x01',dst=b'\xc0\x00\x02\x02',p=17,data=packet);ip.len=len(ip);raw=bytes(dpkt.ethernet.Ethernet(src=b'\1'*6,dst=b'\2'*6,type=0x800,data=ip)).ljust(60,b'\0');stamp=base+i*60;rel='DoHBrw-2020/PCAPs/extracted_pcaps/malicuous/fixture'+str(i)+'_doh1_2020.pcap';path=data/rel;path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('wb') as f:writer=dpkt.pcap.Writer(f);writer.writepkt(raw,stamp)
   metadata.append(dict(dataset='CIC_DOH',source_file=rel,bytes=path.stat().st_size,first_epoch=stamp,last_epoch=stamp))
   row={n:0 for n in names};row.update(SourceIP='192.0.2.1',DestinationIP='192.0.2.2',SourcePort=1000+i,DestinationPort=443,TimeStamp=datetime.datetime.fromtimestamp(stamp,datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),Duration=0,FlowBytesSent=60,FlowBytesReceived=0)
   with (csvdir/name).open('w') as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerow(row)
  metapath=root/'metadata.csv'
  with metapath.open('w') as f:w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
  sys.argv=['test','--data-root',str(data),'--capture-metadata',str(metapath)];d.main();out=r.HERE/'CIC_DOH/alignment_preflight';rows=json.loads((out/'ANCHOR_RESULTS.json').read_text());assert len(rows)==5 and all(x['protocol']==17 and x['decision']=='unique_metadata_anchor' for x in rows);report=json.loads((out/'CALIBRATION_STATUS.json').read_text());assert not report['formal_acceptance_allowed'];assert not (out/'accepted.jsonl').exists();r.HERE=original;sys.argv=argv
 print(json.dumps(dict(passed=True,checks=['all five CSV sources indexed with29 features','UDP observed rather than defaultTCP','unique time/duration/byte anchors','preflight never exports formal accepted samples'],synthetic_only=True)))
if __name__=='__main__':main()

"""End-to-end synthetic PCAP/CSV exercise, no original data or frozen outputs edited."""
import csv, datetime, json, tempfile
from pathlib import Path
import dpkt
import run as r

def main():
 with tempfile.TemporaryDirectory(prefix='v3_full_check_') as td:
  root=Path(td);out=root/'outputs';out.mkdir();pcap=root/'source.pcap';csvpath=root/'source.csv'
  # Use the real feature column order; generated values and packets are synthetic.
  header=next(csv.reader((r.V3.parent.parent.parent/'data/CIC_IDS/GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv').open(encoding='cp1252')))
  names=[x.strip() for x in header];rows=[];base=1499083200
  with pcap.open('wb') as f:
   writer=dpkt.pcap.Writer(f)
   for i in range(22):
    tcp=dpkt.tcp.TCP(sport=10000+i,dport=80,flags=16,seq=i)
    ip=dpkt.ip.IP(src=b'\xc0\x00\x02\x01',dst=b'\xc0\x00\x02\x02',p=6,data=tcp);ip.len=len(ip)
    raw=bytes(dpkt.ethernet.Ethernet(src=b'\x01'*6,dst=b'\x02'*6,type=0x0800,data=ip));raw=raw.ljust(60,b'\0');writer.writepkt(raw,base+i)
    row={n:0 for n in names};row.update({'Flow ID':str(i),'Source IP':'192.0.2.1','Destination IP':'192.0.2.2','Source Port':10000+i,'Destination Port':80,'Protocol':6,'Timestamp':datetime.datetime.fromtimestamp(base+i-10800,datetime.timezone.utc).isoformat(),'Flow Duration':0,'Total Fwd Packets':1,'Total Backward Packets':0,'Total Length of Fwd Packets':6,'Total Length of Bwd Packets':0,'Label':'BENIGN'})
    if i==20:row['Label']='outside_taxonomy'
    if i==21:row['Source Port']='invalid'
    rows.append(row)
  with csvpath.open('w') as f:
   w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
  job={'csvs':[csvpath],'pcaps':[pcap],'domain':'synthetic'};db=r.connect(out)
  registry=[dict(source_relative_path='source.pcap',source_file_id='synthetic_file',content_sha256=r.sha(pcap))]
  r.ingest_csv(job,'CIC_IDS',root,out,db);r.scan_packets(job,'CIC_IDS',root,out,db,registry,[]);policy=r.calibrate(job,'CIC_IDS',out,db)
  assert policy['metric']=='capture_payload_len' and policy['status']=='CERTIFIED_FOR_DECLARED_SCOPE',policy
  r.reconstruct(job,'CIC_IDS',root,out,db,'synthetic_domain',policy)
  result=json.loads((out/'REBUILD_COMPLETE.json').read_text());assert result['accepted']==20 and result['matching_or_scope_rejected']==1 and result['parse_rejected']==1 and result['counts_close'],result
  accepted=[json.loads(s) for s in (out/'accepted.jsonl').open()];assert all(x['packet_count']==1 and x['missing_initial_syn'] for x in accepted)
  assert len(list((out/'accepted_packet_membership.jsonl').open()))==20
  print(json.dumps({'passed':True,'scope':'synthetic end-to-end, not full-source certification','result':result}))
if __name__=='__main__':main()

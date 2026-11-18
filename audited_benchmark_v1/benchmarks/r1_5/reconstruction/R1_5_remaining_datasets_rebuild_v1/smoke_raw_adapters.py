"""Bounded raw-header/session/25-statistic checks, NOT formal dataset generation."""
import ast,collections,itertools,json,mmap,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;V3=HERE.parent/'R1_5_session_rebuild_v3';sys.path.insert(0,str(V3));sys.path.insert(0,str(V3.parent/'R1_5_session_rebuild_v2'))
from rebuild import H
from packet_io_v3 import decode_accounting
from rules_v3 import assign_sessions

def tracker(dataset):
 source=HERE.parent/('USTC/generate_ustc_multimodal.py' if dataset=='USTC' else 'ISCX/generate_iscx_pkl.py');tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlowTracker'];env={'np':np,'MAX_LEN':512};exec(compile(tree,str(source),'exec'),env);return env['FlowTracker']
def main():
 root=Path(sys.argv[1]);jobs=[]
 for dataset,dirs in [('USTC',['USTC-TFC2016-master/Benign','USTC-TFC2016-master/Malware']),('ISCX',['ISCX/pcap'])]:
  jobs.extend((dataset,p) for d in dirs for p in sorted((root/d).iterdir()) if p.suffix.lower() in ['.pcap','.pcapng'])
 doh=root/'DoHBrw-2020/PCAPs/extracted_pcaps';selected={}
 for p in sorted(doh.rglob('*')):
  if p.suffix.lower() not in ['.pcap','.pcapng']:continue
  rel=p.relative_to(doh);group=rel.parts[0]
  if group.lower() in ['malicuous','malicious']:
   name=p.name.lower();tool='dns2tcp' if 'dns2tcp' in name else 'dnscat2' if 'dnscat2' in name or 'dsncat2' in name else 'iodine' if 'iodine' in name else 'other';group+=':'+tool+':'+('baseline' if 'baseline' in name else 'tunnel' if 'tunnel' in name else 'unspecified')
  selected.setdefault(group,p)
 jobs.extend(('CIC_DOH',p) for p in selected.values());results=[]
 for ds,path in jobs:
  counts=collections.Counter();bykey=collections.defaultdict(list);failure=None;n=0
  try:
   with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
    for index,ts,offset,cap,wire,link in itertools.islice(H['indexed_packets'](mm),1000):
     n+=1;key=H['fast_endpoint'](mm,offset,cap,link)
     try:p=decode_accounting(bytes(mm[offset:offset+cap]),link)
     except ValueError as exc:counts[str(exc)]+=1;continue
     p.update(file_id=str(path.relative_to(root)),index=index,ts=ts,offset=offset,caplen=cap,link=link);bykey[key].append(p);counts['decoded_ipv4_tcp_udp']+=1
   session_count=0;finite=True;dims=set()
   for key,ps in bykey.items():
    assigned,events,edges=assign_sessions(ds,'SMOKE_SOURCE_ONLY',key,ps);session_count+=len({p['session_id'] for p in assigned})
    if ds in ['USTC','ISCX']:
     cls=tracker(ds)
     for sid in {p['session_id'] for p in assigned}:
      t=cls(str(key))
      for p in assigned:
       if p['session_id']==sid:t.add_packet(b'\0'*p['caplen'],p['ts'],str(tuple(p['src']))>str(tuple(p['dst'])))
      features=t.compute_statistical_features();dims.add(len(features));finite=finite and bool(np.isfinite(features).all())
   if ds in ['USTC','ISCX']:assert not dims or dims=={25}
  except Exception as exc:failure=repr(exc);session_count=None;finite=False;dims=set()
  results.append(dict(dataset=ds,path=str(path.relative_to(root)),header_records_read=n,counts=dict(counts),observed_sessions_within_prefix=session_count,legacy_stat_dimensions=sorted(dims),finite_statistics=finite,error=failure,scope='first1000 records; session ends at sample boundary not certified; statistic lengths only, zero placeholders not final anonymized inputs'))
 summary={}
 for ds in ['USTC','ISCX','CIC_DOH']:
  rr=[x for x in results if x['dataset']==ds];summary[ds]=dict(files_checked=len(rr),files_with_read_error=sum(x['error'] is not None for x in rr),files_with_decodable_packets=sum(x['counts'].get('decoded_ipv4_tcp_udp',0)>0 for x in rr),decoded_packets=sum(x['counts'].get('decoded_ipv4_tcp_udp',0) for x in rr),unsupported_network_records=sum(x['counts'].get('unsupported_network_family',0) for x in rr),formal_rebuild_validated=False)
 report=dict(time=time.time(),summary=summary,files=results,scope='bounded adapter smoke, no label/window join or independent-split validation; DoH29 statistics not reconstructed',whole_pipeline_pass_claim=False);(HERE/'RAW_ADAPTER_SMOKE.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()

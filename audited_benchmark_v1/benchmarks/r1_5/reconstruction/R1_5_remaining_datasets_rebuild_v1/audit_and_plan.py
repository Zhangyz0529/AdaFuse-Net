"""Read-only source inventory and dataset-specific rebuild prerequisites.
No unsafe pickle loading, model training, full PCAP scan or historical mutation.
Existing capture-header metadata is explicitly previous evidence, not a new join.
"""
import argparse,ast,csv,hashlib,json,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--paper-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 previous=a.paper_root/'R1_5_leakage_audited_reproduction_v1/source_audit';oldmeta=list(csv.DictReader((previous/'capture_metadata.csv').open()));byrel={r['source_file']:r for r in oldmeta}
 specs={'USTC':{'roots':['USTC-TFC2016-master/Benign','USTC-TFC2016-master/Malware'],'generator':a.paper_root/'USTC/generate_ustc_multimodal.py','dim':25},'ISCX':{'roots':['ISCX/pcap'],'generator':a.paper_root/'ISCX/generate_iscx_pkl.py','dim':25},'CIC_DOH':{'roots':['DoHBrw-2020/PCAPs/extracted_pcaps','DoHBrw-2020/Clean_CSVs'],'generator':a.data_root/'DoHBrw-2020/generate_doh2020_multimodal.py','dim':29}}
 combined=[];overview=[]
 for dataset,spec in specs.items():
  out=a.output/dataset;out.mkdir(exist_ok=True);files=[];headers=[];labels=json.loads((previous/(dataset+'_pkl_identity.json')).read_text())['labels'];missing=[]
  for directory in spec['roots']:
   root=a.data_root/directory
   if not root.exists():missing.append(directory);continue
   for f in sorted(root.rglob('*')):
    if not f.is_file() or f.suffix.lower() not in ['.pcap','.pcapng','.csv','.zip','.7z']:continue
    rel=str(f.relative_to(a.data_root));kind='pcap' if f.suffix.lower() in ['.pcap','.pcapng'] else 'csv' if f.suffix.lower()=='.csv' else 'archive';m=byrel.get(rel,{})
    entry=dict(dataset=dataset,relative_path=rel,source_file_id=hashlib.sha256((dataset+'|'+rel).encode()).hexdigest(),kind=kind,bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns,content_sha256='NOT_YET_COMPUTED',previous_header_scan_status=m.get('status','NOT_AVAILABLE'),previous_packet_records=m.get('packet_records',''),previous_first_epoch=m.get('first_epoch',''),previous_last_epoch=m.get('last_epoch',''),provenance_domain='UNRESOLVED_DO_NOT_SPLIT',label_status='REQUIRES_SOURCE_MAPPING')
    files.append(entry)
    if kind=='csv' and (dataset=='CIC_DOH' or len(headers)<5):
     with f.open(encoding='cp1252',newline='') as handle:
      reader=csv.DictReader(handle);sample=next(reader,{})
      headers.append(dict(path=rel,columns=reader.fieldnames,first_record_metadata={k:v for k,v in sample.items() if k is not None and k.strip().lower() in ['timestamp','time_stamp','protocol','duration','label','doh']},first_row_extra_fields=sample.get(None,[]),scope='header and one record only, not full-schema validation'))
  source=spec['generator'];text=source.read_text();ast.parse(text)
  facts=dict(generator=str(source),sha256=sha(source),identity_to_historical_execution='UNCONFIRMED',legacy_stat_dim=spec['dim'],legacy_pkl_labels=labels,missing_roots=missing,raw_files=len([f for f in files if f['kind']=='pcap']),raw_bytes=sum(f['bytes'] for f in files if f['kind']=='pcap'),csv_files=len([f for f in files if f['kind']=='csv']),generator_has_timeout_text=any(s in text for s in ['idle_timeout','tcp_idle','udp_idle']),generator_legacy_min_three='>= 3' in text or '>=3' in text,full_rebuild_started=False,split_generated=False,training_allowed=False)
  if dataset=='CIC_DOH':facts.update(protocol_column_present=any(any(c.strip().lower()=='protocol' for c in h['columns']) for h in headers),legacy_overwrite_expression="global_dict[flow_key] =",overwrite_expression_present='global_dict[flow_key] =' in text,required_join=['retain all CSV candidates','verify raw TCP/UDP protocol per candidate window','calibrate timestamp/duration/byte definitions per source domain','verify sequence +29 statistics from same record','reject ambiguous CSV/packet mappings; do not pick attack label by preference'])
  else:facts.update(required_join=['verify publisher/file-derived class mapping and source shard relations','reconstruct observed TCP/UDP sessions before computing25 statistics','preserve original file/packet ordinals and hashes','retain 1/2-packet observations with explicit feature definitions; no silent >=3 filter','check IPv6 support against raw scope; no unreported IPv4-only shrinkage'])
  if dataset=='ISCX':facts['reference_limitation']='facebook_audio4 has one candidate capture in previous inventory; capture-disjoint three-way claim unavailable without new evidence'
  dump(out/'AUDIT.json',facts);dump(out/'CSV_HEADERS.json',headers)
  with (out/'source_inventory.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(files[0]) if files else ['dataset']);w.writeheader();w.writerows(files)
  combined.extend(files);overview.append(facts)
 with (a.output/'source_inventory_all.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
 dump(a.output/'AUDIT_SUMMARY.json',dict(time=time.time(),datasets=overview,scope='resource/code/header audit, not completed reconstruction or session certification',priority='no new full PCAP scanning while CIC_IDS is prioritized',formal_acceptance_allowed=False))
 print(json.dumps([{k:r[k] for k in ['legacy_stat_dim','raw_files','raw_bytes','csv_files','full_rebuild_started']} for r in overview]))
if __name__=='__main__':main()

"""Inspect local archive integrity and create explicit acquisition-domain registry.
No raw file modified. Archive metadata is not publisher authentication.
"""
import argparse,hashlib,json,csv,struct,zipfile,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2'
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def streamhash(handle):
 h=hashlib.sha256();crc=0;n=0
 for b in iter(lambda:handle.read(4<<20),b''):h.update(b);crc=zlib.crc32(b,crc);n+=len(b)
 return dict(bytes=n,sha256=h.hexdigest(),crc32=crc&0xffffffff)
def boundary(path):
 n=0;last=None;first=None
 with path.open('rb') as f:
  magic=f.read(4);modes={b'\xd4\xc3\xb2\xa1':('<',1e6),b'\xa1\xb2\xc3\xd4':('>',1e6),b'\x4d\x3c\xb2\xa1':('<',1e9),b'\xa1\xb2\x3c\x4d':('>',1e9)}
  if magic not in modes:raise ValueError('Expected classic pcap')
  endian,unit=modes[magic];hdr=f.read(20)
  if len(hdr)!=20:raise ValueError('Incomplete global header')
  size=path.stat().st_size
  while True:
   pos=f.tell();h=f.read(16)
   if not h:return dict(status='COMPLETE',valid_packets=n,complete_record_prefix_bytes=pos,first_epoch=first,last_complete_epoch=last)
   if len(h)!=16:return dict(status='TRUNCATED_RECORD_HEADER',valid_packets=n,complete_record_prefix_bytes=pos,remaining_bytes=len(h),first_epoch=first,last_complete_epoch=last)
   sec,sub,cap,wire=struct.unpack(endian+'IIII',h);ts=sec+sub/unit
   if f.tell()+cap>size:return dict(status='TRUNCATED_RECORD_DATA',valid_packets=n,complete_record_prefix_bytes=pos,partial_record_timestamp=ts,declared_caplen=cap,available_caplen=size-f.tell(),wirelen=wire,first_epoch=first,last_complete_epoch=last)
   f.seek(cap,1);n+=1;first=ts if first is None else min(first,ts);last=ts

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();out=HERE/'diagnostics';out.mkdir(exist_ok=True);results=[]
 for day,name in [('Test_Day_03-11','SAT-03-11-2018_0145.pcap'),('Train_Day_01-12','SAT-01-12-2018_0818.pcap')]:
  local=a.data_root/'CIC-DDoS-2019'/day/'PCAPs'/name
  with local.open('rb') as f:localhash=streamhash(f)
  report=dict(source_file=str(local),local=localhash,record_boundary=boundary(local),archive_matches=[])
  for archive in sorted(local.parent.glob('*.zip')):
   with zipfile.ZipFile(archive) as z:
    matches=[m for m in z.infolist() if Path(m.filename).name in [name,Path(name).stem]]
    for member in matches:
     try:
      with z.open(member) as f:content=streamhash(f)
      entry=dict(archive=str(archive),member=member.filename,declared_size=member.file_size,declared_crc32=member.CRC,decompressed=content,read_to_eof_crc_verified=True,bytes_equal_to_local=content['sha256']==localhash['sha256'])
     except Exception as e:entry=dict(archive=str(archive),member=member.filename,error=str(e),read_to_eof_crc_verified=False)
     report['archive_matches'].append(entry)
  report['conclusion']='LOCAL_ARCHIVE_MEMBER_ALSO_TRUNCATED' if any(e.get('bytes_equal_to_local') and e.get('read_to_eof_crc_verified') for e in report['archive_matches']) and report['record_boundary']['status']!='COMPLETE' else 'REQUIRES_MORE_SOURCE_VERIFICATION'
  report['formal_policy']='BLOCK_ACQUISITION_DOMAIN. Complete-record prefix can be diagnosed but is NOT automatically approved as formal data; scope decision must be frozen separately.'
  results.append(report);print(name,report['conclusion'],report['record_boundary'],flush=True)
 (out/'CIC_DDOS_TRUNCATION_FORENSICS.json').write_text(json.dumps(results,indent=2))
 # Registry uses logical, relocation-stable identities; content hashes are separate evidence.
 rows=[];evidence=[]
 source_meta={}
 for f in (V2/'reconstructed/CIC_IDS').glob('*/source_files.json'):
  for r in json.loads(f.read_text()):source_meta[str(Path(r['path']).resolve())]=r
 for dataset in ['CIC_IDS','CIC_DDOS']:
  if dataset=='CIC_IDS':
   groups=[(day,[r for r in sorted((a.data_root/'CIC_IDS/pcap').glob('*.pcap')) if r.name.lower()==f'{day}-workinghours.pcap'.lower()],[]) for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']]
  else:groups=[(day,sorted((a.data_root/f'CIC-DDoS-2019/{day}/PCAPs').glob('*.pcap')),sorted((a.data_root/f'CIC-DDoS-2019/{day}/PCAPs').glob('*.zip'))) for day in ['Test_Day_03-11','Train_Day_01-12']]
  for family,files,archives in groups:
   domain=stable(['provenance_registry_v3',dataset,family]);members={};archive_evidence=[]
   for archive in archives:
    with zipfile.ZipFile(archive) as z:
     meta=[dict(name=m.filename,size=m.file_size,crc32=m.CRC) for m in z.infolist() if not m.is_dir()];archive_evidence.append(dict(path=str(archive.relative_to(a.data_root)),central_directory_metadata_sha256=stable(meta),members=len(meta)))
     for m in z.infolist():members.setdefault(Path(m.filename).name,[]).append((archive,m))
   for f in files:
    rel=str(f.relative_to(a.data_root));mid=stable(['source_file_v3',dataset,rel]);meta=source_meta.get(str(f.resolve()));matching=members.get(f.name,members.get(f.stem,[]));conflict=len(matching)>1;matched=matching[0] if len(matching)==1 else None
    rows.append(dict(dataset=dataset,source_relative_path=rel,source_file_id=mid,provenance_domain_id=domain,acquisition_family=family,bytes=f.stat().st_size,content_sha256=meta['sha256'] if meta else next((r['local']['sha256'] for r in results if Path(r['source_file'])==f),''),archive_relative_path=str(matched[0].relative_to(a.data_root)) if matched else '',archive_member=matched[1].filename if matched else '',archive_size_matches=(f.stat().st_size==matched[1].file_size) if matched else '',source_evidence_status='v2_full_capture_hash_recorded' if meta else 'local_archive_member_size_only_not_full_content_verified' if matched and not conflict else 'AMBIGUOUS_OR_UNVERIFIED'))
   evidence.append(dict(dataset=dataset,acquisition_family=family,provenance_domain_id=domain,source_files=len(files),archive_metadata=archive_evidence,basis='single named working-day capture and paired day CSVs' if dataset=='CIC_IDS' else 'same local acquisition-day directory; contiguous archive member series; raw timestamp metadata available for review',limitation='acquisition-domain registry, not independent publisher authentication or application-session truth'))
 with (HERE/'source_domain_registry.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (HERE/'source_domain_evidence.json').write_text(json.dumps(evidence,indent=2))
 assert len({r['source_file_id'] for r in rows})==len(rows)
 (out/'DOMAIN_REGISTRY_CHECK.json').write_text(json.dumps(dict(files=len(rows),domains=len(evidence),source_id_duplicates=0,same_tuple_across_unrelated_domains='not merged by tuple identity alone',related_fragments='all fragments in each declared acquisition family share one domain',content_verification_pending_files=sum(not r['content_sha256'] for r in rows),independence_claim='none from ID uniqueness alone'),indent=2))
if __name__=='__main__':main()

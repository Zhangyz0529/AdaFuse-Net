"""Freeze local source identity; verify/restore ISCX from available ZIP members."""
import collections,json,shutil,zipfile,zlib
from pathlib import Path
import packet_rebuild as r

def crc(p):
 h=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h=zlib.crc32(b,h)
 return h&0xffffffff

def freeze(root):
 reports=[];archives=collections.defaultdict(list)
 for archive in sorted((root/'ISCX/pcap').glob('*.zip')):
  with zipfile.ZipFile(archive) as z:
   for m in z.infolist():
    if m.filename.lower().endswith(('.pcap','.pcapng')):archives[Path(m.filename).name.lower()].append(dict(archive=str(archive),member=m.filename,size=m.file_size,crc=m.CRC))
 for dataset in ['USTC','ISCX']:
  plan=r.plan(root,dataset)
  for family,files in plan.items():
   for source in files:
    p=Path(source['path']);source['original_path']=str(p);source['original_sha256']=r.sha(p);source['original_bytes']=p.stat().st_size
    if dataset=='ISCX':
     candidates=archives.get(p.name.lower(),[])
     if not candidates:candidates=[x for name,items in archives.items() if Path(name).stem==p.stem.lower() for x in items]
     if not candidates:raise RuntimeError('ISCX source lacks archive membership: '+str(p))
     currentcrc=crc(p);matched=[x for x in candidates if x['size']==p.stat().st_size and x['crc']==currentcrc]
     if not matched:
      if len({(x['size'],x['crc']) for x in candidates})!=1:raise RuntimeError('Ambiguous archive originals: '+str(p))
      m=candidates[0];target=r.HERE/'verified_sources'/'ISCX'/source['file_id'];target.parent.mkdir(parents=True,exist_ok=True)
      with zipfile.ZipFile(m['archive']) as z,z.open(m['member']) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,8<<20)
      if target.stat().st_size!=m['size'] or crc(target)!=m['crc']:raise RuntimeError('Restored archive CRC/size failure')
      source.update(path=str(target),restored_from_archive=True,restoration_reason='local source CRC/size mismatches available ZIP member',archive=m)
     else:source.update(restored_from_archive=False,archive=matched[0])
    source['sha256']=r.sha(Path(source['path']));source['provenance_domain_id']=r.stable([dataset,family]);reports.append(dict(dataset=dataset,family=family,**source))
  r.write(r.HERE/(dataset+'_SOURCE_PLAN.json'),plan)
 r.write(r.HERE/'SOURCE_IDENTITIES.json',dict(sources=reports,scope='local originals and locally available archive members, not remote publisher authentication',originals_unchanged=True));return reports
if __name__=='__main__':
 import sys
 freeze(Path(sys.argv[1]))

"""Verify Release assets, join numbered parts and extract into one directory."""
import argparse,csv,hashlib,shutil,tarfile,tempfile,fnmatch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--match',action='append',help='Optional filename glob, repeatable, e.g. *ustc*');a=p.parse_args();rows=list(csv.DictReader((a.assets/'artifact_index.csv').open()));groups={};a.output.mkdir(parents=True,exist_ok=True)
for row in rows:
 if a.match and not any(fnmatch.fnmatch(row['file'],pattern) for pattern in a.match):continue
 f=a.assets/row['file'];h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=row['sha256']:raise RuntimeError('Checksum mismatch: '+str(f))
 groups.setdefault(row['reassemble_as'] or row['file'],[]).append(f)
for name,files in groups.items():
 temp=None;path=files[0]
 try:
  if len(files)>1:
   temp=tempfile.NamedTemporaryFile(prefix='joined_',suffix='.tar.gz',dir=a.output,delete=False);path=Path(temp.name)
   for f in sorted(files):
    with f.open('rb') as stream:shutil.copyfileobj(stream,temp,8*1024*1024)
   temp.close()
  with tarfile.open(path,'r:gz') as tar:
   for member in tar.getmembers():
    target=(a.output/member.name).resolve();root=a.output.resolve()
    if root not in target.parents or member.issym() or member.islnk():raise RuntimeError('Unsafe archive entry: '+member.name)
   tar.extractall(a.output)
  print('Extracted',name,flush=True)
 finally:
  if temp is not None:path.unlink(missing_ok=True)

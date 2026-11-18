"""Check downloaded release assets against artifact_index.csv."""
import argparse,csv,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);a=p.parse_args();bad=[]
with (a.assets/'artifact_index.csv').open() as f:
 for row in csv.DictReader(f):
  path=a.assets/row['file']
  if not path.exists():bad.append((row['file'],'missing'));continue
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
  if h.hexdigest()!=row['sha256']:bad.append((row['file'],'hash mismatch'))
print('PASS' if not bad else bad);raise SystemExit(bool(bad))

"""Verify current raw sources against recorded v2 hashes, without rewriting data."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2';rows=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
for day in ['Friday','Tuesday','Wednesday']:
 root=V2/'reconstructed/CIC_IDS'/day;sources=json.loads((root/'source_files.json').read_text())
 for r in sources:
  p=Path(r['path']);actual=sha(p);rows.append(dict(domain=day,kind='pcap',path=str(p),expected_sha256=r['sha256'],actual_sha256=actual,matches=actual==r['sha256']));print(day,p.name,actual==r['sha256'],flush=True)
 for p,d in json.loads((root/'csv_selection_audit.json').read_text())['files'].items():
  actual=sha(Path(p));rows.append(dict(domain=day,kind='csv',path=p,expected_sha256=d['sha256'],actual_sha256=actual,matches=actual==d['sha256']))
(HERE/'diagnostics/SOURCE_HASH_VERIFICATION.json').write_text(json.dumps(dict(all_match=all(r['matches'] for r in rows),sources=rows),indent=2))
if not all(r['matches'] for r in rows):raise RuntimeError('Raw source differs from frozen v2')

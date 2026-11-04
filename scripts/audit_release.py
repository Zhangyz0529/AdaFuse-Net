"""Reject data, outputs, personal absolute paths or publication artifacts."""
from pathlib import Path
import re,sys,hashlib,json
ROOT=Path(__file__).resolve().parent.parent
blocked={'.csv','.png','.jpg','.pdf','.pt','.pth','.ckpt','.pkl','.npz','.npy','.pcap','.pcapng','.log','.db','.sqlite','.tex','.zip'}
violations=[];manifest={}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or '.egg-info' in str(p):continue
    relative=str(p.relative_to(ROOT))
    if p.suffix in blocked:violations.append(relative+': prohibited artifact')
    if p.suffix in {'.py','.yaml','.yml','.json','.md','.sh','.toml','.txt','.cff'}:
        text=p.read_text()
        if re.search(r'[/](?:home|Users|opt|mnt)/',text):violations.append(relative+': personal/system absolute path')
        if re.search(r'(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,})',text):violations.append(relative+': potential credential')
    if relative!='SOURCE_MANIFEST.json':manifest[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
if violations:
    print('\n'.join(violations));sys.exit(1)
(ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(version='0.1.0',files=manifest),indent=2)+'\n')
print('Release audit passed:',len(manifest),'source/configuration files; no data/results/weights/personal paths')

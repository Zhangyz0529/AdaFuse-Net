"""Fixed rule: exclude ALL rows in a cross-label exact-input group; no scores read."""
import argparse,hashlib,json,pickle,csv,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
with a.input.open('rb') as f:data=pickle.load(f)
hashes=[];labels=collections.defaultdict(set)
for r in data:
 h=hashlib.sha256(np.asarray(r['stat_raw'],dtype='float32').tobytes()+np.asarray(r['seq_feature']).tobytes()).hexdigest();hashes.append(h);labels[h].add(str(r['label']))
conflicts={h for h,l in labels.items() if len(l)>1};keep=[];removed=[]
for i,(r,h) in enumerate(zip(data,hashes)):
 if h in conflicts:removed.append(dict(source_row=i,label=str(r['label']),input_hash=h,reason='all_members_of_cross_label_exact_input_group'))
 else:keep.append(r)
with (a.output_dir/'excluded_rows.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['source_row','label','input_hash','reason']);w.writeheader();w.writerows(removed)
with (a.output_dir/'legacy_conflicts_removed_NOT_SESSION_AUDITED.pkl').open('wb') as f:pickle.dump(keep,f,protocol=4)
(a.output_dir/'CLEANUP_REPORT.json').write_text(json.dumps(dict(rule='exclude every member of every cross-label exact input group; no test metrics used',before=len(data),conflict_groups=len(conflicts),removed=len(removed),after=len(keep),purpose='diagnostic cleaned historical input only; not used by v2 reconstruction; no session isolation claim'),indent=2));print(len(conflicts),len(removed),len(keep))

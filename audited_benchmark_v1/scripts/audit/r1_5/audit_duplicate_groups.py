"""Inspect exact trusted-local PKL duplicate groups, preserving label counts."""
import argparse,pickle,hashlib,collections,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--pkl',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
with a.pkl.open('rb') as f:d=pickle.load(f)
g={}
for i,r in enumerate(d):
 s=np.asarray(r['stat_raw'],dtype='float32');q=np.asarray(r['seq_feature']);h=hashlib.sha256(s.tobytes()+q.tobytes()).hexdigest()
 if h not in g:g[h]={'n':0,'labels':collections.Counter(),'first_index':i,'all_zero_stats':bool(np.all(s==0)),'all_zero_sequence':bool(np.all(q==0)),'nonzero_seq_bytes':int(np.count_nonzero(q)),'nonzero_stats':int(np.count_nonzero(s))}
 g[h]['n']+=1;g[h]['labels'][str(r['label'])]+=1
v=[{'hash':k,**x} for k,x in g.items() if x['n']>1]
report={'pkl':str(a.pkl),'n_records':len(d),'duplicate_groups':len(v),'cross_label_duplicate_groups':sum(len(x['labels'])>1 for x in v),'duplicate_groups_detail':v,'limitation':'Input-level fingerprints only; origins and label correctness unresolved.'}
a.output.write_text(json.dumps(report,indent=2));print('groups',len(v),'cross-label',report['cross_label_duplicate_groups']);print('largest',sorted(v,key=lambda x:-x['n'])[:2])

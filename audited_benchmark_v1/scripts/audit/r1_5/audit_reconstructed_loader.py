"""Named loader reconstruction, not recovered historical partitions."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--pkl',type=Path,required=True);p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
os.environ.update(TARGET_DATASET=a.dataset,SEED=str(a.seed),SPLIT_PROTOCOL='class_disjoint');sys.path.insert(0,str(a.snapshot.resolve()))
from data_manager_logo import LOGODataManager
d=LOGODataManager(str(a.pkl),seed=a.seed,split_protocol='class_disjoint')
hashes=[hashlib.sha256(np.asarray(s,dtype='float32').tobytes()+np.asarray(q).tobytes()).hexdigest() for s,q in zip(d.raw_stats,d.seqs)]
results=[]
for fold,parts in enumerate(d.split_indices_by_round):
 for x,y in [('meta_train','meta_val'),('meta_train','meta_test'),('meta_val','meta_test')]:
  shared={hashes[i] for i in parts[x]}&{hashes[i] for i in parts[y]}
  results.append({'fold':fold,'partitions':[x,y],'sample_id_intersection':len(set(parts[x])&set(parts[y])),'model_input_hash_intersection':len(shared),'rows_in_first_with_shared_content':sum(hashes[i] in shared for i in parts[x]),'rows_in_second_with_shared_content':sum(hashes[i] in shared for i in parts[y])})
report={'dataset':a.dataset,'seed':a.seed,'protocol':'class_disjoint','scope':'reconstructed using named source snapshot; NOT recovered historical manifests','pkl':str(a.pkl),'loader_sha256':hashlib.sha256((a.snapshot/'data_manager_logo.py').read_bytes()).hexdigest(),'n_loaded':len(hashes),'overlaps':results,'limitations':['Content overlap verified for this reconstruction only','No capture/session identity available','Not proof all historical runs used this partition']}
(a.output/f'{a.dataset}_seed{a.seed}_reconstructed_split_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

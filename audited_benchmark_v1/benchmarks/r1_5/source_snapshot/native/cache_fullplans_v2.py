"""Prepare missing folds without waiting for an AdaFuse training job to start.

Plans reproduce the initial runner exactly: same Train-only IDs, class order,
GroupShuffleSplit3407, numpy generator3407,10support/15query,100steps. Existing
reference plans are checked for exact equality before new folds are generated.
"""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from cache import HERE,OLD,ROOTS,sha,save

def all_plans(ds):
 root=ROOTS[ds];g=json.loads((root/'PARTITION_GATE.json').read_text());assert g['status']=='PASS_PARTITIONS_AND_EVALUATION_POOLS';csv.field_size_limit(2**30)
 with (root/'group_components.csv').open(newline='') as f:meta={x['sample_id']:(x['label'],x['split_group_id'],i) for i,x in enumerate(csv.DictReader(f))}
 sequence=np.load(root/'sequence.npy',mmap_mode='r')
 for i in range(len(g['folds'])):
  rd=f'round{i}';out=HERE/'cache'/ds/rd
  if (out/'COMPLETE.json').exists():continue
  with (root/rd/'scaler_fit_sample_ids.csv').open(newline='') as f:ids=[x['sample_id'] for x in csv.DictReader(f)]
  classes=sorted({meta[s][0] for s in ids});sp={};qp={}
  for c in classes:
   group=[s for s in ids if meta[s][0]==c];si,qi=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=3407).split(group,groups=[meta[s][1] for s in group]));sp[c]=np.array(group)[si];qp[c]=np.array(group)[qi];assert not {meta[s][1] for s in sp[c]}&{meta[s][1] for s in qp[c]}
   assert len(sp[c])>=10 and len(qp[c])>=15
  rng=np.random.default_rng(3407);tr=[]
  for _ in range(100):
   ep=dict(classes=classes,support={},query={})
   for c in classes:ep['support'][c]=rng.choice(sp[c],10,False).tolist();ep['query'][c]=rng.choice(qp[c],15,False).tolist()
   tr.append(ep)
  refs=list((OLD/'runs'/ds/'AdaFuse-Net').glob(f'seed*/{rd}/shot*/training_episodes.json'))
  for ref in refs:assert json.loads(ref.read_text())==tr, 'Independent native plan differs from established training plan'
  ev=json.loads((root/rd/'evaluation_episodes.json').read_text());used=sorted({s for ep in tr+sum(ev.values(),[]) for field in ['support','query'] for vv in ep[field].values() for s in vv});z=np.load(root/rd/'scaled_statistics.npz');lm={s:j for j,s in enumerate(z['sample_ids'].astype(str))};ix=np.array([meta[s][2] for s in used]);stat=z['stat'][[lm[s] for s in used]];seq=np.asarray(sequence[ix]);out.mkdir(parents=True,exist_ok=True);np.save(out/'stats.npy',stat);np.save(out/'sequence.npy',seq);np.save(out/'global_indices.npy',ix);save(out/'ids.json',used);save(out/'train.json',tr);save(out/'eval.json',ev);save(out/'COMPLETE.json',dict(dataset=ds,fold=rd,records=len(used),source_root=str(root),source_gate_sha256=sha(root/'PARTITION_GATE.json'),source_eval_sha256=sha(root/rd/'evaluation_episodes.json'),training_plan_sha256=sha(out/'train.json'),training_plan_generation='Train-fitIDs+component labels/groups, exact initial episode rule3407; no model selection',existing_reference_plans_exactly_verified=len(refs),stats_sha256=sha(out/'stats.npy'),sequence_sha256=sha(out/'sequence.npy'),generator_sha256=sha(__file__),status='EXACT_SOURCE_INPUTS_AND_PLANS'));print(ds,rd,'independently cached',len(used),flush=True)
if __name__=='__main__':
 import fcntl
 lock=(HERE/'fullplans_v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  for ds in ['CIC_IDS','CIC_DDOS','CIC_DOH']:
   marker=ROOTS[ds]/'PARTITION_GATE.json'
   if not marker.exists():continue
   g=json.loads(marker.read_text())
   if g.get('status')!='PASS_PARTITIONS_AND_EVALUATION_POOLS':continue
   if any(not (HERE/'cache'/ds/f'round{i}'/'COMPLETE.json').exists() for i in range(len(g['folds']))):all_plans(ds)
  time.sleep(30)

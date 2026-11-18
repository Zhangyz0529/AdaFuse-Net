"""Once-only deterministic input cache, preserving every frozen sample identity.

Caches only records actually used by the existing training/evaluation plans, not a
new sample selection. This avoids rereading multi-GB manifests for every run.
"""
import argparse,hashlib,json,sys,os,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'R1_5_audited_benchmark_v1'
sys.path.insert(0,str(OLD));from worker import sha,save
ROOTS={'USTC':HERE.parent/'R1_5_ustc_boundary_rebuild_v2/partitions/USTC','ISCX':HERE.parent/'R1_5_task_role_partitions_v1/partitions/ISCX','CIC_IDS':HERE.parent/'R1_5_task_role_partitions_v1/partitions/CIC_IDS','CIC_DDOS':HERE.parent/'R1_5_task_role_partitions_v2/partitions/CIC_DDOS','CIC_DOH':HERE.parent/'R1_5_task_role_partitions_v2/partitions/CIC_DOH'}
def prepare(ds):
 root=ROOTS[ds];gate=json.loads((root/'PARTITION_GATE.json').read_text());assert gate['status']=='PASS_PARTITIONS_AND_EVALUATION_POOLS'
 # No model outputs enter this cache; only saved sampling plans and prepared inputs.
 needed=[]
 for fold in range(len(gate['folds'])):
  rd=f'round{fold}';out=HERE/'cache'/ds/rd
  if (out/'COMPLETE.json').exists():continue
  plans=list((OLD/'runs'/ds/'AdaFuse-Net').glob(f'seed*/{rd}/shot*/training_episodes.json'))
  if not plans:continue
  tr=json.loads(plans[0].read_text());ev=json.loads((root/rd/'evaluation_episodes.json').read_text());ids=sorted({s for ep in tr+sum(ev.values(),[]) for field in ['support','query'] for vv in ep[field].values() for s in vv});needed.append((rd,out,tr,ev,ids,plans[0]))
 if not needed:return
 csv.field_size_limit(2**30)
 with (root/'group_components.csv').open(newline='') as f:gmap={x['sample_id']:i for i,x in enumerate(csv.DictReader(f))}
 seq=np.load(root/'sequence.npy',mmap_mode='r')
 for rd,out,tr,ev,ids,tp in needed:
  out.mkdir(parents=True,exist_ok=True);z=np.load(root/rd/'scaled_statistics.npz');lm={s:i for i,s in enumerate(z['sample_ids'].astype(str))};ix=np.array([gmap[s] for s in ids]);stat=z['stat'][[lm[s] for s in ids]];sq=np.asarray(seq[ix]);np.save(out/'stats.npy',stat);np.save(out/'sequence.npy',sq);np.save(out/'global_indices.npy',ix);save(out/'ids.json',ids);save(out/'train.json',tr);save(out/'eval.json',ev)
  # Exact byte comparison ensures the cache is storage-only.
  assert np.array_equal(stat,z['stat'][[lm[s] for s in ids]]) and np.array_equal(sq,seq[ix]);save(out/'COMPLETE.json',dict(dataset=ds,fold=rd,records=len(ids),source_root=str(root),source_gate_sha256=sha(root/'PARTITION_GATE.json'),source_eval_sha256=sha(root/rd/'evaluation_episodes.json'),training_plan_sha256=sha(tp),stats_sha256=sha(out/'stats.npy'),sequence_sha256=sha(out/'sequence.npy'),status='EXACT_SOURCE_INPUTS_AND_PLANS'))
  print(ds,rd,'cached',len(ids),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();prepare(a.dataset)

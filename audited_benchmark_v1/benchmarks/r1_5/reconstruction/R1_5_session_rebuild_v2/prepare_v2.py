"""Freeze exact-source reconstructions, conflict exclusions, folds and episode plans."""
import argparse,json,sys,subprocess,collections,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from session_rules import stable
from rebuild import HERE,CLASSES,save,sha
SPLITTER=HERE.parent/'R1_5_leakage_audited_reproduction_v1/split_and_audit.py'

def prepare(dataset,seed):
 src=HERE/'reconstructed'/dataset
 if not (src/'EXTRACTION_COMPLETE.json').exists():raise RuntimeError('Extraction incomplete')
 entries=[];stats=[];seq=[]
 for f in sorted(src.glob('*/records.json')):
  rows=json.loads(f.read_text());z=np.load(f.parent/'inputs.npz',allow_pickle=False)
  if len(rows)!=len(z['stat_raw']):raise RuntimeError('Source identity mismatch')
  for i,r in enumerate(rows):r['stat_index']=len(entries);entries.append(r);stats.append(z['stat_raw'][i]);seq.append(z['seq_feature'][i])
 if not entries:raise RuntimeError('No uniquely matched samples; clock/source audit must be resolved')
 x=np.array(stats,dtype='float32');q=np.array(seq,dtype='uint8');hashes=[hashlib.sha256(s.tobytes()+v.tobytes()).hexdigest() for s,v in zip(x,q)];byhash=collections.defaultdict(set)
 for r,h in zip(entries,hashes):byhash[h].add(r['label'])
 conflicts={h for h,labels in byhash.items() if len(labels)>1};keep=np.array([h not in conflicts for h in hashes]);out=HERE/'datasets'/dataset;out.mkdir(parents=True,exist_ok=True)
 if (out/'FREEZE_COMPLETE.json').exists():return
 excluded=[dict(sample_id=r['sample_id'],label=r['label'],input_hash=h,reason='exclude_all_cross_label_identical_input_group') for r,h in zip(entries,hashes) if h in conflicts]
 save(out/'cross_label_conflict_exclusions.json',excluded)
 selected=[r for r,k in zip(entries,keep) if k];df=pd.DataFrame([{k:v for k,v in r.items() if k!='packet_records'} for r in selected]);df.to_csv(out/'verified_sources.csv',index=False)
 np.savez_compressed(out/'raw_inputs.npz',sample_ids=df.sample_id.to_numpy(dtype=str),stat_raw=x[keep],seq_feature=q[keep]);save(out/'packet_membership.json',selected)
 attacks=CLASSES[dataset][1:];tasks={f'round{i}':{'Train':[CLASSES[dataset][0]]+[a for a in attacks if a not in [task,attacks[(i+1)%len(attacks)]]],'Val':[CLASSES[dataset][0],attacks[(i+1)%len(attacks)]],'Test':[CLASSES[dataset][0],task]} for i,task in enumerate(attacks)}
 save(out/'tasks.json',tasks);save(out/'provenance_evidence.json',dict(provenance_join_verified=True,group_definition='unique CSV packet-window match plus TCP-state/idle session; all identical bidirectional tuples across captures share a conservative source_group_id',evidence_files=['packet_membership.json'],limitations=['observed sessions may be midstream','capture/host independence not claimed']))
 splitout=out/'split_v2'
 subprocess.run([sys.executable,str(SPLITTER),'--samples',str(out/'verified_sources.csv'),'--inputs',str(out/'raw_inputs.npz'),'--evidence',str(out/'provenance_evidence.json'),'--tasks',str(out/'tasks.json'),'--ratios','0.6','0.2','0.2','--seed','42','--min-per-class','50','--output',str(splitout)],check=True)
 m=pd.read_csv(splitout/'split_manifest.csv',dtype={'sample_id':str});m.to_csv(out/'split_manifest_v2.csv',index=False)
 prepared=HERE/'prepared'/dataset/f'seed{seed}';prepared.mkdir(parents=True,exist_ok=True);globalids={s:i for i,s in enumerate(df.sample_id)};manifest=dict(dataset=dataset,seed=seed,stat_dim=77,source=str(out/'raw_inputs.npz'),source_sha256=sha(out/'raw_inputs.npz'),rounds=[],split_manifest_sha256=sha(out/'split_manifest_v2.csv'),protocol='observed-session and canonical-tuple/content grouping; reference grouped; train-only scaler; no capture/host independence claim')
 allx=x[keep];allq=q[keep]
 for rid,(fold,roles) in enumerate(tasks.items()):
  frame=m[m.outer_fold==fold];ids=np.array([globalids[s] for s in frame.sample_id]);scaled=np.load(splitout/fold/'scaled_statistics.npz',allow_pickle=False)
  if not np.array_equal(scaled['sample_ids'],frame.sample_id.to_numpy()):raise RuntimeError('Scaler output identity mismatch')
  local={s:i for i,s in enumerate(frame.sample_id)};rng=np.random.default_rng(int(stable([dataset,seed,rid,'train'])[:16],16));train=[];pools={}
  for role,labels in roles.items():
   for label in labels:pools[role,label]=frame[(frame.partition==role)&(frame.label==label)]
  for _ in range(50):
   classes=roles['Train'];choices=[rng.choice(pools['Train',c].sample_id.to_numpy(),25,replace=False) for c in classes]
   train.append(dict(classes=classes,support=[[local[s] for s in a[:10]] for a in choices],query=[[local[s] for s in a[10:]] for a in choices]))
  specs={};pool_audit=[]
  for role,key in [('Val','validation'),('Test','test')]:
   episodepools=[]
   for label in roles[role]:
    pool=pools[role,label];tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed).split(pool,groups=pool.split_group_id))
    support=pool.iloc[tr];query=pool.iloc[te]
    if len(support)<10 or len(query)<15:raise RuntimeError(f'Insufficient group-disjoint support/query pool: {dataset}/{fold}/{role}/{label}')
    assert not set(support.session_id)&set(query.session_id);assert not set(support.input_hash)&set(query.input_hash)
    m.loc[(m.outer_fold==fold)&m.sample_id.isin(support.sample_id),'pool']='support'
    m.loc[(m.outer_fold==fold)&m.sample_id.isin(query.sample_id),'pool']='query'
    episodepools.append((support.sample_id.to_numpy(),query.sample_id.to_numpy()));pool_audit.append(dict(role=role,label=label,support_ids=support.sample_id.tolist(),query_ids=query.sample_id.tolist(),calibration_ids=[]))
   eps=[]
   for _ in range(100):eps.append(dict(support=[[local[s] for s in rng.choice(a,10,replace=False)] for a,b in episodepools],query=[[local[s] for s in rng.choice(b,15,replace=False)] for a,b in episodepools]))
   specs[key]={roles[role][1]:eps}
  rd=prepared/f'round{rid}';rd.mkdir(exist_ok=True)
  np.savez_compressed(rd/'shared_inputs.npz',stats=scaled['stat'],sequence=allq[ids].astype('int16'),source_indices=ids)
  save(rd/'source_sample_ids.json',frame.sample_id.tolist());save(rd/'episode_pool_audit.json',pool_audit);save(rd/'train_episodes.json',train);save(rd/'eval_episodes.json',specs)
  manifest['rounds'].append(dict(round=rid,shared_sha256=sha(rd/'shared_inputs.npz'),train_sha256=sha(rd/'train_episodes.json'),eval_sha256=sha(rd/'eval_episodes.json')))
 m.loc[m.partition=='Train','pool']='training'
 m.to_csv(out/'split_manifest_v2.csv',index=False);manifest['split_manifest_sha256']=sha(out/'split_manifest_v2.csv')
 save(prepared/'manifest.json',manifest);save(out/'FREEZE_COMPLETE.json',dict(passed=True,cross_label_groups_removed=len(conflicts),cross_label_rows_removed=len(excluded),retained=len(df),split_manifest_sha256=sha(out/'split_manifest_v2.csv'),prepared_manifest_sha256=sha(prepared/'manifest.json'),seed=seed))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CLASSES),required=True);p.add_argument('--seed',type=int,default=3407);a=p.parse_args();prepare(a.dataset,a.seed)

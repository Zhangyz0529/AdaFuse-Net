"""Post-rebuild gate: exact-input conflicts, connected groups, fixed split and audit.

No seed retries, no model selection or training. A failed count/group check blocks
this dataset and leaves a failure report. Arrays and scalers are built only after
sample identities and conservative grouping are fixed.
"""
import argparse, collections, csv, hashlib, itertools, json, sqlite3, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import common as r
from role_assignment import assign_roles

POLICY={'version':'task_role_v1','amendment':'source-count feasibility, before any model training; preserve old global-split failure reports','partition_assignment':'attack/target components assigned whole to their fold role; reference-only components use60/20/20; conflicting-role components all excluded','ratios':[.6,.2,.2],'split_seed':42,'pool_seed':3407,'episode_seed':3407,'minimum_samples_per_partition_class':25,'minimum_components_per_class':2,'shots':[1,3,5,10],'query_per_class':15,'evaluation_episodes_per_task_shot':100,'calibration_pool':'none; no calibration used; no fitted decision threshold in preparation','outer_tasks':'registry target Test; next target Val; remaining targets Train; dataset reference in all roles with disjoint groups','grouping':'session, conservative same-domain bidirectional tuple, exact model input, identical raw packet content, explicit uncertain-continuation edges','blocked_domains':'excluded by fixed source-evidence rule only; named in scope table; never by model score','no_seed_retries':True,'training_allowed':False}
class Union:
 def __init__(self,n):self.p=np.arange(n,dtype=np.int64)
 def find(self,a):
  while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
  return int(a)
 def union(self,a,b):self.p[self.find(a)]=self.find(b)
def prepare(dataset,source_root=None,output=None):
 from types import SimpleNamespace
 a=SimpleNamespace(dataset=dataset);src=source_root or r.source_root(a.dataset);out=output or r.HERE/'partitions'/a.dataset;out.mkdir(parents=True,exist_ok=True)
 if (out/'PARTITION_GATE.json').exists():raise RuntimeError('Gate already executed; do not overwrite or retry seed')
 report=dict(dataset=a.dataset,status='INCOMPLETE',policy=POLICY,training_allowed=False,optuna_allowed=False)
 try:
  expected=r.expected_sources(a.dataset)
  dirs=[src/d for d in expected];summaries=[]
  for d in dirs:
   if not (d/'REBUILD_COMPLETE.json').exists():raise RuntimeError('Incomplete source domain: '+d.name)
   info=json.loads((d/'REBUILD_COMPLETE.json').read_text())
   if not info['counts_close'] or not info.get('session_counts_close',True) or info['status']!='RECONSTRUCTED_UNSPLIT':raise RuntimeError('Unclosed source counts: '+d.name)
   summaries.append(info)
  report['source_scope']=summaries
  db=sqlite3.connect(out/'audit_index.sqlite');db.execute('CREATE TABLE samples (i INTEGER PRIMARY KEY,sample_id TEXT UNIQUE,label TEXT,session TEXT,source_group TEXT,content TEXT,rawhash TEXT,metadata TEXT)')
  n=0
  for d in dirs:
   for line in (d/'accepted.jsonl').open():
    x=json.loads(line);meta={k:v for k,v in x.items() if k not in ['stat_raw','seq_hex']}
    db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)',(n,x['sample_id'],x['label'],x['session_id'],x['source_group_id'],x['content_hash'],None,json.dumps(meta)));n+=1
   db.commit()
   for line in (d/'accepted_packet_membership.jsonl').open():
    x=json.loads(line);h=r.stable([p['raw_sha256'] for p in x['packets']]);db.execute('UPDATE samples SET rawhash=? WHERE sample_id=?',(h,x['sample_id']))
   db.commit()
  if not n:raise RuntimeError('No formally accepted source records')
  conflicts={h for h, in db.execute('SELECT content FROM samples GROUP BY content HAVING count(DISTINCT label)>1')};excluded=set()
  with (out/'cross_label_conflict_exclusions.jsonl').open('w') as f:
   for i,sample,label,h in db.execute('SELECT i,sample_id,label,content FROM samples'):
    if h in conflicts:excluded.add(i);r.emit(f,sample_id=sample,label=label,content_hash=h,reason='exclude_all_members_cross_label_identical_input')
  metadata=[];original_to_local={};sample_to_local={}
  for i,sample,label,sid,sg,ch,rh,js in db.execute('SELECT * FROM samples ORDER BY i'):
   if i in excluded or label not in r.CLASSES[a.dataset]:continue
   if not rh:raise RuntimeError('Missing packet evidence hash')
   m=json.loads(js);m['raw_record_hash']=rh
   if 'record_locator' not in m:m['record_locator']=str(m.get('csv_path',''))+':'+str(m.get('csv_row',''))
   original_to_local[i]=len(metadata);sample_to_local[sample]=len(metadata);metadata.append(m)
  frame=pd.DataFrame(metadata);uf=Union(len(frame));session_first={}
  for col in ['session_id','source_group_id','content_hash','raw_record_hash']:
   first={}
   for i,v in enumerate(frame[col]):
    if v in first:uf.union(i,first[v])
    else:first[v]=i
   if col=='session_id':session_first=first
  # Source-tuple grouping already covers uncertain same-key continuation, including
  # intermediate sessions without samples. Explicit retained-session edges audited too.
  edge_count=0
  for d in dirs:
   for line in (d/'session_relations.jsonl').open():
    e=json.loads(line);edge_count+=1
    if e['first_session'] in session_first and e['second_session'] in session_first:uf.union(session_first[e['first_session']],session_first[e['second_session']])
  frame['split_group_id']=[r.stable([a.dataset,'connected_component',uf.find(i)]) for i in range(len(frame))]
  counts=frame.groupby('label').agg(samples=('sample_id','count'),observed_sessions=('session_id','nunique'),source_groups=('source_group_id','nunique'),components=('split_group_id','nunique')).reset_index();counts.to_csv(out/'class_group_feasibility.csv',index=False)
  report.update(before_conflict_samples=n,cross_label_hash_groups=len(conflicts),excluded_conflict_records=len(excluded),retained=len(frame),uncertain_continuation_edges=edge_count,class_groups=counts.to_dict('records'))
  for label in r.CLASSES[a.dataset]:
   c=counts[counts.label==label]
   if len(c)!=1 or int(c.iloc[0].components)<2 or int(c.iloc[0].samples)<25:raise RuntimeError('Insufficient independent groups/counts: '+label+'; no group splitting or seed retry')
  frame.to_csv(out/'group_components.csv',index=False)
  # Build arrays in fixed sample order after conflict removal; no scaler was used.
  stats=np.lib.format.open_memmap(out/'stat_raw.npy',mode='w+',dtype='float32',shape=(len(frame),r.DIMS[a.dataset]));seq=np.lib.format.open_memmap(out/'sequence.npy',mode='w+',dtype='uint8',shape=(len(frame),512))
  for d in dirs:
   for line in (d/'accepted.jsonl').open():
    x=json.loads(line);i=sample_to_local.get(x['sample_id'])
    if i is not None:
     stats[i]=x['stat_raw'];seq[i]=np.frombuffer(bytes.fromhex(x['seq_hex']),dtype=np.uint8)
     if hashlib.sha256(stats[i].tobytes()+seq[i].tobytes()).hexdigest()!=x['content_hash']:raise RuntimeError('Accepted input hash does not match actual arrays')
  stats.flush();seq.flush();reference=r.CLASSES[a.dataset][0];attacks=r.CLASSES[a.dataset][1:];manifest=[];audit=[]
  for fold,heldout in enumerate(attacks):
   valattack=attacks[(fold+1)%len(attacks)];roles={'Train':[reference]+[x for x in attacks if x not in [heldout,valattack]],'Val':[reference,valattack],'Test':[reference,heldout]};m,excluded_roles=assign_roles(frame,roles,seed=42);excluded_roles['outer_fold']='round'+str(fold);excluded_roles.to_csv(out/('round'+str(fold)+'_role_conflict_exclusions.csv'),index=False);m['outer_fold']='round'+str(fold);m['pool']='training'
   for part,labels in roles.items():
    for label in labels:
     if ((m.partition==part)&(m.label==label)).sum()<25:raise RuntimeError(f'Insufficient fixed assignment: round{fold}/{part}/{label}')
   pairs=[];fields=['sample_id','session_id','source_group_id','split_group_id','content_hash','raw_record_hash']
   for x,y in itertools.combinations(['Train','Val','Test'],2):
    intersections={f:len(set(m.loc[m.partition==x,f])&set(m.loc[m.partition==y,f])) for f in fields};pairs.append(dict(partitions=[x,y],intersections=intersections))
    if any(intersections.values()):raise RuntimeError('Partition overlap')
   foldout=out/('round'+str(fold));foldout.mkdir();scaler=StandardScaler().fit(stats[m.index[m.partition=='Train']]);joblib.dump(scaler,foldout/'scaler.joblib');m.loc[m.partition=='Train',['sample_id']].to_csv(foldout/'scaler_fit_sample_ids.csv',index=False)
   episodeplans={};poolchecks=[]
   for part in ['Val','Test']:
    p=m[m.partition==part];s,q=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=3407).split(p,groups=p.split_group_id));support=p.iloc[s];query=p.iloc[q];m.loc[support.index,'pool']='support';m.loc[query.index,'pool']='query'
    check={f:len(set(support[f])&set(query[f])) for f in fields}
    if any(check.values()):raise RuntimeError('Support/query component overlap')
    poolchecks.append(dict(partition=part,intersections=check,calibration_count=0));rng=np.random.default_rng(int(r.stable([a.dataset,fold,part,3407])[:16],16))
    for shot in [1,3,5,10]:
     episodes=[]
     for _ in range(100):
      ep={'classes':roles[part],'support':{},'query':{}}
      for label in roles[part]:
       ss=support.loc[support.label==label,'sample_id'].to_numpy();qq=query.loc[query.label==label,'sample_id'].to_numpy()
       if len(ss)<10 or len(qq)<15:raise RuntimeError('Insufficient group-disjoint support/query: '+label)
       ep['support'][label]=rng.choice(ss,shot,replace=False).tolist();ep['query'][label]=rng.choice(qq,15,replace=False).tolist()
      episodes.append(ep)
     episodeplans[part+'_'+str(shot)+'shot']=episodes
   r.write(foldout/'evaluation_episodes.json',episodeplans)
   # Verify fitted output, including content collisions introduced by float32 scaling.
   scaled=scaler.transform(stats[m.index]).astype('float32');seen={}
   for j,i in enumerate(m.index):
    h=hashlib.sha256(scaled[j].tobytes()+seq[i].tobytes()).hexdigest();location=(m.at[i,'partition'],m.at[i,'pool'])
    if h in seen and seen[h]!=location:raise RuntimeError('Post-scaler content collision across partitions/pools; no training')
    seen[h]=location
   np.savez_compressed(foldout/'scaled_statistics.npz',sample_ids=m.sample_id.to_numpy(dtype=str),stat=scaled)
   fitids=set(m.loc[m.partition=='Train','sample_id']);assert not fitids & set(m.loc[m.partition!='Train','sample_id'])
   audit.append(dict(outer_fold='round'+str(fold),roles=roles,partition_pairs=pairs,pool_pairs=poolchecks,scaler_fit_partition='Train',scaler_fit_n=len(fitids),counts=m.groupby(['partition','label','pool']).size().reset_index(name='n').to_dict('records')));manifest.append(m)
  pd.concat(manifest).to_csv(out/'split_manifest_v3.csv',index=False);report.update(status='PASS_PARTITIONS_AND_EVALUATION_POOLS',folds=audit,training_allowed=False,training_episode_and_model_selection_audit='NOT_EXECUTED; no training started; common budgets/configs required before training',limitations=['observed connections, not authenticated complete application sessions','capture/host independence not claimed','group-count ratios do not imply exact sample ratios'])
 except Exception as exc:
  report.update(status='BLOCKED_DO_NOT_TRAIN',error=repr(exc));raise
 finally:
  r.write(out/'PARTITION_GATE.json',report);(out/'leakage_audit_report.txt').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=list(r.CLASSES),required=True);a=parser.parse_args();prepare(a.dataset)

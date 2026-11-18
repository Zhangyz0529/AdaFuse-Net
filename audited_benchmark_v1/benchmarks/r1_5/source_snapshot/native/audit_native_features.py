"""Report representation aliases separately from proven source/session overlap.

A many-to-one native representation may map independent records to the same
features; this is not relabeled as proven capture/session leakage or silently
removed. Scope is actual frozen training/evaluation plans, not entire population.
"""
import json,hashlib,sqlite3,itertools
from pathlib import Path
import numpy as np
from cache import HERE,save
for ds in ['USTC','ISCX']:
 for method,sub,table,column in [('FlowPic','flowpic','histograms.sqlite','coords'),('MetaTraffic','packet_patterns','patterns.sqlite','pattern')]:
  dbpath=HERE/sub/ds/table
  if not (HERE/sub/ds/'COMPLETE.json').exists():continue
  conn=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True);hashes={}
  for sid,data in conn.execute(f'SELECT sample_id,{column} FROM samples'):
   if method=='FlowPic':
    p=np.frombuffer(data,dtype='int16').reshape(-1,2);locations=p[:,0].astype('int64')*1500+p[:,1];bins,counts=np.unique(locations,return_counts=True);canonical=bins.tobytes()+counts.astype('float32').tobytes()
   else:
    p=np.frombuffer(data,dtype='float32');fixed=np.zeros(5000,dtype='float32');fixed[:len(p)]=p;canonical=fixed.tobytes()
   hashes[sid]=hashlib.sha256(canonical).hexdigest()
  folds=[]
  for rd in sorted((HERE/'cache'/ds).glob('round*')):
   tr=json.loads((rd/'train.json').read_text());ev=json.loads((rd/'eval.json').read_text());pools={'Train':set(),'Val':set(),'Test':set()}
   for ep in tr:
    for field in ['support','query']:
     for ids in ep[field].values():pools['Train'].update(ids)
   detail={}
   for role in ['Val','Test']:
    support=set();query=set()
    for key,episodes in ev.items():
     if not key.startswith(role+'_'):continue
     for ep in episodes:
      for ids in ep['support'].values():support.update(ids)
      for ids in ep['query'].values():query.update(ids)
    assert not support&query;pools[role]=support|query;detail[role]=dict(sample_id_intersection=0,feature_hash_intersection=len({hashes[x] for x in support}&{hashes[x] for x in query}))
   pairs=[]
   for a,b in itertools.combinations(pools,2):
    idintersection=len(pools[a]&pools[b]);assert idintersection==0;pairs.append(dict(partitions=[a,b],sample_id_intersection=0,representation_hash_intersection=len({hashes[x] for x in pools[a]}&{hashes[x] for x in pools[b]})))
   folds.append(dict(fold=rd.name,partition_pairs=pairs,support_query=detail))
  save(HERE/'audits'/f'{ds}_{method}_representation_aliases.json',dict(dataset=ds,method=method,scope='only records used by frozen plans; source/session separation established by upstream gate; this checks native feature aliases',interpretation='feature collisions are not automatically evidence of shared source/session; do not claim zero exact native input overlap when reported nonzero',folds=folds))
  print(ds,method,'audited',flush=True)

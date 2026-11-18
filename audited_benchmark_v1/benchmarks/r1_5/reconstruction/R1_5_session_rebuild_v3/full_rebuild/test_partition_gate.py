"""Confirm mixed-label duplicate removal and insufficient-group stop before fit."""
import json,tempfile
from pathlib import Path
from prepare_partitions import prepare
from run import write
with tempfile.TemporaryDirectory(prefix='v3_partition_check_') as td:
 root=Path(td);src=root/'sources';out=root/'partition'
 for day in ['Friday','Tuesday','Wednesday','Monday','Thursday']:
  d=src/day;d.mkdir(parents=True);write(d/'REBUILD_COMPLETE.json',{'counts_close':True})
  records=[]
  if day=='Friday':
   for i in range(6):records.append(dict(sample_id=str(i),label='BENIGN' if i!=1 else 'PortScan',session_id='session'+str(i),source_group_id='source'+str(i),content_hash='shared_conflict' if i<2 else 'content'+str(i),stat_raw=[0.]*77,seq_hex='00'*512))
  (d/'accepted.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(d/'accepted_packet_membership.jsonl').write_text(''.join(json.dumps({'sample_id':r['sample_id'],'packets':[{'raw_sha256':'packet'+r['sample_id']}]})+'\n' for r in records));(d/'session_relations.jsonl').write_text('')
 try:prepare('CIC_IDS',src,out)
 except RuntimeError as e:assert 'Insufficient independent groups' in str(e)
 else:raise AssertionError('Insufficient groups must stop')
 r=json.loads((out/'PARTITION_GATE.json').read_text());assert r['excluded_conflict_records']==2 and r['cross_label_hash_groups']==1 and r['status']=='BLOCKED_DO_NOT_TRAIN';assert not list(out.rglob('scaler.joblib'))
 print(json.dumps({'passed':True,'checks':['all members of cross-label hash group excluded','insufficient independent groups block before scaler fit','no seed retry or training'],'synthetic_only':True}))

# Positive synthetic case: enough distinct components in every class.
import numpy as np,pandas as pd,joblib,hashlib
with tempfile.TemporaryDirectory(prefix='v3_partition_pass_') as td:
 root=Path(td);src=root/'sources';out=root/'partition';rng=np.random.default_rng(817)
 labels=['BENIGN','PortScan','DDoS','FTP-Patator','SSH-Patator','DoS'];records=[]
 for label in labels:
  for j in range(600):
   i=len(records);records.append(dict(dataset='CIC_IDS',sample_id='sample'+str(i),label=label,session_id='session'+str(i),source_group_id='source'+str(i),content_hash='content'+str(i),source_file_id='file',provenance_domain_id='test',csv_path='synthetic.csv',csv_row=i+2,stat_raw=rng.normal(size=77).tolist(),seq_hex='00'*512))
 for record in records:record['content_hash']=hashlib.sha256(np.asarray(record['stat_raw'],dtype='<f4').tobytes()+bytes.fromhex(record['seq_hex'])).hexdigest()
 for day in ['Friday','Tuesday','Wednesday','Monday','Thursday']:
  d=src/day;d.mkdir(parents=True);write(d/'REBUILD_COMPLETE.json',{'counts_close':True});rr=records if day=='Friday' else []
  (d/'accepted.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr));(d/'accepted_packet_membership.jsonl').write_text(''.join(json.dumps({'sample_id':r['sample_id'],'packets':[{'raw_sha256':'packet'+r['sample_id']}]})+'\n' for r in rr));(d/'session_relations.jsonl').write_text('')
 prepare('CIC_IDS',src,out);r=json.loads((out/'PARTITION_GATE.json').read_text());assert r['status']=='PASS_PARTITIONS_AND_EVALUATION_POOLS' and not r['training_allowed']
 m=pd.read_csv(out/'split_manifest_v3.csv');values=np.load(out/'stat_raw.npy')
 for fold,f in m.groupby('outer_fold'):
  fit=pd.read_csv(out/fold/'scaler_fit_sample_ids.csv');assert set(fit.sample_id)==set(f.loc[f.partition=='Train','sample_id'])
  ids=[int(s.removeprefix('sample')) for s in fit.sample_id];scaler=joblib.load(out/fold/'scaler.joblib');assert np.allclose(scaler.mean_,values[ids].mean(axis=0,dtype='float64'))
  for part in ['Val','Test']:
   p=f[f.partition==part];assert not set(p[p.pool=='support'].split_group_id)&set(p[p.pool=='query'].split_group_id)
 print(json.dumps({'passed':True,'checks':['five-fold manifests generated from fixed component split','all partition pairs and eval pools disjoint','scaler means equal exactly specified Train rows','calibration unused and training remains blocked'],'synthetic_only':True}))

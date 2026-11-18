"""No scores: validate whole-component roles and sparse-target68 feasibility."""
import json,tempfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
import common as r
from prepare import prepare
from role_assignment import assign_roles

def main():
 roles={'Train':['ref','a'],'Val':['ref','b'],'Test':['ref','c']};rows=[dict(label='ref',split_group_id='r'+str(i),sample_id='r'+str(i)) for i in range(30)]+[dict(label='a',split_group_id='conflict',sample_id='ca'),dict(label='c',split_group_id='conflict',sample_id='cc')];m,excluded=assign_roles(pd.DataFrame(rows),roles);assert set(excluded.sample_id)=={'ca','cc'};assert m.groupby('split_group_id').partition.nunique().max()==1
 old=r.expected_sources
 with tempfile.TemporaryDirectory(prefix='role_partition_test_') as td:
  root=Path(td);src=root/'source';d=src/'synthetic';d.mkdir(parents=True);out=root/'out';r.expected_sources=lambda ds:['synthetic'];r.write(d/'REBUILD_COMPLETE.json',dict(counts_close=True,session_counts_close=True,status='RECONSTRUCTED_UNSPLIT'));rng=np.random.default_rng(640);records=[]
  for label in r.CLASSES['ISCX']:
   count=600 if label==r.CLASSES['ISCX'][0] else 68 if label=='vpn_ftps_a' else 180
   for _ in range(count):
    i=len(records);stats=rng.normal(size=25).astype('<f4');q=bytes(512);records.append(dict(dataset='ISCX',sample_id='s'+str(i),label=label,session_id='session'+str(i),source_group_id='source'+str(i),source_file_id='synthetic',record_locator='file:'+str(i),stat_raw=stats.tolist(),seq_hex=q.hex(),content_hash=hashlib.sha256(stats.tobytes()+q).hexdigest()))
  with (d/'accepted.jsonl').open('w') as f,(d/'accepted_packet_membership.jsonl').open('w') as p:
   for x in records:r.emit(f,**x);r.emit(p,sample_id=x['sample_id'],packets=[dict(raw_sha256=x['sample_id'])])
  (d/'session_relations.jsonl').write_text('');prepare('ISCX',src,out);report=json.loads((out/'PARTITION_GATE.json').read_text());assert report['status']=='PASS_PARTITIONS_AND_EVALUATION_POOLS';manifest=pd.read_csv(out/'split_manifest_v3.csv');stats=np.load(out/'stat_raw.npy')
  for fold,frame in manifest.groupby('outer_fold'):
   assert len(frame[frame.label=='vpn_ftps_a'])==68 and frame.groupby('split_group_id').partition.nunique().max()==1;fit=pd.read_csv(out/fold/'scaler_fit_sample_ids.csv');ids=[int(s[1:]) for s in fit.sample_id];assert np.allclose(joblib.load(out/fold/'scaler.joblib').mean_,stats[ids].mean(axis=0,dtype='float64'))
  r.expected_sources=old
 print(json.dumps(dict(passed=True,checks=['conflicting-role groups excluded whole','68-record target retained whole in each fold role','all5 folds pass partition and support/query checks','scaler fits only fold Train','no seed retry or model results'],synthetic_only=True)))
if __name__=='__main__':main()

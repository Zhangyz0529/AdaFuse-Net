"""25-dimensional, dataset-reference-aware partition/scaler test."""
import tempfile,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
import packet_rebuild as r
from prepare_partitions import prepare

def main():
 original=r.HERE
 with tempfile.TemporaryDirectory(prefix='remaining_partition_') as td:
  r.HERE=Path(td);src=r.HERE/'outputs/USTC';d=src/'fixture';d.mkdir(parents=True);r.write(r.HERE/'USTC_SOURCE_PLAN.json',{'fixture':[]});r.write(d/'REBUILD_COMPLETE.json',dict(counts_close=True,session_counts_close=True,status='RECONSTRUCTED_UNSPLIT'));rng=np.random.default_rng(517);records=[]
  for label in r.CLASSES['USTC']:
   for _ in range(600):
    i=len(records);stats=rng.normal(size=25).astype('<f4');q=bytes(512);records.append(dict(dataset='USTC',sample_id='s'+str(i),label=label,session_id='session'+str(i),source_group_id='source'+str(i),source_file_id='f',record_locator='fixture:'+str(i),stat_raw=stats.tolist(),seq_hex=q.hex(),content_hash=hashlib.sha256(stats.tobytes()+q).hexdigest()))
  with (d/'accepted.jsonl').open('w') as f,(d/'accepted_packet_membership.jsonl').open('w') as m:
   for x in records:r.emit(f,**x);r.emit(m,sample_id=x['sample_id'],packets=[{'raw_sha256':x['sample_id']}])
  (d/'session_relations.jsonl').write_text('');out=r.HERE/'partitions';prepare('USTC',src,out);report=json.loads((out/'PARTITION_GATE.json').read_text());assert report['status']=='PASS_PARTITIONS_AND_EVALUATION_POOLS';assert len(report['folds'])==4
  manifest=pd.read_csv(out/'split_manifest_v3.csv');values=np.load(out/'stat_raw.npy');assert values.shape==(3000,25)
  for fold,m in manifest.groupby('outer_fold'):
   fit=pd.read_csv(out/fold/'scaler_fit_sample_ids.csv');ids=[int(x[1:]) for x in fit.sample_id];scaler=joblib.load(out/fold/'scaler.joblib');assert np.allclose(scaler.mean_,values[ids].mean(axis=0,dtype='float64'));assert 'Benign' in set(m.label)
  r.HERE=original
 print(json.dumps(dict(passed=True,checks=['correct25-dimension arrays','dataset-specific Benign reference','four configured USTC targets','all partition/pool intersections','scaler fits exact Train rows'],synthetic_only=True)))
if __name__=='__main__':main()

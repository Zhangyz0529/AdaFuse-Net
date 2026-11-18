"""Temporary synthetic end-to-end runner test. Never produces benchmark evidence."""
import argparse,hashlib,json,shutil,tempfile
from pathlib import Path
import numpy as np
import prepare_v2 as prep
import search
from rebuild import HERE

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
 with tempfile.TemporaryDirectory(prefix='r15v2_gpu_smoke_') as td:
  t=Path(td);records=[];rng=np.random.default_rng(123);stats=[];seq=[]
  for c in prep.CLASSES['CIC_IDS']:
   for j in range(500):
    sid=hashlib.sha256((c+str(j)).encode()).hexdigest();records.append(dict(dataset='CIC_IDS',sample_id=sid,session_id='s'+sid,source_group_id='g'+sid,source_file_id='SYNTHETIC',record_locator='synthetic:'+sid,label=c,raw_record_hash='r'+sid,packet_records=[]));stats.append(rng.normal(size=77));seq.append(rng.integers(0,256,512,dtype=np.uint8))
  src=t/'reconstructed/CIC_IDS/test';src.mkdir(parents=True);(src/'records.json').write_text(json.dumps(records));np.savez_compressed(src/'inputs.npz',stat_raw=np.array(stats,dtype='float32'),seq_feature=np.array(seq,dtype='uint8'));(src.parent/'EXTRACTION_COMPLETE.json').write_text('{}')
  shutil.copytree(HERE/'source_snapshot',t/'source_snapshot');shutil.copy2(HERE/'search.py',t/'search.py');prep.HERE=t;prep.prepare('CIC_IDS',3407);search.HERE=t
  search.main(argparse.Namespace(dataset='CIC_IDS',seed=3407,device=a.device,smoke=True))
  complete=json.loads((t/'smoke/CIC_IDS/seed3407/COMPLETE.json').read_text());assert complete['passed'] and complete['complete_trials']==1
  (HERE/'GPU_SMOKE_CHECK.json').write_text(json.dumps({'passed':True,'device':a.device,'scope':'temporary SYNTHETIC input only; four shots one round two updates; train/checkpoint/validation selection/test evaluation interfaces; NOT benchmark result','temporary_artifacts_removed':True},indent=2))
if __name__=='__main__':main()

"""Frozen USTC/ISCX reconstruction with two CPU workers per dataset, then gate."""
import concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import packet_rebuild as r

def check():
 for name,h in json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())['code_sha256'].items():
  if r.sha(Path(name))!=h:raise RuntimeError('Frozen code changed: '+name)
def source_job(dataset,family,root):
 check();logs=r.HERE/'execution';logs.mkdir(exist_ok=True);path=logs/(dataset+'_'+family+'.log')
 with path.open('a',buffering=1) as f:
  cmd=[sys.executable,'-u',str(r.HERE/'packet_rebuild.py'),'--data-root',str(root),'--dataset',dataset,'--family',family];p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});r.write(logs/(dataset+'_'+family+'_process.json'),dict(pid=p.pid,command=cmd,start_time=time.time()));return family,p.wait()
def dataset_job(dataset,root):
 families=json.loads((r.HERE/(dataset+'_SOURCE_PLAN.json')).read_text());finished={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  active={pool.submit(source_job,dataset,f,root):f for f in families}
  while active:
   done,_=concurrent.futures.wait(active,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    family,ret=f.result();finished[family]=ret
   for f in done:active.pop(f)
   r.write(r.HERE/(dataset+'_RUN_STATUS.json'),dict(active_or_queued=list(active.values()),finished=finished,time=time.time(),training_started=False))
 if any(finished.values()):return dict(stage='BLOCKED_RECONSTRUCTION',failed={k:v for k,v in finished.items() if v})
 check()
 with (r.HERE/'execution'/(dataset+'_partition.log')).open('w') as f:p=subprocess.run([sys.executable,str(r.HERE/'prepare_partitions.py'),'--dataset',dataset],stdout=f,stderr=subprocess.STDOUT)
 return dict(stage='PARTITION_GATE_FINISHED',returncode=p.returncode,report=str(r.HERE/'partitions'/dataset/'PARTITION_GATE.json'))
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();lock=(r.HERE/'launch_remaining.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  active={pool.submit(dataset_job,d,a.data_root):d for d in ['USTC','ISCX']};finished={}
  while active:
   done,_=concurrent.futures.wait(active,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:finished[active.pop(f)]=f.result()
   r.write(r.HERE/'RUN_STATUS.json',dict(active=list(active.values()),finished=finished,time=time.time(),training_started=False,optuna_started=False))
if __name__=='__main__':main()

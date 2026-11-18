"""Two CPU workers, all seven source domains; no HPO/training stage exists."""
import argparse, concurrent.futures, fcntl, json, os, subprocess, sys, time
from pathlib import Path
import run as r

def worker(dataset,domain,root):
 logdir=r.HERE/'execution';logdir.mkdir(exist_ok=True)
 with (logdir/(dataset+'_'+domain+'.log')).open('a',buffering=1) as log:
  cmd=[sys.executable,'-u',str(r.HERE/'run.py'),'--data-root',str(root),'--dataset',dataset,'--domain',domain]
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  r.write(logdir/(dataset+'_'+domain+'_process.json'),dict(pid=p.pid,command=cmd,start_time=time.time()))
  return dict(dataset=dataset,domain=domain,returncode=p.wait())
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
 lock=(r.HERE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())
 for name,h in protocol['code_sha256'].items():
  if r.sha(Path(name))!=h:raise RuntimeError('Frozen source modified: '+name)
 tasks=[('CIC_IDS',d) for d in ['Friday','Tuesday','Wednesday','Monday','Thursday']]+[('CIC_DDOS',d) for d in ['Test_Day_03-11','Train_Day_01-12']]
 # One sequential worker per dataset: two independent CPU jobs, bounded memory.
 def sequence(dataset):
  results=[]
  for ds,domain in tasks:
   if ds==dataset:results.append(worker(ds,domain,a.data_root))
  return results
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(sequence,ds):ds for ds in ['CIC_IDS','CIC_DDOS']};finished={}
  while futures:
   done,_=concurrent.futures.wait(futures,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:finished[futures.pop(f)]=f.result()
   r.write(r.HERE/'RUN_STATUS.json',dict(pid=os.getpid(),active=list(futures.values()),finished=finished,training_started=False,optuna_started=False,time=time.time(),state='running' if futures else 'reconstruction_jobs_finished_review_required'))
if __name__=='__main__':main()

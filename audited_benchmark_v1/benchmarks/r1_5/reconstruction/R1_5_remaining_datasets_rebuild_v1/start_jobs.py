"""Persistent parent for source reconstruction and DoH alignment preflight."""
import concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import packet_rebuild as r

def child(name,script,root):
 log=r.HERE/'execution'/(name+'.log');log.parent.mkdir(exist_ok=True)
 with log.open('a',buffering=1) as f:
  cmd=[sys.executable,'-u',str(r.HERE/script),'--data-root',str(root)];p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});r.write(r.HERE/(name+'_PROCESS.json'),dict(pid=p.pid,command=cmd,start_time=time.time()));return p.wait()
def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--data-root',type=Path,required=True);args=a.parse_args();lock=(r.HERE/'start_jobs.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,h in json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())['code_sha256'].items():
  if r.sha(Path(name))!=h:raise RuntimeError('Frozen dependency changed: '+name)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  pending={pool.submit(child,n,s,args.data_root):n for n,s in [('USTC_ISCX','launch_remaining.py'),('DOH_PREFLIGHT','doh_alignment_preflight.py')]};finished={}
  while pending:
   done,_=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:finished[pending.pop(f)]=f.result()
   r.write(r.HERE/'ALL_JOBS_STATUS.json',dict(pid=os.getpid(),active=list(pending.values()),finished=finished,time=time.time(),training_started=False,optuna_started=False))
if __name__=='__main__':main()

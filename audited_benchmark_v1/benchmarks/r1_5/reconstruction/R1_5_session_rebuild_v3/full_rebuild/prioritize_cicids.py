"""Execution-only scheduling change: leave frozen scientific rules untouched.
The original mixed supervisor and DDoS child are externally SIGSTOP-paused.
Friday complete; all four remaining source domains run concurrently.
Existing partition watcher performs the same frozen audit when all days finish.
"""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
import run as r

def worker(domain):
 statuspath=r.HERE/'outputs'/'CIC_IDS'/domain/'STATUS.json'
 complete=statuspath.parent/'REBUILD_COMPLETE.json'
 # Attach to already-running Tuesday/Wednesday workers, preserving their work.
 if complete.exists():return {'domain':domain,'returncode':0,'mode':'already_complete'}
 if statuspath.exists():
  status=json.loads(statuspath.read_text());pid=status.get('pid');proc=Path('/proc')/str(pid)/'cmdline'
  if proc.exists():
   command=proc.read_bytes().replace(b'\0',b' ').decode()
   if str(r.HERE/'run.py') in command and '--dataset CIC_IDS' in command and ('--domain '+domain) in command:
    while not complete.exists():
     status=json.loads(statuspath.read_text())
     if status['stage']=='FAILED':return {'domain':domain,'returncode':1,'error':status}
     if not proc.exists():return {'domain':domain,'returncode':1,'error':'existing worker exited without completion'}
     time.sleep(15)
    return {'domain':domain,'returncode':0,'mode':'attached_existing_worker'}
 frozen=json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())
 for name,h in frozen['code_sha256'].items():
  if r.sha(Path(name))!=h:raise RuntimeError('Frozen source changed: '+name)
 cmd=[sys.executable,'-u',str(r.HERE/'run.py'),'--data-root',str(r.HERE.parents[3]/'data'),'--dataset','CIC_IDS','--domain',domain]
 # Resolve workspace data root from the recorded original command, not CWD.
 original=json.loads((r.HERE/'SUPERVISOR_PROCESS.json').read_text())['command'];cmd[cmd.index('--data-root')+1]=original[original.index('--data-root')+1]
 with (r.HERE/'execution'/('CIC_IDS_'+domain+'.log')).open('a',buffering=1) as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  r.write(r.HERE/'execution'/('CIC_IDS_'+domain+'_process.json'),dict(pid=p.pid,command=cmd,start_time=time.time(),scheduling='CIC_IDS_PRIORITY'))
  return {'domain':domain,'returncode':p.wait()}
def main():
 import fcntl
 lock=(r.HERE/'cicids_priority.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  pending={pool.submit(worker,d):d for d in ['Tuesday','Wednesday','Monday','Thursday']};finished={}
  while pending:
   done,_=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:finished[pending.pop(f)]=f.result()
   r.write(r.HERE/'CICIDS_PRIORITY_STATUS.json',dict(pid=os.getpid(),time=time.time(),active_or_queued=list(pending.values()),finished=finished,Friday='completed; see outputs/CIC_IDS/Friday/REBUILD_COMPLETE.json',CIC_DDOS='SIGSTOP paused; no automatic resume',training_started=False))
if __name__=='__main__':main()

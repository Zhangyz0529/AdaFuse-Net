"""User-authorized dependency scheduling; scientific reconstruction stays frozen.
Resume paused DDoS Test only after all five IDS source rebuilds close successfully.
The mixed old supervisor stays paused, avoiding duplicate IDS work. Afterwards
run DDoS Train with the same frozen implementation; existing audit watcher remains.
"""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import run as r

def status(stage,**fields):r.write(r.HERE/'DDOS_AFTER_CICIDS_STATUS.json',dict(stage=stage,time=time.time(),pid=os.getpid(),training_started=False,optuna_started=False,**fields))
def frozen_check():
 for name,h in json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())['code_sha256'].items():
  if r.sha(Path(name))!=h:raise RuntimeError('Frozen code changed: '+name)
def run_domain(domain):
 frozen_check();original=json.loads((r.HERE/'SUPERVISOR_PROCESS.json').read_text())['command'];root=original[original.index('--data-root')+1];cmd=[sys.executable,'-u',str(r.HERE/'run.py'),'--data-root',root,'--dataset','CIC_DDOS','--domain',domain]
 with (r.HERE/'execution'/('CIC_DDOS_'+domain+'.log')).open('a',buffering=1) as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  r.write(r.HERE/'execution'/('CIC_DDOS_'+domain+'_process.json'),dict(pid=p.pid,command=cmd,start_time=time.time(),scheduler='after_cicids'))
  if p.wait():raise RuntimeError('DDoS source job failed: '+domain)
def main():
 lock=(r.HERE/'ddos_after_cicids.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while True:
   ready=[]
   for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
    d=r.HERE/'outputs/CIC_IDS'/day;f=d/'REBUILD_COMPLETE.json';s=d/'STATUS.json'
    if s.exists() and json.loads(s.read_text())['stage']=='FAILED':raise RuntimeError('CIC_IDS source failed: '+day)
    if f.exists():
     x=json.loads(f.read_text())
     if not x['counts_close'] or x['status']!='RECONSTRUCTED_UNSPLIT':raise RuntimeError('CIC_IDS source did not produce complete accepted scope: '+day)
     ready.append(day)
   if len(ready)==5:break
   status('WAITING_FOR_CICIDS_REBUILD',completed_ids_domains=ready,trigger='five complete source rebuilds; IDS partition audit may continue concurrently');time.sleep(20)
  frozen_check();domain='Test_Day_03-11';d=r.HERE/'outputs/CIC_DDOS'/domain;complete=d/'REBUILD_COMPLETE.json'
  if not complete.exists():
   pause=json.loads((r.HERE/'CIC_DDOS_PAUSED_FOR_CICIDS.json').read_text());pid=pause['CIC_DDOS']['pid'];proc=Path('/proc')/str(pid)/'cmdline'
   if proc.exists():
    command=proc.read_bytes().replace(b'\0',b' ').decode()
    if str(r.HERE/'run.py') not in command or '--dataset CIC_DDOS' not in command or '--domain '+domain not in command:raise RuntimeError('Paused PID identity mismatch; refusing signal')
    status('RESUMING_DDOS_TEST',resumed_pid=pid);os.kill(pid,signal.SIGCONT)
    r.write(r.HERE/'CIC_DDOS_RESUMED_AFTER_CICIDS.json',dict(pid=pid,time=time.time(),authorization='User requested automatic continuation after CIC_IDS generation',signal='SIGCONT',mixed_supervisor_remains_paused=True))
    while not complete.exists():
     s=json.loads((d/'STATUS.json').read_text())
     if s['stage']=='FAILED':raise RuntimeError('Resumed DDoS Test failed: '+str(s))
     if not proc.exists():raise RuntimeError('Resumed DDoS Test exited without completion')
     time.sleep(20)
   else:status('RESTARTING_MISSING_DDOS_TEST_PROCESS');run_domain(domain)
  status('GENERATING_DDOS_TRAIN')
  if not (r.HERE/'outputs/CIC_DDOS/Train_Day_01-12/REBUILD_COMPLETE.json').exists():run_domain('Train_Day_01-12')
  status('DDOS_REBUILD_FINISHED_PARTITION_AUDIT_PENDING',audit_status='PARTITION_WATCH_STATUS.json')
 except Exception as e:status('BLOCKED',error=repr(e));raise
if __name__=='__main__':main()

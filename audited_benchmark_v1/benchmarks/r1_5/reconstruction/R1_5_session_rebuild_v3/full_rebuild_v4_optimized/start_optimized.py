"""Preserve the partial old index, restart identical DoH-independent DDoS rules
with sequential key dedup and a4GiB SQLite cache; no model outcomes involved.
"""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import run as r

def matching(needle,extra=None):
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v];cmd=' '.join(argv)
   if not argv or not Path(argv[0]).name.startswith('python'):continue
   cwd=(p/'cwd').resolve();same_script=any((Path(arg) if Path(arg).is_absolute() else cwd/arg).resolve()==Path(needle).resolve() for arg in argv[1:] if arg.endswith('.py'))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if same_script and (extra is None or extra in cmd):found.append((int(p.name),cmd))
 return found
def status(stage,**x):r.write(r.HERE/'OPERATIONS_STATUS.json',dict(stage=stage,time=time.time(),pid=os.getpid(),training_started=False,**x))
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();lock=(r.HERE/'operations.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,h in json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())['code_sha256'].items():
  if r.sha(Path(name))!=h:raise RuntimeError('Optimized source frozen code mismatch')
 old=r.V3/'full_rebuild';paper=r.V3.parent;stopped=[]
 for pid,cmd in matching(str(old/'resume_ddos_after_cicids.py')):os.kill(pid,signal.SIGTERM);stopped.append(dict(pid=pid,command=cmd))
 children=matching(str(old/'run.py'),'--dataset CIC_DDOS')
 for pid,cmd in children:os.kill(pid,signal.SIGTERM);stopped.append(dict(pid=pid,command=cmd))
 while True:
  active=[]
  for pid,_ in children:
   f=Path('/proc')/str(pid)/'status'
   try:state=f.read_text().split('State:')[1].splitlines()[0]
   except FileNotFoundError:continue
   if 'Z' not in state:active.append(pid)
  if not active:break
  status('WAITING_OLD_INDEX_WRITER_EXIT',old_pids=active);time.sleep(5)
 # Leave the running IDS audit child untouched; stop only its old waiting dispatcher.
 paused=[]
 for pid,cmd in matching(str(paper/'R1_5_task_role_partitions_v1/watch.py')):os.kill(pid,signal.SIGSTOP);paused.append(dict(pid=pid,command=cmd))
 public=old/'outputs/CIC_DDOS';destination=r.HERE/'outputs/CIC_DDOS';destination.mkdir(parents=True,exist_ok=True);archive=None
 if public.exists() and not public.is_symlink():
  archive=old/'archives'/('partial_CIC_DDOS_before_storage_v4_'+str(int(time.time())));archive.parent.mkdir(exist_ok=True);public.rename(archive)
 if not public.exists():public.symlink_to(destination.resolve(),target_is_directory=True)
 elif public.resolve()!=destination.resolve():raise RuntimeError('Unexpected existing output link')
 r.write(r.HERE/'MIGRATION.json',dict(time=time.time(),terminated_old=stopped,paused_dispatcher_only=paused,IDS_audit_child_unchanged=True,archived_partial_index=str(archive),public_output=str(public),new_output=str(destination),scientific_settings_unchanged=True,reason='observed severe random-key disk writes; old CSV progress partial, no result/checkpoint existed'))
 status('CIC_DDOS_REBUILD_RUNNING')
 with (r.HERE/'rebuild.log').open('w') as f:
  cmd=[sys.executable,'-u',str(r.HERE/'run.py'),'--dataset','CIC_DDOS','--data-root',str(a.data_root)];p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});r.write(r.HERE/'REBUILD_PROCESS.json',dict(pid=p.pid,command=cmd,time=time.time()));ret=p.wait()
 if ret:status('BLOCKED_REBUILD',returncode=ret);return
 role=paper/'R1_5_task_role_partitions_v2'
 status('CIC_DDOS_ROLE_AUDIT_RUNNING')
 with (r.HERE/'audit.log').open('w') as f:ret=subprocess.run([sys.executable,str(role/'prepare.py'),'--dataset','CIC_DDOS'],stdout=f,stderr=subprocess.STDOUT).returncode
 status('FINISHED_REVIEW_AUDIT_REPORT',returncode=ret,report=str(role/'partitions/CIC_DDOS/PARTITION_GATE.json'))
if __name__=='__main__':main()

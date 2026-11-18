"""Apply one role-aware partition rule to every formally rebuilt dataset.
No model training, seed retries or accepting unsupported DoH source protocols.
"""
import concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import common as r

def task(dataset):
 frozen=json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())
 for name,h in frozen['code_sha256'].items():
  import hashlib
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen role protocol changed: '+name)
 logs=r.HERE/'execution';logs.mkdir(exist_ok=True)
 with (logs/(dataset+'.log')).open('w') as f:
  cmd=[sys.executable,str(r.HERE/'prepare.py'),'--dataset',dataset];p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});r.write(logs/(dataset+'_PROCESS.json'),dict(pid=p.pid,command=cmd,time=time.time()));return p.wait()
def main():
 lock=(r.HERE/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);remaining=['ISCX','USTC','CIC_IDS','CIC_DDOS','CIC_DOH'];active={};finished={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  while remaining or active:
   waiting={}
   for ds in remaining[:]:
    try:names=r.expected_sources(ds)
    except RuntimeError as e:waiting[ds]=str(e);continue
    dirs=[r.source_root(ds)/name for name in names];complete=[d for d in dirs if (d/'REBUILD_COMPLETE.json').exists()]
    if len(complete)!=len(dirs):waiting[ds]=dict(completed=len(complete),expected=len(dirs));continue
    if len(active)>=2:waiting[ds]='waiting for role-audit worker';continue
    active[pool.submit(task,ds)]=ds;remaining.remove(ds)
   for f in [f for f in active if f.done()]:
    ds=active.pop(f)
    try:ret=f.result();finished[ds]=dict(returncode=ret,report=str(r.HERE/'partitions'/ds/'PARTITION_GATE.json'))
    except Exception as e:finished[ds]=dict(error=repr(e))
   r.write(r.HERE/'STATUS.json',dict(time=time.time(),pid=os.getpid(),active=list(active.values()),waiting=waiting,finished=finished,training_started=False,optuna_started=False))
   if remaining or active:time.sleep(15)
if __name__=='__main__':main()

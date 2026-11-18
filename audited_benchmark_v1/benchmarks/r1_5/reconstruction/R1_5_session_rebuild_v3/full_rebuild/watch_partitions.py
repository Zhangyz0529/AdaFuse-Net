"""Run frozen partition gate after each dataset's entire source scan completes."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
import run as r

def main():
 lock=(r.HERE/'partition_watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 frozen=json.loads((r.HERE/'PARTITION_PROTOCOL_FROZEN.json').read_text());remaining=['CIC_IDS','CIC_DDOS'];finished={}
 while remaining:
  for dataset in remaining[:]:
   domains=['Friday','Tuesday','Wednesday','Monday','Thursday'] if dataset=='CIC_IDS' else ['Test_Day_03-11','Train_Day_01-12'];dirs=[r.HERE/'outputs'/dataset/d for d in domains]
   if not all((d/'REBUILD_COMPLETE.json').exists() for d in dirs):
    statuses=[json.loads((d/'STATUS.json').read_text()) for d in dirs if (d/'STATUS.json').exists()]
    if any(s['stage']=='FAILED' for s in statuses):finished[dataset]='BLOCKED_SOURCE_JOB_FAILURE';remaining.remove(dataset)
    continue
   for name,h in frozen['code_sha256'].items():
    if r.sha(Path(name))!=h:raise RuntimeError('Frozen preparation code changed: '+name)
   with (r.HERE/'execution'/(dataset+'_partition_gate.log')).open('w') as f:
    p=subprocess.run([sys.executable,str(r.HERE/'prepare_partitions.py'),'--dataset',dataset],stdout=f,stderr=subprocess.STDOUT)
   finished[dataset]={'returncode':p.returncode,'report':str(r.HERE/'partitions'/dataset/'PARTITION_GATE.json')};remaining.remove(dataset)
  r.write(r.HERE/'PARTITION_WATCH_STATUS.json',dict(remaining=remaining,finished=finished,training_started=False,optuna_started=False,time=time.time()))
  if remaining:time.sleep(20)
if __name__=='__main__':main()

"""Persistent, conservative scheduler: at most2 jobs/GPU,2 RF workers.

Unavailable datasets and unverified external implementations remain visibly queued.
A failed job is recorded, never silently retried or counted as a result.
"""
import json, os, subprocess, sys, time
from pathlib import Path
from worker import HERE,sha,save
PAPER=HERE.parent
ROOTS={'USTC':PAPER/'R1_5_ustc_boundary_rebuild_v2/partitions/USTC','ISCX':PAPER/'R1_5_task_role_partitions_v1/partitions/ISCX','CIC_IDS':PAPER/'R1_5_task_role_partitions_v1/partitions/CIC_IDS','CIC_DDOS':PAPER/'R1_5_task_role_partitions_v2/partitions/CIC_DDOS','CIC_DOH':PAPER/'R1_5_task_role_partitions_v2/partitions/CIC_DOH'}
def main():
 import fcntl
 lock=(HERE/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=json.loads((HERE/'PROTOCOL.json').read_text());frozen=json.loads((HERE/'CODE_SHA256.json').read_text());active={};failed=set();logs=[]
 # Adopt live workers from the prior supervisor without restarting their training.
 class Adopted:
  def __init__(self,pid,job):self.pid=pid;self.job=job
  def poll(self):
   try:
    cmd=Path(f'/proc/{self.pid}/cmdline').read_bytes().split(b'\0')
    live=[x.decode() for x in cmd if x]
    stat=Path(f'/proc/{self.pid}/stat').read_text().split(') ',1)[1].split()[0]
    if live==self.job['command'] and stat!='Z':return None
   except (FileNotFoundError,ProcessLookupError):pass
   j=self.job;dest=Path(j['output_dir']) if 'output_dir' in j else HERE/'runs'/j['dataset']/j['model']/f"seed{j['seed']}"/f"round{j['fold']}"/f"shot{j['shot']}"
   return 0 if (dest/'COMPLETE.json').exists() else 1
 for lp in (HERE/'launches').glob('*.json'):
  job=json.loads(lp.read_text());proc=Adopted(job['pid'],job)
  if proc.poll() is None:
   log=(HERE/'logs'/f"{job['key']}.log").open('a');active[job['key']]=(proc,log,job['gpu'],job)
 def verify():
  for p,h in frozen.items():
   if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
 cycle=0
 while True:
  verify();ready={};waiting={}
  for ds,root in ROOTS.items():
   p=root/'PARTITION_GATE.json'
   if not p.exists():waiting[ds]='WAIT_PARTITION_GATE';continue
   g=json.loads(p.read_text())
   if g.get('status')!='PASS_PARTITIONS_AND_EVALUATION_POOLS':waiting[ds]='BLOCKED: '+str(g.get('error'));continue
   if not (root/'split_manifest_v3.csv').exists():waiting[ds]='WAIT_MANIFEST';continue
   ready[ds]=(root,len(g['folds']))
  for key,(proc,log,gpu,job) in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   log.close();del active[key]
   if rc:failed.add(key);save(HERE/'failures'/f'{key}.json',dict(job=job,returncode=rc,time=time.time(),automatic_retry=False))
  try:
   lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();free={int(x.split(',')[0]):int(x.split(',')[1]) for x in lines}
  except Exception:free={}
  pending=0
  native_order=['Pre-Contrast-Local','MFF','GP-VAE','FlowPic','MetaTraffic','Trafficformer-Light'];shift=cycle%len(native_order);cycle+=1
  for model in native_order[shift:]+native_order[:shift]+['AdaFuse-Net','RandomForest']:
   for ds,(root,nfolds) in ready.items():
    for seed in protocol['seeds']:
     for fold in range(nfolds):
      for shot in protocol['shots']:
       native=model in ['MFF','GP-VAE','FlowPic','MetaTraffic','Trafficformer-Light','Pre-Contrast-Local'];nr=HERE.parent/'R1_5_native_baselines_v1';key=f'{ds}_{model}_{seed}_round{fold}_shot{shot}';dest=(nr if native else HERE)/'runs'/ds/model/f'seed{seed}'/f'round{fold}'/f'shot{shot}'
       if model=='MetaTraffic':key='V4_'+key
       if native:
        if not (nr/'cache'/ds/f'round{fold}'/'COMPLETE.json').exists():pending+=1;continue
        if model=='FlowPic' and not (nr/'flowpic'/ds/'COMPLETE.json').exists():pending+=1;continue
        if model in ['MetaTraffic','Trafficformer-Light'] and not (nr/'packet_patterns'/ds/'COMPLETE.json').exists():pending+=1;continue
       smokebase='smoke_v4' if model=='MetaTraffic' else 'smoke';smoke=native and model in ['FlowPic','MetaTraffic','Trafficformer-Light','Pre-Contrast-Local'] and not (nr/smokebase/'USTC'/model/'seed42'/'round0'/'shot1'/'COMPLETE.json').exists()
       if smoke:
        if ds!='USTC' or seed!=42 or fold!=0 or shot!=1:pending+=1;continue
        key=('SMOKE_V4_' if model=='MetaTraffic' else 'SMOKE_')+key;dest=nr/smokebase/ds/model/f'seed{seed}'/f'round{fold}'/f'shot{shot}'
       if (dest/'COMPLETE.json').exists() or key in active:continue
       if (HERE/'failures'/f'{key}.json').exists() or key in failed:continue
       gpu=None
       if model!='RandomForest':
        if native and sum(v[3]['model']==model and v[2] is not None for v in active.values())>=1:pending+=1;continue
        if not native and sum(v[3]['dataset']==ds and v[2] is not None for v in active.values())>=2:pending+=1;continue
        options=[i for i in [0,1] if sum(v[2]==i for v in active.values())<2 and free.get(i,0)>=protocol['minimum_free_gpu_mib']]
        if not options:pending+=1;continue
        gpu=min(options,key=lambda i:(sum(v[2]==i for v in active.values()),-free[i]))
       elif sum(v[2] is None for v in active.values())>=2:pending+=1;continue
       verify();logpath=HERE/'logs'/f'{key}.log';logpath.parent.mkdir(exist_ok=True);log=logpath.open('a');env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1');device='cpu'
       if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu);device='cuda:0';free[gpu]-=protocol['minimum_free_gpu_mib']
       worker=(nr/('worker_precontrast_local.py' if model=='Pre-Contrast-Local' else 'worker_v4.py' if model=='MetaTraffic' else 'worker_v2.py' if model in ['FlowPic','Trafficformer-Light'] else 'worker.py')) if native else HERE/'worker.py';command=[sys.executable,str(worker),'--dataset',ds,'--fold',f'round{fold}','--shot',str(shot),'--seed',str(seed),'--model',model,'--device',device];command+=[] if native else ['--root',str(root)];command+=['--smoke'] if smoke else [];job=dict(key=key,dataset=ds,model=model,seed=seed,fold=fold,shot=shot,gpu=gpu,command=command,start=time.time(),output_dir=str(dest),smoke=smoke,implementation_version=('original_project_precontrast_local_v1' if model=='Pre-Contrast-Local' else 'native_common_input_fewshot_v4' if model=='MetaTraffic' else 'native_common_input_fewshot_v2' if model in ['FlowPic','Trafficformer-Light'] else 'native_common_input_fewshot_v1') if native else 'audited_benchmark_v1');proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);job['pid']=proc.pid;active[key]=(proc,log,gpu,job);save(HERE/'launches'/f'{key}.json',job)
  save(HERE/'QUEUE_STATUS.json',dict(time=time.time(),pid=os.getpid(),active=[v[3] for v in active.values()],ready=list(ready),waiting=waiting,pending_ready_jobs=pending,failed_this_session=sorted(failed),external_models='5 native adaptations plus user-authorized original Pre-Contrast-Local, all subject to versioned checks',max_per_gpu=2,RF_workers=2))
  time.sleep(20)
if __name__=='__main__':main()

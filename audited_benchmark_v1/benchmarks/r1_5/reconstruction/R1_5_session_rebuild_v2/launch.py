"""Audited source reconstruction -> freeze -> 50 trials. Fail closed per dataset."""
import argparse,concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from rebuild import HERE,save,sha,CLASSES,jobs
from session_rules import RULES

def run_dataset(dataset,gpu,data_root):
 out=HERE/'execution'/dataset;out.mkdir(parents=True,exist_ok=True)
 stages=[('reconstruct',[sys.executable,str(HERE/'rebuild.py'),'--dataset',dataset,'--data-root',str(data_root),'--target-per-class-per-domain','5000']),('freeze_and_audit',[sys.executable,str(HERE/'prepare_v2.py'),'--dataset',dataset,'--seed','3407']),('optuna50',[sys.executable,str(HERE/'search.py'),'--dataset',dataset,'--seed','3407','--device',f'cuda:{gpu}'])]
 try:
  for phase,cmd in stages:
   frozen=json.loads((HERE/'PROTOCOL_FROZEN.json').read_text())
   for path,digest in frozen['code_sha256'].items():
    if sha(path)!=digest:raise RuntimeError('Frozen code changed: '+path)
   save(out/'STATUS.json',dict(dataset=dataset,phase=phase,state='running',gpu=gpu,time=time.time(),pid=os.getpid()))
   with (out/(phase+'.log')).open('a',buffering=1) as log:
    subprocess.run(cmd,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
  save(out/'STATUS.json',dict(dataset=dataset,phase='complete',state='complete',gpu=gpu,time=time.time()))
 except Exception as e:
  save(out/'STATUS.json',dict(dataset=dataset,state='FAILED_DO_NOT_TRAIN_OR_CLAIM_COMPLETION',error=str(e),time=time.time()));return False
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();lock=(HERE/'.launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for dataset in CLASSES:
  for job in jobs(a.data_root,dataset):
   if not job['csvs'] or not job['pcaps'] or any(not p.is_file() for p in job['csvs']+job['pcaps']):raise RuntimeError('Source path preflight failed: '+dataset+'/'+job['domain'])
 import torch
 if not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError('Two visible GPUs required; no silent CPU fallback')
 if not (HERE/'GPU_SMOKE_CHECK.json').exists():subprocess.run([sys.executable,str(HERE/'gpu_smoke.py'),'--device','cuda:0'],cwd=HERE,check=True)
 if not json.loads((HERE/'GPU_SMOKE_CHECK.json').read_text())['passed']:raise RuntimeError('GPU smoke failed')
 code=list(HERE.glob('*.py'))+list((HERE/'code').glob('*.py'))+list((HERE/'source_snapshot').glob('*.py'))+[HERE.parent/'R1_5_leakage_audited_reproduction_v1/split_and_audit.py',HERE.parent/'CIC_loV/ours_b7_optuna400_v1/search.py']
 contract=dict(version='session_rebuild_v2',datasets=list(CLASSES),session_rules=RULES,classes=CLASSES,source_root=str(a.data_root.resolve()),cross_label_input_rule='exclude_ALL_members_before_split',same_label_duplicates='retain together via connected-component grouping',ratios=[.6,.2,.2],split_seed=42,hpo_seed=3407,complete_trials_per_dataset=50,max_attempted_trials_per_dataset=100,updates=50,shots=[1,3,5,10],test_episodes_per_task=100,selection='mean validation F1 equally over five held-out folds and four shots; fixed final update50 checkpoint; test only after study selection',resources=dict(target_rows_per_class_per_domain=5000,max_csv_rows_per_key=100,max_packets_per_key=50000,max_collected_packets_per_domain=3000000),clock=dict(min_anchors=10,min_consensus=.95,duration_tolerance_us=2),scope='AdaFuse-Net only; not a completed full baseline comparison or multiseed principal table',code_sha256={str(f.resolve()):sha(f) for f in code})
 frozen=HERE/'PROTOCOL_FROZEN.json'
 if frozen.exists():
  if json.loads(frozen.read_text())!=contract:raise RuntimeError('Protocol mismatch; do not overwrite frozen experiment')
 else:save(frozen,contract)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  active={pool.submit(run_dataset,d,g,a.data_root):d for d,g in [('CIC_IDS',0),('CIC_DDOS',1)]};results={}
  while active:
   ready,_=concurrent.futures.wait(active,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in ready:results[active.pop(f)]=f.result()
   save(HERE/'STATUS.json',dict(active=list(active.values()),finished=results,state='running' if active else 'complete' if all(results.values()) else 'failed_or_partial',time=time.time()))
if __name__=='__main__':main()

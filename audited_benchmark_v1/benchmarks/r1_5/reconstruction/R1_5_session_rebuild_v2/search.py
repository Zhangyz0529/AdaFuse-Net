"""Per-dataset/seed 50-trial B7 validation-only search over all rotation rounds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,importlib.util,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import optuna
HERE=Path(__file__).resolve().parent
COMMON=HERE.parent/'CIC_loV/ours_b7_optuna400_v1/search.py'
spec=importlib.util.spec_from_file_location('b7_shared_search',COMMON);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
spec=importlib.util.spec_from_file_location('b7_dynamic_model',HERE/'source_snapshot/model.py');dynamic=importlib.util.module_from_spec(spec);spec.loader.exec_module(dynamic)
from losses import factorial_loss
SHOTS=[1,3,5,10]
save=common.save;sha=common.sha

def main(a):
 root=HERE/('smoke' if a.smoke else 'runs')/a.dataset/f'seed{a.seed}';root.mkdir(parents=True,exist_ok=True)
 prep=HERE/'prepared'/a.dataset/f'seed{a.seed}';manifest=json.loads((prep/'manifest.json').read_text());rounds=[]
 freeze=HERE/'datasets'/a.dataset/'FREEZE_COMPLETE.json';assert freeze.exists(),'No audited v2 freeze: DO NOT TRAIN'
 frozen=json.loads(freeze.read_text());assert frozen['passed'] and sha(HERE/'datasets'/a.dataset/'split_manifest_v2.csv')==frozen['split_manifest_sha256']
 assert sha(prep/'manifest.json')==frozen['prepared_manifest_sha256']
 for item in manifest['rounds'][:1 if a.smoke else None]:
  rd=prep/f"round{item['round']}";data=dict(np.load(rd/'shared_inputs.npz'));assert sha(rd/'shared_inputs.npz')==item['shared_sha256']
  train=json.loads((rd/'train_episodes.json').read_text());specs=json.loads((rd/'eval_episodes.json').read_text())
  if a.smoke:
   train=train[:2];specs={role:{task:eps[:2] for task,eps in tasks.items()} for role,tasks in specs.items()}
  rounds.append(dict(id=item['round'],data=data,train=train,specs=specs,val_ids=common.role_ids(specs['validation'])))
 contract=dict(dataset=a.dataset,seed=a.seed,stat_dim=manifest['stat_dim'],shots=SHOTS,target_complete_trials=1 if a.smoke else 50,rounds=[r['id'] for r in rounds],updates=2 if a.smoke else 50,
  prepared_manifest_sha256=sha(prep/'manifest.json'),objective='mean validation MacroF1 across all rounds/tasks and4shots; never test-based selection',
  source_sha256={str(p):sha(p) for p in [HERE/'search.py',COMMON,*sorted((HERE/'source_snapshot').glob('*.py'))]})
 if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==contract
 else:save(root/'protocol.json',contract)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sampler_path=root/'sampler.pkl';sampler=pickle.loads(sampler_path.read_bytes()) if sampler_path.exists() else optuna.samplers.TPESampler(seed=a.seed)
 study=optuna.create_study(study_name=f'B7_{a.dataset}_seed{a.seed}',storage='sqlite:///'+str(root/'study.db'),direction='maximize',sampler=sampler,pruner=optuna.pruners.NopPruner(),load_if_exists=True)
 def objective(trial):
  p=dict(embed_dim=trial.suggest_categorical('embed_dim',[64,128,256]),lr=trial.suggest_float('lr',1e-5,1e-2,log=True),weight_decay=trial.suggest_float('weight_decay',1e-6,1e-3,log=True),temperature=trial.suggest_float('temperature',.05,.2),lambda_scl=trial.suggest_float('lambda_scl',.01,1.,log=True),lambda_regularizer=trial.suggest_float('lambda_regularizer',.001,.1,log=True))
  folder=root/'trials'/f'trial{trial.number:04d}';folder.mkdir(parents=True,exist_ok=True);save(folder/'params.json',p);scores=[]
  for r in rounds:
   data=r['data']
   for k in SHOTS:
    out=folder/f"round{r['id']}"/f'shot{k}';out.mkdir(parents=True,exist_ok=True);common.reset(a.seed)
    model=dynamic.FactorialNet(True,True,True,embed_dim=p['embed_dim'],stat_dim=manifest['stat_dim']).to(a.device)
    opt=torch.optim.Adam(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=len(r['train']),eta_min=1e-6);logs=[]
    for step,ep in enumerate(r['train']):
     model.train();opt.zero_grad(set_to_none=True);s=np.asarray(ep['support'])[:,:k].ravel();q=np.asarray(ep['query']).ravel();labels=torch.arange(len(ep['classes']),device=a.device).repeat_interleave(len(q)//len(ep['classes']))
     loss,parts=factorial_loss(model,model(*common.torch_inputs(data,s,a.device),return_parts=True),model(*common.torch_inputs(data,q,a.device),return_parts=True),labels,k,p)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
     loss.backward();grads=[v.grad for v in model.parameters() if v.grad is not None]
     if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Invalid gradients')
     opt.step();sched.step();logs.append(dict(update=step+1,**{n:float(v.detach()) for n,v in parts.items()}))
     save(root/'STATUS.json',dict(phase='search',dataset=a.dataset,seed=a.seed,trial=trial.number,round=r['id'],shot=k,update=step+1,complete_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)))
    pd.DataFrame(logs).to_csv(out/'training_log.csv',index=False)
    if not any(v['dec']>0 for v in logs):raise FloatingPointError('Inactive dec')
    z=common.encode_ids(model,data,r['val_ids'],a.device,p['embed_dim']);common.evaluate(model,'AdaFuse-Net',data,{'validation':r['specs']['validation']},k,a.device,{},out,z)
    score=float(common.summarize(out).F1.iloc[0]);scores.append(score);trial.set_user_attr(f"round{r['id']}_{k}shot_valF1",score)
    torch.save(dict(state_dict=model.cpu().state_dict(),params=p,seed=a.seed,shot=k,round=r['id'],protocol_sha256=sha(root/'protocol.json')),out/'checkpoint.pt')
    del model,opt,sched,z;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
  return float(np.mean(scores))
 def callback(study,trial):
  sampler_path.write_bytes(pickle.dumps(study.sampler));study.trials_dataframe().to_csv(root/'trials.csv',index=False)
  winners=sorted((t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE),key=lambda t:(-t.value,t.number))[:10];keep={t.number for t in winners}
  save(root/'top10_validation.json',[dict(trial=t.number,validation_score=t.value,params=t.params,per_round_shot=t.user_attrs) for t in winners])
  for f in (root/'trials').glob('trial*/round*/shot*/checkpoint.pt'):
   if int(f.parents[2].name.replace('trial','')) not in keep:f.unlink()
 while sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)<contract['target_complete_trials']:
  
  if len(study.trials)>=100:raise RuntimeError('Frozen maximum100 attempted trials reached before50 COMPLETE trials')
  study.optimize(objective,n_trials=1,callbacks=[callback],catch=(FloatingPointError,),gc_after_trial=True)
 best=study.best_trial;save(root/'SELECTED_BEFORE_TEST.json',dict(trial=best.number,validation_score=best.value,params=best.params));rows=[]
 for r in rounds:
  data=r['data'];ids=common.role_ids(r['specs']['test'])
  for k in SHOTS:
   out=root/'selected_test'/f"round{r['id']}"/f'shot{k}';out.mkdir(parents=True,exist_ok=True)
   ck=torch.load(root/'trials'/f'trial{best.number:04d}'/f"round{r['id']}"/f'shot{k}'/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['protocol_sha256']==sha(root/'protocol.json')
   model=dynamic.FactorialNet(True,True,True,embed_dim=best.params['embed_dim'],stat_dim=manifest['stat_dim']).to(a.device);model.load_state_dict(ck['state_dict']);z=common.encode_ids(model,data,ids,a.device,best.params['embed_dim'])
   common.evaluate(model,'AdaFuse-Net',data,{'test':r['specs']['test']},k,a.device,{},out,z);common.summarize(out);rows.append(pd.read_csv(out/'task_metrics.csv'));del model,z
 task=pd.concat(rows,ignore_index=True);task.to_csv(root/'test_task_metrics.csv',index=False);fields=['F1','Attack_F1','AUC','AUPRC','ACC','REC','PRE','AttackRecall','AttackPrecision','FPR'];task.groupby(['model','shot','role'])[fields].mean().reset_index().to_csv(root/'test_summary.csv',index=False)
 save(root/'COMPLETE.json',dict(passed=True,dataset=a.dataset,seed=a.seed,complete_trials=contract['target_complete_trials'],best_trial=best.number,smoke=a.smoke));save(root/'STATUS.json',dict(phase='complete',dataset=a.dataset,seed=a.seed))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['CIC_IDS','CIC_DDOS'],required=True);p.add_argument('--seed',type=int,choices=[3407,2026],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a)

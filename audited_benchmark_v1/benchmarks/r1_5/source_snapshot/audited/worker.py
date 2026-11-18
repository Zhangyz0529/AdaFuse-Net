"""Audited-fold runner: native support-fitted RF and published AdaFuse implementation.

Never samples outside the supplied manifest. Test is read only after the fixed
last-update checkpoint has been written. This is fixed-configuration evaluation,
not independently tuned best performance or an official external reproduction.
"""
import argparse, csv, gzip, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[2]/'reproducibility/AdaFuse-Net-Repro/src'
sys.path.insert(0,str(PACKAGE))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset',required=True);a.add_argument('--root',required=True);a.add_argument('--fold',required=True);a.add_argument('--shot',type=int,required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--model',choices=['AdaFuse-Net','RandomForest'],required=True);a.add_argument('--device',default='cpu');a.add_argument('--smoke',action='store_true');args=a.parse_args()
 protocol=json.loads((HERE/'PROTOCOL.json').read_text());root=Path(args.root);gate=json.loads((root/'PARTITION_GATE.json').read_text())
 if gate['status']!='PASS_PARTITIONS_AND_EVALUATION_POOLS':raise RuntimeError('Partition gate not passed')
 dest=HERE/('smoke' if args.smoke else 'runs')/args.dataset/args.model/f'seed{args.seed}'/args.fold/f'shot{args.shot}'
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 random.seed(args.seed);np.random.seed(args.seed)
 # stdlib CSV avoids a reproduced pandas C-parser segmentation fault on ISCX.
 csv.field_size_limit(2**30)
 keep=['sample_id','label','partition','session_id','split_group_id','content_hash']
 with (root/'split_manifest_v3.csv').open(newline='') as f:
  m=pd.DataFrame([{k:row[k] for k in keep} for row in csv.DictReader(f) if row['outer_fold']==args.fold])
 with (root/'group_components.csv').open(newline='') as f:
  globalmap={row['sample_id']:i for i,row in enumerate(csv.DictReader(f))}
 for x,y in [('Train','Val'),('Train','Test'),('Val','Test')]:
  for col in ['sample_id','session_id','split_group_id','content_hash']:
   assert not set(m.loc[m.partition==x,col])&set(m.loc[m.partition==y,col]),(x,y,col)
 with (root/args.fold/'scaler_fit_sample_ids.csv').open(newline='') as f:
  fitids={row['sample_id'] for row in csv.DictReader(f)}
 assert fitids==set(m.loc[m.partition=='Train','sample_id']), 'Scaler fit IDs differ from Train'
 archive=np.load(root/args.fold/'scaled_statistics.npz');ids=archive['sample_ids'].astype(str);stat=archive['stat'];local={s:i for i,s in enumerate(ids)};sequence=np.load(root/'sequence.npy',mmap_mode='r')
 def inputs(sids):
  return stat[[local[s] for s in sids]],np.asarray(sequence[[globalmap[s] for s in sids]])
 plans=json.loads((root/args.fold/'evaluation_episodes.json').read_text())
 info=dict(dataset=args.dataset,model=args.model,seed=args.seed,fold=args.fold,shot=args.shot,protocol=protocol,root=str(root.resolve()),gate_sha256=sha(root/'PARTITION_GATE.json'),manifest_sha256=sha(root/'split_manifest_v3.csv'),evaluation_plan_sha256=sha(root/args.fold/'evaluation_episodes.json'),worker_sha256=sha(__file__),scaler_sha256=sha(root/args.fold/'scaler.joblib'),smoke=args.smoke)
 save(dest/'effective_config.json',info)
 model=None
 if args.model=='AdaFuse-Net':
  import torch
  from adafuse.models.network import FactorialNet
  from adafuse.losses.objectives import factorial_loss
  torch.set_num_threads(2);torch.manual_seed(args.seed)
  if args.device.startswith('cuda'):torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  cfg=protocol['adafuse'];model=FactorialNet(True,True,True,embed_dim=cfg['embed_dim'],stat_dim=stat.shape[1]).to(args.device)
  opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['updates'],eta_min=1e-6)
  tr=m[m.partition=='Train'];classes=sorted(tr.label.unique());supportpool={};querypool={}
  for label in classes:
   p=tr[tr.label==label];s,q=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=3407).split(p,groups=p.split_group_id));supportpool[label]=p.iloc[s].sample_id.to_numpy();querypool[label]=p.iloc[q].sample_id.to_numpy()
   assert not set(p.iloc[s].split_group_id)&set(p.iloc[q].split_group_id)
   if len(s)<10 or len(q)<15:raise RuntimeError('Insufficient group-disjoint training pools')
  rng=np.random.default_rng(3407);training=[]
  for step in range(cfg['updates']):
   ep=dict(classes=classes,support={},query={})
   for c in classes:ep['support'][c]=rng.choice(supportpool[c],10,False).tolist();ep['query'][c]=rng.choice(querypool[c],15,False).tolist()
   training.append(ep)
  save(dest/'training_episodes.json',training)
  def tensors(sids):
   xs,xq=inputs(sids);return torch.tensor(xs,device=args.device),torch.tensor(xq.astype('int64'),device=args.device)
  with (dest/'training_trajectory.jsonl').open('w') as log:
   for step,ep in enumerate(training[:2] if args.smoke else training,1):
    model.train();opt.zero_grad(set_to_none=True);si=sum([ep['support'][c][:args.shot] for c in classes],[]);qi=sum([ep['query'][c] for c in classes],[])
    s=model(*tensors(si),return_parts=True);q=model(*tensors(qi),return_parts=True);labels=torch.arange(len(classes),device=args.device).repeat_interleave(15)
    loss,parts=factorial_loss(model,s,q,labels,args.shot,cfg)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradients')
    opt.step();scheduler.step();log.write(json.dumps(dict(step=step,**{k:float(v.detach()) for k,v in parts.items()}))+'\n');log.flush()
  torch.save(dict(state_dict=model.cpu().state_dict(),effective_config=info),dest/'checkpoint.pt');model.to(args.device).eval();save(dest/'parameters.json',dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
  def encode(xs,xq):
   values=[]
   with torch.inference_mode():
    for start in range(0,len(xs),16):values.append(model(torch.tensor(xs[start:start+16],device=args.device),torch.tensor(xq[start:start+16].astype('int64'),device=args.device)).cpu().numpy())
   return np.concatenate(values)
 from adafuse.perturbations.operators import corrupt_sequence
 rows=[]
 with gzip.open(dest/'predictions.jsonl.gz','wt') as predfile:
  for role in ['Val','Test']:
   episodes=plans[f'{role}_{args.shot}shot'];episodes=episodes[:2] if args.smoke else episodes
   # Encode unique records once per condition; no transductive BN or query fitting.
   su=sorted({s for ep in episodes for c in ep['classes'] for s in ep['support'][c]});qu=sorted({s for ep in episodes for c in ep['classes'] for s in ep['query'][c]});assert not set(su)&set(qu)
   ss,sq=inputs(su);qs,qq=inputs(qu);sl={s:i for i,s in enumerate(su)};ql={s:i for i,s in enumerate(qu)}
   zsupport=encode(ss,sq) if model is not None else None
   for condition in ['Clean','IBP30']:
    corrupt=corrupt_sequence(qq,[globalmap[s] for s in qu],args.dataset,'Clean' if condition=='Clean' else 'IBP',0 if condition=='Clean' else .3,protocol['noise_seed'])
    zquery=encode(qs,corrupt) if model is not None else None
    for j,ep in enumerate(episodes):
     si=sum([ep['support'][c] for c in ep['classes']],[]);qi=sum([ep['query'][c] for c in ep['classes']],[]);ys=np.repeat(np.arange(2),args.shot);y=np.repeat(np.arange(2),15);isup=[sl[s] for s in si];iq=[ql[s] for s in qi]
     if model is None:
      rf=RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=args.seed,n_jobs=1);x_s=np.concatenate([ss[isup],sq[isup].astype('float32')/255],axis=1);x_q=np.concatenate([qs[iq],corrupt[iq].astype('float32')/255],axis=1);rf.fit(x_s,ys);pred=rf.predict(x_q);scores=rf.predict_proba(x_q)[:,1]
      if condition=='Clean':
       import joblib
       joblib.dump(rf,dest/f'{role}_episode{j}_forest.joblib',compress=3)
     else:
      proto=zsupport[isup].reshape(2,args.shot,-1).mean(1);distance=((zquery[iq,None,:]-proto[None,:,:])**2).sum(2);pred=distance.argmin(1);v=-distance;v-=v.max(1,keepdims=True);prob=np.exp(v);scores=(prob/prob.sum(1,keepdims=True))[:,1]
     tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();record=dict(dataset=args.dataset,model=args.model,seed=args.seed,fold=args.fold,shot=args.shot,role=role,condition=condition,episode=j,MacroF1=float(f1_score(y,pred,average='macro',zero_division=0)),AttackRecall=float(recall_score(y,pred,zero_division=0)),FPR=float(fp/(tn+fp)))
     rows.append(record);predfile.write(json.dumps(dict(**record,support_ids=si,query_ids=qi,y_true=y.tolist(),y_pred=pred.tolist(),attack_score=scores.tolist(),query_input_sha256=hashlib.sha256(qs[iq].tobytes()+corrupt[iq].tobytes()).hexdigest()))+'\n')
    pd.DataFrame(rows).to_csv(dest/'episode_metrics.csv',index=False)
    print(args.dataset,args.model,args.fold,args.shot,args.seed,role,condition,'done',flush=True)
 save(dest/'COMPLETE.json',dict(status='COMPLETE',time=time.time(),episodes=len(rows),effective_config_sha256=sha(dest/'effective_config.json')))
if __name__=='__main__':main()

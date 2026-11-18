"""Native supervised/generative losses and support-only method-specific adaptation.

Does not force all methods into a ProtoNet. Fixed budgets are an audited benchmark
adaptation, not a claim to replicate the authors' original closed-set tables.
"""
import argparse,copy,gzip,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score,confusion_matrix
from cache import HERE,OLD,sha,save
from precontrast_local import build
sys.path.insert(0,str(OLD));sys.path.insert(0,str(HERE.parents[2]/'reproducibility/AdaFuse-Net-Repro/src'))
from adafuse.perturbations.operators import corrupt_sequence

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',choices=['Pre-Contrast-Local'],required=True);p.add_argument('--fold',required=True);p.add_argument('--shot',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 root=HERE/'cache'/a.dataset/a.fold;attestation=json.loads((root/'COMPLETE.json').read_text());assert attestation['status']=='EXACT_SOURCE_INPUTS_AND_PLANS';dest=HERE/('smoke' if a.smoke else 'runs')/a.dataset/a.model/f'seed{a.seed}'/a.fold/f'shot{a.shot}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 protocol=json.loads((HERE/'PRECONTRAST_LOCAL_PROTOCOL.json').read_text());ids=json.loads((root/'ids.json').read_text());lookup={s:i for i,s in enumerate(ids)};stats=np.load(root/'stats.npy',mmap_mode='r');seq=np.load(root/'sequence.npy',mmap_mode='r');globalidx=np.load(root/'global_indices.npy');tr=json.loads((root/'train.json').read_text());ev=json.loads((root/'eval.json').read_text());classes=tr[0]['classes'];torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model=build(a.model,stats.shape[1],len(classes)).to(a.device);cfg=dict(dataset=a.dataset,model=a.model,implementation_version='original_project_precontrast_local_v1',seed=a.seed,fold=a.fold,shot=a.shot,cache_attestation=attestation,protocol=protocol,worker_sha256=sha(__file__),models_sha256=sha(HERE/'precontrast_local.py'),vendor_versions=json.loads((HERE/'VENDOR_VERSIONS.json').read_text()),torch=torch.__version__,smoke=a.smoke);save(dest/'effective_config.json',cfg)
 def raw(sids):
  ix=[lookup[s] for s in sids];return np.asarray(stats[ix]),np.asarray(seq[ix]),globalidx[ix]
 def tensors(sids):
  ss,sq,gi=raw(sids)
  if a.model=='FlowPic':
   coord=__import__('sqlite3').connect(f"file:{HERE/'flowpic'/a.dataset/'histograms.sqlite'}?mode=ro",uri=True);images=np.zeros((len(sids),1,1500,1500),dtype='float32')
   for i,s in enumerate(sids):
    row=coord.execute('SELECT coords FROM samples WHERE sample_id=?',(s,)).fetchone()
    if row is None:raise RuntimeError('Missing source-derived FlowPic input')
    pairs=np.frombuffer(row[0],dtype='int16').reshape(-1,2);np.add.at(images[i,0],(pairs[:,0],pairs[:,1]),1)
   coord.close();sq=images
  return torch.tensor(ss,device=a.device),torch.tensor(sq,device=a.device)
 opt=torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=0)
 save(dest/'training_episodes.json',tr)
 with (dest/'training_trajectory.jsonl').open('w') as f:
  for step,ep in enumerate(tr[:2] if a.smoke else tr,1):
   si=sum([ep['support'][c][:a.shot] for c in classes],[]);qi=sum([ep['query'][c] for c in classes],[]);sids=si+qi;y=torch.cat([torch.arange(len(classes)).repeat_interleave(a.shot),torch.arange(len(classes)).repeat_interleave(15)]).to(a.device);model.train();opt.zero_grad(set_to_none=True)
   if a.model=='Pre-Contrast-Local':
    support=model.features(*tensors(si));query=model.features(*tensors(qi));loss=model.episode_loss(support,query,y)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite original local contrastive loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();f.write(json.dumps(dict(update=step,loss=float(loss.detach())))+'\n');f.flush();continue
   # Microbatches preserve native loss; BatchNorm uses declared training minibatches.
   total=0.;logs={}
   for begin in range(0,len(sids),16):
    end=min(begin+16,len(sids));loss,parts=model.loss(*tensors(sids[begin:end]),y[begin:end]);weight=(end-begin)/len(sids)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native loss')
    (loss*weight).backward();total+=float(loss.detach())*weight
    for k,v in parts.items():logs[k]=logs.get(k,0)+float(v.detach())*weight
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite native gradient')
   opt.step();f.write(json.dumps(dict(update=step,loss=total,**logs))+'\n');f.flush()
 model.eval();torch.save(dict(state_dict=model.cpu().state_dict(),config=cfg),dest/'checkpoint.pt');model.to(a.device).eval()
 save(dest/'parameters.json',dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
 def encode(sids,condition):
  zs=[];lv=[]
  for start in range(0,len(sids),16):
   batch=sids[start:start+16];s,q=tensors(batch)
   if condition=='IBP30':
    _,rawq,gi=raw(batch);q=torch.tensor(corrupt_sequence(rawq,gi,a.dataset,'IBP',.3,20261028),device=a.device)
   with torch.inference_mode():
    if a.model=='GP-VAE':mu,logvar,_=model.net.encoder(model.image(q));zs.append(mu.cpu());lv.append(logvar.cpu())
    else:zs.append(model.features(s,q).cpu())
  return torch.cat(zs),torch.cat(lv) if lv else None
 rows=[];headstates={}
 with gzip.open(dest/'predictions.jsonl.gz','wt') as fp:
  for role in ['Val','Test']:
   episodes=ev[f'{role}_{a.shot}shot'];episodes=episodes[:2] if a.smoke else episodes;su=sorted({s for ep in episodes for c in ep['classes'] for s in ep['support'][c]});qu=sorted({s for ep in episodes for c in ep['classes'] for s in ep['query'][c]});assert not set(su)&set(qu);sl={s:i for i,s in enumerate(su)};ql={s:i for i,s in enumerate(qu)};zs,_=encode(su,'Clean')
   # Adapt exactly once using support only. Reuse across Clean and corrupted query.
   classifiers=[]
   for j,ep in enumerate(episodes):
    si=sum([ep['support'][c] for c in ep['classes']],[]);features=zs[[sl[s] for s in si]].to(a.device);ys=torch.arange(2,device=a.device).repeat_interleave(a.shot);torch.manual_seed(a.seed+j)
    if a.model=='Pre-Contrast-Local':
     proto=features.reshape(2,a.shot,-1).mean(1).detach().cpu();classifiers.append(proto);headstates[f'{role}_{j}']=proto
    elif a.model=='GP-VAE':
     means=nn.Parameter(torch.stack([features[ys==c].mean(0) for c in range(2)]));adapt=torch.optim.Adam([means],lr=.01)
     for _ in range(25):adapt.zero_grad();loss=F.cross_entropy(-torch.cdist(features,means).square()/2,ys);loss.backward();adapt.step()
     classifiers.append(means.detach().cpu());headstates[f'{role}_{j}']=means.detach().cpu()
    else:
     head=nn.Linear(features.shape[1],2).to(a.device);adapt=torch.optim.Adam(head.parameters(),lr=.01)
     for _ in range(25):adapt.zero_grad();loss=F.cross_entropy(head(features),ys);loss.backward();adapt.step()
     classifiers.append(copy.deepcopy(head.cpu()).eval());headstates[f'{role}_{j}']=head.cpu().state_dict()
   for condition in ['Clean'] if a.model=='FlowPic' else ['Clean','IBP30']:
    zq,lv=encode(qu,condition)
    for j,ep in enumerate(episodes):
     qi=sum([ep['query'][c] for c in ep['classes']],[]);si=sum([ep['support'][c] for c in ep['classes']],[]);ix=[ql[s] for s in qi];feat=zq[ix]
     with torch.inference_mode():
      if a.model=='Pre-Contrast-Local':logits=-torch.cdist(feat,classifiers[j]).square()
      elif a.model=='GP-VAE':
       # Native KL to unit-covariance Gaussian prototypes. Posterior variance term
       # cancels across classes, but is retained in the reported classification.
       logits=-.5*(torch.cdist(feat,classifiers[j]).square()+(lv[ix].exp()-lv[ix]-1).sum(1,keepdim=True))
      else:logits=classifiers[j](feat)
      score=logits.softmax(1)[:,1].numpy();pred=logits.argmax(1).numpy()
     y=np.repeat([0,1],15);tn,fpr,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();row=dict(dataset=a.dataset,model=a.model,implementation_version='original_project_precontrast_local_v1',seed=a.seed,fold=a.fold,shot=a.shot,role=role,condition=condition,episode=j,MacroF1=float(f1_score(y,pred,average='macro')),AttackRecall=float(tp/(tp+fn)),FPR=float(fpr/(tn+fpr)));rows.append(row)
     ss,sq,gi=raw(qi);cq=corrupt_sequence(sq,gi,a.dataset,'IBP' if condition=='IBP30' else 'Clean',.3 if condition=='IBP30' else 0,20261028);fp.write(json.dumps(dict(**row,support_ids=si,query_ids=qi,y_true=y.tolist(),y_pred=pred.tolist(),attack_score=score.tolist(),query_input_sha256=hashlib.sha256(ss.tobytes()+cq.tobytes()).hexdigest()))+'\n')
    pd.DataFrame(rows).to_csv(dest/'episode_metrics.csv',index=False);print(a.dataset,a.model,a.fold,a.shot,a.seed,role,condition,'done',flush=True)
 torch.save(headstates,dest/'support_fitted_classifiers.pt');save(dest/'COMPLETE.json',dict(status='COMPLETE',time=time.time(),episodes=len(rows),effective_config_sha256=sha(dest/'effective_config.json')))
if __name__=='__main__':main()

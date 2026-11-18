"""Portable evaluation using preserved inputs/plans and a published checkpoint."""
import argparse,csv,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score,confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from adafuse.models.network import FactorialNet
from adafuse.perturbations.operators import corrupt_sequence

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--dataset',required=True);p.add_argument('--fold',default='round0');p.add_argument('--seed',type=int,default=42);p.add_argument('--shot',type=int,default=1);p.add_argument('--episodes',type=int,default=100);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 r=a.artifacts/'runs'/a.dataset/'AdaFuse-Net'/f'seed{a.seed}'/a.fold/f'shot{a.shot}';cache=a.artifacts/'cache'/a.dataset/a.fold
 cfg=json.loads((r/'effective_config.json').read_text());ids=json.loads((cache/'ids.json').read_text());lookup={s:i for i,s in enumerate(ids)};stats=np.load(cache/'stats.npy',mmap_mode='r');seq=np.load(cache/'sequence.npy',mmap_mode='r');gi=np.load(cache/'global_indices.npy');plans=json.loads((cache/'eval.json').read_text())
 model=FactorialNet(True,True,True,embed_dim=cfg['protocol']['adafuse']['embed_dim'],stat_dim=stats.shape[1]).to(a.device);state=torch.load(r/'checkpoint.pt',map_location='cpu');model.load_state_dict(state['state_dict'],strict=True);model.eval()
 def encode(sids,condition):
  ix=[lookup[s] for s in sids];s=np.asarray(stats[ix]);q=np.asarray(seq[ix]);q=corrupt_sequence(q,gi[ix],a.dataset,'Clean' if condition=='Clean' else 'IBP',0 if condition=='Clean' else .3,cfg['protocol']['noise_seed']);z=[]
  with torch.inference_mode():
   for start in range(0,len(ix),16):z.append(model(torch.tensor(s[start:start+16],device=a.device),torch.tensor(q[start:start+16].astype('int64'),device=a.device)).cpu().numpy())
  return np.concatenate(z)
 with gzip.open(r/'predictions.jsonl.gz','rt') as f:reference={(x['role'],x['condition'],x['episode']):x for x in map(json.loads,f)}
 rows=[];mismatches=0;maxdiff=0.
 for role in ['Val','Test']:
  eps=plans[f'{role}_{a.shot}shot'][:a.episodes];su=sorted({s for ep in eps for c in ep['classes'] for s in ep['support'][c]});qu=sorted({s for ep in eps for c in ep['classes'] for s in ep['query'][c]});sl={s:i for i,s in enumerate(su)};ql={s:i for i,s in enumerate(qu)};zs=encode(su,'Clean')
  for condition in ['Clean','IBP30']:
   zq=encode(qu,condition)
   for j,ep in enumerate(eps):
    si=sum([ep['support'][c] for c in ep['classes']],[]);qi=sum([ep['query'][c] for c in ep['classes']],[]);proto=zs[[sl[s] for s in si]].reshape(2,a.shot,-1).mean(1);d=((zq[[ql[s] for s in qi],None,:]-proto[None,:,:])**2).sum(2);pred=d.argmin(1);y=np.repeat(np.arange(2),15);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();f1=float(f1_score(y,pred,average='macro',zero_division=0));ref=reference[(role,condition,j)];mismatches+=int(np.sum(pred!=ref['y_pred']));maxdiff=max(maxdiff,abs(f1-ref['MacroF1']));rows.append(dict(dataset=a.dataset,seed=a.seed,fold=a.fold,shot=a.shot,role=role,condition=condition,episode=j,MacroF1=f1,AttackRecall=float(tp/(tp+fn)),FPR=float(fp/(tn+fp))))
 a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/'episode_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 report=dict(dataset=a.dataset,seed=a.seed,fold=a.fold,shot=a.shot,device=a.device,episodes_per_role_condition=a.episodes,rows=len(rows),prediction_mismatches=mismatches,max_abs_f1_difference=maxdiff,passed=mismatches==0,scope='checkpoint reload evaluation, limited to recorded dataset/seed/fold/shot/episodes; not retraining');(a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(mismatches!=0)
if __name__=='__main__':main()

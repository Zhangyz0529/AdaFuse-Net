"""Matched Clean/corrupt evaluation with identical support-defined prototypes."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
import pandas as pd
import torch
from ..training.trainer import build
from ..perturbations.operators import corrupt_sequence,corrupt_statistics
from ..utils.common import save,sha
from .metrics import binary_metrics

def evaluate(checkpoint,prepared,role='test',operator='Clean',rho=0.,output=None):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];ds=saved['dataset'];device=cfg['device'];rid=saved['round'];k=saved['shot'];rd=Path(prepared)/f'round{rid}'
    model=build(cfg,ds['stat_dim']).to(device);model.load_state_dict(saved['state_dict'],strict=True);model.eval().requires_grad_(False)
    buffers={n:v.clone() for n,v in model.named_buffers()};data=dict(np.load(rd/'inputs.npz'));plans=json.loads((rd/'eval.json').read_text())[role];rows=[];raw=[]
    def encode(ids,stats,seq):
        zs=[]
        with torch.inference_mode():
            for start in range(0,len(ids),32):zs.append(model(torch.tensor(stats[start:start+32],device=device),torch.tensor(seq[start:start+32].astype('int64'),device=device)).cpu())
        return torch.cat(zs)
    for task,episodes in plans.items():
        su=np.unique(np.concatenate([np.asarray(e['support'])[:,:k].ravel() for e in episodes]));qu=np.unique(np.concatenate([np.asarray(e['query']).ravel() for e in episodes]));assert not np.intersect1d(su,qu).size
        ss=data['sequence'][su];qs=data['sequence'][qu];stats=data['stats'][qu]
        if operator in ['Gaussian','Joint']:stats=corrupt_statistics(stats,qu,ds['name'],rho,cfg['noise_seed'])
        qs=corrupt_sequence(qs,qu,ds['name'],'INP' if operator=='Joint' else 'Clean' if operator=='Gaussian' else operator,rho,cfg['noise_seed'],cfg.get('preserve_prefix',0))
        zsupport=encode(su,data['stats'][su],ss);zquery=encode(qu,stats,qs)
        for ep in episodes:
            si=np.asarray(ep['support'])[:,:k].ravel();qi=np.asarray(ep['query']).ravel();proto=zsupport[np.searchsorted(su,si)].reshape(2,k,-1).mean(1);distance=torch.cdist(zquery[np.searchsorted(qu,qi)],proto)
            if cfg['distance']=='squared_euclidean':distance=distance.square()
            elif cfg['distance']!='euclidean':raise ValueError('Unsupported distance')
            score=(-distance).softmax(1)[:,1].numpy();pred=distance.argmin(1).numpy();y=np.repeat([0,1],15)
            identity=dict(dataset=ds['name'],model=cfg.get('variant','AdaFuse-Net'),training_seed=cfg['training_seed'],round=rid,shot=k,task=task,episode=ep['episode'],condition=operator if operator=='Clean' else 'IBP30' if operator=='IBP' and rho==.3 else operator,rho=rho,shared_episode_group=f"{ds['name']}/round{rid}/{task}/shot{k}/episode{ep['episode']}",input_sha256=hashlib.sha256(si.tobytes()+qi.tobytes()+data['stats'][si].tobytes()+ss[np.searchsorted(su,si)].tobytes()+stats[np.searchsorted(qu,qi)].tobytes()+qs[np.searchsorted(qu,qi)].tobytes()).hexdigest())
            rows.append(dict(**identity,**binary_metrics(y,score,prediction=pred)))
            raw.append(dict(**identity,support_ids=si.tolist(),query_ids=qi.tolist(),y_true=y.tolist(),y_pred=pred.tolist(),attack_score=score.tolist()))
    assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers())
    frame=pd.DataFrame(rows)
    if output:
        out=Path(output);out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'episode_metrics.csv',index=False)
        with gzip.open(out/'predictions.jsonl.gz','wt') as f:
            for r in raw:f.write(json.dumps(r)+'\n')
        fields=['MacroF1','AttackRecall','FPR','Precision','AUROC','AUPRC'];task=frame.groupby('task')[fields].mean();task.to_csv(out/'task_metrics.csv');save(out/'summary.json',task.mean().to_dict());save(out/'input_audit.json',dict(checkpoint_sha256=sha(checkpoint),plan_sha256=sha(rd/'eval.json'),support_clean=True,support_query_disjoint=True,BN_buffers_unchanged=True))
    return frame

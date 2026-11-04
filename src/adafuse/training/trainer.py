"""Fixed-budget episodic training; validation-only HPO is a separate entry."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from ..models.network import FactorialNet
from ..baselines.registry import REGISTRY,Baseline,local_supcon
from ..models.fusion_variants import FusionNet
from ..models.denoising import DenoisingNet
from ..losses.objectives import factorial_loss
from ..utils.common import reset,stable32,save,sha

def build(cfg,stat_dim):
    v=cfg.get('variant','AdaFuse-Net')
    if v in REGISTRY:return Baseline(v,stat_dim,cfg['embed_dim'])
    if v in ['Denoising-Reconstruction','Concatenation-Noisy-Rec0']:return DenoisingNet(stat_dim)
    if v!='AdaFuse-Net':return FusionNet(v,stat_dim,cfg['embed_dim'])
    return FactorialNet(cfg.get('use_adef',True),cfg.get('use_gating',True),cfg.get('use_dec',True),embed_dim=cfg['embed_dim'],stat_dim=stat_dim)

def train(dataset,cfg,prepared,round_index,shot):
    dataset=dict(dataset)
    if dataset.get('input_contract')=='NF57/IP512':
        contract=json.loads(Path(dataset['input']).with_suffix('.json').read_text())
        if contract.get('input_contract')!='NF57/IP512' or contract.get('feature_order')!=dataset.get('feature_order'):raise ValueError('Source input attestation mismatch')
        dataset['native_contract']={k:v for k,v in contract.items() if k in ['feature_order','frontend_sha256','nfstream_version']}
    reset(cfg['training_seed']);device=cfg['device'];rd=Path(prepared)/f'round{round_index}';dest=Path(cfg['output'])/'checkpoints'/f'round{round_index}'/f'shot{shot}';dest.mkdir(parents=True,exist_ok=True)
    checkpoint=dest/'model.pt'
    if checkpoint.exists():raise FileExistsError('Refusing to overwrite existing checkpoint; use a new output directory')
    data=dict(np.load(rd/'inputs.npz'));plans=json.loads((rd/'train.json').read_text());model=build(cfg,dataset['stat_dim']).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['updates'],eta_min=1e-6)
    def inp(ids):return torch.tensor(data['stats'][ids],device=device),torch.tensor(data['sequence'][ids].astype('int64'),device=device)
    logs=[]
    for step,ep in enumerate(plans[:cfg['updates']],1):
        if cfg.get('seed_per_update',False):reset(stable32(cfg['training_seed'],round_index,step,'matched_baseline_training' if isinstance(model,Baseline) else 'R1_1_training'))
        model.train();si=np.asarray(ep['support'])[:,:shot].ravel();qi=np.asarray(ep['query']).ravel();y=torch.arange(len(ep['classes']),device=device).repeat_interleave(15)
        opt.zero_grad(set_to_none=True);support=model(*inp(si),return_parts=True);qstat,cleanseq=inp(qi);qseq=cleanseq;mask=None
        if cfg.get('training_noise','Clean')=='alternating_INP_IBP':
            rng=np.random.default_rng(stable32(cfg['training_seed'],round_index,step,'denoising_training'));replacement=rng.integers(0,256,tuple(qseq.shape))
            if step%2:mask=rng.random(tuple(qseq.shape))<.3
            else:mask=np.zeros(tuple(qseq.shape),bool);mask[:,179:332]=True
            mask=torch.tensor(mask,device=device);qseq=torch.where(mask,torch.tensor(replacement,device=device),cleanseq)
        query=model(qstat,qseq,return_parts=True)
        if isinstance(model,Baseline):
            if model.name=='FC-Net':
                proto=support['z'].reshape(len(ep['classes']),shot,-1).mean(1);q=query['z'];n=len(ep['classes']);scores=model.encoder.compare(proto[None].expand(len(q),n,-1),q[:,None].expand(len(q),n,-1));loss=F.mse_loss(scores,F.one_hot(y,n).float())
            else:
                labels=torch.cat([torch.arange(len(ep['classes']),device=device).repeat_interleave(shot),y]);loss=local_supcon(torch.cat([support['z'],query['z']]),labels,cfg['temperature'])
            parts={'baseline_objective':loss}
        else:loss,parts=factorial_loss(model,support,query,y,shot,cfg)
        if hasattr(model,'reconstruction_head'):
            if mask is None:raise ValueError('Reconstruction control requires noisy training')
            rec=F.cross_entropy(model.reconstruction_head(query['h_seq_raw'])[mask],cleanseq[mask]);loss=loss+cfg['reconstruction_weight']*rec;parts['reconstruction']=rec
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if not all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite gradient')
        opt.step();sched.step();logs.append(dict(update=step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}))
    save(dest/'training_log.json',logs)
    scaler=dict(np.load(rd/'split_scaler.npz'))
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,dataset=dataset,round=round_index,shot=shot,mean=scaler['mean'],scale=scaler['scale'],training_plan_sha256=sha(rd/'train.json')),checkpoint)
    return checkpoint

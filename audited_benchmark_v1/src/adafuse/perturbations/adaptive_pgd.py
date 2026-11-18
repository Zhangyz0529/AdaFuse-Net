"""Exact discrete forward, embedding-interpolation BPDA/STE backward; fixed prototypes."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

class PrototypeClassifier:
    def __init__(self,model,prototypes,embedding_module,threshold=.5,is_b7=False):
        self.model=model;self.prototypes=prototypes.detach();self.embedding_module=embedding_module
        self.threshold=threshold;self.is_b7=is_b7
        assert 0<threshold<1
    def forward(self,stat,seq,parts=False):
        if self.is_b7:
            result=self.model(stat,seq,return_parts=True);z=result['z']
        else:
            result={};out=self.model(stat,seq);z=out[0] if isinstance(out,(tuple,list)) else out
        logits=-torch.cdist(z,self.prototypes)
        # Shift the attack logit to match a previously fixed score threshold.
        bias=torch.logit(torch.tensor(self.threshold,device=z.device,dtype=z.dtype))
        adjusted=logits-torch.stack([bias*0,bias])[None]
        return (adjusted,result) if parts else adjusted

@contextmanager
def discrete_byte_ste(embedding,byte_float):
    """Keep original integer lookup forward; replace its missing byte derivative.

    Surrogate: linear interpolation between embeddings of adjacent byte values.
    This is a BPDA choice, not the true derivative of round/long or a physical attack.
    """
    def hook(module,args,output):
        q=byte_float.clamp(0,255)
        low=q.floor().detach().long().clamp(max=254);high=low+1
        fraction=(q-low.to(q.dtype)).unsqueeze(-1)
        interp=F.embedding(low,module.weight)*(1-fraction)+F.embedding(high,module.weight)*fraction
        assert output.shape==interp.shape
        return output+(interp-interp.detach())
    handle=embedding.register_forward_hook(hook)
    try:yield
    finally:handle.remove()

def attack(classifier,stat,seq,y,*,mode='joint',gate_target=None,beta=.5,
           epsilon_stat=.1,epsilon_seq=10.,alpha_stat=.01,alpha_seq=1.,steps=20,restarts=5,seed=42):
    assert mode in ['stat','seq','joint'] and restarts>=1
    if gate_target is not None:assert classifier.is_b7 and gate_target in ['w_seq','w_stat']
    original_stat=stat.detach();original_seq=seq.detach().float();active=y.eq(1)
    m=active[:,None];stat_on=mode in ['stat','joint'];seq_on=mode in ['seq','joint']
    if not active.any():raise ValueError('No malicious Query samples')
    with torch.no_grad():clean_logits,parts=classifier.forward(stat,seq,parts=True)
    def objective(logits,parts):
        value=F.cross_entropy(logits,y,reduction='none')
        if gate_target is not None:value=value+beta*parts[gate_target].reshape(-1)
        return value
    best_stat=stat.detach().clone();best_seq=seq.detach().long().clone()
    best_success=clean_logits.argmax(1).eq(0)&active;best_value=objective(clean_logits,parts).detach()
    gradients=[]
    def consider(xs,xq):
        nonlocal best_stat,best_seq,best_success,best_value
        integer=xq.round().clamp(0,255).long()
        with torch.no_grad():logits,part=classifier.forward(xs,integer,parts=True);value=objective(logits,part);success=logits.argmax(1).eq(0)&active
        # Keep any successful candidate; among equal success states maximize attack objective.
        choose=active&((success&~best_success)|(success.eq(best_success)&value.gt(best_value)))
        best_stat=torch.where(choose[:,None],xs.detach(),best_stat);best_seq=torch.where(choose[:,None],integer,best_seq)
        best_success=torch.where(choose,success,best_success);best_value=torch.where(choose,value,best_value)
    for restart in range(restarts):
        g=torch.Generator(device='cpu').manual_seed(seed+restart)
        xs=original_stat.clone();xq=original_seq.clone()
        if restart:
            if stat_on:xs=xs+torch.where(m,(torch.rand(xs.shape,generator=g)*2-1).to(xs.device)*epsilon_stat,0.)
            if seq_on:xq=xq+torch.where(m,(torch.rand(xq.shape,generator=g)*2-1).to(xq.device)*epsilon_seq,0.)
        xq=xq.clamp(0,255)
        for step in range(steps):
            consider(xs,xq)
            xs=xs.detach().requires_grad_(stat_on);xq=xq.detach().requires_grad_(seq_on)
            if seq_on:
                with discrete_byte_ste(classifier.embedding_module,xq):logits,part=classifier.forward(xs,xq.round().clamp(0,255).long(),parts=True)
            else:logits,part=classifier.forward(xs,seq,parts=True)
            loss=objective(logits,part)[active].mean()
            variables=([xs] if stat_on else [])+([xq] if seq_on else [])
            grads=torch.autograd.grad(loss,variables,allow_unused=False)
            if not all(torch.isfinite(v).all() for v in grads):raise FloatingPointError('Nonfinite attack gradient')
            diagnostics=dict(restart=restart,step=step)
            index=0
            if stat_on:
                grad=grads[index];index+=1;diagnostics['stat_grad_abs_mean']=float(grad[active].abs().mean())
                xs=xs.detach()+alpha_stat*grad.sign()*m
                xs=torch.maximum(torch.minimum(xs,original_stat+epsilon_stat),original_stat-epsilon_stat)
            if seq_on:
                grad=grads[index];diagnostics['seq_grad_abs_mean']=float(grad[active].abs().mean())
                xq=xq.detach()+alpha_seq*grad.sign()*m
                xq=torch.maximum(torch.minimum(xq,original_seq+epsilon_seq),original_seq-epsilon_seq).clamp(0,255)
            gradients.append(diagnostics)
        consider(xs,xq)
    assert (best_stat-original_stat).abs().max()<=epsilon_stat+1e-5
    assert (best_seq.float()-original_seq).abs().max()<=epsilon_seq+1e-5
    assert torch.equal(best_stat[~active],original_stat[~active]) and torch.equal(best_seq[~active],seq.long()[~active])
    return best_stat,best_seq,dict(gradient_diagnostics=gradients,actual_stat_Linf=float((best_stat-original_stat).abs().max()),actual_seq_Linf=float((best_seq.float()-original_seq).abs().max()),changed_bytes=int(best_seq.ne(seq).sum()))

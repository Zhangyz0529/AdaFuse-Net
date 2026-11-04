"""Fixed-length input-space corruptions; not packet insertion or legal PCAP edits."""
import numpy as np
from ..utils.common import stable32

def corrupt_sequence(sequence, ids, dataset, operator, rho, noise_seed=20261028, preserve_prefix=0):
    x=np.asarray(sequence)
    if x.ndim!=2 or not np.issubdtype(x.dtype,np.integer) or np.any(x<0) or np.any(x>256):raise ValueError('Expected integer [N,L] tokens0..256')
    if operator not in ['Clean','INP','IBP','RTO','CTO']:raise ValueError(operator)
    if not 0<=rho<=1 or len(ids)!=len(x) or not 0<=preserve_prefix<=x.shape[1]:raise ValueError('Invalid corruption arguments')
    out=x.copy();length=x.shape[1];n=int(length*rho)
    if operator=='Clean' or rho==0:return out
    for i,sid in enumerate(ids):
        # IBP draws mask BEFORE replacement, exactly the matched-input evaluation contract.
        tag='matched_IBP30' if operator=='IBP' else 'R1_1_INP30' if operator=='INP' else operator
        rng=np.random.default_rng(stable32(noise_seed,dataset,int(sid),tag))
        if operator=='IBP':mask=rng.random(length)<rho;values=rng.integers(0,256,length)
        else:
            values=rng.integers(0,256,length);mask=np.zeros(length,dtype=bool)
            if operator=='INP':start=(length-n)//2;mask[start:start+n]=True
            elif n:mask[-n:]=True
        mask[:preserve_prefix]=False
        out[i,mask]=255 if operator=='CTO' else values[mask]
    return out

def corrupt_statistics(stats,ids,dataset,rho,noise_seed=20261028):
    if rho<0:raise ValueError('Negative Gaussian SD')
    out=np.asarray(stats,dtype=np.float32).copy()
    for i,sid in enumerate(ids):out[i]+=rho*np.random.default_rng(stable32(noise_seed,dataset,int(sid),'Gaussian')).normal(size=out.shape[1])
    return out

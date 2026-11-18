"""Nested BENIGN pools; fixed attack count and scores; FP count is explicit."""
import numpy as np
import pandas as pd
from .metrics import binary_metrics

def sweep(frame,threshold,attack_count=15,betas=(1,10,50,100),seed=42):
    rng=np.random.default_rng(seed);attack=frame[frame.label==1].iloc[:attack_count]
    benign=frame[frame.label==0].iloc[rng.permutation(sum(frame.label==0))]
    if len(attack)<attack_count or len(benign)<attack_count*max(betas):raise ValueError('Insufficient fixed pools')
    rows=[]
    for beta in betas:
        b=benign.iloc[:attack_count*beta];x=pd.concat([b,attack]);rows.append(dict(beta=beta,N_BENIGN=len(b),N_ATTACK=len(attack),**binary_metrics(x.label,x.score,threshold)))
    return pd.DataFrame(rows)

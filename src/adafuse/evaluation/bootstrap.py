"""Task-stratified paired percentile bootstrap conditional on fixed checkpoints."""
import numpy as np
import pandas as pd
KEY=['dataset','training_seed','shot','task','episode','condition']
def paired_bootstrap(ours,baseline,repeats=10000,seed=20261104):
    for f in [ours,baseline]:
        if f.duplicated(KEY).any():raise ValueError('Duplicate episode-condition keys')
        if 'shared_episode_group' not in f or 'input_sha256' not in f:raise ValueError('Need shared episode plan identity')
    a=ours.set_index(KEY);b=baseline.set_index(KEY)
    if set(a.index)!=set(b.index):raise ValueError('Missing pairs')
    b=b.reindex(a.index)
    if not a.shared_episode_group.equals(b.shared_episode_group) or not a.input_sha256.equals(b.input_sha256):raise ValueError('Unmatched episode plans')
    if len(a.index.get_level_values('training_seed').unique())!=1 or len(a.index.get_level_values('dataset').unique())!=1 or len(a.index.get_level_values('shot').unique())!=1:raise ValueError('Run separately per dataset/seed/shot')
    col='MacroF1' if 'MacroF1' in a else 'F1';delta=(a[col]-b[col]).rename('delta').reset_index();rng=np.random.default_rng(seed);tasks=[]
    for task,x in delta.groupby('task'):
        z=x.pivot(index='episode',columns='condition',values='delta')
        if z[['Clean','IBP30']].isna().any().any():raise ValueError('Missing Clean/corrupt episode')
        values=z[['Clean','IBP30']].to_numpy();tasks.append(values)
    estimates=np.mean([x.mean(0) for x in tasks],axis=0);replicates=np.zeros((repeats,2))
    for values in tasks:replicates+=values[rng.integers(0,len(values),(repeats,len(values)))].mean(1)/len(tasks)
    samples=np.c_[replicates,replicates[:,1]-replicates[:,0]];point=np.r_[estimates,estimates[1]-estimates[0]]
    return [dict(metric=m,estimate=float(v),CI_low=float(np.quantile(samples[:,i],.025)),CI_high=float(np.quantile(samples[:,i],.975)),bootstrap_repeats=repeats,bootstrap_seed=seed,unit='episode conditional on fixed checkpoints and candidate pools') for i,(m,v) in enumerate(zip(['Delta_Clean','Delta_IBP30','Delta_Drop'],point))]

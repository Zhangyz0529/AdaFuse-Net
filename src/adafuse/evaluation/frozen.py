"""Frozen-source NF57 target prototypes; same target scores, three thresholds."""
from pathlib import Path
import hashlib,pickle
import numpy as np
import pandas as pd
import torch
from ..training.trainer import build
from ..utils.common import stable32
from .metrics import calibrate,binary_metrics

def frozen_pilot(checkpoint,source_calibration,target,normal_label,output,shot=10,repeats=100,queries_per_class=1500):
    from ..deployment.nfstream_frontend import FEATURES
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];ds=saved['dataset']
    if ds.get('input_contract')!='NF57/IP512' or ds.get('feature_order')!=FEATURES:raise ValueError('Not a certified native source input contract')
    import json
    contract=json.loads(Path(target).with_suffix('.json').read_text())
    if contract.get('input_contract')!='NF57/IP512' or contract.get('feature_order')!=FEATURES:raise ValueError('Target native contract mismatch')
    source_contract=ds.get('native_contract',{})
    for key in ['frontend_sha256','nfstream_version']:
        if not source_contract.get(key) or source_contract[key]!=contract.get(key):raise ValueError('Unverified source/target frontend identity: '+key)
    with Path(target).open('rb') as f:records=pickle.load(f)
    grouped={}
    for i,r in enumerate(records):grouped.setdefault(hashlib.sha256(np.asarray(r['stat_raw']).tobytes()+np.asarray(r['seq_feature']).tobytes()).hexdigest(),[]).append(i)
    keep=[ids[0] for ids in grouped.values() if len({records[i]['label'] for i in ids})==1];records=[records[i] for i in sorted(keep)]
    if not all('group' in r for r in records):raise ValueError('Need canonical-group metadata')
    cal=pd.read_csv(source_calibration);source_threshold=calibrate(cal.label,cal.score)
    labels=np.array([r['label'] for r in records]);roles=np.array([int.from_bytes(hashlib.sha256(('target_roles|'+r['group']).encode()).digest()[:8],'big')%10 for r in records]);raw=np.stack([r['stat_raw'] for r in records]);seq=np.stack([r['seq_feature'] for r in records])
    model=build(cfg,57).to(cfg['device']).eval().requires_grad_(False);model.load_state_dict(saved['state_dict']);zs=[];before={k:v.clone() for k,v in model.named_buffers()}
    with torch.inference_mode():
        for i in range(0,len(raw),64):zs.append(model(torch.tensor(((raw[i:i+64]-saved['mean'])/saved['scale']).astype('float32'),device=cfg['device']),torch.tensor(seq[i:i+64].astype('int64'),device=cfg['device'])).cpu())
    z=torch.cat(zs);rows=[]
    for attack in sorted(set(labels)-{normal_label}):
        pools=[]
        for role in [0,1,2]:
            part=[]
            for label in [normal_label,attack]:
                mask=roles<2 if role==0 else (roles>=2)&(roles<6) if role==1 else roles>=6
                ids=np.flatnonzero((labels==label)&mask);need=shot if role==0 else queries_per_class
                if len(ids)<need:raise ValueError(f'Insufficient target pool: {label},role{role}')
                part.append(ids if role==0 else ids[:need])
            pools.append(part)
        ci=np.concatenate(pools[1]);ti=np.concatenate(pools[2]);y=np.repeat([0,1],queries_per_class);rng=np.random.default_rng(stable32(42,attack,'target_support'))
        for repeat in range(repeats):
            support=np.stack([rng.choice(ids,shot,replace=False) for ids in pools[0]]);proto=z[support.ravel()].reshape(2,shot,-1).mean(1)
            def score(ids):
                d=torch.cdist(z[ids],proto)
                if cfg['distance']=='squared_euclidean':d=d.square()
                return (-d).softmax(1)[:,1].numpy()
            calibration_score=score(ci);test_score=score(ti);target_threshold=calibrate(y,calibration_score)
            for name,t in [('Fixed0.5',.5),('Source_calibrated',source_threshold),('Target_task_calibrated',target_threshold)]:rows.append(dict(task=attack,support_repeat=repeat,scheme=name,Threshold=t,**binary_metrics(y,test_score,t)))
    assert all(torch.equal(v,before[k]) for k,v in model.named_buffers())
    out=Path(output);out.mkdir(parents=True,exist_ok=True);f=pd.DataFrame(rows);f.to_csv(out/'per_support.csv',index=False)
    fields=['MacroF1','AttackRecall','FPR','Precision','AUROC','AUPRC'];tasks=f.groupby(['task','scheme'])[fields].mean();tasks.to_csv(out/'per_task.csv');tasks.groupby('scheme').mean().to_csv(out/'equal_task_mean.csv')

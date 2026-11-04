"""Persist model-independent episode IDs BEFORE model training or scoring."""
from pathlib import Path
import json
import numpy as np
from ..data.legacy_manager import LOGODataManager
from ..utils.common import stable,stable32,save,sha

def prepare(dataset,run):
    root=Path(run['output'])/'prepared';root.mkdir(parents=True,exist_ok=True)
    cfg=dict(dataset,seed=dataset.get('split_seed',42))
    dm=LOGODataManager(cfg['input'],cfg,seed=cfg['seed'],split_protocol='class_rotate')
    if dm.raw_stats.shape[1]!=cfg['stat_dim'] or dm.seqs.shape[1]!=512:raise ValueError('Input contract mismatch')
    seq=np.asarray(dm.seqs)
    if np.max(seq)<=1:seq=np.round(seq*255.)
    if not np.isfinite(seq).all() or seq.min()<0 or seq.max()>256:raise ValueError('Invalid byte values')
    seq=seq.astype('int16');manifest=[]
    for rid,(classes,split) in enumerate(zip(dm.class_rounds,dm.split_indices_by_round)):
        dest=root/f'round{rid}';dest.mkdir(exist_ok=True)
        for a,ids in split.items():
            for b,j in split.items():
                if a!=b:assert not np.intersect1d(ids,j).size
        train=split['meta_train'];mean=dm.raw_stats[train].mean(0);scale=dm.raw_stats[train].std(0)+1e-8
        np.savez_compressed(dest/'inputs.npz',stats=((dm.raw_stats-mean)/scale).astype('float32'),sequence=seq,source_indices=np.arange(len(seq)))
        np.savez_compressed(dest/'split_scaler.npz',**split,mean=mean,scale=scale)
        pools={(role,c):ids[dm.labels[ids]==c] for role,ids in split.items() for c in classes[role]}
        eligible=[c for c in classes['meta_train'] if len(pools['meta_train',c])>=25]
        if len(eligible)<2:raise ValueError('Need >=2 classes with25 samples for source training')
        rng=np.random.default_rng(stable(run['training_seed'],rid,'train'));trainplans=[]
        for step in range(run['updates']):
            selected=rng.choice(eligible,min(5,len(eligible)),replace=False).tolist();draws=[rng.choice(pools['meta_train',c],25,replace=False) for c in selected]
            trainplans.append(dict(classes=selected,support=np.stack([z[:10] for z in draws]).tolist(),query=np.stack([z[10:] for z in draws]).tolist()))
        save(dest/'train.json',trainplans);evaluation={}
        for role,part in [('validation','meta_val'),('test','meta_test')]:
            evaluation[role]={}
            for task in classes['val_attack_classes' if role=='validation' else 'test_attack_classes']:
                if role=='validation' or run.get('evaluation_plan','matched')=='source':
                    rng=np.random.default_rng(stable(run['training_seed'],rid,role,task));roles=[]
                    for c in [cfg['normal_label'],task]:
                        ids=rng.permutation(pools[part,c]);cut=len(ids)//2
                        if cut<10 or len(ids)-cut<15:raise ValueError('Insufficient independent role pools')
                        roles.append((ids[:cut],ids[cut:]))
                    evaluation[role][task]=[dict(episode=e,support=np.stack([rng.choice(s,10,replace=False) for s,q in roles]).tolist(),query=np.stack([rng.choice(q,15,replace=False) for s,q in roles]).tolist()) for e in range(run['episodes'])]
                    continue
                # Reconstruct outcome-independent candidate unions from reference seed3407.
                ref=np.random.default_rng(stable(3407,rid,role,task));roles=[]
                for c in [cfg['normal_label'],task]:
                    ids=ref.permutation(pools[part,c]);cut=len(ids)//2
                    if cut<10 or len(ids)-cut<15:raise ValueError('Too few samples for disjoint support/query pools')
                    roles.append((ids[:cut],ids[cut:]))
                reference=[]
                for _ in range(100):reference.append(([ref.choice(s,10,replace=False) for s,q in roles],[ref.choice(q,15,replace=False) for s,q in roles]))
                sp=[np.unique(np.concatenate([e[0][c] for e in reference])) for c in range(2)]
                qp=[np.unique(np.concatenate([e[1][c] for e in reference])) for c in range(2)]
                tag='shared_evaluation' if role=='test' else 'shared_validation'
                rng=np.random.default_rng(stable32(run['episode_seed'],cfg['name'],rid,task,tag));eps=[]
                for e in range(run['episodes']):
                    si=np.stack([rng.choice(v,10,replace=False) for v in sp]);qi=np.stack([rng.choice(v,15,replace=False) for v in qp]);assert not np.intersect1d(si,qi).size
                    eps.append(dict(episode=e,support=si.tolist(),query=qi.tolist()))
                evaluation[role][task]=eps
        save(dest/'eval.json',evaluation);manifest.append(dict(round=rid,classes=classes,input_sha256=sha(dest/'inputs.npz'),train_sha256=sha(dest/'train.json'),eval_sha256=sha(dest/'eval.json')))
    save(root/'manifest.json',dict(dataset=cfg['name'],input_sha256=sha(cfg['input']),run=run,rounds=manifest,split_scope='sample ID, not certified session independence',candidate_scope='training-seed-specific disjoint role pools' if run.get('evaluation_plan','matched')=='source' else 'reference seed3407 pre-existing support/query draw unions; not full test partition'))
    return root

def prepare_binary(dataset,run):
    """Separate native-flow binary deployment training, group-isolated5 folds.

    This is explicitly a new portable source-training protocol, not a numerical
    reproduction claim for the historical single-capture pilot.
    """
    import pickle
    from sklearn.model_selection import StratifiedGroupKFold
    with Path(dataset['input']).open('rb') as f:records=pickle.load(f)
    if not all('group' in r for r in records):raise ValueError('Binary deployment requires retained canonical flow groups')
    y=np.array([0 if r['label']==dataset['normal_label'] else 1 for r in records]);groups=np.array([r['group'] for r in records]);raw=np.stack([r['stat_raw'] for r in records]).astype('float32');seq=np.stack([r['seq_feature'] for r in records])
    if raw.shape[1]!=dataset['stat_dim'] or not np.isfinite(raw).all():raise ValueError('Invalid binary source features')
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=dataset.get('split_seed',42)).split(raw,y,groups))
    split={'meta_test':folds[0][1],'meta_val':folds[1][1],'meta_train':np.concatenate([v[1] for v in folds[2:]])}
    for a,ids in split.items():
        for b,j in split.items():
            if a!=b:assert not set(groups[ids])&set(groups[j])
    train=split['meta_train'];mean=raw[train].mean(0);scale=raw[train].std(0)+1e-8;root=Path(run['output'])/'prepared';rd=root/'round0';rd.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(rd/'inputs.npz',stats=((raw-mean)/scale).astype('float32'),sequence=seq.astype('int16'),source_indices=np.arange(len(raw)));np.savez_compressed(rd/'split_scaler.npz',**split,mean=mean,scale=scale)
    rng=np.random.default_rng(stable(run['training_seed'],'binary_training'));pools=[train[y[train]==c] for c in [0,1]]
    if min(map(len,pools))<35:raise ValueError('Need at least35 source training samples per class')
    reserved=[rng.choice(v,10,replace=False) for v in pools];pools=[np.setdiff1d(v,s) for v,s in zip(pools,reserved)];plans=[]
    for _ in range(run['updates']):
        draw=[rng.choice(v,25,replace=False) for v in pools];plans.append(dict(classes=[dataset['normal_label'],'ATTACK'],support=np.stack([v[:10] for v in draw]).tolist(),query=np.stack([v[10:] for v in draw]).tolist()))
    save(rd/'train.json',plans);evaluation={}
    for role,key in [('validation','meta_val'),('test','meta_test')]:
        ids=split[key];qs=[ids[y[ids]==c] for c in [0,1]]
        if min(map(len,qs))<15:raise ValueError('Need15 per class in held-out binary pools')
        rng=np.random.default_rng(stable(run['episode_seed'],role));evaluation[role]={'ATTACK':[dict(episode=e,support=np.stack(reserved).tolist(),query=np.stack([rng.choice(v,15,replace=False) for v in qs]).tolist()) for e in range(run['episodes'])]}
    save(rd/'eval.json',evaluation);save(root/'manifest.json',dict(dataset=dataset['name'],input_sha256=sha(dataset['input']),run=run,rounds=[dict(round=0,input_sha256=sha(rd/'inputs.npz'))],split_scope='canonical flow-group disjoint; one capture may still be shared',source_support='reserved clean training-role support, excluded from gradient updates'))
    return root

def prepare_internal(dataset,run):
    """CICIoT within-domain all-class training on precomputed packet-block splits."""
    import pickle
    with Path(dataset['input']).open('rb') as f:records=pickle.load(f)
    raw=np.stack([r['stat_raw'] for r in records]).astype('float32');seq=np.stack([r['seq_feature'] for r in records]);labels=np.array([r['label'] for r in records]);groups=np.array([r['group_id'] for r in records]);split=dict(np.load(dataset['split_indices']))
    for a,ids in split.items():
        for b,j in split.items():
            if a!=b:assert not set(groups[ids])&set(groups[j])
    mean=raw[split['train']].mean(0,dtype=np.float64);scale=np.maximum(raw[split['train']].std(0,dtype=np.float64),1e-8);root=Path(run['output'])/'prepared';rd=root/'round0';rd.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(rd/'inputs.npz',stats=((raw-mean)/scale).astype('float32'),sequence=seq.astype('int16'),source_indices=np.arange(len(raw)));np.savez_compressed(rd/'split_scaler.npz',**split,mean=mean,scale=scale)
    classes=[dataset['normal_label']]+dataset['attack_classes'];rng=np.random.default_rng(stable(run['training_seed'],'internal_training'));train=[]
    pools={c:split['train'][labels[split['train']]==c] for c in classes}
    if any(len(v)<25 for v in pools.values()):raise ValueError('Need25 samples per source class')
    for _ in range(run['updates']):
        draw=[rng.choice(pools[c],25,replace=False) for c in classes];train.append(dict(classes=classes,support=np.stack([v[:10] for v in draw]).tolist(),query=np.stack([v[10:] for v in draw]).tolist()))
    save(rd/'train.json',train);evaluation={}
    for role in ['validation','test']:
        evaluation[role]={}
        for task in dataset['attack_classes']:
            rng=np.random.default_rng(stable32(run['episode_seed'],role,task));pools=[]
            for c in [dataset['normal_label'],task]:
                ids=rng.permutation(split[role][labels[split[role]]==c]);cut=len(ids)//2
                if cut<10 or len(ids)-cut<15:raise ValueError('Insufficient internal held-out pools')
                pools.append((ids[:cut],ids[cut:]))
            evaluation[role][task]=[dict(episode=e,support=np.stack([rng.choice(s,10,replace=False) for s,q in pools]).tolist(),query=np.stack([rng.choice(q,15,replace=False) for s,q in pools]).tolist()) for e in range(run['episodes'])]
    save(rd/'eval.json',evaluation);save(root/'manifest.json',dict(dataset=dataset['name'],run=run,rounds=[dict(round=0,input_sha256=sha(rd/'inputs.npz'))],split_scope='precomputed raw packet blocks, not sessions; allclasses in every role; NOT LOCO'))
    return root

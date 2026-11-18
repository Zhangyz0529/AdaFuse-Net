"""Support-fitted RF on statistics+sequence; no pretrained source encoder."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ..perturbations.operators import corrupt_sequence
from .metrics import binary_metrics

def evaluate_forest(prepared,dataset,output,seed=42,shots=(1,3,5,10),trees=100):
    root=Path(prepared);rows=[];records=[]
    for rd in sorted(root.glob('round*')):
        data=dict(np.load(rd/'inputs.npz'));plans=json.loads((rd/'eval.json').read_text())['test'];rid=int(rd.name[5:])
        for task,episodes in plans.items():
            for shot in shots:
                for ep in episodes:
                    si=np.asarray(ep['support'])[:,:shot].ravel();qi=np.asarray(ep['query']).ravel();assert not np.intersect1d(si,qi).size
                    support=np.concatenate([data['stats'][si],data['sequence'][si]],axis=1);model=RandomForestClassifier(n_estimators=trees,random_state=seed,n_jobs=1);model.fit(support,np.repeat([0,1],shot));y=np.repeat([0,1],15)
                    for op,rho,condition in [('Clean',0.,'Clean'),('IBP',.3,'IBP30')]:
                        seq=corrupt_sequence(data['sequence'][qi],qi,dataset['name'],op,rho);x=np.concatenate([data['stats'][qi],seq],axis=1);score=model.predict_proba(x)[:,list(model.classes_).index(1)];pred=model.predict(x)
                        identity=dict(dataset=dataset['name'],model='RandomForest',training_seed=seed,round=rid,shot=shot,task=task,episode=ep['episode'],condition=condition,rho=rho,shared_episode_group=f"{dataset['name']}/round{rid}/{task}/shot{shot}/episode{ep['episode']}",input_sha256=hashlib.sha256(si.tobytes()+qi.tobytes()+data['stats'][si].tobytes()+data['sequence'][si].tobytes()+data['stats'][qi].tobytes()+seq.tobytes()).hexdigest())
                        rows.append(dict(**identity,**binary_metrics(y,score,prediction=pred)));records.append(dict(**identity,support_ids=si.tolist(),query_ids=qi.tolist(),y_true=y.tolist(),y_pred=pred.tolist(),attack_score=score.tolist()))
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(dest/'episode_metrics.csv',index=False)
    with gzip.open(dest/'predictions.jsonl.gz','wt') as f:
        for r in records:f.write(json.dumps(r)+'\n')

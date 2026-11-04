"""Flatten fixed-prototype prediction arrays; reject changing scores for one ID."""
import argparse,gzip,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--condition',default='Clean');a=p.parse_args();rows={}
with gzip.open(a.predictions,'rt') as f:
    for line in f:
        r=json.loads(line)
        if r['condition']!=a.condition:continue
        for sid,y,s in zip(r['query_ids'],r['y_true'],r['attack_score']):
            v=(int(y),float(s))
            if sid in rows and (rows[sid][0]!=v[0] or abs(rows[sid][1]-v[1])>1e-7):raise ValueError('Changing prototypes/scores: cannot form a fixed-prototype calibration pool')
            rows[sid]=v
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame([dict(query_id=k,label=v[0],score=v[1]) for k,v in sorted(rows.items())]).to_csv(out,index=False)

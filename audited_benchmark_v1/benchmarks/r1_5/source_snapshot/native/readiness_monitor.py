"""Current evidence dashboard: distinguish short checks, full gates and formal jobs."""
import json,time,collections
from pathlib import Path
from cache import HERE,OLD,save
while True:
 q=json.loads((OLD/'QUEUE_STATUS.json').read_text());counts=collections.Counter()
 for p in (HERE/'runs').glob('*/*/seed*/round*/shot*/COMPLETE.json'):
  bits=p.relative_to(HERE/'runs').parts;counts[(bits[0],bits[1])]+=1
 checks={}
 for model in ['MFF','GP-VAE','FlowPic','MetaTraffic','Trafficformer-Light','Pre-Contrast-Local']:
  base=HERE/('smoke_v4' if model=='MetaTraffic' else 'smoke')/'USTC'/model/'seed42/round0/shot1';marker=base/'COMPLETE.json';trajectory=base/'training_trajectory.jsonl'
  checks[model]=dict(gate='PASS' if marker.exists() else 'PENDING_OR_FAILED',required_updates=100 if model=='MetaTraffic' else 4 if model in ['FlowPic','Trafficformer-Light'] else 2,observed_updates=sum(1 for _ in trajectory.open()) if trajectory.exists() else 0,formal_complete={ds:n for (ds,m),n in counts.items() if m==model})
 checks['MetaTraffic']['prior_v3_full_run_failed']=True;checks['MetaTraffic']['prior_failures_retained']='META_V3_RETIREMENT.json'
 save(HERE/'CURRENT_STATUS.json',dict(time=time.time(),models=checks,active_jobs=q['active'],ready_datasets=q['ready'],waiting=q['waiting'],scope='marker/log evidence, queue refreshed by supervisor; retired versions never aggregated'))
 time.sleep(30)

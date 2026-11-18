"""Recompute unified source/label counts; incomplete sources never get fake zeros."""
import argparse,collections,csv,json,time
from pathlib import Path
import run as r

def report():
 out=r.HERE/'reports';out.mkdir(exist_ok=True);policies=[];counts=[];status=[]
 for dataset,domains in [('CIC_IDS',['Monday','Tuesday','Wednesday','Thursday','Friday']),('CIC_DDOS',['Test_Day_03-11','Train_Day_01-12'])]:
  for domain in domains:
   d=r.HERE/'outputs'/dataset/domain;p=d/'SOURCE_POLICY.json';completed=d/'REBUILD_COMPLETE.json'
   state=json.loads((d/'STATUS.json').read_text()) if (d/'STATUS.json').exists() else {'stage':'not_started'}
   status.append(dict(dataset=dataset,domain=domain,state=state))
   policy=json.loads(p.read_text()) if p.exists() else {}
   policies.append(dict(dataset=dataset,domain=domain,status=policy.get('status','PENDING'),byte_metric=policy.get('metric'),byte_consensus=policy.get('byte_consensus'),clock_offsets=json.dumps(policy.get('offset_options_seconds',[])),clock_consensus=policy.get('clock_consensus'),clock_residual_seconds=json.dumps(policy.get('residual_seconds',{})),source_shards=len(policy.get('source_files',[])),packet_min=1,packet_max='none',source_policy_file=str(p)))
   if completed.exists():
    cache=out/(dataset+'_'+domain+'_final_counts.json')
    if cache.exists():
     rows=json.loads(cache.read_text());counts.extend(rows);continue
    c=collections.Counter()
    with (d/'counts_by_label_reason.csv').open() as f:
     for x in csv.DictReader(f):c[x['label'],x['decision'],x['stage'],x['reason']]+=int(x['count'])
    with (d/'parse_rejections.jsonl').open() as f:
     for line in f:
      x=json.loads(line);c[x['label'],'reject',x['stage'],x['reason']]+=1
    rows=[dict(dataset=dataset,domain=domain,label=key[0],decision=key[1],stage=key[2],reason=key[3],count=n) for key,n in sorted(c.items())];counts.extend(rows)
    info=json.loads(completed.read_text());assert sum(c.values())==info['csv_total'];r.write(cache,rows)
 def writecsv(name,rows,fields):
  p=out/name;tmp=p.with_suffix('.tmp')
  with tmp.open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
  tmp.replace(p)
 writecsv('source_domain_policies.csv',policies,list(policies[0]));writecsv('all_sources_labels_decisions.csv',counts,['dataset','domain','label','decision','stage','reason','count'])
 r.write(out/'LIVE_STATUS.json',dict(time=time.time(),sources=status,completed_source_domains=len({(x['dataset'],x['domain']) for x in counts}),training_started=False,optuna_started=False,warning='Missing count rows mean source not complete, never zero accepted/rejected'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  report()
  if not a.watch:break
  s=r.HERE/'PARTITION_WATCH_STATUS.json'
  if s.exists() and not json.loads(s.read_text())['remaining']:break
  time.sleep(30)

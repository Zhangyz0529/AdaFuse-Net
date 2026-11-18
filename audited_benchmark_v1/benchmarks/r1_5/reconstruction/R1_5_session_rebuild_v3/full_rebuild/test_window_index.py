"""Compare optimized search with frozen v3 on real fixed cases and random data."""
import json,sys
import numpy as np
from window_index import WindowIndex
import run as r
from rules_v3 import windows,stable,normalize_row
rng=np.random.default_rng(10571);checks=0
for trial in range(300):
 n=int(rng.integers(1,200));ps=[];ts=np.cumsum(rng.uniform(0,2,n))+10000
 for j in range(n):ps.append(dict(ts=float(ts[j]),src=['a',1] if rng.random()<.5 else ['b',2],payload_len=int(rng.integers(0,25)),capture_payload_len=int(rng.integers(25,50))))
 index=WindowIndex(ps);a=int(rng.integers(n));b=int(rng.integers(a+1,n+1));origin=tuple(ps[a]['src']);unit=ps[a:b];field=['payload_len','capture_payload_len'][trial%2]
 row={'n_packets':b-a,'Source IP':origin[0],'Source Port':origin[1],'Total Fwd Packets':sum(tuple(p['src'])==origin for p in unit),'Total Backward Packets':sum(tuple(p['src'])!=origin for p in unit),'Total Length of Fwd Packets':sum(p[field] for p in unit if tuple(p['src'])==origin),'Total Length of Bwd Packets':sum(p[field] for p in unit if tuple(p['src'])!=origin),'Flow Duration':(unit[-1]['ts']-unit[0]['ts'])*1e6,'clock':unit[0]['ts']-60,'clock_precision_s':60.}
 for offsets in [[60.],[0.,60.],[600.]]:
  assert index.windows(row,offsets,field)==windows(row,ps,offsets,field);checks+=1
 full=r.metadata_windows(row,ps);fast=index.metadata(row);assert (len(full)==0)==(len(fast)==0) and (len(full)==1)==(len(fast)==1)
 if len(full)==1:assert full==fast
plan=json.loads((r.V3/'diagnostics/FIXED_CASES.json').read_text())['cases']
class Log:
 def write(self,*args):raise AssertionError(args)
for domain in ['Friday','Tuesday','Wednesday']:
 d=r.V3/'diagnostics'/domain;groups={stable(g['key']):g['packets'] for g in json.loads((d/'RAW_PACKETS_BY_KEY.json').read_text())};offsets=json.loads((d/'CLOCK_QUALITY.json').read_text())['v2_clock_status']['offset_options_seconds']
 for case in plan:
  if case['domain']!=domain:continue
  raw=case['csv_record'];row=normalize_row(raw,'case',raw['csv_row'],raw['label'],Log());ps=groups[stable(r.H['csv_key'](raw))];ps=sorted(ps,key=lambda p:(p['ts'],p['file_id'],p['index']));index=WindowIndex(ps)
  for field in ['payload_len','capture_payload_len']:assert index.windows(row,offsets,field)==windows(row,ps,offsets,field);checks+=1
print(json.dumps({'passed':True,'exact_window_comparisons':checks,'scope':'300 random packet groups + all66 fixed real cases, both byte metrics'}))

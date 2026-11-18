"""Complete-source, disk-backed v3 rebuild. No model training or Optuna imports.

All CSV rows enter ambiguity checks; requested labels alone enter model samples.
Stages commit only on complete scans. No per-class sampling or hot-key truncation.
Domain calibration is label-blind and precedes all row acceptance decisions.
"""
import argparse, collections, csv, hashlib, json, mmap, os, sqlite3, struct, sys, time, zlib
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
V3=HERE.parent
sys.path.insert(0,str(V3))
sys.path.insert(0,str(V3.parent/'R1_5_session_rebuild_v2'))
from rebuild import H, jobs, CLASSES, MAP, META, anonymous_frame
from rules_v3 import RULES, normalize_row, stable, assign_sessions, windows
from packet_io_v3 import decode_accounting
from window_index import WindowIndex

POLICY={
 'version':'v3_full_source_2', 'all_csv_rows':True,'per_class_sample_cap':None,
 'anchor_selection':'lowest SHA256(canonical key), first 4096 keys; all labels, no scores; unique count/direction/duration window before bytes/time',
 'anchor_keys':4096,'minimum_clock_anchors':10,'clock_consensus':.95,
 'minimum_informative_byte_anchors':8,'byte_consensus':.95,
 'clock_rule':'integer-minute offset from metadata-unique window, residual within CSV precision (+/-2us numeric tolerance); CIC_IDS normalize 12h ambiguity, retain both offsets; no per-record fitting',
 'byte_rule':'single domain metric from independent physical byte accounting; contradictory anchors count against consensus',
 'packet_min':1,'packet_max':None,'overflow_policy':'disk spill, or fail stage/domain; never accept incomplete candidate searches',
 'prefix_policy':'use verified complete-record prefix only; reject matching CSV uncertainty interval touching unavailable range, including overlaps supplied by another shard',
 'sample_input':'official 77 unscaled float32 statistics; first512 bytes of anonymized frames, zero padded; no fitted preprocessing',
 'duplicate_policy':'exclude ALL records in cross-label identical-input components; same-label hashes + sessions + uncertain continuation edges union before partition',
 'split_gate':'no automatic training; require class/group feasibility and explicit episode/split protocol before manifest; no fabricated independence',
}
def write(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(path)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pack(x):return zlib.compress(json.dumps(x,separators=(',',':'),allow_nan=False).encode(),1)
def unpack(x):return json.loads(zlib.decompress(x))
def kh(key):return stable(key)
def emit(f,**r):f.write(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n')
def progress(out,stage,**kw):
 write(out/'STATUS.json',dict(stage=stage,pid=os.getpid(),time=time.time(),training_started=False,**kw));print(out.name,stage,kw,flush=True)
class ParseLog:
 def __init__(self,f):self.f=f
 def write(self,source_file_id,csv_row_id,label,stage,reason,detail,decision='reject'):
  emit(self.f,source_file_id=source_file_id,csv_row_id=csv_row_id,label=label,stage=stage,reason=reason,detail=detail,decision=decision)
def connect(out):
 db=sqlite3.connect(out/'index.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL');db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-131072')
 return db

def ingest_csv(job,dataset,root,out,db):
 if (out/'CSV_COMPLETE.json').exists():return
 # A failed stage is not resumable mid-row; its database belongs to this domain only.
 for name in ['rows','keys','packets','badkeys']:db.execute('DROP TABLE IF EXISTS '+name)
 db.execute('CREATE TABLE rows (id INTEGER PRIMARY KEY, k TEXT, valid INTEGER, label TEXT, data BLOB)')
 db.execute('CREATE TABLE keys (k TEXT PRIMARY KEY, endpoints TEXT, rank TEXT)')
 names=None;total=valid=0;files=[]
 with (out/'parse_rejections.jsonl').open('w') as log:
  logger=ParseLog(log)
  for f in job['csvs']:
   rowidx=2;sourceid=stable(['csv_file_v3',dataset,str(f.relative_to(root))])
   for frame in pd.read_csv(f,encoding='cp1252',chunksize=20000,low_memory=False):
    frame.columns=frame.columns.str.strip()
    cols=[c for c in frame.columns if c not in META and not(dataset=='CIC_DDOS' and c in ['Unnamed: 0','SimillarHTTP','Inbound'])]
    if names is None:names=cols
    if cols!=names or len(cols)!=77:raise ValueError('Unexpected feature schema: '+str(f)+' '+str(len(cols)))
    pending=[];keyitems=[]
    for values in frame.itertuples(index=False,name=None):
     raw=dict(zip(frame.columns,values));label=MAP.get(str(raw['Label']).strip(),str(raw['Label']).strip());total+=1
     raw.update(csv_path=str(f),csv_row=rowidx,label=label,source_file_id=sourceid)
     try:key=H['csv_key'](raw);keyhash=kh(key)
     except (ValueError,TypeError,OSError,OverflowError):
      logger.write(sourceid,rowidx,label,'parse','invalid_endpoint_key',{});rowidx+=1;continue
     rr=normalize_row(raw,sourceid,rowidx,label,logger,[raw[c] for c in cols]);rowidx+=1
     # Invalid keyed candidates must still veto ambiguous acceptance.
     if rr is not None:
      rr={k:rr[k] for k in ['Source IP','Destination IP','Source Port','Destination Port','Protocol','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Flow Duration','clock','clock_precision_s','n_packets','stat','nonfinite_features_replaced','csv_path','csv_row','label','source_file_id']};valid+=1
     else:rr=dict(csv_path=str(f),csv_row=rowidx-1,label=label,source_file_id=sourceid)
     pending.append((keyhash,int('stat' in rr),label,pack(rr)));keyitems.append((keyhash,json.dumps(key),keyhash))
    db.executemany('INSERT INTO rows(k,valid,label,data) VALUES(?,?,?,?)',pending);db.executemany('INSERT OR IGNORE INTO keys VALUES(?,?,?)',keyitems);db.commit()
    progress(out,'csv_ingest',csv_rows=total,valid_rows=valid,file=f.name)
   files.append(dict(path=str(f),rows=rowidx-2,sha256=sha(f)))
 db.execute('CREATE INDEX rows_key ON rows(k)');db.commit()
 write(out/'CSV_COMPLETE.json',dict(total_rows=total,valid_rows=valid,feature_names=names,files=files,per_class_cap=None))

def scan_packets(job,dataset,root,out,db,registry,prefixes):
 if (out/'PACKETS_COMPLETE.json').exists():return
 db.execute('DROP TABLE IF EXISTS packets');db.execute('DROP TABLE IF EXISTS badkeys')
 db.execute('CREATE TABLE packets(k TEXT,ts REAL,file_id TEXT,ordinal INTEGER,data BLOB)')
 db.execute('CREATE TABLE badkeys(k TEXT PRIMARY KEY, reason TEXT)');db.commit()
 keymap={}
 for k,e in db.execute('SELECT k,endpoints FROM keys'):
  x=json.loads(e);keymap[(x[0],tuple(x[1]),tuple(x[2]))]=k
 registry={r['source_relative_path']:r for r in registry};prefixes={r['original_relative_path']:r for r in prefixes};sources=[];badpairs=set();global_errors=[]
 for f in job['pcaps']:
  rel=str(f.relative_to(root));src=registry[rel];sid=src['source_file_id'];prefix=prefixes.get(rel);readpath=V3/prefix['working_prefix'] if prefix else f
  if prefix and sha(readpath)!=prefix['working_prefix_sha256']:raise ValueError('Prefix hash changed')
  n=retained=0;counts=collections.Counter();batch=[];bad=[];first=last=None
  with readpath.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as mm:
   for ordinal,ts,offset,cap,wire,link in H['indexed_packets'](mm):
    n+=1;first=ts if first is None else min(first,ts);last=ts if last is None else max(last,ts)
    key=H['fast_endpoint'](mm,offset,cap,link)
    if key is None:
     if wire>cap:global_errors.append('unassignable_truncated_capture_record')
     # A short noninitial fragment may not expose transport-port bytes.
     net=offset+(14 if link==1 else 16 if link==113 else 0)
     if link==1 and cap>=14:
      eth=struct.unpack_from('!H',mm,offset+12)[0]
      while eth in (0x8100,0x88a8) and net+4<=offset+cap:eth=struct.unpack_from('!H',mm,net+2)[0];net+=4
     if net+20<=offset+cap and mm[net]>>4==4 and struct.unpack_from('!H',mm,net+6)[0]&0x3fff:
      ips=struct.unpack_from('!II',mm,net+12);badpairs.add(tuple(sorted(ips)));counts['fragment_requires_reassembly']+=1
     counts['out_of_ipv4_tcp_udp_scope']+=1
     if link not in [1,113,101,228]:global_errors.append('unsupported_link_type')
     continue
    # Detect fragments before looking up transport ports (noninitial ports are meaningless).
    net=offset+(14 if link==1 else 16 if link==113 else 0)
    if link==1:
     eth=struct.unpack_from('!H',mm,offset+12)[0]
     while eth in (0x8100,0x88a8):eth=struct.unpack_from('!H',mm,net+2)[0];net+=4
    if net+8<=offset+cap and struct.unpack_from('!H',mm,net+6)[0]&0x3fff:
     badpairs.add(tuple(sorted([key[1][0],key[2][0]])));counts['fragment_requires_reassembly']+=1;continue
    keyhash=keymap.get(key)
    if keyhash is None:continue
    try:
     raw=bytes(mm[offset:offset+cap]);p=decode_accounting(raw,link)
     p.update(file_id=sid,index=ordinal,ts=ts,offset=offset,caplen=cap,link=link)
     batch.append((keyhash,ts,sid,ordinal,pack(p)));retained+=1
    except ValueError as e:
     counts[str(e)]+=1;bad.append((keyhash,str(e)))
     if str(e)=='fragment_requires_reassembly':badpairs.add(tuple(sorted([key[1][0],key[2][0]])))
    if len(batch)>=10000:
     db.executemany('INSERT INTO packets VALUES(?,?,?,?,?)',batch);batch=[]
     db.executemany('INSERT OR IGNORE INTO badkeys VALUES(?,?)',bad);bad=[];db.commit()
    if n%2000000==0:progress(out,'pcap_scan',file=f.name,packet_records=n,retained=retained)
  db.executemany('INSERT INTO packets VALUES(?,?,?,?,?)',batch);db.executemany('INSERT OR IGNORE INTO badkeys VALUES(?,?)',bad);db.commit()
  h=sha(f)
  if src['content_sha256'] and h!=src['content_sha256']:raise ValueError('Source hash changed: '+str(f))
  sources.append(dict(path=str(f),read_path=str(readpath),file_id=sid,sha256=h,read_sha256=sha(readpath) if prefix else h,records=n,retained=retained,first_ts=first,last_ts=last,errors=dict(counts),prefix=prefix))
  write(out/'SCAN_PROGRESS.json',dict(sources=sources))
  progress(out,'pcap_file_complete',file=f.name,packet_records=n,retained=retained)
 # Fragment transport ports are unreliable. All candidate tuples sharing these IPs fail closed.
 if badpairs:
  affected=[]
  for k,e in db.execute('SELECT k,endpoints FROM keys'):
   x=json.loads(e)
   if tuple(sorted([x[1][0],x[2][0]])) in badpairs:affected.append((k,'ip_pair_contains_unreassembled_fragment'))
  db.executemany('INSERT OR IGNORE INTO badkeys VALUES(?,?)',affected)
 db.execute('CREATE INDEX packet_key_time ON packets(k,ts,file_id,ordinal)');db.commit()
 write(out/'PACKETS_COMPLETE.json',dict(sources=sources,global_errors=sorted(set(global_errors)),fragment_ip_pairs=len(badpairs),keys_with_packet_errors=db.execute('SELECT count(*) FROM badkeys').fetchone()[0],search_complete=not global_errors))

def perkey(db,k):
 rows=[(i,v,unpack(d)) for i,v,d in db.execute('SELECT id,valid,data FROM rows WHERE k=? ORDER BY id',(k,))]
 packets=[unpack(d) for d, in db.execute('SELECT data FROM packets WHERE k=? ORDER BY ts,file_id,ordinal',(k,))]
 return rows,packets

def metadata_windows(r,ps):
 n=r['n_packets']
 if len(ps)<n:return []
 direction=np.array([tuple(p['src'])==(r['Source IP'],r['Source Port']) for p in ps],dtype=np.int64);pref=np.r_[0,np.cumsum(direction)];a=np.arange(len(ps)-n+1);b=a+n;a=a[pref[b]-pref[a]==r['Total Fwd Packets']]
 return [(int(x),int(x+n)) for x in a if abs((ps[x+n-1]['ts']-ps[x]['ts'])*1e6-r['Flow Duration'])<=2.]

def calibrate(job,dataset,out,db):
 if (out/'SOURCE_POLICY.json').exists():return json.loads((out/'SOURCE_POLICY.json').read_text())
 anchors=[];bv=collections.Counter();ov=collections.Counter();contradictory=uninformative=0
 for (k,) in db.execute('SELECT k FROM keys ORDER BY rank LIMIT ?',(POLICY['anchor_keys'],)).fetchall():
  if db.execute('SELECT 1 FROM badkeys WHERE k=?',(k,)).fetchone():continue
  rows,ps=perkey(db,k);index=WindowIndex(ps)
  if any(not v for _,v,_ in rows):continue
  candidates=collections.defaultdict(list)
  for _,_,r in rows:
   wins=index.metadata(r)
   if len(wins)==1:candidates[wins[0]].append(r)
  for (a,b),rr in candidates.items():
   if len(rr)!=1:continue
   r=rr[0];unit=ps[a:b];origin=(r['Source IP'],r['Source Port']);metrics={};match=[]
   for field in ['payload_len','capture_payload_len']:
    fw=sum(p[field] for p in unit if tuple(p['src'])==origin);bw=sum(p[field] for p in unit if tuple(p['src'])!=origin)
    metrics[field]=[fw,bw]
    if [fw,bw]==[r['Total Length of Fwd Packets'],r['Total Length of Bwd Packets']]:match.append(field)
   if len(match)==1:bv[match[0]]+=1
   elif len(match)==2:uninformative+=1
   else:contradictory+=1
   precision=r['clock_precision_s'];delta=unit[0]['ts']-r['clock']
   offset=int(np.floor(delta/60)*60) if precision>=60 else int(round(delta/60)*60)
   if not -2e-6<=delta-offset<precision+2e-6:continue
   # Fixed 12h representation ambiguity, not record-specific clock fitting.
   phase=offset%43200 if dataset=='CIC_IDS' else offset;ov[phase]+=1
   anchors.append(dict(csv_path=r['csv_path'],csv_row=r['csv_row'],key=k,window=[a,b],observed_start=unit[0]['ts'],csv_clock=r['clock'],precision=precision,offset_bin=offset,phase=phase,metrics=metrics,matching_metrics=match,duration_error_us=(unit[-1]['ts']-unit[0]['ts'])*1e6-r['Flow Duration']))
 mode,clockn=ov.most_common(1)[0] if ov else (None,0);metric,byten=bv.most_common(1)[0] if bv else (None,0)
 clockok=clockn>=10 and clockn/max(1,sum(ov.values()))>=.95
 byteok=sum(bv.values())>=8 and byten/max(1,sum(bv.values())+contradictory)>=.95
 offsets=([mode,mode+43200] if dataset=='CIC_IDS' else [mode]) if clockok else []
 residual=[r['observed_start']-r['csv_clock']-r['offset_bin'] for r in anchors if r['phase']==mode]
 status='CERTIFIED_FOR_DECLARED_SCOPE' if clockok and byteok else 'BLOCKED_CALIBRATION_EVIDENCE'
 source=json.loads((out/'PACKETS_COMPLETE.json').read_text())
 if not source['search_complete']:status='BLOCKED_SOURCE_SEARCH'
 result=dict(dataset=dataset,domain=job['domain'],status=status,metric=metric if byteok else None,metric_votes=dict(bv),contradictory=contradictory,uninformative=uninformative,byte_consensus=byten/max(1,sum(bv.values())+contradictory),offset_options_seconds=offsets,clock_votes=dict(ov),clock_consensus=clockn/max(1,sum(ov.values())),clock_anchor_count=sum(ov.values()),residual_seconds=dict(min=min(residual) if residual else None,max=max(residual) if residual else None,median=float(np.median(residual)) if residual else None),clock_precision_warning='residuals belong to CSV representation bins, not subsecond accuracy proof',anchors=anchors,policy=POLICY,source_files=source['sources'],domain_scope='registered acquisition domain and all its shards; no per-label or per-record metric choice')
 write(out/'SOURCE_POLICY.json',result);progress(out,'source_policy_fixed',status=status,metric=result['metric'],anchors=len(anchors));return result

def reconstruct(job,dataset,root,out,db,domain,policy):
 if (out/'REBUILD_COMPLETE.json').exists():return
 sources=json.loads((out/'PACKETS_COMPLETE.json').read_text());paths={s['file_id']:Path(s['read_path']) for s in sources['sources']};cutoffs=[s['prefix']['available_time_cutoff_epoch'] for s in sources['sources'] if s['prefix']];counts=collections.Counter();flags=collections.Counter();sessions=collections.defaultdict(set);sourcegroups=collections.defaultdict(set)
 accepted=out/'accepted.jsonl';reject=out/'matching_rejections.jsonl';membership=out/'accepted_packet_membership.jsonl';relations=out/'session_relations.jsonl'
 handles={};keycount=0
 try:
  with accepted.open('w') as af,reject.open('w') as rf,membership.open('w') as mf,relations.open('w') as ef:
   for (k,e) in db.execute('SELECT k,endpoints FROM keys ORDER BY k').fetchall():
    rows,rawps=perkey(db,k);key=json.loads(e);ps,events,edges=assign_sessions(dataset,domain,key,rawps);bad=db.execute('SELECT reason FROM badkeys WHERE k=?',(k,)).fetchone()
    for edge in edges:emit(ef,**edge)
    for event in events:
     if event['event']=='duplicate_observation_across_files':flags['cross_file_duplicate_packet']+=1
    index=WindowIndex(ps);matches={};winowners=collections.defaultdict(list)
    if policy['status']=='CERTIFIED_FOR_DECLARED_SCOPE' and not bad and all(v for _,v,_ in rows):
     for i,v,r in rows:
      wins=index.windows(r,policy['offset_options_seconds'],policy['metric']);matches[i]=wins
      for win in wins:winowners[win].append(i)
    for i,valid,r in rows:
     if not valid:continue # Already present in parse_rejections.jsonl.
     label=r['label'];reason=stage=None;detail={}
     if label not in CLASSES[dataset]:reason='label_outside_predeclared_experiment';stage='scope'
     elif policy['status']!='CERTIFIED_FOR_DECLARED_SCOPE':reason=policy['status'];stage='source_policy'
     elif bad:reason='incomplete_packet_candidate_search';stage='packet_decode';detail={'error':bad[0]}
     elif any(not v for _,v,_ in rows):reason='invalid_candidate_record_for_key';stage='matching'
     else:
      wins=matches[i]
      if len(wins)!=1:reason='no_consistent_packet_window' if not wins else 'ambiguous_packet_window';stage='matching';detail={'candidate_count':len(wins)}
      else:
       a,b=wins[0];unit=ps[a:b];detail={'window':[a,b],'sessions':sorted({p['session_id'] for p in unit})}
       if len(winowners[(a,b)])!=1:reason='ambiguous_csv_record_or_label';stage='matching'
       elif len(detail['sessions'])!=1:reason='window_spans_multiple_observed_sessions';stage='session'
       elif any(any(r['clock']+o<=unit[0]['ts']<r['clock']+o+r['clock_precision_s'] and r['clock']+o+r['clock_precision_s']+r['Flow Duration']/1e6>=cut for o in policy['offset_options_seconds']) and unit[0]['ts']<=cut for cut in cutoffs):reason='csv_time_uncertainty_touches_truncated_coverage';stage='coverage'
     if reason:
      emit(rf,source_file_id=r['source_file_id'],csv_row_id=r['csv_row'],csv_path=r['csv_path'],label=label,stage=stage,reason=reason,detail=detail,decision='reject');counts[(label,'reject',stage,reason)]+=1;continue
     # Use raw unscaled features. Scaler is deliberately absent at rebuild time.
     prefix=bytearray();packets=[]
     for p in unit:
      packets.append({n:p[n] for n in ['file_id','index','offset','caplen','ts','raw_sha256','session_id','missing_initial_syn','closed_control_tail']})
      if len(prefix)<512:
       fid=p['file_id']
       if fid not in handles:handles[fid]=paths[fid].open('rb')
       h=handles[fid];h.seek(p['offset']);prefix.extend(anonymous_frame(h.read(p['caplen']),p['link']))
     seq=bytes(prefix[:512]).ljust(512,b'\0');stat=np.asarray(r['stat'],dtype='<f4');content=hashlib.sha256(stat.tobytes()+seq).hexdigest();sample=stable([dataset,r['source_file_id'],r['csv_row']]);sid=unit[0]['session_id'];group=stable([dataset,domain,key]);missing=unit[0]['missing_initial_syn'];tail=sum(p['closed_control_tail'] for p in unit)
     record=dict(dataset=dataset,sample_id=sample,label=label,session_id=sid,source_group_id=group,provenance_domain_id=domain,source_file_id=r['source_file_id'],csv_path=r['csv_path'],csv_row=r['csv_row'],start_time=unit[0]['ts'],end_time=unit[-1]['ts'],missing_initial_syn=missing,closed_control_tail_packets=tail,packet_count=len(unit),content_hash=content,stat_raw=stat.tolist(),seq_hex=seq.hex(),nonfinite_features_replaced=r['nonfinite_features_replaced'],overflow=False)
     emit(af,**record);emit(mf,sample_id=sample,csv_path=r['csv_path'],csv_row=r['csv_row'],session_id=sid,packets=packets);counts[(label,'accept','matching','unique_complete_observed_session_window')]+=1;sessions[label].add(sid);sourcegroups[label].add(group);flags['missing_initial_syn']+=bool(missing);flags['closed_control_tail_packets']+=tail;flags['one_or_two_packet_records']+=len(unit)<3
    keycount+=1
    if keycount%1000==0:progress(out,'reconstruct',keys=keycount,accepted=sum(v for k,v in counts.items() if k[1]=='accept'))
 finally:
  for h in handles.values():h.close()
 with (out/'counts_by_label_reason.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['label','decision','stage','reason','count']);w.writerows([(*k,v) for k,v in sorted(counts.items())])
 # Parse failures are separately enumerated and included in closed accounting.
 parsed=collections.Counter()
 for line in (out/'parse_rejections.jsonl').open():
  r=json.loads(line);parsed[(r['label'],r['stage'],r['reason'])]+=1
 summary=dict(dataset=dataset,domain=job['domain'],status='RECONSTRUCTED_UNSPLIT' if policy['status']=='CERTIFIED_FOR_DECLARED_SCOPE' else 'BLOCKED_NO_FORMAL_ACCEPTANCE',source_policy=policy['status'],accepted=sum(v for k,v in counts.items() if k[1]=='accept'),matching_or_scope_rejected=sum(v for k,v in counts.items() if k[1]=='reject'),parse_rejected=sum(parsed.values()),csv_total=json.loads((out/'CSV_COMPLETE.json').read_text())['total_rows'],independent_observed_sessions_by_label={k:len(v) for k,v in sessions.items()},conservative_tuple_source_groups_by_label={k:len(v) for k,v in sourcegroups.items()},flags=dict(flags),training_allowed=False,scaler_fitted=False,formal_split_generated=False)
 summary['counts_close']=summary['csv_total']==summary['accepted']+summary['matching_or_scope_rejected']+summary['parse_rejected']
 if not summary['counts_close']:raise RuntimeError('Row counts do not close')
 write(out/'REBUILD_COMPLETE.json',summary);progress(out,'finished_unsplit',**summary)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--dataset',choices=['CIC_IDS','CIC_DDOS'],required=True);p.add_argument('--domain');a=p.parse_args()
 registry=list(csv.DictReader((V3/'source_domain_registry.csv').open()));prefixes=json.loads((V3/'sources/PREFIX_PROVENANCE.json').read_text())
 for job in jobs(a.data_root,a.dataset):
  if a.domain and job['domain']!=a.domain:continue
  out=HERE/'outputs'/a.dataset/job['domain'];out.mkdir(parents=True,exist_ok=True)
  with (out/'worker.lock').open('w') as lock:
   import fcntl
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   try:
    progress(out,'starting');db=connect(out);ingest_csv(job,a.dataset,a.data_root,out,db);scan_packets(job,a.dataset,a.data_root,out,db,registry,prefixes);policy=calibrate(job,a.dataset,out,db);domain=next(r['provenance_domain_id'] for r in registry if r['dataset']==a.dataset and r['acquisition_family']==job['domain']);reconstruct(job,a.dataset,a.data_root,out,db,domain,policy);db.close()
   except Exception as exc:
    progress(out,'FAILED',error=repr(exc));raise
if __name__=='__main__':main()

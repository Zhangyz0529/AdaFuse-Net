"""Legacy29-preserving DoH preparation; no guessed protocols or feature schema.
Build all-CSV candidate index, select fixed hash anchors, scan only PCAP files
whose previously audited times can cover anchors under predeclared clock offsets.
A calibration report is not a formal sample or split acceptance certificate.
"""
import argparse,collections,csv,datetime,hashlib,heapq,json,mmap,os,re,sqlite3,time
from pathlib import Path
import numpy as np
import packet_rebuild as r
OFFSETS=[x*3600 for x in range(-12,13)]
POLICY=dict(version='doh29_alignment_preflight_v1',csv_features=29,protocol='unknown until observed in original PCAP; retain TCP and UDP candidates',anchors='64 lowest SHA256(relative CSV,row number) per each of5 known CSV files; no model scores',timestamp='CSV second-precision interval; fixed integer-hour offset candidates -12..12h',duration='seconds, candidate tolerance2us; test before deciding formal units',byte_candidates=['captured_frame_len','ip_total_len','transport_len','transport_payload_len','captured_transport_payload_len'],source_domains='browser provider (casefold); malicious filename scenario preceding _dohN_; provisional until shard/archive/overlap audit',minimum_informative_anchors=8,minimum_consensus=.95,formal_acceptance_allowed=False,sequence_schema='legacy512 anonymized frame prefix, not alternate25-dimension IP pilot')
LABELS={'chrome_all.csv':'Benign','firefox_all.csv':'Benign','dns2tcp.csv':'dns2tcp','dnscat2.csv':'dnscat2','iodine.csv':'iodine'}
def status(out,stage,**x):r.write(out/'STATUS.json',dict(stage=stage,time=time.time(),pid=os.getpid(),formal_acceptance_allowed=False,**x));print(stage,x,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--capture-metadata',type=Path,default=r.HERE.parent/'R1_5_leakage_audited_reproduction_v1/source_audit/capture_metadata.csv');a=p.parse_args();out=r.HERE/'CIC_DOH/alignment_preflight';out.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 db=sqlite3.connect(out/'candidates.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-65536')
 if not (out/'CSV_COMPLETE.json').exists():
  db.execute('DROP TABLE IF EXISTS rows');db.execute('CREATE TABLE rows(k TEXT,csv TEXT,rowno INTEGER,label TEXT,data BLOB)');anchors=[];totals={};schemas={}
  with (out/'csv_rejections.jsonl').open('w') as log:
   for path in sorted((a.data_root/'DoHBrw-2020/Clean_CSVs').glob('*.csv')):
    if path.name not in LABELS:raise RuntimeError('Unspecified CSV label: '+path.name)
    heap=[];n=0;batch=[]
    with path.open(encoding='cp1252',newline='') as f:
     reader=csv.DictReader(f);fields=reader.fieldnames;feature=[c for c in fields if c not in ['SourceIP','DestinationIP','SourcePort','DestinationPort','TimeStamp','Label','DoH']]
     if len(feature)!=29:raise RuntimeError('Unexpected DoH feature dimension')
     schemas[path.name]=feature
     for rowno,raw in enumerate(reader,2):
      n+=1
      try:
       ends=sorted([(raw['SourceIP'].strip(),int(raw['SourcePort'])),(raw['DestinationIP'].strip(),int(raw['DestinationPort']))]);key=json.dumps(ends);clock=datetime.datetime.fromisoformat(raw['TimeStamp']).replace(tzinfo=datetime.timezone.utc).timestamp();stats=np.asarray([float(raw[c]) for c in feature],dtype='float64');bad=int((~np.isfinite(stats)).sum());stats=np.nan_to_num(stats,nan=0.,posinf=0.,neginf=0.).astype('float32')
       if not np.isfinite(stats).all():raise ValueError('float32 overflow')
       row=dict(csv_path=str(path),csv_row=rowno,label=LABELS[path.name],SourceIP=raw['SourceIP'].strip(),SourcePort=int(raw['SourcePort']),clock=clock,duration=float(raw['Duration']),sent=float(raw['FlowBytesSent']),received=float(raw['FlowBytesReceived']),stat_raw=stats.tolist(),nonfinite_replaced=bad,protocol=None)
       if not all(np.isfinite(row[k]) and row[k]>=0 for k in ['duration','sent','received']):raise ValueError('invalid duration/byte values')
       batch.append((key,str(path),rowno,row['label'],r.pack(row)));rank=int(r.stable([path.name,rowno]),16);entry=(-rank,rowno,key,row)
       if len(heap)<64:heapq.heappush(heap,entry)
       elif rank < -heap[0][0]:heapq.heapreplace(heap,entry)
      except (KeyError,ValueError,OverflowError) as exc:r.emit(log,csv_path=str(path),csv_row=rowno,stage='parse',reason=repr(exc))
      if len(batch)>=20000:db.executemany('INSERT INTO rows VALUES(?,?,?,?,?)',batch);batch=[];db.commit();status(out,'csv_index',file=path.name,rows=n)
    db.executemany('INSERT INTO rows VALUES(?,?,?,?,?)',batch);db.commit();totals[path.name]=n;anchors.extend(dict(key=k,**row) for _,_,k,row in sorted(heap,reverse=True))
   if any(v!=next(iter(schemas.values())) for v in schemas.values()):raise RuntimeError('DoH CSV feature order differs')
  db.execute('CREATE INDEX rows_key ON rows(k)');db.commit();r.write(out/'CSV_COMPLETE.json',dict(rows=totals,schemas=schemas,sha256={str(f):r.sha(f) for f in (a.data_root/'DoHBrw-2020/Clean_CSVs').glob('*.csv')}));r.write(out/'FIXED_ANCHORS.json',anchors)
 anchors=json.loads((out/'FIXED_ANCHORS.json').read_text());keys={x['key'] for x in anchors};meta=list(csv.DictReader(a.capture_metadata.open()));selected=[];unknown=[]
 for m in meta:
  if m['dataset']!='CIC_DOH':continue
  try:first=float(m['first_epoch']);last=float(m['last_epoch'])
  except ValueError:unknown.append(m);continue
  if any(first<=x['clock']+o+1+x['duration'] and last>=x['clock']+o for x in anchors for o in OFFSETS):selected.append(m)
 r.write(out/'ANCHOR_SOURCE_SCOPE.json',dict(selected=selected,unknown_time_sources=unknown,scope='previous header-audited time ranges only; no whole-source identity claim; unknown coverage blocks formal acceptance'))
 if not (out/'PACKET_ANCHORS_COMPLETE.json').exists():
  db.execute('DROP TABLE IF EXISTS packets');db.execute('CREATE TABLE packets(k TEXT,domain TEXT,ts REAL,data BLOB)');db.commit();errors=collections.Counter();sources=[]
  with (out/'packet_rejections.jsonl').open('w') as log:
   for m in selected:
    path=a.data_root/m['source_file'];rel=Path(m['source_file']);prefix=rel.parts[3].casefold() if len(rel.parts)>3 else ''
    if prefix in ['malicuous','malicious']:
     family=re.split(r'_doh\d+_',path.stem)[0]
    else:family='browser_provider:'+prefix
    domain=r.stable(['DoH',family]);fileid=r.stable(['DoH',str(rel)]);batch=[];n=kept=0
    if path.stat().st_size!=int(m['bytes']):raise RuntimeError('Source size changed since time audit: '+str(rel))
    try:
     with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
      for ordinal,ts,offset,cap,wire,link in r.H['indexed_packets'](mm):
       n+=1
       try:
        p=r.decode(bytes(mm[offset:offset+cap]),link);key=json.dumps(sorted([tuple(p['src']),tuple(p['dst'])]))
        if key not in keys:continue
        p.update(ts=ts,file_id=fileid,index=ordinal,offset=offset,caplen=cap,link=link,domain=domain,family=family,captured_frame_len=cap,ip_total_len=p['ip_end']-p['network_offset'],transport_len=p['ip_end']-p['transport_offset'],transport_payload_len=p['payload_len'],captured_transport_payload_len=p['payload_len']+cap-p['ip_end']);batch.append((key,domain,ts,r.pack(p)));kept+=1
       except ValueError as exc:errors[str(exc)]+=1
       if len(batch)>=10000:db.executemany('INSERT INTO packets VALUES(?,?,?,?)',batch);batch=[];db.commit()
     db.executemany('INSERT INTO packets VALUES(?,?,?,?)',batch);db.commit();sources.append(dict(path=str(path),file_id=fileid,sha256=r.sha(path),domain=domain,family=family,scanned=n,kept=kept));status(out,'anchor_packet_scan',sources=len(sources),of=len(selected),file=path.name,kept=kept)
    except Exception as exc:r.emit(log,source=str(path),stage='source_scan',reason=repr(exc));raise
  db.execute('CREATE INDEX packet_key ON packets(k,domain,ts)');db.commit();r.write(out/'PACKET_ANCHORS_COMPLETE.json',dict(sources=sources,decode_counts=dict(errors)))
 # Inspect unique count-free time/duration windows; do not choose a byte metric per row.
 votes=collections.defaultdict(collections.Counter);records=[]
 for row in anchors:
  packets=[r.unpack(x) for x, in db.execute('SELECT data FROM packets WHERE k=? ORDER BY domain,ts',(row['key'],))];groups=collections.defaultdict(dict)
  for p in packets:groups[p['domain'],p['proto']].setdefault((p['ts'],p['raw_sha256']),p)
  matches=[]
  for (domain,proto),byobs in groups.items():
   ps=sorted(byobs.values(),key=lambda p:(p['ts'],p['file_id'],p['index']));ts=np.asarray([p['ts'] for p in ps])
   for offset in OFFSETS:
    starts=range(int(np.searchsorted(ts,row['clock']+offset)),int(np.searchsorted(ts,row['clock']+offset+1)))
    for a0 in starts:
     target=ts[a0]+row['duration'];lo=max(a0,int(np.searchsorted(ts,target-2e-6)));hi=int(np.searchsorted(ts,target+2e-6,side='right'))
     for b in range(lo,hi):matches.append((domain,proto,offset,ps[a0:b+1]))
  result=dict(csv_path=row['csv_path'],csv_row=row['csv_row'],candidate_count=len(matches),decision='ambiguous_metadata' if len(matches)>1 else 'no_metadata_window' if not matches else 'unique_metadata_anchor')
  if len(matches)==1:
   domain,proto,offset,unit=matches[0];origin=(row['SourceIP'],row['SourcePort']);metrics={};matching=[]
   for field in POLICY['byte_candidates']:
    sent=sum(p[field] for p in unit if tuple(p['src'])==origin);received=sum(p[field] for p in unit if tuple(p['src'])!=origin);metrics[field]=[sent,received]
    if sent==row['sent'] and received==row['received']:matching.append(field)
   if len(matching)==1:votes[domain][matching[0]]+=1
   elif not matching:votes[domain]['contradictory']+=1
   else:votes[domain]['uninformative']+=1
   result.update(domain=domain,family=unit[0]['family'],protocol=proto,offset=offset,clock_residual=unit[0]['ts']-row['clock']-offset,byte_metrics=metrics,matching_metrics=matching,packet_count=len(unit),duration_error_us=((unit[-1]['ts']-unit[0]['ts'])-row['duration'])*1e6)
  records.append(result)
 r.write(out/'ANCHOR_RESULTS.json',records);r.write(out/'CALIBRATION_STATUS.json',dict(stage='ANCHOR_EVIDENCE_COMPLETE_NOT_FORMAL_REBUILD',domain_votes={k:dict(v) for k,v in votes.items()},metadata_counts=dict(collections.Counter(x['decision'] for x in records)),unknown_time_sources=len(unknown),formal_acceptance_allowed=False,remaining=['certify domain boundaries and complete raw search coverage','freeze supported per-domain byte/time definitions from sufficient anchors','implement and verify all-row29-dimension label/window matching','run final session/content grouping and partition gate'],policy=POLICY));status(out,'ANCHOR_EVIDENCE_COMPLETE_NOT_FORMAL_REBUILD')
if __name__=='__main__':main()

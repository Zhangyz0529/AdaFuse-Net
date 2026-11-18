"""Disk-backed source-label observed-session rebuild for USTC and ISCX.
Never imports or executes historical generator top-level code. IPv4 and IPv6
TCP/UDP supported; fragmented/unparsed tuples fail closed. No fitted processing.
"""
import argparse,ast,collections,csv,hashlib,ipaddress,json,mmap,os,re,socket,sqlite3,struct,sys,time,zlib
from pathlib import Path
import numpy as np,dpkt
HERE=Path(__file__).resolve().parent;CODE=HERE;V3=HERE.parent/'R1_5_session_rebuild_v3';sys.path.insert(0,str(V3));sys.path.insert(0,str(V3.parent/'R1_5_session_rebuild_v2'))
from rebuild import H
from rules_v3 import RULES,assign_sessions,stable
DIMS={'USTC':25,'ISCX':25}
CLASSES={'USTC':['Benign','Htbot','Neris','Miuref','Virut'],'ISCX':['facebook_audio4','vpn_bittorrent','vpn_hangouts_audio2','vpn_facebook_audio2','vpn_email2b','vpn_ftps_a']}
POLICY=dict(version='remaining_observed_v1',stat_dim=25,sequence_bytes=512,min_packets=1,max_packets=None,raw_scope='IPv4/IPv6 TCP/UDP; no fragment reassembly; dependent tuples reject',direction='legacy lexicographically descending endpoint string ip:port',bytes='length-preserving anonymized captured frame, with recomputed network/transport checksums',time='source timestamps retained; no synthetic UTC alignment; unrelated captures never joined by timestamp',labels='source-file activity/class cohort; mixed-label observed sessions rejected',family='USTC source file; ISCX lowercase name with final a/b suffix conservatively merged; naming relationship is grouping precaution not authenticated publisher provenance',source_group='same domain and bidirectional tuple; uncertain continuations never split',unsupported='log every unsupported packet; a fragmented or undecodable transport packet with identifiable IP pair blocks all candidate tuples in that IP pair',training_allowed=False)
def write(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(path)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def emit(f,**r):f.write(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n')
def pack(x):return zlib.compress(json.dumps(x,separators=(',',':')).encode(),1)
def unpack(x):return json.loads(zlib.decompress(x))
def status(out,stage,**kw):write(out/'STATUS.json',dict(stage=stage,time=time.time(),pid=os.getpid(),training_started=False,**kw));print(out.name,stage,kw,flush=True)
def network(raw,link):
 if link==1:
  if len(raw)<14:raise ValueError('truncated_ethernet')
  net=14;eth=struct.unpack_from('!H',raw,12)[0]
  while eth in [0x8100,0x88a8]:
   if net+4>len(raw):raise ValueError('truncated_vlan')
   eth=struct.unpack_from('!H',raw,net+2)[0];net+=4
  if eth not in [0x800,0x86dd]:raise ValueError('non_IP_frame')
 elif link==113:
  net=16
  if len(raw)<16 or struct.unpack_from('!H',raw,14)[0] not in [0x800,0x86dd]:raise ValueError('non_IP_cooked')
 elif link in [101,228,229]:net=0
 else:raise ValueError('unsupported_link')
 return net
class Reject(ValueError):
 def __init__(self,reason,pair=None):super().__init__(reason);self.pair=pair

def decode(raw,link):
 net=network(raw,link)
 if net>=len(raw):raise Reject('truncated_IP')
 version=raw[net]>>4
 if version==4:
  if net+20>len(raw):raise Reject('truncated_IPv4')
  src=socket.inet_ntoa(raw[net+12:net+16]);dst=socket.inet_ntoa(raw[net+16:net+20]);pair=tuple(sorted([src,dst]));ihl=(raw[net]&15)*4;end=net+struct.unpack_from('!H',raw,net+2)[0];proto=raw[net+9];tr=net+ihl
  if struct.unpack_from('!H',raw,net+6)[0]&0x3fff:raise Reject('fragment_requires_reassembly',pair)
  if ihl<20 or end<tr or end>len(raw):raise Reject('invalid_or_truncated_IPv4',pair)
 elif version==6:
  if net+40>len(raw):raise Reject('truncated_IPv6')
  src=socket.inet_ntop(socket.AF_INET6,raw[net+8:net+24]);dst=socket.inet_ntop(socket.AF_INET6,raw[net+24:net+40]);pair=tuple(sorted([src,dst]));plen=struct.unpack_from('!H',raw,net+4)[0];end=net+40+plen;proto=raw[net+6];tr=net+40;ihl=40
  if not plen or end>len(raw):raise Reject('jumbo_or_truncated_IPv6',pair)
  for _ in range(16):
   if proto==44:raise Reject('fragment_requires_reassembly',pair)
   if proto not in [0,43,60,51]:break
   if tr+2>end:raise Reject('truncated_IPv6_extension',pair)
   size=(raw[tr+1]+1)*8 if proto!=51 else (raw[tr+1]+2)*4;proto=raw[tr];tr+=size
   if tr>end:raise Reject('invalid_IPv6_extension',pair)
  else:raise Reject('excess_IPv6_extension_chain',pair)
 else:raise Reject('unsupported_IP_version')
 if proto not in [6,17]:raise Reject('non_TCP_UDP')
 minimum=20 if proto==6 else 8
 if tr+minimum>end:raise Reject('truncated_transport',pair)
 sport,dport=struct.unpack_from('!HH',raw,tr);hdr=(raw[tr+12]>>4)*4 if proto==6 else 8
 if hdr<minimum or tr+hdr>end:raise Reject('invalid_transport_header',pair)
 if proto==17 and struct.unpack_from('!H',raw,tr+4)[0] != end-tr:raise Reject('UDP_length_disagrees_IP_payload',pair)
 return dict(src=[src,sport],dst=[dst,dport],proto=proto,flags=raw[tr+13] if proto==6 else 0,isn=struct.unpack_from('!I',raw,tr+4)[0] if proto==6 else None,payload_len=end-tr-hdr,raw_sha256=hashlib.sha256(raw).hexdigest(),ip_version=version,network_offset=net,transport_offset=tr,ip_end=end,ip_header_length=ihl)
def anonymous(raw,link,p):
 b=bytearray(raw);net=p['network_offset'];tr=p['transport_offset'];end=p['ip_end'];v=p['ip_version'];proto=p['proto']
 if link==1:b[:12]=b'\0'*12
 if v==4:
  b[net+12:net+20]=b'\0'*8;b[net+10:net+12]=b'\0\0';struct.pack_into('!H',b,net+10,dpkt.in_cksum(bytes(b[net:net+p['ip_header_length']])));pseudo=b'\0'*8+struct.pack('!BBH',0,proto,end-tr)
 else:b[net+8:net+40]=b'\0'*32;pseudo=b'\0'*32+struct.pack('!I3xB',end-tr,proto)
 b[tr:tr+4]=b'\0'*4;check=tr+(16 if proto==6 else 6);b[check:check+2]=b'\0\0';checksum=dpkt.in_cksum(pseudo+bytes(b[tr:end]));struct.pack_into('!H',b,check,checksum or (65535 if proto==17 else 0));return bytes(b)
def tracker(dataset):
 f=CODE.parent/('USTC/generate_ustc_multimodal.py' if dataset=='USTC' else 'ISCX/generate_iscx_pkl.py');tree=ast.parse(f.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlowTracker'];env={'np':np,'MAX_LEN':512};exec(compile(tree,str(f),'exec'),env);return env['FlowTracker']
def plan(root,dataset):
 if dataset=='USTC':
  files=[p for d in ['Benign','Malware'] for p in sorted((root/'USTC-TFC2016-master'/d).iterdir()) if p.suffix.lower() in ['.pcap','.pcapng']]
 else:files=[p for p in sorted((root/'ISCX/pcap').iterdir()) if p.suffix.lower() in ['.pcap','.pcapng']]
 groups=collections.defaultdict(list)
 for f in files:
  label=('Benign' if f.parent.name=='Benign' else f.stem) if dataset=='USTC' else f.stem.lower();family=f.stem.lower() if dataset=='USTC' else re.sub(r'[_-]?[ab]$','',f.stem.lower());groups[family].append(dict(path=str(f),relative_path=str(f.relative_to(root)),label=label,file_id=stable([dataset,str(f.relative_to(root))])))
 return groups

def build(dataset,family,files):
 out=HERE/'outputs'/dataset/family;out.mkdir(parents=True,exist_ok=True)
 if (out/'REBUILD_COMPLETE.json').exists():return
 import fcntl
 lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);status(out,'starting');domain=stable([dataset,family]);db=sqlite3.connect(out/'packets.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-65536')
 try:
  if not (out/'SCAN_COMPLETE.json').exists():
   db.execute('DROP TABLE IF EXISTS packets');db.execute('CREATE TABLE packets(k TEXT,pair TEXT,ts REAL,fid TEXT,ordinal INTEGER,data BLOB)');db.execute('DROP TABLE IF EXISTS badpairs');db.execute('CREATE TABLE badpairs(pair TEXT PRIMARY KEY)');total=decoded=0;sources=[];reasons=collections.Counter();global_errors=[]
   with (out/'packet_rejections.jsonl').open('w') as log:
    for source in files:
     f=Path(source['path']);batch=[];n=0
     with f.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as mm:
      for ordinal,ts,offset,cap,wire,link in H['indexed_packets'](mm):
       n+=1;total+=1
       try:
        raw=bytes(mm[offset:offset+cap]);p=decode(raw,link);key=(p['proto'],*sorted([tuple(p['src']),tuple(p['dst'])]));pair=json.dumps(sorted([p['src'][0],p['dst'][0]]));p.update(file_id=source['file_id'],index=ordinal,ts=ts,offset=offset,caplen=cap,link=link,label=source['label']);batch.append((json.dumps(key),pair,ts,p['file_id'],ordinal,pack(p)));decoded+=1
       except ValueError as exc:
        reasons[str(exc)]+=1;emit(log,source_file_id=source['file_id'],packet_index=ordinal,label=source['label'],stage='decode',reason=str(exc))
        if isinstance(exc,Reject) and exc.pair:db.execute('INSERT OR IGNORE INTO badpairs VALUES(?)',(json.dumps(exc.pair),))
        elif str(exc) not in ['non_IP_frame','non_IP_cooked','non_TCP_UDP']:global_errors.append(str(exc))
       if len(batch)>=10000:db.executemany('INSERT INTO packets VALUES(?,?,?,?,?,?)',batch);batch=[];db.commit()
       if total%500000==0:status(out,'scan',records=total,decoded=decoded)
     db.executemany('INSERT INTO packets VALUES(?,?,?,?,?,?)',batch);db.commit();current_hash=sha(f)
     if source.get('sha256') and current_hash!=source['sha256']:raise RuntimeError('Frozen source changed: '+str(f))
     sources.append({**source,'sha256':current_hash,'records':n})
   db.execute('CREATE INDEX packet_key_order ON packets(k,ts,fid,ordinal)');db.commit();write(out/'SCAN_COMPLETE.json',dict(records=total,decoded=decoded,rejected=total-decoded,reasons=dict(reasons),sources=sources,global_errors=sorted(set(global_errors)),counts_close=total==decoded+sum(reasons.values())))
  scan=json.loads((out/'SCAN_COMPLETE.json').read_text());handles={s['file_id']:open(s['path'],'rb') for s in scan['sources']};accepted=rejected=observed_total=0;counts=collections.Counter();observed=collections.defaultdict(set);groups=collections.defaultdict(set);flags=collections.Counter();klass=tracker(dataset)
  try:
   with (out/'accepted.jsonl').open('w') as af,(out/'accepted_packet_membership.jsonl').open('w') as mf,(out/'session_rejections.jsonl').open('w') as rf,(out/'session_relations.jsonl').open('w') as ef:
    for key,pair in db.execute('SELECT k,pair FROM packets GROUP BY k').fetchall():
     ps=[unpack(b) for b, in db.execute('SELECT data FROM packets WHERE k=? ORDER BY ts,fid,ordinal',(key,))];raw_labels=collections.defaultdict(set)
     for p in ps:raw_labels[p['ts'],p['raw_sha256']].add(p['label'])
     duplicate_label_conflict=any(len(v)>1 for v in raw_labels.values())
     ps,events,edges=assign_sessions(dataset,domain,json.loads(key),ps)
     for edge in edges:emit(ef,**edge)
     bysession=collections.defaultdict(list)
     for p in ps:bysession[p['session_id']].append(p)
     observed_total+=len(bysession)
     bad=duplicate_label_conflict or bool(scan['global_errors']) or db.execute('SELECT 1 FROM badpairs WHERE pair=?',(pair,)).fetchone() is not None
     for sid,unit in bysession.items():
      labels={p['label'] for p in unit};reason='incomplete_source_or_IP_pair_decode' if bad else 'mixed_source_labels_in_observed_session' if len(labels)!=1 else None
      if reason:
       rejected+=1;emit(rf,session_id=sid,labels=sorted(labels),stage='source_or_session',reason=reason,source_packets=[{'file_id':p['file_id'],'index':p['index']} for p in unit]);continue
      label=next(iter(labels));t=klass(key)
      for p in unit:
       h=handles[p['file_id']];h.seek(p['offset']);raw=h.read(p['caplen']);frame=anonymous(raw,p['link'],p);assert len(frame)==p['caplen'];forward=(str(p['src'][0])+':'+str(p['src'][1]))>(str(p['dst'][0])+':'+str(p['dst'][1]));t.add_packet(frame,p['ts'],forward)
      stats=t.compute_statistical_features();q=bytes(t.payload[:512]).ljust(512,b'\0')
      if stats.shape!=(25,) or not np.isfinite(stats).all():rejected+=1;emit(rf,session_id=sid,label=label,stage='features',reason='nonfinite_or_invalid_statistics');continue
      sample=stable([dataset,domain,sid]);sg=stable([dataset,domain,key]);record=dict(dataset=dataset,sample_id=sample,label=label,session_id=sid,source_group_id=sg,source_file_id=';'.join(sorted({p['file_id'] for p in unit})),provenance_domain_id=domain,record_locator=';'.join(f"{p['file_id']}:{p['index']}" for p in [unit[0],unit[-1]]),start_time=unit[0]['ts'],end_time=unit[-1]['ts'],packet_count=len(unit),stat_raw=stats.tolist(),seq_hex=q.hex(),content_hash=hashlib.sha256(stats.astype('<f4').tobytes()+q).hexdigest(),missing_initial_syn=unit[0]['missing_initial_syn'],closed_control_tail_packets=sum(p['closed_control_tail'] for p in unit),ip_versions=sorted({p['ip_version'] for p in unit}),overflow=False)
      emit(af,**record);emit(mf,sample_id=sample,session_id=sid,packets=[{k:p[k] for k in ['file_id','index','offset','caplen','ts','raw_sha256','session_id']} for p in unit]);accepted+=1;counts[label]+=1;observed[label].add(sid);groups[label].add(sg);flags['one_or_two_packet_samples']+=len(unit)<3;flags['missing_initial_syn']+=bool(unit[0]['missing_initial_syn'])
     if (accepted+rejected)%1000<10:status(out,'reconstruct',accepted=accepted,rejected_sessions=rejected)
  finally:
   for h in handles.values():h.close()
  result=dict(dataset=dataset,family=family,accepted=accepted,rejected_observed_sessions=rejected,by_label=dict(counts),sessions_by_label={k:len(v) for k,v in observed.items()},source_groups_by_label={k:len(v) for k,v in groups.items()},flags=dict(flags),packet_counts=scan,session_counts_close=accepted+rejected==observed_total,counts_close=scan['counts_close'],status='RECONSTRUCTED_UNSPLIT' if not scan['global_errors'] else 'BLOCKED_SOURCE_DECODE',training_allowed=False);write(out/'REBUILD_COMPLETE.json',result);status(out,'finished_unsplit',accepted=accepted,status=result['status'])
 except Exception as exc:status(out,'FAILED',error=repr(exc));raise
 finally:db.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(DIMS),required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--family');a=p.parse_args();planfile=HERE/(a.dataset+'_SOURCE_PLAN.json');config=json.loads(planfile.read_text()) if planfile.exists() else plan(a.data_root,a.dataset)
 if not planfile.exists():write(planfile,config)
 for family,files in config.items():
  if a.family and family!=a.family:continue
  build(a.dataset,family,files)
if __name__=='__main__':main()

"""Rebuild official-statistics/512-byte samples from unique CSV packet windows.
Ambiguous clocks, windows, labels, overflows and cross-label content are rejected.
No historical PKL is used as model input. Resources and rejection rules are frozen.
"""
import argparse,ast,collections,hashlib,heapq,json,mmap,socket,struct,sys,gc
from pathlib import Path
import numpy as np,pandas as pd,dpkt
from session_rules import assign_sessions,stable,RULES
HERE=Path(__file__).resolve().parent

def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def helpers():
 env=dict(np=np,struct=struct,socket=socket,Counter=collections.Counter,dpkt=dpkt)
 for name,functions in [('packet_window_reference.py',['indexed_packets','fast_endpoint','csv_key','window_candidates','row_identity','calibrated_matches']),('packet_decode_reference.py',['decode'])]:
  t=ast.parse((HERE/'code'/name).read_text());t.body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in functions];exec(compile(t,name,'exec'),env)
 return env
H=helpers()
CLASSES={'CIC_IDS':['BENIGN','PortScan','DDoS','FTP-Patator','SSH-Patator','DoS'],'CIC_DDOS':['BENIGN','Syn','DrDoS_MSSQL','DrDoS_NetBIOS','DrDoS_UDP','UDP-lag']}
MAP={'DoS Hulk':'DoS','DoS GoldenEye':'DoS','UDPLag':'UDP-lag','MSSQL':'DrDoS_MSSQL','NetBIOS':'DrDoS_NetBIOS','UDP':'DrDoS_UDP'}
META=['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Label']

def jobs(root,dataset):
 if dataset=='CIC_IDS':
  return [dict(domain=day,csvs=sorted((root/'CIC_IDS/GeneratedLabelledFlows/TrafficLabelling').glob(day+'*.csv')),pcaps=[next(p for p in (root/'CIC_IDS/pcap').glob('*.pcap') if p.name.lower()==f'{day}-workinghours.pcap'.lower())]) for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']]
 return [dict(domain=day,csvs=sorted((root/f'CIC-DDoS-2019/{day}/CSVs/{sub}').glob('*.csv')),pcaps=sorted((root/f'CIC-DDoS-2019/{day}/PCAPs').glob('*.pcap'))) for day,sub in [('Test_Day_03-11','03-11'),('Train_Day_01-12','01-12')]]

def frames(f):
 for chunk in pd.read_csv(f,encoding='cp1252',chunksize=50000,low_memory=False):
  chunk.columns=chunk.columns.str.strip();yield chunk

def load_rows(job,dataset,out,cap):
 heaps=collections.defaultdict(list);files=job['csvs'];report={'files':{},'selection':'smallest SHA256 of source-file-relative-name and row, per class per domain; no model scores'}
 for f in files:
  offset=0
  for df in frames(f):
   for j,label in enumerate(df.Label.astype(str).str.strip()):
    label=MAP.get(label,label)
    if label not in CLASSES[dataset]:continue
    rid=offset+j+2;rank=int(stable([f.name,rid]),16);entry=(-rank,str(f),rid)
    if len(heaps[label])<cap:heapq.heappush(heaps[label],entry)
    elif rank < -heaps[label][0][0]:heapq.heapreplace(heaps[label],entry)
   offset+=len(df)
  report['files'][str(f)]={'rows':offset,'sha256':sha(f)};print(dataset,job['domain'],'CSV candidates',f.name,offset,flush=True)
 selected={(p,r) for h in heaps.values() for rank,p,r in h};target=[];keys=set();feature_names=None;invalid=0
 # Load statistics only for selected rows, using unchanged official feature order.
 for f in files:
  offset=0
  for df in frames(f):
   pick=[j for j in range(len(df)) if (str(f),offset+j+2) in selected]
   cols=[c for c in df.columns if c not in META and not (dataset=='CIC_DDOS' and c in ['Unnamed: 0','SimillarHTTP','Inbound'])]
   if feature_names is None:feature_names=cols
   if cols!=feature_names:raise ValueError('Official feature schema differs across source CSVs')
   for j in pick:
    r=df.iloc[j].to_dict();r['csv_path']=str(f);r['csv_row']=offset+j+2;r['label']=MAP.get(str(r['Label']).strip(),str(r['Label']).strip())
    try:
     vals=np.array([r[c] for c in cols],dtype='float32');vals=np.nan_to_num(vals,nan=0.,posinf=0.,neginf=0.)
     for c in ['Source Port','Destination Port','Protocol','Total Fwd Packets','Total Backward Packets']:r[c]=int(float(r[c]))
     r['Flow Duration']=float(r['Flow Duration']);clock=pd.to_datetime(str(r['Timestamp']),format='mixed',dayfirst=dataset=='CIC_IDS',errors='raise');r['clock']=float(clock.timestamp())
     n=r['Total Fwd Packets']+r['Total Backward Packets']
     if n<3 or n>10000 or r['Flow Duration']<0 or len(vals)!=77:raise ValueError('Ineligible row')
     r['_invalid_stats']=False;r['stat']=vals.tolist();key=H['csv_key'](r);keys.add(key);target.append(r)
    except (ValueError,TypeError,OverflowError):invalid+=1
   offset+=len(df)
 # Include nonselected rows and all labels in ambiguity checks. Reject hot keys.
 bykey=collections.defaultdict(list);overflow=set()
 for f in files:
  offset=0
  for df in frames(f):
   for j,rr in enumerate(df[['Source IP','Source Port','Destination IP','Destination Port','Protocol']].itertuples(index=False,name=None)):
    try:key=H['csv_key'](dict(zip(['Source IP','Source Port','Destination IP','Destination Port','Protocol'],rr)))
    except (ValueError,TypeError,OSError):continue
    if key not in keys or key in overflow:continue
    if len(bykey[key])>=100:overflow.add(key);bykey.pop(key,None);continue
    r=df.iloc[j].to_dict();r['csv_path']=str(f);r['csv_row']=offset+j+2;r['label']=MAP.get(str(r['Label']).strip(),str(r['Label']).strip())
    try:
     for c in ['Source Port','Destination Port','Protocol','Total Fwd Packets','Total Backward Packets']:r[c]=int(float(r[c]))
     r['Flow Duration']=float(r['Flow Duration']);r['clock']=float(pd.to_datetime(str(r['Timestamp']),format='mixed',dayfirst=dataset=='CIC_IDS',errors='raise').timestamp());r['_invalid_stats']=r['Flow Duration']<0
    except (ValueError,TypeError,OverflowError):r['_invalid_stats']=True;r['clock']=0.
    bykey[key].append(r)
   offset+=len(df)
 for key in overflow:bykey.pop(key,None)
 report.update(selected_before_eligibility=len(selected),invalid_selected=invalid,overflow_keys=len(overflow),target_rows=len(target),feature_names=feature_names)
 save(out/'csv_selection_audit.json',report);save(out/'targets.json',target)
 return target,bykey,report

def anonymous_frame(raw,link):
 from scapy.all import Ether,IP,TCP,UDP
 from scapy.layers.l2 import CookedLinux
 p=Ether(raw) if link==1 else CookedLinux(raw) if link==113 else IP(raw)
 if Ether in p:p[Ether].src=p[Ether].dst='00:00:00:00:00:00'
 p[IP].src=p[IP].dst='0.0.0.0'
 tr=p[TCP] if TCP in p else p[UDP];tr.sport=tr.dport=0
 if hasattr(tr,'chksum'):del tr.chksum
 if hasattr(p[IP],'chksum'):del p[IP].chksum
 return bytes(p)

def extract(job,dataset,out,cap):
 out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():return
 targets,bykey,csvreport=load_rows(job,dataset,out,cap);packets=collections.defaultdict(list);overflow=set();collected=0;source=[]
 for f in job['pcaps']:
  if not f.exists():raise FileNotFoundError(f)
  file_id=stable([dataset,str(f)]);fileinfo={'path':str(f),'file_id':file_id,'sha256':sha(f)};source.append(fileinfo);total=0
  with f.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as mm:
   for ordinal,ts,offset,n,wire,link in H['indexed_packets'](mm):
    total+=1
    if total%2000000==0:print(dataset,job['domain'],f.name,'packets',total,'retained',collected,flush=True)
    key=H['fast_endpoint'](mm,offset,n,link)
    if key not in bykey or key in overflow:continue
    if len(packets[key])>=50000:overflow.add(key);packets.pop(key,None);continue
    if collected>=3000000:raise RuntimeError('Frozen packet memory cap; no silent truncation')
    raw=bytes(mm[offset:offset+n])
    try:
     p=H['decode'](raw,link);ip=dpkt.ethernet.Ethernet(raw).data if link==1 else dpkt.sll.SLL(raw).data if link==113 else dpkt.ip.IP(raw)
     item=dict(index=ordinal,ts=ts,offset=offset,caplen=n,link=link,src=p['src'],dst=p['dst'],proto=p['proto'],flags=p['flags'],isn=p['isn'],payload_len=len(ip.data.data),raw_sha256=hashlib.sha256(raw).hexdigest(),file_id=file_id)
    except (ValueError,dpkt.UnpackError,AttributeError,IndexError):overflow.add(key);packets.pop(key,None);continue
    packets[key].append(item);collected+=1
  fileinfo['packet_records']=total
 for k in overflow:packets.pop(k,None)
 duplicate_packets=0
 for key,ps in list(packets.items()):packets[key],nd=assign_sessions(dataset,key,ps);duplicate_packets+=nd
 config={'duration_tolerance_us':2.,'min_clock_anchors':10,'min_clock_consensus':.95,'allow_12h_clock_ambiguity':dataset=='CIC_IDS'}
 accepted,rejected,clock=H['calibrated_matches'](targets,bykey,packets,config)
 statmap={(r['csv_path'],r['csv_row']):r['stat'] for r in targets};stats=[];seqs=[];records=[];rejections=list(rejected);handles={s['file_id']:open(s['path'],'rb') for s in source}
 try:
  for row,unit in accepted:
   sessions={p['session_id'] for p in unit}
   if len(sessions)!=1:rejections.append({'csv_path':row['csv_path'],'csv_row':row['csv_row'],'reason':'CSV_window_spans_multiple_observed_sessions'});continue
   prefix=bytearray()
   for p in unit:
    if len(prefix)>=512:break
    h=handles[p['file_id']];h.seek(p['offset']);prefix.extend(anonymous_frame(h.read(p['caplen']),p['link']))
   s=np.array(statmap[(row['csv_path'],row['csv_row'])],dtype='float32');q=np.frombuffer(bytes(prefix[:512]).ljust(512,b'\0'),dtype='uint8').copy();sid=stable([dataset,row['csv_path'],row['csv_row'],[(p['file_id'],p['index']) for p in unit]])
   records.append(dict(dataset=dataset,sample_id=sid,session_id=next(iter(sessions)),source_group_id=stable([dataset,H['csv_key'](row)]),source_file_id=';'.join(sorted({p['file_id'] for p in unit})),record_locator=f"{row['csv_path']}:{row['csv_row']}",label=row['label'],start_time=unit[0]['ts'],end_time=unit[-1]['ts'],missing_initial_syn=unit[0]['missing_initial_syn'],raw_record_hash=stable([p['raw_sha256'] for p in unit]),packet_records=[{'file_id':p['file_id'],'index':p['index'],'offset':p['offset'],'caplen':p['caplen']} for p in unit]));stats.append(s);seqs.append(q)
 finally:
  for h in handles.values():h.close()
 save(out/'records.json',records);save(out/'rejected.json',rejections);save(out/'source_files.json',source);save(out/'clock_audit.json',clock)
 np.savez_compressed(out/'inputs.npz',sample_ids=np.array([r['sample_id'] for r in records],dtype=str),stat_raw=np.array(stats,dtype='float32').reshape(-1,77),seq_feature=np.array(seqs,dtype='uint8').reshape(-1,512))
 save(out/'COMPLETE.json',dict(accepted=len(records),rejected=len(rejections),clock=clock['status'],source_files=len(source),observed_duplicate_packets=duplicate_packets,overflow_packet_keys=len(overflow),session_rules=RULES))
 print(dataset,job['domain'],'ACCEPTED',len(records),'CLOCK',clock['status'],flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CLASSES),required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--target-per-class-per-domain',type=int,default=5000);a=p.parse_args()
 for job in jobs(a.data_root,a.dataset):
  extract(job,a.dataset,HERE/'reconstructed'/a.dataset/job['domain'],a.target_per_class_per_domain);gc.collect()
 save(HERE/'reconstructed'/a.dataset/'EXTRACTION_COMPLETE.json',{'dataset':a.dataset,'jobs':[j['domain'] for j in jobs(a.data_root,a.dataset)],'status':'extraction_finished_not_split_audited'})
if __name__=='__main__':main()

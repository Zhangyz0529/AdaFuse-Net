"""Native size/direction patterns and burst boundaries; no relabeling or split changes."""
import argparse,json,hashlib,sqlite3,mmap,sys,time
from pathlib import Path
import numpy as np
from cache import HERE,save
sys.path.insert(0,str(HERE.parent/'R1_5_ustc_boundary_rebuild_v2'));import packet_rebuild as parser
SOURCES={'USTC':HERE.parent/'R1_5_ustc_boundary_rebuild_v2/outputs/USTC','ISCX':HERE.parent/'R1_5_remaining_datasets_rebuild_v1/outputs/ISCX','CIC_IDS':HERE.parent/'R1_5_session_rebuild_v3/full_rebuild/outputs/CIC_IDS','CIC_DDOS':HERE.parent/'R1_5_session_rebuild_v3/full_rebuild/outputs/CIC_DDOS'}
def run(ds):
 out=HERE/'packet_patterns'/ds;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():return
 wanted={s for p in (HERE/'cache'/ds).glob('round*/ids.json') for s in json.loads(p.read_text())};assert wanted
 conn=sqlite3.connect(out/'patterns.sqlite');conn.execute('CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,pattern BLOB,bursts TEXT)');done={x[0] for x in conn.execute('SELECT sample_id FROM samples')};handles={};maps={};links={};n=0
 for domain in sorted(SOURCES[ds].iterdir()):
  if not (domain/'REBUILD_COMPLETE.json').exists():continue
  summary=json.loads((domain/'REBUILD_COMPLETE.json').read_text())
  source_list=summary['packet_counts']['sources'] if 'packet_counts' in summary else json.loads((domain/'PACKETS_COMPLETE.json').read_text())['sources']
  for src in source_list:
   fid=src['file_id']
   if fid not in maps:handles[fid]=open(src.get('read_path',src['path']),'rb');maps[fid]=mmap.mmap(handles[fid].fileno(),0,access=mmap.ACCESS_READ);links[fid]=next(parser.H['indexed_packets'](maps[fid]))[-1]
  for line in (domain/'accepted_packet_membership.jsonl').open():
   x=json.loads(line);sid=x['sample_id']
   if sid not in wanted or sid in done:continue
   sizes=[];spans=[];offset=0;previous=None;firstsrc=None
   for p in x['packets']:
    raw=bytes(maps[p['file_id']][p['offset']:p['offset']+p['caplen']]);assert hashlib.sha256(raw).hexdigest()==p['raw_sha256'];meta=parser.decode(raw,links[p['file_id']]);endpoint=meta['src'];firstsrc=endpoint if firstsrc is None else firstsrc;sign=1 if endpoint==firstsrc else -1;sizes.append(sign*(meta['ip_end']-meta['network_offset']))
    if offset<512:
     end=min(offset+len(raw),512)
     if sign==previous:spans[-1][1]=end
     else:spans.append([offset,end]);previous=sign
    offset+=len(raw)
   conn.execute('INSERT INTO samples VALUES(?,?,?)',(sid,np.array(sizes[:5000],dtype='float32').tobytes(),json.dumps(spans)));done.add(sid);n+=1
   if n%1000==0:conn.commit();save(out/'STATUS.json',dict(processed=len(done),wanted=len(wanted),source_domain=domain.name,time=time.time()))
  conn.commit();print(ds,domain.name,len(done),'/',len(wanted),flush=True)
 assert wanted<=done;save(out/'COMPLETE.json',dict(status='SOURCE_PACKET_HASHES_VERIFIED_NATIVE_PATTERNS',records=len(done),direction='first observed endpoint positive, reverse negative',pattern='signed IP packet length first5000, zero padded by model',burst='maximal same-direction runs in provided first512 anonymized frame bytes; no added bytes/payload; outside512 ignored'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(SOURCES),required=True);a=p.parse_args();run(a.dataset)

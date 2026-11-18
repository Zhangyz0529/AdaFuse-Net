"""Source-derived sparse native time/IP-length histograms for shared sample IDs.

Retains audited bidirectional sample boundaries (explicit adaptation), never
reshapes payload bytes into images. Reads only packet-membership-proven bytes.
"""
import argparse,json,hashlib,sqlite3,mmap,sys,time
from pathlib import Path
import numpy as np
from cache import HERE,OLD,save
sys.path.insert(0,str(HERE.parent/'R1_5_ustc_boundary_rebuild_v2'));import packet_rebuild as parser
SOURCES={'USTC':HERE.parent/'R1_5_ustc_boundary_rebuild_v2/outputs/USTC','ISCX':HERE.parent/'R1_5_remaining_datasets_rebuild_v1/outputs/ISCX','CIC_IDS':HERE.parent/'R1_5_session_rebuild_v3/full_rebuild/outputs/CIC_IDS','CIC_DDOS':HERE.parent/'R1_5_session_rebuild_v3/full_rebuild/outputs/CIC_DDOS'}
def run(ds):
 out=HERE/'flowpic'/ds;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():return
 wanted={s for p in (HERE/'cache'/ds).glob('round*/ids.json') for s in json.loads(p.read_text())};assert wanted
 conn=sqlite3.connect(out/'histograms.sqlite');conn.execute('CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,coords BLOB,packets INTEGER,zero_duration INTEGER,oversize_packets INTEGER)');done={x[0] for x in conn.execute('SELECT sample_id FROM samples')};handles={};maps={};links={};processed=0
 for domain in sorted(SOURCES[ds].iterdir()):
  if not (domain/'REBUILD_COMPLETE.json').exists():continue
  summary=json.loads((domain/'REBUILD_COMPLETE.json').read_text())
  source_list=summary['packet_counts']['sources'] if 'packet_counts' in summary else json.loads((domain/'PACKETS_COMPLETE.json').read_text())['sources']
  for src in source_list:
   fid=src['file_id']
   if fid not in maps:
    handles[fid]=open(src.get('read_path',src['path']),'rb');maps[fid]=mmap.mmap(handles[fid].fileno(),0,access=mmap.ACCESS_READ);links[fid]=next(parser.H['indexed_packets'](maps[fid]))[-1]
  with (domain/'accepted_packet_membership.jsonl').open() as f:
   for line in f:
    x=json.loads(line);sid=x['sample_id']
    if sid not in wanted or sid in done:continue
    times=[];sizes=[]
    for p in x['packets']:
     raw=bytes(maps[p['file_id']][p['offset']:p['offset']+p['caplen']]);assert hashlib.sha256(raw).hexdigest()==p['raw_sha256'];meta=parser.decode(raw,links[p['file_id']]);times.append(p['ts']);sizes.append(meta['ip_end']-meta['network_offset'])
    times=np.array(times);sizes=np.array(sizes);duration=times.max()-times.min();tb=np.minimum(((times-times.min())/duration*1500).astype('int64'),1499) if duration>0 else np.zeros(len(times),dtype='int64');mask=(sizes>=0)&(sizes<=1500);coords=np.stack([np.minimum(sizes[mask],1499),tb[mask]],1).astype('int16');conn.execute('INSERT INTO samples VALUES(?,?,?,?,?)',(sid,coords.tobytes(),len(times),int(duration==0),int((~mask).sum())));done.add(sid);processed+=1
    if processed%1000==0:conn.commit();save(out/'STATUS.json',dict(processed=len(done),wanted=len(wanted),source_domain=domain.name,time=time.time()))
  conn.commit()
  print(ds,domain.name,len(done),'/',len(wanted),flush=True)
 assert wanted<=done, f'Missing source samples: {len(wanted-done)}';counts=conn.execute('SELECT count(*),sum(zero_duration),sum(oversize_packets) FROM samples').fetchone();save(out/'COMPLETE.json',dict(status='SOURCE_PACKET_HASHES_VERIFIED_NATIVE_HISTOGRAM_COORDINATES',records=counts[0],zero_duration_samples=counts[1],oversize_packets_excluded_by_original_histogram_range=counts[2],scope='audited bidirectional sample window, no original60s filtering; all fixed benchmark sampleIDs retained'))
 for mm in maps.values():mm.close()
 for f in handles.values():f.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(SOURCES),required=True);a=p.parse_args();run(a.dataset)

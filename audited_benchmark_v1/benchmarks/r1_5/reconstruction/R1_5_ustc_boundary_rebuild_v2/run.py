"""Reuse complete parsed-record indexes, rebuild only six affected USTC sources.
Verify the exact leading defect from the original. Never mutate old indexes.
The14 unaffected source outputs are read-only links, not recomputed samples.
"""
import concurrent.futures,json,os,shutil,sqlite3,struct,subprocess,sys,time
from pathlib import Path
import packet_rebuild as r

def prepare_indexes():
 old=r.HERE.parent/'R1_5_remaining_datasets_rebuild_v1';allplan=json.loads((old/'USTC_SOURCE_PLAN.json').read_text());affected=json.loads((r.HERE/'USTC_SOURCE_PLAN.json').read_text());root=r.HERE/'outputs/USTC';root.mkdir(parents=True,exist_ok=True);evidence=[]
 for family,files in allplan.items():
  source=old/'outputs/USTC'/family;dest=root/family
  if family not in affected:
   if not dest.exists():dest.symlink_to(source.resolve(),target_is_directory=True)
   continue
  if (dest/'SCAN_COMPLETE.json').exists():continue
  scan=json.loads((source/'SCAN_COMPLETE.json').read_text());events=[json.loads(line) for line in (source/'packet_rejections.jsonl').open() if 'truncated_ethernet' in line]
  if len(files)!=1 or scan['global_errors']!=['truncated_ethernet'] or len(events)!=1 or events[0]['packet_index']!=0:raise RuntimeError('Unspecified defect shape: '+family)
  raw=Path(files[0]['path'])
  if r.sha(raw)!=files[0]['sha256']:raise RuntimeError('Original changed')
  with raw.open('rb') as f:
   hdr=f.read(24);endian='<' if hdr[:4]==b'\xd4\xc3\xb2\xa1' else '>';sec,frac,cap,wire=struct.unpack(endian+'IIII',f.read(16));data=f.read(cap)
   if cap!=4 or wire!=60 or data!=bytes(4):raise RuntimeError('Boundary differs')
  dest.mkdir(exist_ok=True);srcdb=sqlite3.connect('file:'+str((source/'packets.sqlite').resolve())+'?mode=ro',uri=True);dstdb=sqlite3.connect(dest/'packets.sqlite');srcdb.backup(dstdb);srcdb.close();dstdb.close();shutil.copy2(source/'packet_rejections.jsonl',dest/'packet_rejections.jsonl');scan['global_errors']=[];scan['leading_boundaries']=[dict(file_id=files[0]['file_id'],packet_index=0,ts=sec+frac/1e6,caplen=4,wirelen=60,available_records_begin_byte=44,reason='unassignable leading observation; not restored')];scan['reused_complete_index']=dict(source=str(source/'packets.sqlite'),read_only_backup=True,reason='identical decoder and all valid packet records; only boundary-scope policy amended');r.write(dest/'SCAN_COMPLETE.json',scan);evidence.append(dict(family=family,original_sha256=files[0]['sha256'],boundary=scan['leading_boundaries'][0],old_report=str(source/'REBUILD_COMPLETE.json')))
 r.write(r.HERE/'BOUNDARY_EVIDENCE.json',dict(affected=evidence,unaffected_sources_reused=14,old_outputs_unchanged=True))
def task(family):
 log=r.HERE/'execution';log.mkdir(exist_ok=True)
 with (log/(family+'.log')).open('w') as f:
  cmd=[sys.executable,str(r.HERE/'packet_rebuild.py'),'--dataset','USTC','--data-root',str(r.HERE.parents[2]/'data'),'--family',family];p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});return family,p.wait()
def main():
 import fcntl,hashlib
 lock=(r.HERE/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,h in json.loads((r.HERE/'PROTOCOL_FROZEN.json').read_text())['code_sha256'].items():
  if r.sha(Path(name))!=h:raise RuntimeError('Boundary protocol changed')
 prepare_indexes();families=json.loads((r.HERE/'USTC_SOURCE_PLAN.json').read_text());finished={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  active={pool.submit(task,k):k for k in families}
  while active:
   done,_=concurrent.futures.wait(active,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    family,ret=f.result();finished[family]=ret;active.pop(f)
   r.write(r.HERE/'STATUS.json',dict(stage='boundary_rebuild',active_or_queued=list(active.values()),finished=finished,time=time.time(),training_started=False))
 if any(finished.values()):return
 # Reuse the frozen common role algorithm with explicit new inputs/output path.
 role=r.HERE.parent/'R1_5_task_role_partitions_v1';cmd=[sys.executable,str(r.HERE/'audit_roles.py'),'--source-root',str(r.HERE/'outputs/USTC'),'--output',str(r.HERE/'partitions/USTC')]
 with (r.HERE/'execution/partition.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 r.write(r.HERE/'STATUS.json',dict(stage='ROLE_AUDIT_FINISHED',returncode=p.returncode,report=str(r.HERE/'partitions/USTC/PARTITION_GATE.json'),time=time.time(),training_started=False))
if __name__=='__main__':main()

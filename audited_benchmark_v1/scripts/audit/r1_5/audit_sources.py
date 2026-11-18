"""Exact CSV tuple-reuse audit on declared source directories, streaming into SQLite.
No timestamps inferred; raw timestamps retained as strings. No model data modified.
"""
import argparse,csv,hashlib,json,sqlite3,time
from pathlib import Path

def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def norm(s):return ''.join(s.lower().split()).replace('_','')
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--datasets',nargs='+',default=['CIC_IDS','CIC_DOH','CIC_DDOS']);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 sources={'CIC_IDS':['CIC_IDS/GeneratedLabelledFlows/TrafficLabelling'],'CIC_DOH':['DoHBrw-2020/Clean_CSVs'],'CIC_DDOS':['CIC-DDoS-2019/Test_Day_03-11/CSVs/03-11','CIC-DDoS-2019/Train_Day_01-12/CSVs/01-12']}
 for dataset in a.datasets:
  target=a.output/dataset;target.mkdir(exist_ok=True)
  if (target/'COMPLETE.json').exists():continue
  db=sqlite3.connect(target/'tuple_records.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS records(file TEXT,rowid_source INTEGER,k TEXT,k_no_proto TEXT,ts TEXT,label TEXT,PRIMARY KEY(file,rowid_source))');db.execute('CREATE TABLE IF NOT EXISTS done(file TEXT PRIMARY KEY,rows INTEGER)');db.commit()
  log=[];total=0
  for directory in sources[dataset]:
   files=sorted((a.data_root/directory).glob('*.csv'))
   if not files:log.append({'directory':directory,'error':'NO_FILES'})
   for f in files:
    rel=str(f.relative_to(a.data_root));prior=db.execute('SELECT rows FROM done WHERE file=?',(rel,)).fetchone()
    if prior:total+=prior[0];continue
    n=invalid=0;batch=[]
    with f.open(encoding='cp1252',errors='replace',newline='') as h:
     reader=csv.reader(h);cols=next(reader);lookup={norm(c):i for i,c in enumerate(cols)}
     def col(*names):return next((lookup[n] for n in names if n in lookup),None)
     idx=[col('sourceip','srcip'),col('sourceport','srcport'),col('destinationip','dstip'),col('destinationport','dstport')];pi=col('protocol');ti=col('timestamp');li=col('label')
     if any(i is None for i in idx):log.append({'file':rel,'error':'MISSING_ENDPOINT_FIELDS','columns':cols});continue
     # Reruns replace partial file to avoid stale fragments.
     db.execute('DELETE FROM records WHERE file=?',(rel,));db.commit()
     for number,row in enumerate(reader,2):
      try:
       if max(idx)>=len(row):raise ValueError('short row')
       ends=sorted([(row[idx[0]].strip(),str(int(float(row[idx[1]])))),(row[idx[2]].strip(),str(int(float(row[idx[3]]))))])
       proto=row[pi].strip() if pi is not None and pi<len(row) else 'UNKNOWN'
       if proto!='UNKNOWN':proto=str(int(float(proto)))
       base=json.dumps(ends,separators=(',',':'));key=base+'|'+proto
       stamp=row[ti].strip() if ti is not None and ti<len(row) else ''
       label=row[li].strip() if li is not None and li<len(row) else 'NO_LABEL_COLUMN'
       batch.append((rel,number,key,base,stamp,label));n+=1
      except (ValueError,IndexError):invalid+=1
      if len(batch)>=20000:
       db.executemany('INSERT OR REPLACE INTO records VALUES(?,?,?,?,?,?)',batch);db.commit();batch=[]
       save(a.output/'STATUS.json',dict(stage='csv_tuple_audit',dataset=dataset,file=rel,file_rows=n,dataset_previous_rows=total,time=time.time()))
     if batch:db.executemany('INSERT OR REPLACE INTO records VALUES(?,?,?,?,?,?)',batch)
    db.execute('INSERT OR REPLACE INTO done VALUES(?,?)',(rel,n));db.commit();total+=n
    log.append({'file':rel,'valid_rows':n,'invalid_rows':invalid,'protocol_column':pi is not None,'timestamp_column':ti is not None,'label_column':li is not None})
    print(dataset,rel,n,'invalid',invalid,flush=True)
  save(target/'FILES_AUDIT.json',log)
  save(a.output/'STATUS.json',dict(stage='aggregating',dataset=dataset,rows=total,time=time.time()))
  db.execute('CREATE INDEX IF NOT EXISTS key_idx ON records(k)');db.commit()
  groups=db.execute('SELECT count(*),sum(n-1),sum(n>1),sum(nt>1),sum(nl>1),sum(nf>1) FROM (SELECT k,count(*) n,count(DISTINCT ts) nt,count(DISTINCT label) nl,count(DISTINCT file) nf FROM records GROUP BY k)').fetchone()
  rows=db.execute('SELECT count(*) FROM records').fetchone()[0]
  report={'dataset':dataset,'rows':rows,'unique_bidirectional_tuple_keys':groups[0],'excess_rows_over_unique_keys':groups[1] or 0,'repeated_tuple_keys':groups[2] or 0,'keys_with_distinct_timestamp_strings':groups[3] or 0,'keys_with_distinct_original_labels':groups[4] or 0,'keys_across_csv_files':groups[5] or 0,'limitations':['CSV keys only, not PCAP matching results','Timestamp strings not parsed; no session truth inference','Distinct labels may be aliases; not necessarily true conflicts','Protocol UNKNOWN where unavailable; DoH legacy key omits protocol','No association to current PKL or historical partitions established'],'source_directories':sources[dataset]}
  with (target/'repeated_key_examples.csv').open('w',newline='') as h:
   w=csv.writer(h);w.writerow(['key_sha256','rows','timestamp_variants','label_variants','source_files'])
   for k,n,nt,nl,nf in db.execute('SELECT k,count(*),count(DISTINCT ts),count(DISTINCT label),count(DISTINCT file) FROM records GROUP BY k HAVING count(*)>1 LIMIT 100'):w.writerow([hashlib.sha256(k.encode()).hexdigest(),n,nt,nl,nf])
  if dataset=='CIC_DOH':
   q=db.execute('SELECT count(*),sum(n-1),sum(np>1) FROM (SELECT k_no_proto,count(*) n,count(DISTINCT k) np FROM records GROUP BY k_no_proto)').fetchone();report['legacy_no_protocol_unique_keys']=q[0];report['legacy_no_protocol_excess_rows']=q[1] or 0;report['endpoint_keys_with_multiple_protocols']=q[2] or 0
  save(target/'COMPLETE.json',report);db.close();print(json.dumps(report),flush=True)
 save(a.output/'STATUS.json',dict(stage='complete_requested_csv_scans',datasets=a.datasets,time=time.time()))
if __name__=='__main__':main()

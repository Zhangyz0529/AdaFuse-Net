"""Exact raw-statistic candidate links only; does not certify sequence provenance."""
import argparse,collections,csv,gc,hashlib,json,pickle
from pathlib import Path
import numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--pkl',type=Path,required=True);p.add_argument('--csv-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 with a.pkl.open('rb') as f:data=pickle.load(f)
 wanted={};rows=[]
 for i,r in enumerate(data):
  stat=np.asarray(r['stat_raw'],dtype=np.float32);key=hashlib.sha256(stat.tobytes()).digest();wanted.setdefault(key,{'count':0,'first':None,'labels':set(),'shape':stat.shape});rows.append((i,key,str(r['label'])))
 meta=['SourceIP','DestinationIP','SourcePort','DestinationPort','TimeStamp','Label','DoH'];audit=[]
 for f in sorted(a.csv_dir.glob('*.csv')):
  name=f.name.lower();label='Benign' if any(v in name for v in ['chrome','firefox','benign']) else 'dns2tcp' if 'tcp' in name else 'dnscat2' if 'cat' in name else 'iodine' if 'iodine' in name else 'Benign'
  offset=0;matched=0
  for chunk in pd.read_csv(f,chunksize=50000,low_memory=False):
   chunk['Label']=label;features=[c for c in chunk.columns if c not in meta]
   values=chunk[features].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(dtype=np.float32)
   for j,stat in enumerate(values):
    key=hashlib.sha256(stat.tobytes()).digest()
    if key in wanted and wanted[key]['shape']==stat.shape:
     v=wanted[key];v['count']+=1;v['labels'].add(label);matched+=1
     if v['first'] is None:v['first']=(str(f.resolve()),offset+j+2)
   offset+=len(chunk)
  audit.append({'file':str(f),'rows':offset,'feature_dim':len(features),'candidate_rows_matching_any_pkl_stat':matched});print(audit[-1],flush=True)
 counts=collections.Counter();mismatch=0
 with (a.output/'CIC_DOH_feature_candidate_links.csv').open('w',newline='') as h:
  w=csv.writer(h);w.writerow(['pkl_row','pkl_label','matching_csv_rows','csv_file_if_unique','csv_row_if_unique','candidate_injected_labels','status'])
  for i,key,label in rows:
   v=wanted[key];n=v['count'];status='NO_EXACT_STAT_MATCH' if not n else 'UNIQUE_STAT_MATCH_SEQUENCE_UNVERIFIED' if n==1 else 'AMBIGUOUS_STAT_MATCH_SEQUENCE_UNVERIFIED';counts[status]+=1
   mismatch+=bool(n and label not in v['labels']);f,r=v['first'] if n==1 else ('','')
   w.writerow([i,label,n,f,r,';'.join(sorted(v['labels'])),status])
 report={'pkl':str(a.pkl),'n_pkl_rows':len(rows),'status_counts':dict(counts),'rows_with_no_candidate_label_equal_to_pkl':mismatch,'files':audit,'limitations':['Raw statistics only; bytes and packet positions not matched','File labels reproduce candidate generator filename rule, not independent label certification','Even unique feature match is not a verified session/source mapping','No session_id generated, no split approved']}
 (a.output/'CIC_DOH_feature_links_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()

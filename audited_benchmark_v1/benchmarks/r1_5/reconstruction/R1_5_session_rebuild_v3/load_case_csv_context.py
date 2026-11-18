"""Read all raw CSV candidates sharing fixed-case keys; no class/top-k filtering."""
import argparse,collections,json,sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2';sys.path.insert(0,str(V2));from rebuild import H
from rules_v3 import DecisionLog,normalize_row,stable

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();cases=json.loads((HERE/'diagnostics/FIXED_CASES.json').read_text())['cases'];registry=list(__import__('csv').DictReader((HERE/'source_domain_registry.csv').open()));domainmap={r['acquisition_family']:r['provenance_domain_id'] for r in registry if r['dataset']=='CIC_IDS'}
 for domain in ['Friday','Tuesday','Wednesday']:
  keys={H['csv_key'](c['csv_record']) for c in cases if c['domain']==domain};files=sorted((a.data_root/'CIC_IDS/GeneratedLabelledFlows/TrafficLabelling').glob(domain+'*.csv'));context=collections.defaultdict(list);n=invalid_key=0;log=DecisionLog(HERE/f'diagnostics/{domain}/CSV_CONTEXT_LOG.jsonl')
  names=['Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Label']
  for f in files:
   header=pd.read_csv(f,encoding='cp1252',nrows=0).columns;mapping={c.strip():c for c in header};sourceid=stable(['csv_file_v3','CIC_IDS',str(f.relative_to(a.data_root))]);offset=0
   for frame in pd.read_csv(f,encoding='cp1252',usecols=[mapping[c] for c in names],chunksize=50000,low_memory=False):
    frame.columns=frame.columns.str.strip()
    for j,ends in enumerate(frame[['Source IP','Source Port','Destination IP','Destination Port','Protocol']].itertuples(index=False,name=None)):
     n+=1
     try:key=H['csv_key'](dict(zip(['Source IP','Source Port','Destination IP','Destination Port','Protocol'],ends)))
     except (TypeError,ValueError,OSError):invalid_key+=1;continue
     if key not in keys:continue
     raw=frame.iloc[j].to_dict();raw['csv_path']=str(f);raw['csv_row']=offset+j+2;raw['provenance_domain_id']=domainmap[domain];raw['source_file_id']=sourceid;raw['label']=str(raw['Label']).strip();row=normalize_row(raw,sourceid,raw['csv_row'],raw['label'],log)
     context[stable(key)].append(row)
    offset+=len(frame)
  log.close();out=HERE/'diagnostics'/domain;(out/'CSV_CONTEXT.json').write_text(json.dumps(dict(complete=True,csv_rows_scanned=n,invalid_endpoint_rows_not_assignable_to_any_key=invalid_key,keys_selected=len(keys),context=context,scope='all CSV records for fixed keys, all labels, no key overflow truncation'),indent=2));print(domain,n,'context rows',sum(len(r) for r in context.values()),flush=True)
if __name__=='__main__':main()

"""Inspect trusted local model PKLs, hash files and input contents; no training."""
import argparse,collections,gc,hashlib,json,pickle,time
from pathlib import Path
import numpy as np

def filehash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 names={'USTC':'USTC/multimodal_dataset/ustc_multimodal_512.pkl','ISCX':'ISCX/multimodal_dataset/iscx_multimodal_512.pkl','CIC_DOH':'CIC_DOH/multimodal_dataset/doh2020_multimodal_512.pkl','CIC_DDOS':'CICDDoS/multimodal_dataset/cicddos_multimodal_512.pkl','CIC_IDS':'CICIDS/multimodal_dataset/cicids_multimodal_512.pkl'}
 for name,rel in names.items():
  path=a.workspace_root/rel;out=a.output/(name+'_pkl_identity.json')
  if out.exists():continue
  report={'dataset':name,'path':str(path),'sha256':filehash(path),'bytes':path.stat().st_size,'scope':'registered 512-byte PKL; identity to historical published table not yet proven'}
  with path.open('rb') as f:data=pickle.load(f)
  report['container']=type(data).__name__
  if not isinstance(data,list):report['status']='UNSUPPORTED_CONTAINER'
  else:
   labels=collections.Counter();keys=collections.Counter();shapes=collections.Counter();digests={};duplicates=conflicts=invalid=0;metadata=collections.Counter()
   for i,r in enumerate(data):
    if not isinstance(r,dict) or not all(k in r for k in ['stat_raw','seq_feature','label']):invalid+=1;continue
    labels[str(r['label'])]+=1;keys[tuple(sorted(r))]+=1
    for k in r:
     if k not in ['stat_raw','seq_feature','label']:metadata[k]+=1
    stat=np.asarray(r['stat_raw'],dtype='float32');seq=np.asarray(r['seq_feature']);shapes[(str(stat.shape),str(seq.shape),str(seq.dtype))]+=1
    h=hashlib.sha256(str(stat.shape).encode()+stat.tobytes()+str(seq.shape).encode()+str(seq.dtype).encode()+seq.tobytes()).digest()
    if h in digests:
     duplicates+=1;conflicts+=digests[h]!=str(r['label'])
    else:digests[h]=str(r['label'])
   report.update(n_records=len(data),labels=dict(labels),schema_counts=[{'keys':list(k),'n':v} for k,v in keys.items()],shape_counts=[{'stat':k[0],'seq':k[1],'seq_dtype':k[2],'n':v} for k,v in shapes.items()],extra_field_coverage=dict(metadata),duplicate_model_input_rows=duplicates,duplicate_inputs_with_label_different_from_first=conflicts,invalid_schema_rows=invalid,status='INSPECTED_NO_SPLIT_AUDIT',limitations=['Input duplication is not proof of cross-partition leakage','No source mapping inferred from row index','Model statistics cast to float32 before content hash'])
  out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(name,report.get('n_records'),report.get('duplicate_model_input_rows'),flush=True)
  del data;gc.collect()
if __name__=='__main__':main()

"""Group-isolated splits for verified source-linked samples; never infers session truth.
Run separately for each dataset. Task JSON explicitly preserves held-out label roles.
NPZ rows must match sample_id and contain stat_raw, seq_feature, sample_ids.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib

class Components:
    def __init__(self,n): self.p=list(range(n))
    def find(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def union(self,a,b):self.p[self.find(a)]=self.find(b)

def digest_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(a):
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use an empty output directory; no overwrites')
    a.output.mkdir(parents=True,exist_ok=True)
    report={'status':'INCOMPLETE','scope':'declared source groups and exact provided input content; not universal session-independence proof','selection_audit':'NOT_EXECUTED: no model selection/training has run','configuration':vars(a).copy()}
    report['configuration']={k:str(v) if isinstance(v,Path) else v for k,v in report['configuration'].items()}
    try:
        evidence=json.loads(a.evidence.read_text())
        if not evidence.get('provenance_join_verified') or not evidence.get('group_definition') or not evidence.get('evidence_files'):raise ValueError('Verified provenance join and documented group evidence required')
        # Evidence assertion is traceable, not independently authenticated by this script.
        evidence_hashes={}
        for name in evidence['evidence_files']:
            p=(a.evidence.parent/name).resolve()
            if not p.is_file():raise ValueError('Missing provenance evidence '+str(p))
            evidence_hashes[str(p)]=sha(p)
        d=pd.read_csv(a.samples,dtype=str,keep_default_na=False)
        required=['dataset','sample_id','session_id','source_file_id','record_locator','label','source_group_id']
        if any(k not in d for k in required):raise ValueError('Required columns: '+','.join(required))
        if (d[required]=='').any().any() or d.sample_id.duplicated().any() or d.dataset.nunique()!=1:raise ValueError('Missing identities, duplicate sample IDs or multiple datasets')
        z=np.load(a.inputs,allow_pickle=False);stats=z['stat_raw'];seq=z['seq_feature']
        if not np.array_equal(z['sample_ids'].astype(str),d.sample_id.to_numpy()):raise ValueError('NPZ sample_ids do not exactly match manifest order')
        if stats.ndim!=2 or len(stats)!=len(d) or len(seq)!=len(d) or not np.isfinite(stats).all():raise ValueError('Invalid input dimensions/nonfinite statistics')
        # Canonical float32 is the actual model statistics representation in this pipeline.
        stats=stats.astype('float32')
        if not np.isfinite(stats).all():raise ValueError('float32 overflow')
        d['input_hash']=[hashlib.sha256((digest_array(x)+digest_array(y)).encode()).hexdigest() for x,y in zip(stats,seq)]
        if 'raw_record_hash' not in d:report['raw_content_audit']='NOT_AVAILABLE: supplied model inputs only'
        else:
            if (d.raw_record_hash=='').any():raise ValueError('Partial raw-record hash coverage')
            report['raw_content_audit']='provided upstream raw-record hashes, provenance-dependent'
        # Merge source groups, declared sessions AND duplicate content before splitting.
        comp=Components(len(d));keys=['session_id','source_group_id','input_hash']+(['raw_record_hash'] if 'raw_record_hash' in d else [])
        for k in keys:
            first={}
            for i,v in enumerate(d[k]):
                if v in first:comp.union(i,first[v])
                else:first[v]=i
        d['split_group_id']=[str(comp.find(i)) for i in range(len(d))]
        report['before']={'samples':len(d),'sessions':d.session_id.nunique(),'source_groups':d.source_group_id.nunique(),'merged_groups':d.split_group_id.nunique(),'duplicate_model_input_rows':int(d.input_hash.duplicated().sum())}
        ratios=np.array(a.ratios,dtype=float)
        if len(ratios)!=3 or (ratios<=0).any() or not np.isclose(ratios.sum(),1):raise ValueError('Ratios must be positive Train Val Test fractions summing to 1')
        idx=np.arange(len(d));train,rest=next(GroupShuffleSplit(n_splits=1,test_size=float(ratios[1:].sum()),random_state=a.seed).split(idx,groups=d.split_group_id))
        vi,ti=next(GroupShuffleSplit(n_splits=1,test_size=float(ratios[2]/ratios[1:].sum()),random_state=a.seed+1).split(rest,groups=d.iloc[rest].split_group_id))
        d['partition']='Train';d.loc[rest[vi],'partition']='Val';d.loc[rest[ti],'partition']='Test'
        tasks=json.loads(a.tasks.read_text());manifests=[];exclusions=[];audits=[]
        for fold,roles in tasks.items():
            if set(roles)!=set(['Train','Val','Test']):raise ValueError('Explicit label lists for Train Val Test required')
            keep=np.array([label in roles[part] for label,part in zip(d.label,d.partition)])
            m=d[keep].copy();m['outer_fold']=fold
            excluded=d[~keep].copy();excluded['outer_fold']=fold;excluded['reason']='label_not_allowed_in_assigned_partition';exclusions.append(excluded)
            for part,labels in roles.items():
                for label in labels:
                    if int(((m.partition==part)&(m.label==label)).sum())<a.min_per_class:raise ValueError(f'Insufficient {fold}/{part}/{label}; do not retry seed based on performance')
            pairs=[]
            for x,y in [('Train','Val'),('Train','Test'),('Val','Test')]:
                q={k:len(set(m.loc[m.partition==x,k])&set(m.loc[m.partition==y,k])) for k in ['sample_id',*keys,'split_group_id']}
                if any(q.values()):raise ValueError('Cross-partition overlap '+str(q))
                pairs.append({'partitions':[x,y],'intersection_counts':q})
            trainids=m.index[m.partition=='Train'].to_numpy();scaler=StandardScaler().fit(stats[trainids]);out=a.output/str(fold)
            if out.parent.resolve()!=a.output.resolve():raise ValueError('Unsafe fold name')
            out.mkdir();joblib.dump(scaler,out/'scaler.joblib')
            pd.DataFrame({'sample_id':d.loc[trainids,'sample_id']}).to_csv(out/'scaler_fit_sample_ids.csv',index=False)
            # Transform uses the same frozen scaler; no validation/test fit.
            transformed=scaler.transform(stats[m.index]).astype('float32')
            scaled_hashes=[hashlib.sha256((digest_array(v)+digest_array(seq[i])).encode()).hexdigest() for v,i in zip(transformed,m.index)]
            hash_part={}
            for h,part in zip(scaled_hashes,m.partition):
                if h in hash_part and hash_part[h]!=part:raise ValueError('Content collision after scaler: revise grouping and refreeze, do not train')
                hash_part[h]=part
            np.savez_compressed(out/'scaled_statistics.npz',sample_ids=m.sample_id.to_numpy(dtype=str),stat=transformed)
            m['pool']='unassigned: episode support/query/calibration plans still required'
            manifests.append(m);audits.append({'fold':fold,'counts':[{'partition':p,'label':l,'n':int(n)} for (p,l),n in m.groupby(['partition','label']).size().items()],'overlaps':pairs,'scaler_fit_n':len(trainids)})
        pd.concat(manifests).to_csv(a.output/'split_manifest.csv',index=False);pd.concat(exclusions).to_csv(a.output/'excluded_samples.csv',index=False)
        d.to_csv(a.output/'base_group_assignment.csv',index=False)
        report.update(status='PASS_DECLARED_GROUPS_AND_MODEL_CONTENT_ONLY',folds=audits,provenance_evidence=evidence,evidence_sha256=evidence_hashes,input_sha256={str(p):sha(p) for p in [a.samples,a.inputs,a.evidence,a.tasks]},limitations=['No independent authentication of upstream provenance','No packet-time interval or TCP boundary reconstruction in this stage','Raw content only checked when supplied','Episode isolation and future model-selection logs require separate audit','GroupShuffleSplit proportions target group counts, not exact sample counts'])
    except Exception as e:
        report.update(status='FAILED_DO_NOT_TRAIN',error=str(e));raise
    finally:
        (a.output/'leakage_audit_report.txt').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ratios',type=float,nargs=3,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--min-per-class',type=int,required=True);run(p.parse_args())

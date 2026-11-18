"""Unified tables, exact identities, no approximate/historical baseline mixing."""
import json
from pathlib import Path
import pandas as pd
import argparse
p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();R=a.artifacts;frames=[];counts=[]
for base,version in [(R,'audited_benchmark_v1')]:
 for marker in (base/'runs').glob('*/*/seed*/round*/shot*/COMPLETE.json'):
  f=pd.read_csv(marker.parent/'episode_metrics.csv')
  if 'implementation_version' not in f:f['implementation_version']=version
  frames.append(f);parts=marker.relative_to(base/'runs').parts;counts.append(dict(dataset=parts[0],model=parts[1],seed=parts[2],fold=parts[3],shot=parts[4]))
if not frames:raise SystemExit('No formal completed jobs')
out=R/'results'/'recomputed_from_snapshot';out.mkdir(parents=True,exist_ok=True);ep=pd.concat(frames,ignore_index=True);fields=['MacroF1','AttackRecall','FPR'];keys=['dataset','model','implementation_version','seed','fold','shot','role','condition'];task=ep.groupby(keys)[fields].mean().reset_index();task.to_csv(out/'per_task_seed_condition.csv',index=False);seed=task.groupby([k for k in keys if k!='fold'])[fields].mean().reset_index();seed.to_csv(out/'per_seed_condition.csv',index=False);seed.groupby(['dataset','model','implementation_version','shot','role','condition'])[fields].agg(['mean','std','count']).to_csv(out/'mean_sampleSD.csv')
pivot=task[task.role=='Test'].pivot(index=['dataset','model','implementation_version','seed','fold','shot'],columns='condition',values='MacroF1').reset_index()
if 'IBP30' in pivot:pivot['Clean_minus_IBP30']=pivot['Clean']-pivot['IBP30']
pivot.to_csv(out/'absolute_F1_and_drop.csv',index=False);pd.DataFrame(counts).groupby(['dataset','model']).size().rename('complete_jobs').to_csv(out/'job_completion.csv');(out/'SCOPE.json').write_text(json.dumps(dict(partial_results=True,unit='F1 in0..1; multiply100 for percent/pp',aggregation='episodes->task->equal tasks per seed->seed mean/sampleSD',FlowPic_MetaTraffic_IBP30='not applicable to native size/time input; missing is intentional, not zero drop evidence',RF_seed_unit='estimator repeat, not independent pretrained training run',comparison='audited common evaluation partitions; documented native/fewshot adaptations; not exact author table reproduction'),indent=2))

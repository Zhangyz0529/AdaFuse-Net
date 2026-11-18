"""Task-equal aggregation then independent-training-seed sample SD."""
from pathlib import Path
import argparse
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
f=pd.concat([pd.read_csv(x) for x in a.inputs],ignore_index=True);key=['dataset','model','training_seed','shot','round','task','episode','condition','rho']
if f.duplicated(key).any():raise ValueError('Duplicate episode records; do not double count exports')
fields=['MacroF1','AttackRecall','FPR','Precision','AUROC','AUPRC'];taskkey=['dataset','model','training_seed','shot','round','task','condition','rho'];task=f.groupby(taskkey)[fields].mean().reset_index();seedkey=['dataset','model','training_seed','shot','condition','rho'];perseed=task.groupby(seedkey)[fields].mean().reset_index();summary=perseed.groupby(['dataset','model','shot','condition','rho'])[fields].agg(['mean','std','count']);summary.columns=['_'.join(c) for c in summary.columns]
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);task.to_csv(out/'per_task.csv',index=False);perseed.to_csv(out/'per_seed.csv',index=False);summary.reset_index().to_csv(out/'mean_sampleSD_count.csv',index=False)
# Separate cumulative prediction-occurrence counts from task-mean rates.
f.groupby(seedkey)[['TP','FP','TN','FN']].sum().reset_index().to_csv(out/'prediction_occurrence_totals.csv',index=False)

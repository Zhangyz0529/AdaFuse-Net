"""Recompute all tables from completed jobs only, preserving repeated-run units."""
from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parent
frames=[]
for p in (R/'runs').glob('*/*/seed*/round*/shot*/COMPLETE.json'):
 f=pd.read_csv(p.parent/'episode_metrics.csv');frames.append(f)
if not frames:raise SystemExit('No completed formal jobs; no fabricated summary')
ep=pd.concat(frames,ignore_index=True);out=R/'tables';out.mkdir(exist_ok=True)
fields=['MacroF1','AttackRecall','FPR'];keys=['dataset','model','seed','fold','shot','role','condition'];task=ep.groupby(keys)[fields].mean().reset_index();task.to_csv(out/'per_task_seed_condition.csv',index=False)
seed=task.groupby([x for x in keys if x!='fold'])[fields].mean().reset_index();seed.to_csv(out/'per_seed_condition.csv',index=False)
summary=seed.groupby(['dataset','model','shot','role','condition'])[fields].agg(['mean','std','count']);summary.to_csv(out/'mean_sampleSD.csv')
test=task[task.role=='Test'];drop=test.pivot(index=['dataset','model','seed','fold','shot'],columns='condition',values='MacroF1').reset_index();drop['Clean_minus_IBP30']=drop['Clean']-drop['IBP30'];drop.to_csv(out/'absolute_F1_and_drop.csv',index=False)
# Only compare exact matched task/seed/shot pairs. Incomplete datasets remain partial.
pair=test.pivot(index=['dataset','seed','fold','shot','condition'],columns='model',values='MacroF1').reset_index()
if {'AdaFuse-Net','RandomForest'}.issubset(pair.columns):
 pair=pair.dropna(subset=['AdaFuse-Net','RandomForest']);pair['AdaFuse_minus_RF_pp']=100*(pair['AdaFuse-Net']-pair['RandomForest']);pair.to_csv(out/'paired_task_differences.csv',index=False)
status=json.loads((R/'QUEUE_STATUS.json').read_text()) if (R/'QUEUE_STATUS.json').exists() else {}
(out/'COMPLETENESS.json').write_text(json.dumps({'complete_jobs':len(frames),'partial_results':True,'expected_repeats':5,'remaining_queue':status.get('pending_ready_jobs'),'caution':'Check all folds/shots/seeds complete before any overall ranking; RF seeds are estimator repeats, not independent pretrained runs.'},indent=2))

"""Freeze score-independent diagnosis/holdout cases and v2 status; no training."""
from pathlib import Path
import json,hashlib,collections,csv,datetime
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(r):return hashlib.sha256(('v3_fixed_diagnostics_v1|'+Path(r['csv_path']).name+'|'+str(r['csv_row'])).encode()).hexdigest()
def main():
 out=HERE/'diagnostics';out.mkdir(exist_ok=True);dest=out/'FIXED_CASES.json'
 if dest.exists():raise RuntimeError('Fixed selection already exists; do not silently resample')
 rows=[];counts=collections.Counter();reasons=collections.Counter();markers={}
 for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
  p=V2/'reconstructed/CIC_IDS'/day;targets=json.loads((p/'targets.json').read_text());lookup={(r['csv_path'],r['csv_row']):r for r in targets};rejected=json.loads((p/'rejected.json').read_text());accepted=json.loads((p/'records.json').read_text());done=json.loads((p/'COMPLETE.json').read_text());sel=json.loads((p/'csv_selection_audit.json').read_text());markers[day]={str(f.relative_to(V2)):sha(f) for f in [p/'COMPLETE.json',p/'clock_audit.json',p/'csv_selection_audit.json']}
  for key,n in [('selected',sel['selected_before_eligibility']),('ineligible',sel['invalid_selected']),('targets',sel['target_rows']),('accepted',done['accepted']),('rejected',done['rejected'])]:counts[key]+=n
  for r in rejected:reasons[r['reason']]+=1
  groups=[]
  if day=='Friday':groups=[('DDoS','no_consistent_packet_window')]
  if day=='Tuesday':groups=[('FTP-Patator','CSV_window_spans_multiple_observed_sessions')]
  if day=='Wednesday':groups=[('DoS','no_consistent_packet_window'),('DoS','CSV_window_spans_multiple_observed_sessions')]
  for label,reason in groups:
   candidates=[lookup[r['csv_path'],r['csv_row']] for r in rejected if r['reason']==reason and lookup[r['csv_path'],r['csv_row']]['label']==label];candidates.sort(key=rank)
   for i,r in enumerate(candidates[:12]):rows.append(dict(case_id=rank(r),domain=day,role='diagnosis' if i<8 else 'heldout_rule_check',v2_status='rejected',v2_reason=reason,full_stratum_n=len(candidates),csv_record=r))
  if day in ['Friday','Tuesday','Wednesday']:
   candidates=[lookup[str(Path(r['record_locator'].rsplit(':',1)[0])),int(r['record_locator'].rsplit(':',1)[1])] for r in accepted];candidates.sort(key=rank)
   for i,r in enumerate(candidates[:6]):rows.append(dict(case_id=rank(r),domain=day,role='diagnosis' if i<4 else 'heldout_rule_check',v2_status='accepted',v2_reason=None,full_stratum_n=len(accepted),csv_record=r))
 assert counts['selected']==counts['ineligible']+counts['targets'] and counts['targets']==counts['accepted']+counts['rejected']
 payload=dict(selection='Fixed SHA256 filename/CSV-row order, no model scores. First8 reject/4 accept for diagnosis; next4 reject/2 accept held out. Counts reflect source strata, not acceptance targets.',training_or_test_metrics_used=False,cases=rows,v2_evidence_hashes=markers,v2_counts=dict(counts),v2_rejection_counts=dict(reasons))
 dest.write_text(json.dumps(payload,indent=2));(out/'FIXED_CASES.sha256').write_text(sha(dest)+'\n')
 status=f'''# V2_STATUS（冻结）

CIC-IDS试运行：50000预选＝17954筛选未通过＋32046有效候选；32046＝16799接受＋15247匹配/会话拒绝。逐域和逐原因数量已核对闭合。v2接受记录不能作为最终训练数据。

4996条DDoS无窗口是**CIC-IDS Friday的DDoS类别**，不是CIC-DDoS数据集。CIC-IDS划分因round0/Val/DDoS样本不足停止。

CIC-DDoS数据集因Test_Day_03-11/PCAPs/SAT-03-11-2018_0145.pcap末尾截断而停止，未完成接受/拒绝统计；Train_Day_01-12的SAT-01-12-2018_0818.pcap也已独立检出截断。不得将未完成部分记0。

未启动正式Optuna或正式训练。GPU冒烟仅为临时合成数据，不是模型实验结果。v2不是正式审核数据，两个任务均停止，不自动恢复训练。

未解决：标签窗口匹配失败、FIN尾ACK分段、capture来源域未纳入session ID、3至10000包依据不足、资源overflow与逐条拒绝日志不足。v2脚本及PROTOCOL_FROZEN保留；诊断不修改这些代码。

冻结时间：{datetime.datetime.now().astimezone().isoformat()}。证据及固定样例：../R1_5_session_rebuild_v3/diagnostics/FIXED_CASES.json。
'''
 (V2/'V2_STATUS.md').write_text(status);(out/'V2_STATUS.md').write_text(status)
 (out/'V2_STATUS.sha256').write_text(sha(out/'V2_STATUS.md')+'\n')
 print('fixed cases',len(rows),'strata',dict(collections.Counter((r['domain'],r['v2_reason'] or 'accepted') for r in rows)))
if __name__=='__main__':main()

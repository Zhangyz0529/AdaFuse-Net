"""Seal rules and diagnosis-only choices before held-out checks. No training."""
from pathlib import Path
import datetime,hashlib,json,os
from rules_v3 import RULES
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'R1_5_session_rebuild_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 dest=HERE/'V3_RULES_FROZEN.json'
 if dest.exists():raise RuntimeError('Already frozen: new changes require a new version')
 assert sha(HERE/'diagnostics/FIXED_CASES.json')==(HERE/'diagnostics/FIXED_CASES.sha256').read_text().strip()
 assert json.loads((HERE/'diagnostics/BOUNDARY_TESTS.json').read_text())['passed']
 assert json.loads((HERE/'diagnostics/SOURCE_HASH_VERIFICATION.json').read_text())['all_match']
 choices=json.loads((HERE/'diagnostics/PAYLOAD_SCHEMA_DIAGNOSIS_ONLY.json').read_text());assert all(d['status']=='DIAGNOSTIC_SCHEMA_ESTIMATE_PASS' for d in choices)
 files=list(HERE.glob('*.py'))+list((HERE/'code').glob('*.py'))+[HERE/'source_domain_registry.csv',HERE/'source_domain_evidence.json',HERE/'sources/PREFIX_PROVENANCE.json',HERE/'diagnostics/FIXED_CASES.json',HERE/'diagnostics/V2_STATUS.md',HERE/'diagnostics/PAYLOAD_SCHEMA_DIAGNOSIS_ONLY.json',HERE/'diagnostics/SOURCE_HASH_VERIFICATION.json',HERE/'diagnostics/BOUNDARY_TESTS.json']
 for day in ['Friday','Tuesday','Wednesday']:
  files +=[HERE/f'diagnostics/{day}/{name}' for name in ['RAW_PACKETS_BY_KEY.json','CSV_CONTEXT.json','CLOCK_QUALITY.json','COMPLETE.json']]
  files +=[V2/f'reconstructed/CIC_IDS/{day}/{name}' for name in ['source_files.json','csv_selection_audit.json']]
 files +=[V2/'rebuild.py',V2/'session_rules.py',V2/'code/packet_window_reference.py',V2/'code/packet_decode_reference.py']
 frozen=dict(version='R1_5_observed_session_v3_rules',frozen_at=datetime.datetime.now().astimezone().isoformat(),training_allowed=False,optuna_allowed=False,formal_data_generated=False,mode='rule_freeze_and_small_fixed_case_validation',rules=RULES,byte_schema_estimates={d['domain']:{'metric':d['metric'],'informative_diagnosis_anchors':d['informative'],'consensus':d['consensus']} for d in choices},choice_input_role='diagnosis only; heldout_rule_check outcomes not inspected for choices',formal_acceptance_gate='full-source metadata-only schema and clock evidence, complete candidate search, verified coverage and source hashes; this small-sample freeze is not that certification',frozen_files_sha256={os.path.relpath(p,HERE):sha(p) for p in files})
 dest.write_text(json.dumps(frozen,indent=2));(HERE/'V3_RULES_FROZEN.sha256').write_text(sha(dest)+'\n');print('FROZEN',len(files),'files; models/Optuna disabled')
if __name__=='__main__':main()

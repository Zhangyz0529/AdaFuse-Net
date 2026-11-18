"""Prepare ready datasets and native input evidence once; no training selection."""
import json,subprocess,sys,time
from cache import HERE,ROOTS,prepare
while True:
 for ds in ['USTC','ISCX','CIC_IDS','CIC_DDOS','CIC_DOH']:
  gate=ROOTS[ds]/'PARTITION_GATE.json'
  if not gate.exists() or json.loads(gate.read_text()).get('status')!='PASS_PARTITIONS_AND_EVALUATION_POOLS':continue
  prepare(ds)
  if ds in ['USTC','ISCX']:
   for script,sub in [('flowpic_inputs.py','flowpic'),('packet_patterns.py','packet_patterns')]:
    if not (HERE/sub/ds/'COMPLETE.json').exists():subprocess.run([sys.executable,str(HERE/script),'--dataset',ds],check=True)
 time.sleep(30)

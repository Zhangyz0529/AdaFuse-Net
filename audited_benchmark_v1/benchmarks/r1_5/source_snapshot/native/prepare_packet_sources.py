import json,subprocess,sys,time
from cache import HERE,ROOTS,save
while True:
 for ds in ['CIC_IDS','CIC_DDOS']:
  g=ROOTS[ds]/'PARTITION_GATE.json'
  if not g.exists():continue
  gate=json.loads(g.read_text())
  if gate.get('status')!='PASS_PARTITIONS_AND_EVALUATION_POOLS':continue
  if any(not (HERE/'cache'/ds/f'round{i}'/'COMPLETE.json').exists() for i in range(len(gate['folds']))):continue
  for script,sub in [('flowpic_inputs.py','flowpic'),('packet_patterns.py','packet_patterns')]:
   if (HERE/sub/ds/'COMPLETE.json').exists():continue
   rc=subprocess.run([sys.executable,str(HERE/script),'--dataset',ds]).returncode
   if rc:save(HERE/'native_source_failures'/f'{ds}_{sub}.json',dict(returncode=rc,time=time.time()));raise SystemExit(rc)
 time.sleep(30)

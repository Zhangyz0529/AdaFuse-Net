"""Storage-only optimization preserves every normalized row and unique key."""
import csv,importlib.util,json,tempfile
from pathlib import Path
import run as new
oldpath=new.V3/'full_rebuild/run.py';spec=importlib.util.spec_from_file_location('old_rebuild_for_equivalence',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def main():
 pnew=dict(new.POLICY);pold=dict(old.POLICY);pnew.pop('version');pold.pop('version');assert pnew==pold and new.RULES==old.RULES
 header=next(csv.reader((new.V3.parents[2]/'data/CIC_IDS/GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv').open(encoding='cp1252')));names=[n.strip() for n in header]
 with tempfile.TemporaryDirectory(prefix='key_equivalence_') as td:
  root=Path(td);path=root/'fixture.csv';rows=[]
  for i in range(4):
   row={n:0 for n in names};row.update({'Flow ID':str(i),'Source IP':'192.0.2.1','Destination IP':'192.0.2.2','Source Port':1000 if i<2 else 1001 if i==2 else 'invalid','Destination Port':80,'Protocol':6,'Timestamp':'2020-01-01 00:00:00','Flow Duration':0,'Total Fwd Packets':1,'Total Backward Packets':0,'Total Length of Fwd Packets':6,'Total Length of Bwd Packets':0,'Label':'BENIGN'});rows.append(row)
  with path.open('w') as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
  outputs=[]
  for module,name in [(old,'old'),(new,'new')]:
   out=root/name;out.mkdir();db=module.connect(out);module.ingest_csv({'csvs':[path]},'CIC_IDS',root,out,db);outputs.append((db.execute('SELECT k,valid,label,data FROM rows ORDER BY id').fetchall(),db.execute('SELECT * FROM keys ORDER BY k').fetchall(),(out/'parse_rejections.jsonl').read_text()));db.close()
  assert outputs[0]==outputs[1] and len(outputs[1][1])==2
 print(json.dumps(dict(passed=True,checks=['scientific policies unchanged except version identity','all normalized rows byte-identical','duplicate-key dedup exactly equivalent','parse rejection log exactly equivalent'],synthetic_only=True)))
if __name__=='__main__':main()

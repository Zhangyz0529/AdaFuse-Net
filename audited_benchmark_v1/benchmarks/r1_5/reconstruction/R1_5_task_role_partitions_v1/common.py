"""Explicit shared classes/dimensions and source-only accepted-record locations."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PAPER=HERE.parent
CLASSES={'CIC_IDS':['BENIGN','PortScan','DDoS','FTP-Patator','SSH-Patator','DoS'],'CIC_DDOS':['BENIGN','Syn','DrDoS_MSSQL','DrDoS_NetBIOS','DrDoS_UDP','UDP-lag'],'USTC':['Benign','Htbot','Neris','Miuref','Virut'],'ISCX':['facebook_audio4','vpn_bittorrent','vpn_hangouts_audio2','vpn_facebook_audio2','vpn_email2b','vpn_ftps_a'],'CIC_DOH':['Benign','dns2tcp','dnscat2','iodine']}
DIMS={'CIC_IDS':77,'CIC_DDOS':77,'USTC':25,'ISCX':25,'CIC_DOH':29}
def source_root(dataset):
 parent=PAPER/('R1_5_session_rebuild_v3/full_rebuild' if dataset in ['CIC_IDS','CIC_DDOS'] else 'R1_5_remaining_datasets_rebuild_v1');return parent/'outputs'/dataset
def expected_sources(dataset):
 if dataset=='CIC_IDS':return ['Monday','Tuesday','Wednesday','Thursday','Friday']
 if dataset=='CIC_DDOS':return ['Test_Day_03-11','Train_Day_01-12']
 p=PAPER/'R1_5_remaining_datasets_rebuild_v1'/(dataset+'_SOURCE_PLAN.json')
 if not p.exists():raise RuntimeError('No frozen formal source plan for '+dataset)
 return list(json.loads(p.read_text()))
def write(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(path)
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def emit(f,**x):f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')

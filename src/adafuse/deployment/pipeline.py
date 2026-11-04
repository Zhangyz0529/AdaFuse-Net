"""NF57/IP512 extraction and one-clock offline PCAP inference."""
from pathlib import Path
import csv,json,pickle,time,hashlib,importlib.metadata,platform
import numpy as np
import torch
from .nfstream_frontend import FEATURES,configuration,streamer,inputs,metadata
from ..training.trainer import build
from ..evaluation.metrics import binary_metrics
from ..utils.common import save,sha

def extract(pcap,label,output):
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True);records=[];failed=0
    for flow in streamer(pcap):
        try:
            m=metadata(flow)
            if m['ip_version']!=4 or m['vlan_id']!=0 or m['protocol'] not in [6,17]:raise ValueError('Outside certified source contract')
            stat,seq=inputs(flow)
        except (ValueError,AttributeError):failed+=1;continue
        records.append(dict(stat_raw=stat.astype('float32'),seq_feature=seq.copy(),label=label,group=m['key'],source_index=len(records)))
    with dest.open('wb') as f:pickle.dump(records,f,protocol=4)
    save(dest.with_suffix('.json'),dict(input_contract='NF57/IP512',feature_order=FEATURES,frontend_sha256=sha(Path(__file__).with_name('nfstream_frontend.py')),nfstream_version=importlib.metadata.version('nfstream'),nfstream_config=configuration(pcap),pcap_sha256=sha(pcap),label_rule='capture-file category; not certified per-flow',completed=len(records),failed=failed))

def export_support(prepared,round_index,shot,output):
    rd=Path(prepared)/f'round{round_index}';data=dict(np.load(rd/'inputs.npz'));plans=json.loads((rd/'eval.json').read_text())['test'];ep=next(iter(plans.values()))[0];ids=np.asarray(ep['support'])[:,:shot].ravel()
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,stats=data['stats'][ids],sequence=data['sequence'][ids],labels=np.repeat([0,1],shot),source_ids=ids,scaler_sha256=np.array(hashlib.sha256(np.load(rd/'split_scaler.npz')['mean'].tobytes()+np.load(rd/'split_scaler.npz')['scale'].tobytes()).hexdigest()))

def deploy(checkpoint,support,pcap,output,threshold=.5):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];ds=saved['dataset']
    if ds['stat_dim']!=57 or ds.get('input_contract')!='NF57/IP512' or ds.get('feature_order')!=FEATURES:raise ValueError('Requires a separately trained, explicitly certified NF57/IP512 model;77D/39D models are not compatible')
    device=cfg['device'];model=build(cfg,57).to(device).eval().requires_grad_(False);model.load_state_dict(saved['state_dict']);s=dict(np.load(support));expected=hashlib.sha256(saved['mean'].tobytes()+saved['scale'].tobytes()).hexdigest()
    if 'scaler_sha256' not in s or str(s['scaler_sha256'])!=expected:raise ValueError('Support set uses a different source scaler')
    y=s['labels'];xs=torch.tensor(s['stats'],device=device);xq=torch.tensor(s['sequence'].astype('int64'),device=device)
    with torch.inference_mode():z=model(xs,xq);proto=torch.stack([z[torch.tensor(y==c,device=device)].mean(0) for c in [0,1]])
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True);rows=[];failed=0
    sync();start=time.perf_counter();it=iter(streamer(pcap))
    while True:
        t0=time.perf_counter()
        try:flow=next(it)
        except StopIteration:break
        try:
            m=metadata(flow)
            if m['ip_version']!=4 or m['vlan_id']!=0 or m['protocol'] not in [6,17]:raise ValueError('Outside source contract')
            stat,seq=inputs(flow)
        except (ValueError,AttributeError):failed+=1;continue
        t1=time.perf_counter();xs=torch.tensor(((stat-saved['mean'])/saved['scale']).astype('float32')[None],device=device);xq=torch.tensor(seq.astype('int64')[None],device=device);sync();t2=time.perf_counter()
        with torch.inference_mode():
            z=model(xs,xq);sync();t3=time.perf_counter();d=torch.cdist(z,proto)
            if cfg['distance']=='squared_euclidean':d=d.square()
            score=(-d).softmax(1)[0,1];pred=score>=threshold;sync();t4=time.perf_counter()
        rows.append(dict(flow=len(rows),front_end_ms=1000*(t1-t0),preprocess_ms=1000*(t2-t1),T_encoder=1000*(t3-t2),T_decision=1000*(t4-t3),T_model=1000*(t4-t2),attack_score=float(score),prediction=int(pred)))
    sync();wall=time.perf_counter()-start
    if not rows:raise ValueError('No predictions completed')
    with (dest/'per_flow_stages.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save(dest/'summary.json',dict(completed=len(rows),failed=failed,hardware=dict(platform=platform.platform(),processor=platform.processor(),device=str(device),torch_threads=torch.get_num_threads(),batch_size=1,warmup_flows=0),software=dict(torch=torch.__version__,nfstream=importlib.metadata.version('nfstream')),wall_seconds=wall,flows_per_second=len(rows)/wall,wall_ms_per_completed_flow=1000*wall/len(rows),front_end_definition='time in iterator advancement plus input assembly; may include IPC/prefetch waiting; not isolated worker CPU cost',boundary='offline start before NFStreamer construction, end after all predictions; excludes model/support loading',flow_lifetime_latency='not measured; do not subtract PCAP timestamps from monotonic clock',nfstream_config=configuration(pcap)))

def extract_labelled(pcap,csv_dir,output,utc_offset_seconds=0,afternoon_adjust=False):
    """Retained tuple/time-unanimity labelling; weak correspondence, not exact truth."""
    import pandas as pd
    from .nfstream_frontend import canonical
    frames=[]
    for path in sorted(Path(csv_dir).glob('Thursday*.csv')):
        f=pd.read_csv(path,encoding='cp1252',low_memory=False);f.columns=f.columns.str.strip()
        cols=['Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Label'];f=f[cols].dropna().copy();t=pd.to_datetime(f.Timestamp,format='%d/%m/%Y %H:%M',errors='raise')
        if afternoon_adjust and 'Afternoon' in path.name:t+=pd.to_timedelta(np.where(t.dt.hour<12,12,0),unit='h')
        f['start_s']=t.astype('int64')/1e9+utc_offset_seconds;f['end_upper_s']=f.start_s+60+pd.to_numeric(f['Flow Duration'])/1e6;f['y']=(f.Label!='BENIGN').astype(int)
        f['key']=[canonical(*r) for r in f[['Source IP','Source Port','Destination IP','Destination Port','Protocol']].itertuples(index=False,name=None)];frames.append(f)
    if not frames:raise ValueError('No Thursday official identifying CSVs')
    official=pd.concat(frames,ignore_index=True);groups={k:g for k,g in official.groupby('key',sort=False)};records=[];uncertain=failed=0
    for flow in streamer(pcap):
        try:
            m=metadata(flow)
            if m['ip_version']!=4 or m['vlan_id']!=0 or m['protocol'] not in [6,17]:raise ValueError('Outside source contract')
            stat,seq=inputs(flow)
        except (ValueError,AttributeError):failed+=1;continue
        g=groups.get(m['key'])
        if g is not None:g=g[(g.start_s<=m['end_ms']/1000+.001)&(g.end_upper_s>=m['start_ms']/1000-.001)]
        if g is None or not len(g) or g.y.nunique()!=1:uncertain+=1;continue
        records.append(dict(stat_raw=stat.astype('float32'),seq_feature=seq.copy(),label='ATTACK' if int(g.y.iloc[0]) else 'BENIGN',group=m['key'],source_index=len(records)))
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:pickle.dump(records,f,protocol=4)
    save(dest.with_suffix('.json'),dict(input_contract='NF57/IP512',feature_order=FEATURES,frontend_sha256=sha(Path(__file__).with_name('nfstream_frontend.py')),nfstream_version=importlib.metadata.version('nfstream'),nfstream_config=configuration(pcap),pcap_sha256=sha(pcap),label_rule='unanimous binary official tuple/minute-uncertain time-overlap candidates; NOT exact per-flow certification',utc_offset_seconds=utc_offset_seconds,afternoon_adjust=afternoon_adjust,completed=len(records),label_uncertain_excluded=uncertain,frontend_excluded=failed))

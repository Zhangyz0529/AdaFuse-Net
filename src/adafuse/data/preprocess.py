"""Explicit data conversion. No dimensional padding/truncation of statistics."""
from pathlib import Path
import json,pickle,socket
import numpy as np
import pandas as pd
from .ciciot2023_builder import read_records,decode,anonymize
from ..utils.common import save,sha

def validate_records(records,stat_dim):
    for i,r in enumerate(records):
        x=np.asarray(r['stat_raw'],dtype=np.float32);q=np.asarray(r['seq_feature'])
        if x.shape!=(stat_dim,) or not np.isfinite(x).all():raise ValueError(f'Invalid statistics at record{i}')
        if q.shape!=(512,) or not np.issubdtype(q.dtype,np.integer) or q.min()<0 or q.max()>256:raise ValueError(f'Invalid sequence at record{i}')
    if not records:raise ValueError('No records retained')

def import_legacy(source,destination,stat_dim):
    # Pickle is the historical trusted local format; never load untrusted downloads.
    with Path(source).open('rb') as f:old=pickle.load(f)
    records=[]
    for i,r in enumerate(old):
        stat=np.asarray(r.get('stat_raw',r.get('stat')))
        if stat.shape!=(stat_dim,):raise ValueError('Feature schema mismatch; supply an explicit dataset-specific extractor')
        q=np.asarray(r.get('seq_feature',r.get('seq')))
        if q.max()<=1:q=np.round(q*255)
        if q.shape!=(512,):raise ValueError('Expected length512; explicit reconstruction required')
        records.append(dict(stat_raw=stat.astype('float32'),seq_feature=q.astype('int16'),label=str(r['label']),source_index=i))
    validate_records(records,stat_dim);dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:pickle.dump(records,f,protocol=4)
    save(dest.with_suffix('.json'),dict(input_sha256=sha(source),stat_dim=stat_dim,sequence_length=512,feature_semantics_certified=False,operation='trusted existing dataset conversion; no claim to reconstruct missing historical extraction'))

def csv_pcap(csv_path,pcap,feature_names,output,time_format,utc_offset_seconds=0):
    """CSV intervals plus bidirectional5tuple matching, reject ambiguous packets.

    This strict adapter is separately documented, not a claim of bitwise equality
    to undocumented historical preprocessing. Official CSV has precomputed stats.
    """
    import dpkt
    fields=json.loads(Path(feature_names).read_text());df=pd.read_csv(csv_path,low_memory=False);df.columns=df.columns.str.strip()
    required=['Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Label']
    if len(set(fields))!=len(fields) or not set(required+fields)<=set(df):raise ValueError('Need explicit feature order and identifying metadata')
    times=pd.to_datetime(df.Timestamp,format=time_format,errors='raise').astype('int64').to_numpy()/1e9+utc_offset_seconds
    duration=pd.to_numeric(df['Flow Duration']).to_numpy()/1e6
    values=df[fields].apply(pd.to_numeric,errors='raise').to_numpy(dtype='float32')
    # Nonfinite rows are excluded and counted, never silently zero-filled.
    valid=np.isfinite(values).all(1);index={};buffers={};packets=excluded=ambiguous=0
    def key(src,sp,dst,dp,protocol):return (tuple(sorted(((str(src),int(sp)),(str(dst),int(dp))))),int(protocol))
    for i,row in df.iterrows():
        if valid[i]:index.setdefault(key(row['Source IP'],row['Source Port'],row['Destination IP'],row['Destination Port'],row['Protocol']),[]).append(i)
    for _,offset,ts,frame,wire in read_records(Path(pcap)):
        packets+=1
        try:
            eth=dpkt.ethernet.Ethernet(frame);ip=eth.data
            if isinstance(ip,dpkt.ip.IP):src=socket.inet_ntop(socket.AF_INET,ip.src);dst=socket.inet_ntop(socket.AF_INET,ip.dst);proto=ip.p
            elif isinstance(ip,dpkt.ip6.IP6):src=socket.inet_ntop(socket.AF_INET6,ip.src);dst=socket.inet_ntop(socket.AF_INET6,ip.dst);proto=ip.nxt
            else:raise ValueError('Not IP')
            tr=ip.data
            if not isinstance(tr,(dpkt.tcp.TCP,dpkt.udp.UDP)):raise ValueError('Not TCP/UDP')
            candidates=[i for i in index.get(key(src,tr.sport,dst,tr.dport,proto),[]) if times[i]<=ts<=times[i]+duration[i]]
            if len(candidates)!=1:ambiguous+=int(len(candidates)>1);excluded+=1;continue
            i=candidates[0];buf=buffers.setdefault(i,bytearray())
            if len(buf)<512:buf.extend(anonymize(frame)[:512-len(buf)])
        except (ValueError,IndexError,dpkt.UnpackError):excluded+=1
    records=[dict(stat_raw=values[i],seq_feature=np.frombuffer(bytes(buf).ljust(512,b'\0'),dtype=np.uint8).copy(),label=str(df.iloc[i]['Label']).strip(),source_index=int(i)) for i,buf in sorted(buffers.items())]
    validate_records(records,len(fields));dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:pickle.dump(records,f,protocol=4)
    save(dest.with_suffix('.json'),dict(csv_sha256=sha(csv_path),pcap_sha256=sha(pcap),feature_order=fields,raw_packets=packets,excluded_packets=excluded,ambiguous_packets=ambiguous,retained_records=len(records),nonfinite_csv_rows=int((~valid).sum()),timestamp_format=time_format,utc_offset_seconds=utc_offset_seconds,sequence='anonymous captured Ethernet bytes in capture order; mask before truncate; rightzero-pad512',historical_equivalence_certified=False))

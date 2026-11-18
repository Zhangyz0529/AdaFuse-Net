"""CIC-IDS strict CSV packet-window adapter and ISCX activity-cohort adapter.

No labels are inferred by globally overwriting an endpoint dictionary. CIC
matches directional counts, transport payload bytes and microsecond duration,
then calibrates minute-precision CSV time from unambiguous unlabeled anchors.
ISCX labels denote capture activities, never malicious/normal VPN semantics.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import mmap
from pathlib import Path
import re
import shutil
import socket
import struct
import zipfile
import zlib

import dpkt
import numpy as np
import pandas as pd
import doh_pilot as pilot

CIC_CLASSES=['BENIGN','DoS','DDoS','PortScan','FTP-Patator','SSH-Patator']
ISCX_CLASSES=['Chat','Email','VoIP','Video','VPN_Chat','VPN_Email','VPN_VoIP','VPN_FileTransfer','VPN_P2P']
CIC_URL='https://www.unb.ca/cic/datasets/ids-2017.html'
ISCX_URL='https://www.unb.ca/cic/datasets/vpn.html'
META=['Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration',
      'Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Label']


def read(path):return json.loads(Path(path).read_text())


def save(path,v):pilot.save(path,v)


def iscx_category(stem):
    name=stem.lower();vpn=name.startswith('vpn_');base=name[4:] if vpn else name
    if 'chat' in base:category='Chat'
    elif base.startswith('email'):category='Email'
    elif '_audio' in base:category='VoIP'
    elif '_video' in base:category='Video'
    elif base.startswith('ftps'):category='FileTransfer'
    elif base.startswith('bittorrent'):category='P2P'
    else:return None
    label=('VPN_' if vpn else '')+category
    return label if label in ISCX_CLASSES else None


def iscx_family(stem):
    # a/b and A/B suffixes can be pieces of the same released recording.
    return 'ISCX_recording:'+re.sub(r'[_-]?[ab]$','',stem.lower())


def crc_hash(path):
    h=hashlib.sha256();crc=0
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block);crc=zlib.crc32(block,crc)
    return h.hexdigest(),f'{crc&0xffffffff:08x}'


def prepare(name,raw,out,config,thresholds):
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing started version')
    out.mkdir(parents=True,exist_ok=True);jobs=[];excluded=[]
    if name=='ISCX':
        pcaps=raw/'pcap';members=defaultdict(list)
        for archive in sorted(pcaps.glob('*.zip')):
            with zipfile.ZipFile(archive) as z:
                for m in z.infolist():
                    if m.filename.lower().endswith(('.pcap','.pcapng')):
                        members[Path(m.filename).name].append(dict(archive=str(archive.resolve()),member=m.filename,bytes=m.file_size,crc32=f'{m.CRC:08x}'))
        for p in sorted(pcaps.iterdir()):
            if p.suffix.lower() not in ('.pcap','.pcapng'):continue
            label=iscx_category(p.stem)
            if label not in config['classes']:continue
            if p.stat().st_size>config['max_capture_bytes']:
                excluded.append(dict(path=str(p),label=label,reason='fixed_file_size_limit'));continue
            h,crc=crc_hash(p)
            matched=[m for m in members[p.name] if m['bytes']==p.stat().st_size and m['crc32']==crc]
            restored=None;capture=p
            if not matched:
                candidates=members[p.name]
                identities={(m['bytes'],m['crc32']) for m in candidates}
                if not config.get('restore_from_archive_on_mismatch',False) or len(identities)!=1:
                    excluded.append(dict(path=str(p),label=label,reason='release_archive_crc_size_mismatch'));continue
                member=candidates[0];restore_dir=out/'restored_sources';restore_dir.mkdir(exist_ok=True)
                capture=restore_dir/p.name
                with zipfile.ZipFile(member['archive']) as archive,archive.open(member['member']) as src,capture.open('wb') as dst:
                    for block in iter(lambda:src.read(1<<20),b''):dst.write(block)
                restored=dict(original_path=str(p.resolve()),original_sha256=h,original_crc32=crc,reason='release_crc_size_mismatch',
                              archive=member['archive'],member=member['member'])
                h,crc=crc_hash(capture)
                matched=[m for m in candidates if m['bytes']==capture.stat().st_size and m['crc32']==crc]
                if not matched:raise ValueError('Restored archive member verification failed')
            jobs.append(dict(capture=str(capture.resolve()),label=label,application=p.stem,acquisition_family=iscx_family(p.stem),
                             original_sha256=h,bytes=capture.stat().st_size,archive_members=matched,crc32=crc,restored_source=restored))
        label_scope='publisher_capture_activity_cohort; background browsing may occur; VPN is not an attack'
    else:
        for day in config['capture_days']:
            sources=sorted((raw/'GeneratedLabelledFlows/TrafficLabelling').glob(day+'*.csv'))
            captures=sorted((raw/'pcap').glob(day+'*.pcap'))
            if len(captures)!=1 or not sources:raise ValueError('Ambiguous/missing capture-day association: '+day)
            jobs.append(dict(capture=str(captures[0].resolve()),csvs=[str(p.resolve()) for p in sources],day=day,
                             csv_hashes={str(p.resolve()):pilot.sha(p) for p in sources},acquisition_family='CIC_capture_day:'+day))
        label_scope='unique CSV packet-window correspondence; no preferred attack label'
    return dict(dataset=name,version='source_suite_v4_20261109',raw_root=str(raw),jobs=jobs,excluded_files=excluded,
                requested_classes=config['classes'],effective_config=config,thresholds=thresholds,
                label_policy=label_scope,rules=pilot.RULES,training_started=False)


def indexed_packets(mm):
    """Yield actual original frame offsets for classic PCAP and PCAPNG.

    PCAPNG timestamps obey per-interface tsresol/tsoffset; unsupported untimed
    blocks fail closed, not assigned synthetic timestamps. No payload rewrite.
    """
    size=len(mm);magic=bytes(mm[:4]);index=0
    if magic!=b'\x0a\x0d\x0d\x0a':
        modes={b'\xd4\xc3\xb2\xa1':('<',1e-6),b'\xa1\xb2\xc3\xd4':('>',1e-6),
               b'\x4d\x3c\xb2\xa1':('<',1e-9),b'\xa1\xb2\x3c\x4d':('>',1e-9)}
        if magic not in modes:raise ValueError('Unsupported PCAP header')
        endian,res=modes[magic];link=struct.unpack_from(endian+'I',mm,20)[0]&0xffff;pos=24
        unpack=struct.Struct(endian+'IIII').unpack_from
        while pos<size:
            if pos+16>size:raise ValueError('Truncated record header')
            sec,frac,cap,wire=unpack(mm,pos);offset=pos+16
            if offset+cap>size:raise ValueError('Truncated record data')
            yield index,sec+frac*res,offset,cap,wire,link
            index+=1;pos=offset+cap
        return
    pos=0;interfaces=[];endian=None
    while pos<size:
        if pos+12>size:raise ValueError('Truncated pcapng block')
        if bytes(mm[pos:pos+4])==b'\x0a\x0d\x0d\x0a':
            bom=bytes(mm[pos+8:pos+12]);endian='<' if bom==b'\x4d\x3c\x2b\x1a' else '>' if bom==b'\x1a\x2b\x3c\x4d' else None
            if endian is None:raise ValueError('Bad pcapng byte order')
            interfaces=[]
        if endian is None:raise ValueError('No section header')
        kind,total=struct.unpack_from(endian+'II',mm,pos)
        if total<12 or total%4 or pos+total>size or struct.unpack_from(endian+'I',mm,pos+total-4)[0]!=total:
            raise ValueError('Invalid pcapng block size')
        if kind==1:
            link=struct.unpack_from(endian+'H',mm,pos+8)[0];resolution=1e-6;time_offset=0
            opt=pos+16
            while opt+4<=pos+total-4:
                code,n=struct.unpack_from(endian+'HH',mm,opt);opt+=4
                if code==0:break
                if opt+n>pos+total-4:raise ValueError('Truncated interface option')
                if code==9 and n==1:
                    value=mm[opt];resolution=(2.**-(value&127)) if value&128 else (10.**-value)
                if code==14 and n==8:time_offset=struct.unpack_from(endian+'q',mm,opt)[0]
                opt+=(n+3)&~3
            interfaces.append((link,resolution,time_offset))
        elif kind in (6,2):
            iface=struct.unpack_from(endian+('I' if kind==6 else 'H'),mm,pos+8)[0]
            hi,lo,cap,wire=struct.unpack_from(endian+'IIII',mm,pos+12)
            if iface>=len(interfaces) or pos+28+cap>pos+total-4:raise ValueError('Bad packet interface/data')
            link,res,time_offset=interfaces[iface]
            yield index,((hi<<32)|lo)*res+time_offset,pos+28,cap,wire,link
            index+=1
        elif kind==3:raise ValueError('Untimed simple packet blocks cannot support CSV matching')
        pos+=total


def fast_endpoint(mm,offset,cap,link):
    end=offset+cap;net=offset
    if link==1:
        if cap<14:return None
        eth=struct.unpack_from('!H',mm,offset+12)[0];net+=14
        while eth in (0x8100,0x88a8):
            if net+4>end:return None
            eth=struct.unpack_from('!H',mm,net+2)[0];net+=4
        if eth!=0x0800:return None
    elif link==113:
        if cap<16 or struct.unpack_from('!H',mm,offset+14)[0]!=0x0800:return None
        net+=16
    elif link not in (101,228):return None
    if net+20>end or mm[net]>>4!=4:return None
    ihl=(mm[net]&15)*4;proto=mm[net+9];trans=net+ihl
    if proto not in (6,17) or trans+4>end:return None
    src,dst=struct.unpack_from('!II',mm,net+12);sport,dport=struct.unpack_from('!HH',mm,trans)
    a=(src,sport);b=(dst,dport)
    return (proto,*sorted((a,b)))


def csv_key(row):
    a=(struct.unpack('!I',socket.inet_aton(str(row['Source IP'])))[0],int(row['Source Port']))
    b=(struct.unpack('!I',socket.inet_aton(str(row['Destination IP'])))[0],int(row['Destination Port']))
    return (int(row['Protocol']),*sorted((a,b)))


def load_targets(job,config):
    frames=[]
    for source in job['csvs']:
        p=Path(source);names=pd.read_csv(p,encoding='cp1252',nrows=0).columns
        mapping={c.strip():c for c in names}
        df=pd.read_csv(p,encoding='cp1252',usecols=[mapping[k] for k in META]);df.columns=df.columns.str.strip()
        df['csv_path']=str(p);df['csv_row']=np.arange(len(df))+2
        df['label']=df.Label.astype(str).str.strip().replace({'DoS Hulk':'DoS','DoS GoldenEye':'DoS'})
        # Other DoS labels are not silently merged into the requested legacy taxonomy.
        df['clock']=pd.to_datetime(df.Timestamp,dayfirst=True,errors='raise').astype('int64')/1e9
        frames.append(df)
    frame=pd.concat(frames,ignore_index=True)
    numeric=['Source Port','Destination Port','Protocol','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets']
    for col in numeric:frame[col]=pd.to_numeric(frame[col],errors='coerce')
    frame['_invalid_stats']=~np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)|(frame[numeric]<0).any(axis=1)
    requested=frame[frame.label.isin(config['classes'])].copy()
    n=requested['Total Fwd Packets']+requested['Total Backward Packets']
    eligible=requested[(n>=3)&(n<=config['max_packets_per_row'])&~requested['_invalid_stats']]
    targets=[]
    for label,part in eligible.groupby('label'):
        values=list(part.to_dict('records'))
        values.sort(key=lambda r:hashlib.sha256((r['csv_path']+':'+str(r['csv_row'])).encode()).digest())
        targets.extend(values[:config['target_rows_per_class_per_capture']])
    keys={csv_key(r) for r in targets};rows_by_key=defaultdict(list)
    # All labels/rows with a selected key participate in ambiguity checks.
    for row in frame.to_dict('records'):
        key=csv_key(row)
        if key in keys:rows_by_key[key].append(row)
    report=dict(csv_rows=len(frame),requested_rows=len(requested),eligible_rows=len(eligible),selected_rows=len(targets),
                selected_by_class=dict(Counter(r['label'] for r in targets)),selected_keys=len(keys))
    return targets,rows_by_key,report


def window_candidates(row,packets,tolerance_us):
    if row.get('_invalid_stats',False): return []
    n=int(row['Total Fwd Packets'])+int(row['Total Backward Packets'])
    if n<1 or n>len(packets):return []
    origin=(str(row['Source IP']),int(row['Source Port']))
    direction=np.array([int(tuple(p['src'])==origin) for p in packets],dtype=np.int64)
    payload=np.array([p['payload_len'] for p in packets],dtype=np.int64)
    fc=np.r_[0,np.cumsum(direction)];fb=np.r_[0,np.cumsum(payload*direction)];bb=np.r_[0,np.cumsum(payload*(1-direction))]
    ends=np.arange(n,len(packets)+1);starts=ends-n
    good=(fc[ends]-fc[starts]==int(row['Total Fwd Packets']))&(fb[ends]-fb[starts]==int(row['Total Length of Fwd Packets']))&(bb[ends]-bb[starts]==int(row['Total Length of Bwd Packets']))
    matches=[]
    for start in starts[good]:
        start=int(start);end=start+n
        duration=(packets[end-1]['ts']-packets[start]['ts'])*1e6
        if abs(duration-float(row['Flow Duration']))<=tolerance_us:
            matches.append((start,end))
    return matches


def row_identity(row):return row['csv_path'],int(row['csv_row'])


def calibrated_matches(targets,rows_by_key,packets_by_key,config):
    allmatches={};anchors=[]
    for key,rows in rows_by_key.items():
        packets=packets_by_key.get(key,[])
        for row in rows:
            matches=window_candidates(row,packets,config['duration_tolerance_us']);allmatches[row_identity(row)]=matches
            if len(rows)==1 and len(matches)==1:
                t=packets[matches[0][0]]['ts'];offset=int(np.floor((t-row['clock'])/3600))*3600
                residual=t-row['clock']-offset
                if 0<=residual<60:anchors.append(dict(offset_seconds=offset,csv_path=row['csv_path'],csv_row=int(row['csv_row'])))
    ambiguous_clock=config.get('allow_12h_clock_ambiguity',False)
    votes=Counter(a['offset_seconds']%43200 if ambiguous_clock else a['offset_seconds'] for a in anchors);offset=None
    if votes:
        candidate,count=votes.most_common(1)[0]
        if len(anchors)>=config['min_clock_anchors'] and count/len(anchors)>=config['min_clock_consensus']:offset=candidate
    offsets=([offset,offset+43200] if ambiguous_clock else [offset]) if offset is not None else []
    def time_consistent(row,t):return any(row['clock']+o<=t<row['clock']+o+60 for o in offsets)
    accepted=[];rejected=[]
    for row in targets:
        key=csv_key(row);packets=packets_by_key.get(key,[]);reason='invalid_csv_record_sharing_endpoint' if any(r.get('_invalid_stats',False) for r in rows_by_key[key]) else None
        if offset is None:reason='clock_calibration_not_certified'
        matches=[]
        if reason is None:
            matches=[(a,b) for a,b in allmatches.get(row_identity(row),[]) if time_consistent(row,packets[a]['ts'])]
            if not matches:reason='no_consistent_packet_window'
            elif len(matches)>1:reason='ambiguous_packet_window'
        if reason is None:
            a,b=matches[0];alternatives=[]
            for other in rows_by_key[key]:
                if (a,b) in allmatches[row_identity(other)] and time_consistent(other,packets[a]['ts']):
                    alternatives.append(other)
            if len(alternatives)!=1:reason='ambiguous_csv_record_or_label'
        if reason:rejected.append(dict(csv_path=row['csv_path'],csv_row=int(row['csv_row']),label=row['label'],reason=reason))
        else:accepted.append((row,packets[matches[0][0]:matches[0][1]]))
    occupied=Counter((p['file_id'],p['index']) for row,unit in accepted for p in unit)
    disjoint=[]
    for row,unit in accepted:
        if any(occupied[(p['file_id'],p['index'])]>1 for p in unit):
            rejected.append(dict(csv_path=row['csv_path'],csv_row=int(row['csv_row']),label=row['label'],reason='overlapping_accepted_packet_windows'))
        else:disjoint.append((row,unit))
    return disjoint,rejected,dict(status='CALIBRATED' if offset is not None else 'BLOCKED_CLOCK',offset_seconds=offset,
                                  offset_options_seconds=offsets,clock_representation='minute precision with explicit two-phase 12h ambiguity' if ambiguous_clock else '24h minute precision',anchor_count=len(anchors),votes={str(k):v for k,v in votes.items()},anchors=anchors)


def cic_one(out,index,job,config):
    cache=out/'captures'/f'{index:03d}'
    if (cache/'complete.json').exists():return read(cache/'complete.json')
    if cache.exists():raise RuntimeError('Inspect incomplete source before resume')
    cache.mkdir(parents=True);source=Path(job['capture']);before=source.stat();source_hash=pilot.sha(source)
    targets,rows_by_key,sampling=load_targets(job,config);keys=set(rows_by_key);packets=defaultdict(list);overflow=set();total=0;collected=0;first=None;last=None;linktypes=set()
    with source.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        if hasattr(mm,'madvise'):mm.madvise(mmap.MADV_SEQUENTIAL)
        for ordinal,ts,offset,cap,wire,link in indexed_packets(mm):
            total+=1;first=ts if first is None else min(first,ts);last=ts if last is None else max(last,ts);linktypes.add(link)
            if total%2000000==0:print(f'CIC {job["day"]}: scanned={total} selected-key packets={collected}',flush=True)
            key=fast_endpoint(mm,offset,cap,link)
            if key not in keys or key in overflow:continue
            if len(packets[key])>=config['max_packets_per_key']:
                overflow.add(key);packets[key]=[];continue
            if collected>=config['max_collected_packets']:raise RuntimeError('Prespecified packet memory cap reached; no silent truncation')
            raw=bytes(mm[offset:offset+cap])
            try:
                p=pilot.decode(raw,link)
                # Payload statistics are measured before anonymous reserialization.
                if link==1:ip=dpkt.ethernet.Ethernet(raw).data
                elif link==113:ip=dpkt.sll.SLL(raw).data
                else:ip=dpkt.ip.IP(raw)
                payload_len=len(ip.data.data)
            except (ValueError,dpkt.UnpackError,AttributeError,IndexError):
                overflow.add(key);packets[key]=[];continue
            packets[key].append(dict(index=ordinal,ts=ts,offset=offset,caplen=cap,link=link,src=p['src'],dst=p['dst'],proto=p['proto'],
                                     flags=p['flags'],isn=p['isn'],size=p['size'],payload_len=payload_len,network=p['network'][:512],
                                     raw_sha256=hashlib.sha256(raw).hexdigest()))
            collected+=1
    for key in overflow:packets.pop(key,None)
    accepted,rejected,clock=calibrated_matches(targets,rows_by_key,packets,config)
    records=[];arrays=[];observed=[]
    for values in packets.values():observed.extend(values)
    for row,unit in accepted:
        origin=unit[0]['src'];segment=pilot.Segment(origin,unit[0]['proto'],0,unit[0]['flags'],unit[0]['isn'])
        for p in unit:segment.add(p['index'],p['ts'],p)
        if segment.bad_time:
            rejected.append(dict(csv_path=row['csv_path'],csv_row=int(row['csv_row']),label=row['label'],reason='timestamp_regression'));continue
        sid=hashlib.sha256(json.dumps(['CIC_IDS',source_hash,row_identity(row),segment.indices]).encode()).hexdigest()
        records.append(dict(sample_id=sid,session_id=sid,dataset='CIC_IDS',label=row['label'],capture_index=index,capture_hash=source_hash,
                            source_group_id='CIC_capture_day:'+job['day'],acquisition_family=job['acquisition_family'],packet_indices=segment.indices,
                            packet_offsets=[p['offset'] for p in unit],packet_caplens=[p['caplen'] for p in unit],packet_directions=segment.directions,
                            start_timestamp=unit[0]['ts'],end_timestamp=unit[-1]['ts'],protocol=unit[0]['proto'],packet_count=len(unit),
                            sequence_valid_length=len(segment.prefix),sequence_boundaries=segment.prefix_boundaries,
                            endpoints=dict(first_ip=origin[0],first_port=origin[1],peer_ip=unit[0]['dst'][0],peer_port=unit[0]['dst'][1]),
                            csv_path=row['csv_path'],csv_hash=job['csv_hashes'][row['csv_path']],csv_row=int(row['csv_row']),
                            match_status='UNIQUE_ENDPOINT_TIME_DIRECTIONAL_COUNTS_PAYLOAD_BYTES_DURATION',label_source='matched_release_CSV_record',
                            publisher=CIC_URL,extractor_version='cic_unique_csv_window_v4_1'))
        arrays.append((segment.features(),np.frombuffer(bytes(segment.prefix).ljust(512,b'\0'),dtype=np.uint8).copy()))
    pd.DataFrame(records).to_parquet(cache/'provenance.parquet',index=False);pd.DataFrame(rejected).to_parquet(cache/'quarantine.parquet',index=False)
    if observed:pd.DataFrame(observed).to_parquet(cache/'observed_packets.parquet',index=False)
    np.savez_compressed(cache/'model_inputs.npz',stat_raw=np.stack([x for x,s in arrays]) if arrays else np.empty((0,25),np.float32),
                        seq_feature=np.stack([s for x,s in arrays]) if arrays else np.empty((0,512),np.uint8),sample_id=np.array([r['sample_id'] for r in records]),label=np.array([r['label'] for r in records]))
    save(cache/'clock_calibration.json',clock);save(cache/'sampling.json',sampling)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or pilot.sha(source)!=source_hash:raise ValueError('Original capture changed')
    if any(pilot.sha(Path(p))!=h for p,h in job['csv_hashes'].items()):raise ValueError('Source CSV changed')
    meta=dict(index=index,dataset='CIC_IDS',capture=str(source),sha256=source_hash,bytes=before.st_size,raw_packets_scanned=total,
              capture_start_time=first,capture_end_time=last,link_types=sorted(linktypes),source_group_id='CIC_capture_day:'+job['day'],
              accepted=len(records),labels=dict(Counter(r['label'] for r in records)),rejected_rows=len(rejected),clock=clock,
              overflow_or_invalid_keys=len(overflow),sampling=sampling,original_hash_unchanged=True,
              raw_copy_policy='read-only mmap; selected raw packet byte offsets/hashes retained; no 35GB duplicate or virtual packet insertion')
    save(cache/'complete.json',meta)
    print(f'CIC {job["day"]}: accepted={len(records)}, labels={meta["labels"]}, clock={clock["status"]}',flush=True)
    return meta


def run_cic(out,plan):
    metadata=[];frames=[];xs=[];ss=[];ids=[];labels=[];rejected=[]
    for i,job in enumerate(plan['jobs']):
        metadata.append(cic_one(out,i,job,plan['effective_config']))
        cache=out/'captures'/f'{i:03d}';f=pd.read_parquet(cache/'provenance.parquet')
        if not f.empty:frames.append(f)
        q=pd.read_parquet(cache/'quarantine.parquet')
        if not q.empty:rejected.append(q)
        with np.load(cache/'model_inputs.npz',allow_pickle=False) as a:
            xs.append(a['stat_raw']);ss.append(a['seq_feature']);ids.extend(a['sample_id']);labels.extend(a['label'])
        save(out/'progress.json',dict(completed_captures=i+1,total=len(plan['jobs']),training_started=False))
    columns=['sample_id','label','capture_index','source_group_id','acquisition_family','split']
    data=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)
    assignment={}
    for label in plan['requested_classes']:
        groups=sorted(data.loc[data.label==label,'source_group_id'].unique())
        # One capture day is indivisible even when it contains multiple labels.
        for group in groups:
            if group not in assignment:assignment[group]=('train','val','test')[min(len(assignment),2)]
    data['split']=data.source_group_id.map(assignment).fillna('unassigned')
    data.to_parquet(out/'provenance.parquet',index=False)
    pd.concat(rejected,ignore_index=True).to_parquet(out/'quarantine.parquet',index=False)
    np.savez_compressed(out/'model_inputs.npz',stat_raw=np.concatenate(xs),seq_feature=np.concatenate(ss),sample_id=np.array(ids,dtype=str),label=np.array(labels,dtype=str))
    save(out/'file_manifest.json',metadata)
    save(out/'feature_schema.json',dict(version=pilot.RULES['schema'],fields=pilot.FEATURES,legacy_77_compatible=False,
                                      study_unit='uniquely CSV-matched consecutive same-endpoint packet window',rules=pilot.RULES))
    save(out/'audit_report.json',dict(dataset='CIC_IDS',status='STRICT_MATCHED_CANDIDATES' if len(data) else 'BLOCKED_NO_CERTIFIED_LABEL_MATCHES',
                                     samples=len(data),labels=dict(Counter(data.label)),captures=len(metadata),source_groups=data.source_group_id.nunique(),
                                     label_scope='unique_release_CSV_record',raw_scanning='full selected captures, bounded CSV targets; not full dataset reconstruction',
                                     source_group_policy='capture-day indivisible',training_started=False))
    verify_cic(out,metadata)


def verify_cic(out,metadata):
    frame=pd.read_parquet(out/'provenance.parquet');checks=[]
    with np.load(out/'model_inputs.npz',allow_pickle=False) as a:
        idx={s:i for i,s in enumerate(a['sample_id'])}
        for i,m in enumerate(metadata):
            for row in frame[frame.capture_index==i].sort_values('sample_id').head(5).itertuples():
                unit=[]
                with Path(m['capture']).open('rb') as f:
                    for ordinal,offset,cap in zip(row.packet_indices,row.packet_offsets,row.packet_caplens):
                        f.seek(int(offset));raw=f.read(int(cap));unit.append((int(ordinal),raw))
                observed=pd.read_parquet(out/'captures'/f'{i:03d}'/'observed_packets.parquet').set_index('index')
                origin=None;seg=None
                for ordinal,raw in unit:
                    p=pilot.decode(raw,int(observed.loc[ordinal,'link']))
                    assert hashlib.sha256(raw).hexdigest()==observed.loc[ordinal,'raw_sha256']
                    if seg is None:origin=p['src'];seg=pilot.Segment(origin,p['proto'],0,p['flags'],p['isn'])
                    seg.add(ordinal,float(observed.loc[ordinal,'ts']),p)
                k=idx[row.sample_id];sequence=np.frombuffer(bytes(seg.prefix).ljust(512,b'\0'),dtype=np.uint8)
                ok=np.array_equal(seg.features(),a['stat_raw'][k]) and np.array_equal(sequence,a['seq_feature'][k])
                if not ok:raise ValueError('CIC raw-offset recomputation failed')
                checks.append(dict(sample_id=row.sample_id,capture_index=i,recomputed_equal=True))
    save(out/'independent_recompute_checks.json',checks)
    report=read(out/'audit_report.json');report['independent_recomputed_samples']=len(checks);save(out/'audit_report.json',report)

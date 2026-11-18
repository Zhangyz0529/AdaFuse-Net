"""Source-preserving DoH pilot: same-packet dual inputs, no training.

New ip_packet_stats25_v1 schema, NOT the legacy DoHLyzer 29 dimensions.
Only publisher-documented endpoint roles receive labels. Unknown/background
traffic is quarantined. Source-day split is exploratory and never called
independent acquisition certification. No dataset.pkl if coverage gate fails.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import struct
import sys

import dpkt
import numpy as np
import pandas as pd

PUBLISHER='https://www.unb.ca/cic/datasets/dohbrw-2020.html'
RESOLVERS={'1.1.1.1','8.8.4.4','8.8.8.8','9.9.9.9','9.9.9.10','9.9.9.11','176.103.130.131','176.103.130.130','149.112.112.10','149.112.112.112','104.16.248.249','104.16.249.249'}
BROWSERS={'192.168.20.191','192.168.20.111','192.168.20.112','192.168.20.113'}
TUNNEL_CLIENTS={'192.168.20.144',*[f'192.168.20.{x}' for x in range(204,213)]}
LABELS=('Benign','dns2tcp','dnscat2','iodine')
RULES={'version':'doh_ip_segment_v2_pilot_1','schema':'ip_packet_stats25_v1',
       'tcp_idle_seconds':120.,'udp_idle_seconds':30.,'active_seconds':300.,
       'direction':'first observed packet; not necessarily initiator',
       'tcp_boundary':'new non-retransmitted SYN without ACK; RST; both-direction FIN; idle/active limit',
       'midstream':'allowed with explicit missing_initial_syn flag; observed packet segment, not guaranteed complete connection',
       'retransmissions':'included; identical SYN sequence is not a new connection',
       'out_of_order':'quarantine the affected segment; no timestamp clipping',
       'packet_filter':'IPv4/IPv6 TCP/UDP only; exclude IP fragments, truncated and unsupported link-layer packets',
       'sequence':'512-byte prefix of concatenated anonymized IP packets in original capture order; no inserted boundary/direction tokens',
       'anonymization':'before truncation: zero IP src/dst, IPv4 ID/checksum, IPv6 flow label, transport ports/checksum, TCP seq/ack; preserve flags/options',
       'sequence_values':'uint8 0..255, zero padded; valid length and contributing packet boundaries in provenance',
       'stats':'IP total packet length, all eligible packets incl pure ACK and retransmissions; seconds; std ddof=0; zero duration gives zero rates',
       'label':'publisher endpoint-role rule AND named tool group AND observed TLS record header; unknown is quarantined; CSV is audit evidence, not label override',
       'split':'UTC-day connected components: overlapping capture days, duplicate capture hashes or exact timestamp+raw-packet overlaps cannot cross splits',
       'split_target':'exploratory source-day separation, not proven independent acquisition or chronological holdout',
       'training_allowed':False}
FEATURES=['duration_s','fwd_packets','bwd_packets','fwd_ip_bytes','bwd_ip_bytes',
          'fwd_len_mean','fwd_len_std','fwd_len_max','fwd_len_min',
          'bwd_len_mean','bwd_len_std','bwd_len_max','bwd_len_min',
          'all_len_mean','all_len_std','all_len_max','all_len_min',
          'iat_mean_s','iat_std_s','iat_max_s','iat_min_s','packets_per_s','ip_bytes_per_s','fwd_packets_per_s','bwd_packets_per_s']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def reader(handle):
    magic=handle.read(4);handle.seek(0)
    return dpkt.pcapng.Reader(handle) if magic==b'\x0a\x0d\x0d\x0a' else dpkt.pcap.Reader(handle)


def decode(raw,link):
    if link==1:ip=dpkt.ethernet.Ethernet(raw).data
    elif link==113:ip=dpkt.sll.SLL(raw).data
    elif link in (101,228,229):
        ip=dpkt.ip.IP(raw) if raw[0]>>4==4 else dpkt.ip6.IP6(raw)
    else:raise ValueError('unsupported_link')
    if not isinstance(ip,(dpkt.ip.IP,dpkt.ip6.IP6)):raise ValueError('non_ip')
    if isinstance(ip,dpkt.ip.IP):
        if ip.offset or ip.mf:raise ValueError('fragment')
        if ip.len>len(bytes(ip)):raise ValueError('truncated_ip')
    else:
        if getattr(ip,'fragment_hdr',None) is not None or any(isinstance(e,dpkt.ip6.IP6FragmentHeader) for e in getattr(ip,'all_extension_headers',[])):
            raise ValueError('fragment')
        if ip.plen+40>len(bytes(ip)):raise ValueError('truncated_ip')
    tr=ip.data
    if not isinstance(tr,(dpkt.tcp.TCP,dpkt.udp.UDP)):raise ValueError('non_tcp_udp')
    proto=6 if isinstance(tr,dpkt.tcp.TCP) else 17
    fam=socket.AF_INET if isinstance(ip,dpkt.ip.IP) else socket.AF_INET6
    src=(socket.inet_ntop(fam,ip.src),int(tr.sport));dst=(socket.inet_ntop(fam,ip.dst),int(tr.dport))
    flags=tr.flags if proto==6 else 0;isn=tr.seq if proto==6 else None
    payload=bytes(tr.data)
    tls_seen=proto==6 and len(payload)>=5 and payload[0] in (20,21,22,23) and payload[1]==3 and payload[2]<=4
    size=ip.len if isinstance(ip,dpkt.ip.IP) else ip.plen+40
    ip.src=bytes(len(ip.src));ip.dst=bytes(len(ip.dst))
    if isinstance(ip,dpkt.ip.IP):ip.id=0;ip.sum=0
    else:ip.flow=0
    tr.sport=tr.dport=0;tr.sum=0
    if proto==6:tr.seq=tr.ack=0
    # dpkt may recompute checksums on serialization; explicitly zero the wire fields.
    network=bytearray(bytes(ip))
    if isinstance(ip,dpkt.ip.IP):
        network[10:12]=b'\0\0';offset=ip.hl*4
    else:
        offset=40+sum(len(bytes(e)) for e in getattr(ip,'all_extension_headers',[]))
    checksum_offset=offset+(16 if proto==6 else 6)
    network[checksum_offset:checksum_offset+2]=b'\0\0'
    return dict(src=src,dst=dst,proto=proto,flags=flags,isn=isn,size=size,network=bytes(network[:size]),tls_seen=tls_seen)


def label_for(src,dst,proto,group):
    if proto!=6:return None,'not_tls_transport'
    for client,server in ((src,dst),(dst,src)):
        if server[0] not in RESOLVERS or server[1]!=443:continue
        if group=='browser' and client[0] in BROWSERS:return 'Benign','publisher_browser_client_and_doh_resolver_tcp443'
        if group in LABELS[1:] and client[0] in TUNNEL_CLIENTS:return group,'publisher_tunnel_client_and_doh_resolver_tcp443_plus_tool_source'
    return None,'unknown_endpoint_role'


@dataclass
class Segment:
    endpoint:tuple
    proto:int
    ordinal:int
    initial_flags:int
    initial_isn:object
    indices:list=field(default_factory=list)
    times:list=field(default_factory=list)
    sizes:list=field(default_factory=list)
    directions:list=field(default_factory=list)
    prefix:bytearray=field(default_factory=bytearray)
    prefix_boundaries:list=field(default_factory=list)
    fin_dirs:set=field(default_factory=set)
    ended:bool=False
    bad_time:bool=False
    tls_seen:bool=False

    def add(self,index,ts,p):
        direction=0 if p['src']==self.endpoint else 1
        self.tls_seen=self.tls_seen or p.get('tls_seen',False)
        if self.times and ts<self.times[-1]:self.bad_time=True
        self.indices.append(index);self.times.append(ts);self.sizes.append(p['size']);self.directions.append(direction)
        if len(self.prefix)<512:
            count=min(512-len(self.prefix),len(p['network']))
            self.prefix_boundaries.append(dict(packet_index=index,direction=direction,sequence_offset=len(self.prefix),bytes_retained=count))
            self.prefix.extend(p['network'][:count])
        if p['flags'] & dpkt.tcp.TH_FIN:self.fin_dirs.add(direction)
        if p['flags'] & dpkt.tcp.TH_RST or len(self.fin_dirs)==2:self.ended=True

    def features(self):
        times=np.asarray(self.times);sizes=np.asarray(self.sizes,dtype=float);directions=np.asarray(self.directions)
        f=sizes[directions==0];b=sizes[directions==1];iat=np.diff(times);duration=float(times[-1]-times[0])
        def stats(a):return [float(np.mean(a)),float(np.std(a)),float(np.max(a)),float(np.min(a))] if len(a) else [0.]*4
        rates=[len(sizes)/duration,sizes.sum()/duration,len(f)/duration,len(b)/duration] if duration>0 else [0.]*4
        return np.asarray([duration,len(f),len(b),f.sum(),b.sum(),*stats(f),*stats(b),*stats(sizes),*stats(iat),*rates],dtype=np.float32)


def inspect_capture(path,group,capture_hash,quarantine=None,label_resolver=None,require_tls=True):
    active={};finished=[];ordinals=Counter();excluded=Counter();packets=0;first=None;last=None
    packet_fingerprints=set();overlap_sketch=set()
    with path.open('rb') as f:
        rd=reader(f);link=rd.datalink()
        for index,(ts,raw) in enumerate(rd):
            packets+=1;ts=float(ts);first=ts if first is None else min(first,ts);last=ts if last is None else max(last,ts)
            # Full timestamp+raw digest is exact observed overlap evidence, before anonymization.
            fp=hashlib.sha256(struct.pack('!d',ts)+raw).digest()
            if fp in packet_fingerprints:excluded['repeated_raw_timestamp_packet']+=1
            packet_fingerprints.add(fp)
            try:p=decode(raw,link)
            except (ValueError,dpkt.UnpackError,IndexError,AttributeError) as e:
                reason=str(e) if isinstance(e,ValueError) else 'decode_error'
                excluded[reason]+=1
                if quarantine is not None:
                    quarantine.append(dict(rejection_scope='packet',reason=reason,packet_indices=[index],start_timestamp=ts,end_timestamp=ts,label=None))
                continue
            key=(p['proto'],*sorted((p['src'],p['dst'])))
            prior=active.get(key);idle=RULES['tcp_idle_seconds'] if p['proto']==6 else RULES['udp_idle_seconds']
            syn=bool(p['flags']&dpkt.tcp.TH_SYN and not p['flags']&dpkt.tcp.TH_ACK)
            new_syn=prior is not None and syn and (prior.initial_isn!=p['isn'] or not prior.initial_flags&dpkt.tcp.TH_SYN)
            new=prior is None or prior.ended or new_syn or ts-prior.times[-1]>idle or ts-prior.times[0]>RULES['active_seconds']
            if new:
                if prior:finished.append((key,prior))
                segment=Segment(p['src'],p['proto'],ordinals[key],p['flags'],p['isn']);ordinals[key]+=1;active[key]=segment
            active[key].add(index,ts,p)
    finished.extend(active.items())
    records=[];arrays=[];statuses=Counter();label_exclusions=Counter()
    for key,s in finished:
        peer=key[2] if key[1]==s.endpoint else key[1]
        label,evidence=(label_resolver or label_for)(s.endpoint,peer,s.proto,group)
        reason='timestamp_regression' if s.bad_time else 'too_few_packets' if len(s.indices)<3 else evidence if label is None else 'no_tls_record_header_observed' if require_tls and not s.tls_seen else None
        if reason:
            statuses[reason]+=1;label_exclusions[label or 'unknown']+=1
            if quarantine is not None:
                quarantine.append(dict(rejection_scope='segment',reason=reason,packet_indices=s.indices,
                                       start_timestamp=s.times[0],end_timestamp=s.times[-1],label=label,
                                       endpoints=dict(first_ip=s.endpoint[0],first_port=s.endpoint[1],peer_ip=peer[0],peer_port=peer[1]),protocol=s.proto))
            continue
        sample_id=hashlib.sha256(json.dumps([capture_hash,RULES['version'],s.indices],separators=(',',':')).encode()).hexdigest()
        records.append(dict(sample_id=sample_id,capture_hash=capture_hash,packet_indices=s.indices,
                            packet_directions=s.directions,source_group_id=None,session_id=sample_id,
                            endpoints=dict(first_ip=s.endpoint[0],first_port=s.endpoint[1],peer_ip=peer[0],peer_port=peer[1]),protocol=s.proto,start_timestamp=s.times[0],end_timestamp=s.times[-1],
                            packet_count=len(s.indices),sequence_valid_length=len(s.prefix),sequence_boundaries=s.prefix_boundaries,
                            tls_record_header_observed=s.tls_seen,missing_initial_syn=not bool(s.initial_flags&dpkt.tcp.TH_SYN and not s.initial_flags&dpkt.tcp.TH_ACK) if s.proto==6 else None,
                            csv_hash=None,csv_row=None,csv_match_status='ROUTE_A_CSV_NOT_USED_FOR_FEATURES_OR_LABELS',
                            match_status='SAME_PACKET_SET',label=label,label_source=evidence,publisher=PUBLISHER,
                            extractor_version=RULES['version']))
        arrays.append((s.features(),np.frombuffer(bytes(s.prefix).ljust(512,b'\0'),dtype=np.uint8).copy()))
        statuses['accepted_same_packet_inputs']+=1
    metadata=dict(capture_start_time=first,capture_end_time=last,link_type=link,total_packets=packets,
                  packet_exclusion_reasons=dict(excluded),segment_status_counts=dict(statuses),excluded_by_candidate_label=dict(label_exclusions),
                  accepted_label_counts=dict(Counter(r['label'] for r in records)))
    return records,arrays,metadata,packet_fingerprints


def build_plan(raw_root,count):
    malicious=raw_root/'PCAPs/extracted_pcaps/malicuous'
    pcaps={p.stem:p for p in malicious.glob('*.pcap')}
    jobs=[]
    for tool in LABELS[1:]:
        candidates=[]
        for csvpath in (raw_root/'CSVs/CSVs'/tool/'Separate').glob('*.csv'):
            p=pcaps.get(csvpath.stem)
            if p is not None and 100000<p.stat().st_size<3000000:candidates.append((p.stat().st_size,str(p),csvpath))
        for _,p,c in sorted(candidates)[:count]:jobs.append(dict(capture=str(Path(p).resolve()),csv=str(c.resolve()),group=tool,
                                                               file_relation='exact stem within tool-specific separate release'))
    benign=raw_root/'PCAPs/extracted_pcaps/Google/small/dump.pcap'
    if not benign.exists():raise FileNotFoundError('Pilot benign capture unavailable; no arbitrary fallback')
    jobs.append(dict(capture=str(benign.resolve()),csv=None,group='browser',file_relation='CSV association unresolved; route A uses publisher IP roles'))
    return jobs


def component_groups(metadata,fingerprints):
    parents=list(range(len(metadata)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):parents[find(j)]=find(i)
    daysets=[]
    for m in metadata:
        start=int(m['capture_start_time']//86400);end=int(m['capture_end_time']//86400)
        daysets.append(set(range(start,end+1)))
    overlap=[]
    for i in range(len(metadata)):
        for j in range(i):
            shared=len(fingerprints[i]&fingerprints[j])
            same_day=bool(daysets[i]&daysets[j]);same_hash=metadata[i]['sha256']==metadata[j]['sha256']
            if same_day or same_hash or shared:union(i,j)
            if shared or same_hash:overlap.append(dict(left=i,right=j,exact_timestamp_raw_overlaps=shared,same_capture_hash=same_hash))
    ids={i:'source_'+str(find(i)) for i in range(len(metadata))}
    return ids,overlap


def run(raw_root,out,count,verify_only=False):
    if verify_only:return verify(out)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing nonempty output; original and generated versions remain separate')
    out.mkdir(parents=True,exist_ok=True);(out/'raw_working_copies').mkdir()
    jobs=build_plan(raw_root,count)
    save(out/'pilot_plan.json',dict(rules=RULES,selected_files=jobs,selection='smallest 3 captures per tool over 100KB; one fixed benign capture; resource-bounded, not representative',training_started=False))
    shutil.copy2(__file__,out/'extractor_snapshot.py')
    save(out/'feature_schema.json',dict(version=RULES['schema'],fields=FEATURES,stat_dim=25,compatible_with_legacy=False,
                                      numeric_transform='none fitted; no standardized dataset exported',rules=RULES))
    save(out/'label_rules.json',dict(publisher=PUBLISHER,retrieved='2026-11-08',resolvers=sorted(RESOLVERS),browser_clients=sorted(BROWSERS),
                                   tunnel_clients=sorted(TUNNEL_CLIENTS),acceptance=RULES['label'],
                                   limitations='Publisher topology ground truth, not independent packet content verification of malicious behavior; no blanket file-level benign label'))
    allrecords=[];allarrays=[];metadata=[];fingerprints=[];originals=[];allquarantine=[]
    for i,job in enumerate(jobs):
        original=Path(job['capture']);h=sha(original)
        target=out/'raw_working_copies'/f'{i:02d}_{original.name}'
        shutil.copyfile(original,target)
        if sha(target)!=h:raise ValueError('Copy hash mismatch')
        originals.append((original,h))
        csvinfo=None
        if job['csv']:
            c=Path(job['csv']);ch=sha(c);ct=out/'raw_working_copies'/f'{i:02d}_{c.name}';shutil.copyfile(c,ct)
            with ct.open(newline='') as f:
                rows=list(csv.DictReader(f));csvinfo=dict(sha256=ch,relative_path=str(c.relative_to(raw_root)),file_size=c.stat().st_size,
                                                       rows=len(rows),row_number_convention='parsed data-row ordinal plus header = 2 for first row',
                                                       columns=list(rows[0]) if rows else [],association=job['file_relation'],match_status='FILE_CANDIDATE_ONLY')
            originals.append((c,ch))
        with target.open('rb') as magic_handle: magic=magic_handle.read(4)
        rejected=[]
        records,arrays,meta,fps=inspect_capture(target,job['group'],h,rejected)
        for row in rejected:
            row.update(capture_index=i,capture_hash=h,extractor_version=RULES['version'])
        allquarantine.extend(rejected)
        meta.update(dataset='CIC_DOH',relative_path=str(original.relative_to(raw_root)),sha256=h,file_size=original.stat().st_size,
                    file_type='PCAPNG' if magic==b'\x0a\x0d\x0d\x0a' else 'PCAP',candidate_related_files=[job['csv']] if job['csv'] else [],
                    csv=csvinfo,group=job['group'])
        for record in records:record['capture_index']=i
        allrecords.extend(records);allarrays.extend(arrays);metadata.append(meta);fingerprints.append(fps)
        print(f'{i+1}/{len(jobs)} {job["group"]}: packets={meta["total_packets"]} accepted={len(records)}',flush=True)
    groupids,overlap=component_groups(metadata,fingerprints)
    for row in allrecords:row['source_group_id']=groupids[row['capture_index']]
    groups=sorted(set(groupids.values()));per_group=defaultdict(Counter)
    for row in allrecords:per_group[row['source_group_id']][row['label']]+=1
    # Prespecified exploratory allocation; do not optimize against model results or hide missing classes.
    split_for={g:('train','val','test')[i%3] for i,g in enumerate(groups)}
    for row in allrecords:row['proposed_split']=split_for[row['source_group_id']]
    support={split:dict(Counter(r['label'] for r in allrecords if r['proposed_split']==split)) for split in ('train','val','test')}
    missing={split:[label for label in LABELS if not support[split].get(label)] for split in support}
    X=np.stack([a for a,b in allarrays]) if allarrays else np.empty((0,25),dtype=np.float32)
    S=np.stack([b for a,b in allarrays]) if allarrays else np.empty((0,512),dtype=np.uint8)
    np.savez_compressed(out/'model_inputs.npz',stat_raw=X,seq_feature=S,sample_id=np.array([r['sample_id'] for r in allrecords]),label=np.array([r['label'] for r in allrecords]))
    frame=pd.DataFrame(allrecords)
    frame.to_parquet(out/'provenance.parquet',index=False)
    frame[['sample_id','label','source_group_id','proposed_split','packet_count','sequence_valid_length']].to_parquet(out/'samples.parquet',index=False)
    pd.DataFrame(allquarantine).to_parquet(out/'quarantine.parquet',index=False)
    save(out/'file_manifest.json',metadata)
    save(out/'split_manifest.json',dict(policy=RULES['split'],source_groups=dict(per_group),proposed_group_assignment=split_for,
                                      support=support,missing_class_support=missing,verified_independent_acquisition=False,
                                      dataset_pkl_exported=False,training_allowed=False))
    save(out/'overlap_report.json',dict(exact_packet_overlap_pairs=overlap,scope='full selected capture contents; unselected capture overlap not examined'))
    unchanged=all(sha(p)==h for p,h in originals)
    if not unchanged: raise ValueError('Original input changed during extraction')
    if len({r['sample_id'] for r in allrecords})!=len(allrecords): raise ValueError('Duplicate sample IDs; source duplication must be resolved')
    duplicates=defaultdict(list)
    for i,(a,b) in enumerate(allarrays):duplicates[hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()].append(i)
    cross=[ids for ids in duplicates.values() if len({allrecords[i]['proposed_split'] for i in ids})>1]
    audit=dict(status='PILOT_INPUTS_BUILT_NOT_TRAINING_READY',route='A',new_feature_version=RULES['schema'],accepted_samples=len(allrecords),
               labels=dict(Counter(r['label'] for r in allrecords)),quarantined_records=len(allquarantine),exclusion_reasons=dict(Counter(r['reason'] for r in allquarantine)),selected_captures=len(jobs),source_groups=len(groups),
               raw_originals_unchanged=unchanged,complete_input_duplicate_groups=sum(len(v)>1 for v in duplicates.values()),
               cross_split_duplicate_groups=len(cross),source_groups_split_disjoint=True,
               blockers=['independent acquisition groups not certified beyond selected source-day components','pilot not representative; file-scoped CSV time/statistic matching not certified']+
                        (['class coverage missing in proposed split; no dataset.pkl exported'] if any(missing.values()) else [])+
                        (['cross-split identical inputs require investigation'] if cross else []),
               training_allowed=False,independent_recomputation='pending verify subcommand')
    save(out/'audit_report.json',audit)
    save(out/'extraction_manifest.json',dict(extractor_sha256=sha(Path(__file__)),snapshot_sha256=sha(out/'extractor_snapshot.py'),
                                           dependencies={k:importlib.metadata.version(k) for k in ['numpy','dpkt','pandas','pyarrow']},rules=RULES,
                                           command=sys.argv,original_files=[dict(path=str(p),sha256=h) for p,h in originals]))
    verify(out)
    write_report(out)


def verify(out):
    """Independent packet-index replay of a deterministic sample from each capture.

    Reconstructs features/sequence from provenance-selected raw packets, without
    trusting the session accumulator output. Whole-file hashes checked again.
    """
    metadata=json.loads((out/'file_manifest.json').read_text());frame=pd.read_parquet(out/'provenance.parquet')
    checks=[]
    with np.load(out/'model_inputs.npz',allow_pickle=False) as arrays:
        ids={s:i for i,s in enumerate(arrays['sample_id'])}
        for capture_index,m in enumerate(metadata):
            selected=frame[frame.capture_index==capture_index].sort_values('sample_id').head(3)
            if selected.empty:continue
            p=out/'raw_working_copies'/f'{capture_index:02d}_{Path(m["relative_path"]).name}'
            if sha(p)!=m['sha256']:raise ValueError('Working capture changed')
            wanted=set(int(i) for row in selected.itertuples() for i in row.packet_indices)
            observed={}
            with p.open('rb') as f:
                rd=reader(f);link=rd.datalink()
                for index,(ts,raw) in enumerate(rd):
                    if index in wanted:observed[index]=(float(ts),decode(raw,link))
                    if len(observed)==len(wanted):break
            for row in selected.itertuples():
                packets=[observed[int(i)] for i in row.packet_indices]
                origin=packets[0][1]['src'];ts=np.array([t for t,p in packets]);length=np.array([p['size'] for t,p in packets],dtype=float)
                dirs=np.array([int(p['src']!=origin) for t,p in packets]);fwd=length[dirs==0];bwd=length[dirs==1];iat=np.diff(ts);duration=ts[-1]-ts[0]
                def moments(v):return [float(v.mean()),float(v.std()),float(v.max()),float(v.min())] if len(v) else [0.]*4
                rate=[len(length)/duration,length.sum()/duration,len(fwd)/duration,len(bwd)/duration] if duration else [0.]*4
                stat=np.array([duration,len(fwd),len(bwd),fwd.sum(),bwd.sum(),*moments(fwd),*moments(bwd),*moments(length),*moments(iat),*rate],dtype=np.float32)
                seq=np.frombuffer(b''.join(p['network'] for t,p in packets)[:512].ljust(512,b'\0'),dtype=np.uint8)
                index=ids[row.sample_id];ok=np.array_equal(stat,arrays['stat_raw'][index]) and np.array_equal(seq,arrays['seq_feature'][index])
                checks.append(dict(sample_id=row.sample_id,capture_index=capture_index,recomputed_equal=bool(ok)))
    save(out/'independent_recompute_checks.json',checks)
    if not checks or not all(r['recomputed_equal'] for r in checks):raise ValueError('Independent recomputation failed')
    audit=json.loads((out/'audit_report.json').read_text());audit['independent_recomputation']=f'{len(checks)} deterministic provenance-selected samples passed'
    save(out/'audit_report.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)


def write_report(out):
    a=json.loads((out/'audit_report.json').read_text());s=json.loads((out/'split_manifest.json').read_text())
    text=['# DoH 来源明确的双模态试点','','状态：**仅完成试点输入构造，不允许训练**。没有覆盖或重新使用旧29维特征，也未生成 dataset.pkl。',
          '',f'已处理 {a["selected_captures"]} 个原始采集，生成 {a["accepted_samples"]} 个同包集合双模态样本。',
          f'类别支持：{a["labels"]}。独立重算：{a["independent_recomputation"]}。',
          '', '## 已固定的定义','',
          '- 路线 A：统计和序列都来自 provenance 中同一组原始包索引；统计为新25维 IP 长度/时间定义。',
          '- 序列为匿名化 IP 包字节的512前缀，先匿名化再截断，元数据保存边界和有效长度，不称纯 payload。',
          '- 样本是明确规则下的双向包段，缺少 SYN 的中途捕获保留标记，不声称完整 TCP 会话。',
          '- 原件只读；选中文件复制到 raw_working_copies，复制前后及处理后核对哈希。',
          '- 来源字段只在旁表，不进入模型数组；未拟合任何标准化。',
          '', '## 标签依据','',
          f'[发布方说明]({PUBLISHER})给出了浏览器/隧道客户端和公共 DoH 服务器地址。仅角色吻合且 TCP443 的包段进入候选类别；未知流隔离。工具类别另依赖原始工具目录及文件语义。',
          '这是发布方拓扑下的标签规则，不是独立证明每段内容实施了恶意行为，也不是把采集文件全部标为某类。',
          '', '## 当前未通过的门槛','']
    text.extend('- '+x for x in a['blockers'])
    text+=['',f'建议分区支持：{s["support"]}。',
           '同一天、重复采集或精确重叠包合并为来源组，只检验选中文件。文件来源独立性尚未认证；不能将这份 source-day 探索划分称为无泄漏或泛化认证。',
           '正常候选目前只取一个文件，不能为了覆盖三个 split 将其随机拆开。需先补足独立来源证据，才导出联邦训练数据。',
           '', '## 输出','',
           'samples.parquet：样本索引；provenance.parquet：包索引、方向、时间、来源和标签依据；model_inputs.npz：模型输入；quarantine.parquet：逐条拒绝原因及包索引。',
           'file_manifest.json、feature_schema.json、label_rules.json、split_manifest.json、extraction_manifest.json、audit_report.json：固定规则与审计。',
           'overlap_report.json：文件间重复/精确重叠；independent_recompute_checks.json：原包索引重算。',
           '', '下一步先扩展正常来源关联并检查独立采集关系；不启动训练、不批量覆盖三套数据。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-tool',type=int,default=3)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();run(args.raw_root.resolve(),args.output.resolve(),args.per_tool,args.verify)

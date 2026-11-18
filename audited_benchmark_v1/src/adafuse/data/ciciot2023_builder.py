"""PCAP-native39D statistics + SAME-packet anonymous Ethernet byte sequence.

Uniform non-overlapping100-packet windows are used in every capture/class.
This is a documented re-extraction version, NOT a claim of reproducing official
CSV values or of5tuple flow segmentation. SourceCSV rows are never paired here.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import pickle
import struct
import time
import dpkt
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_DATA=Path('data/raw/ciciot2023/PCAP')
FEATURES=['Header_Length','Protocol Type','Time_To_Live','Rate',
 'fin_flag_number','syn_flag_number','rst_flag_number','psh_flag_number','ack_flag_number','ece_flag_number','cwr_flag_number',
 'ack_count','syn_count','fin_count','rst_count','HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC',
 'TCP','UDP','DHCP','ARP','ICMP','IGMP','IPv','LLC','Tot sum','Min','Max','AVG','Std','Tot size','IAT','Number','Variance']
assert len(FEATURES)==39
FLAGS=[1,2,4,8,16,64,128]
PORTS=[{80,8080},{443,8443},{53},{23,992},{25,465,587},{22},{6667,6697}]

def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def decode(frame):
    """Small deterministic parser; return fields for one captured Ethernet frame.

    Statistics use original full captured-frame sizes, not anonymized sizes.
    Protocol indicators are fractions; transport port indicators are heuristic
    port associations, not DPI application labels.
    """
    if len(frame)<14:raise ValueError('shortEthernet')
    ethertype=struct.unpack_from('!H',frame,12)[0];off=14
    while ethertype in (0x8100,0x88a8,0x9100):
        if len(frame)<off+4:raise ValueError('shortVLAN')
        ethertype=struct.unpack_from('!H',frame,off+2)[0];off+=4
    llc=int(ethertype<=1500)
    if llc and frame[off:off+3]==b'\xaa\xaa\x03':
        if len(frame)<off+8:raise ValueError('shortSNAP')
        ethertype=struct.unpack_from('!H',frame,off+6)[0];off+=8
    proto=ttl=header=flags=sp=dp=0;ipv=0;fragment=False
    if ethertype==0x0800:
        if len(frame)<off+20:raise ValueError('shortIPv4')
        ihl=(frame[off]&15)*4
        if frame[off]>>4!=4 or ihl<20 or len(frame)<off+ihl:raise ValueError('badIPv4')
        ipv=1;ttl=frame[off+8];proto=frame[off+9];fragment=bool(struct.unpack_from('!H',frame,off+6)[0]&0x1fff);off+=ihl
    elif ethertype==0x86dd:
        if len(frame)<off+40 or frame[off]>>4!=6:raise ValueError('shortIPv6')
        ipv=1;proto=frame[off+6];ttl=frame[off+7];off+=40
        for _ in range(12):
            if proto not in (0,43,44,51,60):break
            if len(frame)<off+2:raise ValueError('shortIPv6extension')
            nxt=frame[off];size=8 if proto==44 else (frame[off+1]+2)*4 if proto==51 else (frame[off+1]+1)*8
            if len(frame)<off+size:raise ValueError('shortIPv6extension')
            if proto==44:fragment=bool(struct.unpack_from('!H',frame,off+2)[0]&0xfff8)
            proto=nxt;off+=size
    if ipv and not fragment:
        if proto==6:
            if len(frame)<off+20:raise ValueError('shortTCP')
            sp,dp=struct.unpack_from('!HH',frame,off);header=(frame[off+12]>>4)*4;flags=frame[off+13]
            if header<20 or len(frame)<off+header:raise ValueError('badTCPheader')
        elif proto==17:
            if len(frame)<off+8:raise ValueError('shortUDP')
            sp,dp=struct.unpack_from('!HH',frame,off);header=8
        elif proto in (1,2,58):header=min(8,max(0,len(frame)-off))
    ports={sp,dp} if proto in (6,17) and not fragment else set()
    apps=[int(bool(ports&p)) for p in PORTS]
    indicators=[int(ipv and proto==6),int(ipv and proto==17),int(bool(ports&{67,68,546,547})),int(ethertype==0x0806),int(ipv and proto in (1,58)),int(ipv and proto==2),ipv,llc]
    return [header,proto,ttl]+[int(bool(flags&v)) for v in FLAGS]+apps+indicators

def anonymize(frame):
    """Same header-masking intent as originalCICIDS; include Ethernet headers.

    DPKT recomputes IPv4/TCP/UDP checksums after address/port zeroing where it
    supports the parsed packet. ARP addresses are also zeroed when present.
    """
    eth=dpkt.ethernet.Ethernet(frame);eth.src=eth.dst=b'\0'*6
    ip=eth.data
    if isinstance(ip,dpkt.ip.IP):
        ip.src=ip.dst=b'\0'*4;ip.sum=0
    elif isinstance(ip,dpkt.ip6.IP6):ip.src=ip.dst=b'\0'*16
    elif isinstance(ip,dpkt.arp.ARP):
        ip.sha=ip.tha=b'\0'*len(ip.sha);ip.spa=ip.tpa=b'\0'*len(ip.spa)
    if isinstance(ip,(dpkt.ip.IP,dpkt.ip6.IP6)):
        transport=ip.data
        if isinstance(transport,(dpkt.tcp.TCP,dpkt.udp.UDP)):
            transport.sport=transport.dport=0;transport.sum=0
    result=bytes(eth)
    if result[:12]!=b'\0'*12:raise AssertionError('MACmaskfailed')
    return result

def aggregate(fields,sizes,times):
    a=np.asarray(fields,dtype=np.float64);size=np.asarray(sizes,dtype=np.float64);ts=np.asarray(times,dtype=np.float64);n=len(a)
    duration=float(ts.max()-ts.min());rate=(n-1)/duration if n>1 and duration>0 else 0.
    codes,counts=np.unique(a[:,1].astype('int64'),return_counts=True);proto=float(codes[np.argmax(counts)])
    flags=a[:,3:10].mean(0);counts_flags=a[:,3:10].sum(0)
    variance=float(size.var(ddof=1)) if n>1 else 0.
    result=np.r_[a[:,0].mean(),proto,a[:,2].mean(),rate,flags,
       counts_flags[4],counts_flags[1],counts_flags[0],counts_flags[2],
       a[:,10:17].mean(0),a[:,17:25].mean(0),size.sum(),size.min(),size.max(),size.mean(),np.sqrt(variance),size.mean(),np.diff(ts).mean() if n>1 else 0.,n,variance].astype('float32')
    assert result.shape==(39,) and np.isfinite(result).all()
    return result

def read_records(path):
    """Yield captured packet records; keep raw packet numbering and byteoffsets."""
    with path.open('rb') as f:
        header=f.read(24);formats={b'\xd4\xc3\xb2\xa1':('<',1e6),b'\xa1\xb2\xc3\xd4':('>',1e6),b'\x4d\x3c\xb2\xa1':('<',1e9),b'\xa1\xb2\x3c\x4d':('>',1e9)}
        if len(header)!=24 or header[:4] not in formats:raise ValueError('ExpectedclassicPCAP')
        endian,scale=formats[header[:4]]
        if struct.unpack_from(endian+'I',header,20)[0]!=1:raise ValueError('ExpectedEthernetlinktype1')
        i=0
        while True:
            offset=f.tell();raw=f.read(16)
            if not raw:break
            if len(raw)!=16:raise ValueError('TruncatedPCAPrecordheader')
            sec,frac,cap,wire=struct.unpack(endian+'IIII',raw);frame=f.read(cap)
            if len(frame)!=cap:raise ValueError('TruncatedPCAPrecordpayload')
            yield i,offset,sec+frac/scale,frame,wire
            i+=1

def extract(job):
    path=Path(job['pcap']);dest=Path(job['part']);label='BENIGN' if path.stem=='BenignTraffic' else path.stem
    if dest.exists():
        info=json.loads(dest.with_suffix('.json').read_text())
        if info['generator_sha256']!=job['generator_sha256'] or info['pcap_sha256']!=sha(path) or info['window_packets']!=job['window_packets'] or info['block_packets']!=job['block_packets'] or info['part_sha256']!=sha(dest):raise ValueError('Resumehashmismatch')
        return info
    start=time.perf_counter();records=[];errors=Counter();fields=[];sizes=[];times=[];packet_ids=[];seq=bytearray();current_block=None;window_invalid=False
    raw_count=accepted=discarded_tail=truncated=regressions=0;last_time=None
    for i,offset,ts,frame,wire in read_records(path):
        raw_count+=1;block=i//job['block_packets']
        if block!=current_block:
            discarded_tail+=len(fields);fields=[];sizes=[];times=[];packet_ids=[];seq=bytearray();window_invalid=False;current_block=block
        if last_time is not None and ts<last_time:regressions+=1
        last_time=ts
        if wire>len(frame):truncated+=1
        try:field=decode(frame)
        except Exception as e:
            errors[type(e).__name__+':'+str(e)]+=1;continue
        fields.append(field);sizes.append(len(frame));times.append(ts);packet_ids.append(i);accepted+=1
        if len(seq)<1024 and not window_invalid:
            try:seq.extend(anonymize(frame)[:1024-len(seq)])
            except Exception as e:errors['sequence:'+type(e).__name__+':'+str(e)]+=1;window_invalid=True
        if len(fields)==job['window_packets']:
            if not window_invalid:
                records.append(dict(stat_raw=aggregate(fields,sizes,times),sequence1024=bytes(seq).ljust(1024,b'\0'),
                    label=label,source_pcap=path.name,group_id=f'{path.stem}:block{block}',
                    packet_indices=np.asarray(packet_ids,dtype=np.int64),start_time=float(min(times)),end_time=float(max(times)),
                    sample_unit='nonoverlapping_packet_window',label_scope='capture_scenario'))
            else:errors['discarded_sequence_invalid_windows']+=1
            fields=[];sizes=[];times=[];packet_ids=[];seq=bytearray();window_invalid=False
        if raw_count%1000000==0:print(path.name,'rawpackets',raw_count,'samples',len(records),flush=True)
    discarded_tail+=len(fields)
    temp=dest.with_suffix('.pkl.tmp')
    with temp.open('wb') as f:pickle.dump(records,f,protocol=4)
    temp.replace(dest)
    info=dict(pcap=str(path),pcap_bytes=path.stat().st_size,pcap_sha256=sha(path),generator_sha256=job['generator_sha256'],label=label,
        raw_packets=raw_count,decoded_packets=accepted,short_tail_packets_discarded=discarded_tail,capture_truncated_packets=truncated,
        timestamp_regressions=regressions,samples=len(records),groups=len(set(r['group_id'] for r in records)),errors=dict(errors),
        seconds=time.perf_counter()-start,part=str(dest),part_sha256=sha(dest),window_packets=job['window_packets'],block_packets=job['block_packets'])
    write_json(dest.with_suffix('.json'),info);print('FINISHED',path.name,info['samples'],'samples',flush=True);return info

def definitions():
    desc={k:'window mean of packet indicator; fraction in[0,1]' for k in FEATURES}
    desc.update(Header_Length='mean TCP/UDP transport header bytes; ICMP/IGMP=min(8,capturedremaining); nontransport0',
       **{'Protocol Type':'modal IP protocol number in window; ties use smaller number; nonIP0','Time_To_Live':'mean IPv4TTL/IPv6hoplimit; nonIP0','Rate':'(N-1)/(max_timestamp-min_timestamp) packets/sec; zero forzero-duration',
       'Tot sum':'sum of ORIGINAL captured Ethernet frame sizes, bytes','Min':'minimum captured frame size, bytes','Max':'maximum captured frame size, bytes',
       'AVG':'mean captured frame size, bytes','Std':'sample standard deviation of captured frame sizes, ddof1','Tot size':'mean captured frame size, bytes (intentionalsameasAVG)',
       'IAT':'mean signed adjacent captured timestamp differences, seconds','Number':'packets in window; same fixedwindow size for EVERY class',
       'Variance':'sample variance of captured frame sizes, ddof1', 'LLC':'fraction of Ethernet802.3 LLC frames; NOT a generic Ethernetindicator',
       'IPv':'fraction IPv4orIPv6','DHCP':'fraction TCP/UDP packets withport67/68/546/547'})
    for name in ['ack_count','syn_count','fin_count','rst_count']:desc[name]='number of TCP packets withcorrespondingflag in window'
    for name,ports in zip(['HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC'],PORTS):desc[name]='TCP/UDP port association fraction; ports'+str(sorted(ports))+'; notDPI'
    return [dict(index=i,name=k,definition=desc[k]) for i,k in enumerate(FEATURES)]

def selftest():
    frames=[]
    for port in [80,443]:
        tcp=dpkt.tcp.TCP(sport=10000,dport=port,seq=1,flags=dpkt.tcp.TH_SYN,win=1024,data=b'abc');tcp.off=5
        ip=dpkt.ip.IP(src=b'\x01\x02\x03\x04',dst=b'\x05\x06\x07\x08',p=6,ttl=64,data=tcp);ip.len=len(ip)
        frames.append(bytes(dpkt.ethernet.Ethernet(src=b'\x01'*6,dst=b'\x02'*6,type=0x0800,data=ip)))
    stat=aggregate([decode(f) for f in frames],list(map(len,frames)),[1.,2.])
    assert stat.shape==(39,) and stat[FEATURES.index('syn_count')]==2 and stat[FEATURES.index('syn_flag_number')]==1
    assert stat[FEATURES.index('HTTP')]==.5 and stat[FEATURES.index('HTTPS')]==.5
    assert stat[FEATURES.index('Number')]==2 and stat[FEATURES.index('Rate')]==1
    assert stat[FEATURES.index('Tot sum')]==sum(map(len,frames)) and stat[FEATURES.index('Variance')]==0
    for frame in frames:
        anon=anonymize(frame);eth=dpkt.ethernet.Ethernet(anon);assert eth.src==eth.dst==b'\0'*6 and eth.data.src==eth.data.dst==b'\0'*4
        assert eth.data.data.sport==eth.data.data.dport==0 and eth.data.data.data==b'abc'
        assert dpkt.in_cksum(anon[14:34])==0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',default='CICIoT2023',choices=['CICIoT2023'])
    p.add_argument('--pcap-dir',type=Path,default=DEFAULT_DATA);p.add_argument('--output-dir',type=Path,default=Path('data/processed/ciciot2023'))
    p.add_argument('--packets-per-window',type=int,default=100);p.add_argument('--block-packets',type=int,default=10000)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--split-seed',type=int,default=42);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    selftest()
    if a.self_test:print('SYNTHETIC FEATURE/ANONYMIZATION TESTS PASSED');return
    if a.packets_per_window<2 or a.block_packets<a.packets_per_window:raise ValueError('Invalidwindow/block')
    out=a.output_dir.resolve();out.mkdir(exist_ok=True,parents=True);parts=out/'pcap_stat39_parts';parts.mkdir(exist_ok=True)
    manifest_path=out/'manifest_pcap_stat39_v1.json'
    if manifest_path.exists():raise FileExistsError('Completedversionalreadyexists; chooseanewoutputdirectory')
    for length in [512,784,1024]:
        if (out/f'ciciov_multimodal_{length}.pkl').exists():raise FileExistsError('Outputalreadyexists')
    pcaps=sorted(p for p in a.pcap_dir.resolve().iterdir() if p.is_file() and p.suffix.lower()=='.pcap')
    if not pcaps:raise ValueError('NoPCAPfiles')
    code_sha=sha(Path(__file__).resolve());jobs=[dict(pcap=str(p),part=str(parts/(p.stem+'.pkl')),window_packets=a.packets_per_window,block_packets=a.block_packets,generator_sha256=code_sha) for p in pcaps]
    write_json(out/'PROTOCOL_pcap_stat39.json',dict(version='pcap_stat39_window_v1',sample_unit='nonoverlapping fixedpacket windows; not5tuple flows',
        window_packets=a.packets_per_window,block_packets=a.block_packets,all_classes_use_samewindow=True,feature_definitions=definitions(),
        source_official_CSV_used=False,official_CSV_equivalence_claimed=False,sequence='anonymouswholeEthernetframes from SAMEwindow; concat then rightpad/truncate',
        addresses_ports='MAC/IP/TCP/UDPports zeroed; parsedIP/TCP/UDPchecksums recomputed byDPKT',label_scope='capture_scenario',generator_sha256=code_sha))
    infos=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(extract,j) for j in jobs]
        for f in as_completed(futures):infos.append(f.result())
    infos=sorted(infos,key=lambda x:x['pcap']);records=[]
    for info in infos:
        with Path(info['part']).open('rb') as f:records.extend(pickle.load(f))
    original_count=len(records);seen={};conflicts=set();dedup=Counter()
    for i,r in enumerate(records):
        digest=hashlib.sha256(r['stat_raw'].tobytes()+r['sequence1024'][:512]).hexdigest()
        r['input_hash_stat39_seq512']=digest;r['source_window_index']=i
        if digest in seen:
            if records[seen[digest]]['label']!=r['label']:conflicts.add(digest)
            dedup[r['label']]+=1
        else:seen[digest]=i
    records=[records[i] for digest,i in seen.items() if digest not in conflicts]
    labels=np.asarray([r['label'] for r in records]);groups=np.asarray([r['group_id'] for r in records]);split={k:[] for k in ['train','validation','test']}
    for label in sorted(set(labels)):
        gs=np.unique(groups[labels==label]);rng=np.random.default_rng(a.split_seed);rng.shuffle(gs)
        if len(gs)<3:raise ValueError((label,'need3independent rawpacketblocks'))
        ntest=max(1,int(round(.15*len(gs))));nval=max(1,int(round(.15*len(gs))));ntrain=len(gs)-ntest-nval
        if ntrain<1:raise ValueError('Insufficienttraininggroups')
        for name,selected in [('train',gs[:ntrain]),('validation',gs[ntrain:ntrain+nval]),('test',gs[ntrain+nval:])]:
            split[name].extend(np.flatnonzero((labels==label)&np.isin(groups,selected)).tolist())
    split={k:np.asarray(sorted(v),dtype=np.int64) for k,v in split.items()}
    for i,name in enumerate(split):
        for other in list(split)[i+1:]:assert not set(groups[split[name]])&set(groups[split[other]])
    stats=np.stack([r['stat_raw'] for r in records]);mean=stats[split['train']].mean(0,dtype=np.float64);std=np.maximum(stats[split['train']].std(0,dtype=np.float64),1e-8)
    np.savez_compressed(out/f'split_indices_seed{a.split_seed}.npz',**split)
    np.savez_compressed(out/'scaler_train_only_stat39.npz',mean=mean,std=std)
    output_info=[]
    for length in [512,784,1024]:
        target=out/f'ciciov_multimodal_{length}.pkl';temp=target.with_suffix('.pkl.tmp')
        result=[]
        for i,r in enumerate(records):
            sample={k:v for k,v in r.items() if k!='sequence1024'};sample['seq_feature']=np.frombuffer(r['sequence1024'][:length],dtype=np.uint8).copy();sample['source_index']=i;result.append(sample)
        with temp.open('wb') as f:pickle.dump(result,f,protocol=4)
        temp.replace(target);output_info.append(dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,samples=len(result),seq_len=length))
        print('WROTE',target.name,len(result),'samples',flush=True);del result
    counts=dict(Counter(labels));splitcounts={k:dict(Counter(labels[v])) for k,v in split.items()}
    write_json(manifest_path,dict(dataset=a.dataset,version='pcap_stat39_window_v1',stat_dim=39,feature_order=FEATURES,feature_definitions=definitions(),
        sample_count=len(records),label_counts=counts,source_pcaps=infos,outputs=output_info,original_window_count=original_count,
        exact_input_deduplication='keepfirst identicalstat39+anonymousseq512; common retainedsamples for allsequence lengths; dropallconflictinglabel groups',
        repeated_windows_by_label=dict(dedup),conflicting_input_groups=len(conflicts),
        window_packets=a.packets_per_window,block_packets=a.block_packets,sample_unit='packetwindow; not5tupleflow',
        label_scope='capture_scenario',source_CSV_used=False,official_CSV_equivalence_claimed=False,
        split_seed=a.split_seed,split_label_counts=splitcounts,split_scope='rawpacketblockdisjoint within availablecaptures; notindependentcapture/session',
        scaler_fit_partition='trainindices only',generator_sha256=code_sha))
    write_json(out/'STATUS_pcap_stat39.json',dict(phase='generated_pending_independent_validation',samples=len(records),labels=counts))
    print('PCAP MULTIMODAL GENERATION COMPLETE',len(records),counts,flush=True)

if __name__=='__main__':main()

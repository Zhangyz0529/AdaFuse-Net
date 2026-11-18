"""New, explicitly NFStream-native inputs. Not the historical B7 77D adapter."""
import struct
from nfstream import NFPlugin, NFStreamer
import numpy as np

DIRECTIONS=['bidirectional','src2dst','dst2src']
FEATURES=[f'{d}_{s}' for d in DIRECTIONS for s in ['duration_ms','packets','bytes']]
FEATURES += [f'{d}_{s}' for d in DIRECTIONS for s in
             ['min_ps','mean_ps','stddev_ps','max_ps','min_piat_ms','mean_piat_ms','stddev_piat_ms','max_piat_ms']]
FEATURES += [f'{d}_{flag}_packets' for d in DIRECTIONS for flag in ['syn','cwr','ece','urg','ack','psh','rst','fin']]
assert len(FEATURES)==57


def canonical(src,sp,dst,dp,proto):
    return repr((tuple(sorted(((str(src),int(sp)),(str(dst),int(dp))))),int(proto)))


class AnonymousIPBytes(NFPlugin):
    """Concatenate first 512 IP-layer bytes, address/port/checksum bytes zeroed.

    No Ethernet reconstruction, checksum recomputation or microsecond claim.
    Source differs from the original anonymous Ethernet-packet representation.
    """
    def on_init(self,packet,flow):
        flow.udps.sequence=bytearray()
        flow.udps.sequence_error=''
        self.on_update(packet,flow)

    def on_update(self,packet,flow):
        if len(flow.udps.sequence)>=512 or flow.udps.sequence_error:return
        raw=bytearray(packet.ip_packet)
        try:
            if not raw:raise ValueError('empty IP packet')
            version=raw[0]>>4
            if version==4:
                ihl=(raw[0]&15)*4
                if ihl<20 or len(raw)<ihl:raise ValueError('short IPv4')
                proto=raw[9];fragment=struct.unpack_from('!H',raw,6)[0]&0x1fff
                raw[10:12]=b'\0'*2;raw[12:20]=b'\0'*8
                offset=ihl if not fragment else None
            elif version==6:
                if len(raw)<40:raise ValueError('short IPv6')
                proto=raw[6];raw[8:40]=b'\0'*32
                if proto not in (6,17):raise ValueError('IPv6 extension/unsupported L4')
                offset=40
            else:raise ValueError('unsupported IP version')
            if offset is not None and proto in (6,17):
                required=20 if proto==6 else 8
                if len(raw)<offset+required:raise ValueError('short transport header')
                raw[offset:offset+4]=b'\0'*4
                check=offset+(16 if proto==6 else 6);raw[check:check+2]=b'\0'*2
            flow.udps.sequence.extend(raw[:512-len(flow.udps.sequence)])
        except Exception as error:
            flow.udps.sequence_error=str(error)

    def on_expire(self,flow):
        flow.udps.sequence=bytes(flow.udps.sequence).ljust(512,b'\0')[:512]


def configuration(pcap):
    return dict(source=str(pcap),statistical_analysis=True,n_meters=1,
      idle_timeout=120,active_timeout=1800,accounting_mode=3,decode_tunnels=False,
      snapshot_length=1536,n_dissections=0,splt_analysis=0,max_nflows=0,system_visibility_mode=0)


def streamer(pcap):
    return NFStreamer(**configuration(pcap),udps=AnonymousIPBytes())


def inputs(flow):
    if flow.udps.sequence_error:raise ValueError(flow.udps.sequence_error)
    stat=np.asarray([getattr(flow,k) for k in FEATURES],dtype=np.float64)
    if stat.shape!=(57,) or not np.isfinite(stat).all():raise ValueError('invalid native statistics')
    seq=np.frombuffer(flow.udps.sequence,dtype=np.uint8)
    if seq.shape!=(512,):raise ValueError('invalid sequence')
    return stat,seq


def metadata(flow):
    return dict(key=canonical(flow.src_ip,flow.src_port,flow.dst_ip,flow.dst_port,flow.protocol),
      start_ms=int(flow.bidirectional_first_seen_ms),end_ms=int(flow.bidirectional_last_seen_ms),
      packets=int(flow.bidirectional_packets),ip_version=int(flow.ip_version),vlan_id=int(flow.vlan_id),
      protocol=int(flow.protocol),src_ip=flow.src_ip,dst_ip=flow.dst_ip,
      src_port=int(flow.src_port),dst_port=int(flow.dst_port))


def identity(m):
    return (m['key'],m['start_ms'],m['end_ms'],m['packets'])

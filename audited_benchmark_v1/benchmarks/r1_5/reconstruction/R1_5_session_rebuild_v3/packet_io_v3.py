"""Explicit raw-frame accounting for current IPv4 TCP/UDP source adapter."""
import hashlib,socket,struct

def decode_accounting(raw,link):
 end=len(raw);net=0
 if link==1:
  if end<14:raise ValueError('truncated_ethernet_header')
  eth=struct.unpack_from('!H',raw,12)[0];net=14
  while eth in (0x8100,0x88a8):
   if net+4>end:raise ValueError('truncated_vlan_header')
   eth=struct.unpack_from('!H',raw,net+2)[0];net+=4
  if eth!=0x0800:raise ValueError('unsupported_network_family')
 elif link==113:
  if end<16:raise ValueError('truncated_linux_cooked_header')
  if struct.unpack_from('!H',raw,14)[0]!=0x0800:raise ValueError('unsupported_network_family')
  net=16
 elif link not in [101,228]:raise ValueError('unsupported_link_type')
 if net+20>end or raw[net]>>4!=4:raise ValueError('invalid_or_unsupported_ip_header')
 ihl=(raw[net]&15)*4;total=struct.unpack_from('!H',raw,net+2)[0];frag=struct.unpack_from('!H',raw,net+6)[0]
 if frag&0x3fff:raise ValueError('fragment_requires_reassembly')
 if ihl<20 or total<ihl or net+total>end:raise ValueError('truncated_or_invalid_ip_packet')
 proto=raw[net+9];tr=net+ihl
 if proto not in [6,17]:raise ValueError('unsupported_transport')
 if tr+(20 if proto==6 else 8)>net+total:raise ValueError('truncated_transport_header')
 hdr=(raw[tr+12]>>4)*4 if proto==6 else 8
 if hdr<(20 if proto==6 else 8) or tr+hdr>net+total:raise ValueError('invalid_transport_header_length')
 sport,dport=struct.unpack_from('!HH',raw,tr);src=(socket.inet_ntoa(raw[net+12:net+16]),sport);dst=(socket.inet_ntoa(raw[net+16:net+20]),dport)
 return dict(src=src,dst=dst,proto=proto,flags=raw[tr+13] if proto==6 else 0,isn=struct.unpack_from('!I',raw,tr+4)[0] if proto==6 else None,payload_len=total-ihl-hdr,capture_payload_len=end-tr-hdr,ethernet_padding_len=end-net-total,ip_header_len=ihl,transport_header_len=hdr,raw_sha256=hashlib.sha256(raw).hexdigest())

"""Capture record-header audit. Reads all record headers; does not infer sessions."""
import argparse,csv,json,struct,time
from pathlib import Path
import dpkt

def scan(path):
 n=0;lo=hi=None;capbytes=0
 with path.open('rb') as f:
  magic=f.read(4);formats={b'\xd4\xc3\xb2\xa1':('<',1e6),b'\xa1\xb2\xc3\xd4':('>',1e6),b'\x4d\x3c\xb2\xa1':('<',1e9),b'\xa1\xb2\x3c\x4d':('>',1e9)}
  if magic in formats:
   order,unit=formats[magic];header=f.read(20)
   if len(header)!=20:raise ValueError('Truncated global header')
   size=path.stat().st_size
   while True:
    h=f.read(16)
    if not h:break
    if len(h)!=16:raise ValueError('Truncated packet header')
    sec,sub,cap,wire=struct.unpack(order+'IIII',h)
    if f.tell()+cap>size:raise ValueError('Truncated captured packet')
    stamp=sec+sub/unit;lo=stamp if lo is None else min(lo,stamp);hi=stamp if hi is None else max(hi,stamp);n+=1;capbytes+=cap;f.seek(cap,1)
  else:
   f.seek(0)
   for stamp,packet in dpkt.pcapng.Reader(f):
    lo=float(stamp) if lo is None else min(lo,float(stamp));hi=float(stamp) if hi is None else max(hi,float(stamp));n+=1;capbytes+=len(packet)
 return n,lo,hi,capbytes

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 roots={'USTC':['USTC-TFC2016-master/Benign','USTC-TFC2016-master/Malware'],'ISCX':['ISCX/pcap'],'CIC_IDS':['CIC_IDS/pcap'],'CIC_DDOS':['CIC-DDoS-2019/Test_Day_03-11/PCAPs','CIC-DDoS-2019/Train_Day_01-12/PCAPs'],'CIC_DOH':['DoHBrw-2020/PCAPs/extracted_pcaps']}
 with (a.output/'capture_metadata.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=['dataset','source_file','bytes','packet_records','first_epoch','last_epoch','captured_bytes','status']);w.writeheader();summ=[]
  for dataset,dirs in roots.items():
   ok=bad=0
   for directory in dirs:
    for f in sorted((a.data_root/directory).rglob('*')):
     if not f.is_file() or f.suffix.lower() not in ['.pcap','.pcapng']:continue
     row=dict(dataset=dataset,source_file=str(f.relative_to(a.data_root)),bytes=f.stat().st_size)
     try:
      n,lo,hi,cap=scan(f);row.update(packet_records=n,first_epoch=lo,last_epoch=hi,captured_bytes=cap,status='record_headers_read');ok+=1
     except Exception as e:row['status']='ERROR '+str(e);bad+=1
     w.writerow(row);h.flush()
   summ.append(dict(dataset=dataset,files_read=ok,errors=bad));print(summ[-1],flush=True)
  (a.output/'capture_scan_summary.json').write_text(json.dumps({'datasets':summ,'limitation':'Record headers only. No packet payload or flow reconstruction; no capture independence inference.'},indent=2))
if __name__=='__main__':main()

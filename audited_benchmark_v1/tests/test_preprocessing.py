import unittest,tempfile,json,pickle,socket
from pathlib import Path
import dpkt
import numpy as np
import pandas as pd
from adafuse.data.preprocess import csv_pcap
class PreprocessingTests(unittest.TestCase):
    def test_field_order_and_packet_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);u=dpkt.udp.UDP(sport=1234,dport=80,data=b'abc');u.ulen=len(u);ip=dpkt.ip.IP(src=socket.inet_aton('192.0.2.1'),dst=socket.inet_aton('192.0.2.2'),p=17,data=u);ip.len=len(ip);eth=dpkt.ethernet.Ethernet(src=b'ABCDEF',dst=b'UVWXYZ',type=0x800,data=ip)
            with (p/'test.pcap').open('wb') as f:w=dpkt.pcap.Writer(f);w.writepkt(bytes(eth),ts=1.)
            row={'Source IP':'192.0.2.1','Source Port':1234,'Destination IP':'192.0.2.2','Destination Port':80,'Protocol':17,'Timestamp':'01/01/1970 00:00:01','Flow Duration':1000000,'Label':'BENIGN','fa':1.,'fb':2.};pd.DataFrame([row]).to_csv(p/'test.csv',index=False);(p/'fields.json').write_text(json.dumps(['fb','fa']))
            csv_pcap(p/'test.csv',p/'test.pcap',p/'fields.json',p/'out.pkl','%d/%m/%Y %H:%M:%S')
            with (p/'out.pkl').open('rb') as f:r=pickle.load(f)[0]
            self.assertTrue(np.array_equal(r['stat_raw'],[2.,1.]));self.assertEqual(r['seq_feature'].shape,(512,));self.assertTrue((r['seq_feature'][:12]==0).all())

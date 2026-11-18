"""Portable historical dataset extractor; see docs/data_preparation.md limitations."""
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--raw-dir',required=True)
_parser.add_argument('--output-dir',required=True)
_args=_parser.parse_args()
import os
import numpy as np
import pickle
from scapy.all import PcapReader, IP, IPv6, TCP, UDP
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')




BASE_DIR = _args.raw_dir
PCAP_DIRS = [os.path.join(BASE_DIR, "Benign"), os.path.join(BASE_DIR, "Malware")]
OUTPUT_DIR = _args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


SEQ_LENGTHS = [512, 784, 1024]
MAX_LEN = max(SEQ_LENGTHS)


USTC_BENIGN_APPS = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'Weibo', 'WorldOfWarcraft']




class FlowTracker:
    def __init__(self, flow_key):
        self.flow_key = flow_key
        self.payload = bytearray()
        self.timestamps = []
        self.fwd_sizes = []
        self.bwd_sizes = []

    def add_packet(self, pkt_bytes, ts, is_forward):

        if len(self.payload) < MAX_LEN:
            self.payload.extend(pkt_bytes)


        pkt_len = len(pkt_bytes)
        self.timestamps.append(ts)
        if is_forward:
            self.fwd_sizes.append(pkt_len)
        else:
            self.bwd_sizes.append(pkt_len)

    def compute_statistical_features(self):
        def safe_stat(arr, func, default=0.0):
            return float(func(arr)) if len(arr) > 0 else default

        fwd_mean, fwd_std = safe_stat(self.fwd_sizes, np.mean), safe_stat(self.fwd_sizes, np.std)
        fwd_max, fwd_min = safe_stat(self.fwd_sizes, np.max), safe_stat(self.fwd_sizes, np.min)

        bwd_mean, bwd_std = safe_stat(self.bwd_sizes, np.mean), safe_stat(self.bwd_sizes, np.std)
        bwd_max, bwd_min = safe_stat(self.bwd_sizes, np.max), safe_stat(self.bwd_sizes, np.min)

        all_sizes = self.fwd_sizes + self.bwd_sizes
        all_mean, all_std = safe_stat(all_sizes, np.mean), safe_stat(all_sizes, np.std)
        all_max, all_min = safe_stat(all_sizes, np.max), safe_stat(all_sizes, np.min)

        iats = np.diff(self.timestamps) if len(self.timestamps) > 1 else []
        iat_mean, iat_std = safe_stat(iats, np.mean), safe_stat(iats, np.std)
        iat_max, iat_min = safe_stat(iats, np.max), safe_stat(iats, np.min)

        duration = safe_stat([self.timestamps[-1] - self.timestamps[0]], np.mean) if len(self.timestamps) > 1 else 0.0
        fwd_pkts, bwd_pkts = len(self.fwd_sizes), len(self.bwd_sizes)
        fwd_bytes, bwd_bytes = sum(self.fwd_sizes), sum(self.bwd_sizes)

        flow_pkts_s = (fwd_pkts + bwd_pkts) / duration if duration > 0 else 0.0
        flow_bytes_s = (fwd_bytes + bwd_bytes) / duration if duration > 0 else 0.0
        fwd_pkts_s = fwd_pkts / duration if duration > 0 else 0.0
        bwd_pkts_s = bwd_pkts / duration if duration > 0 else 0.0

        return np.array([
            duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes,
            fwd_mean, fwd_std, fwd_max, fwd_min,
            bwd_mean, bwd_std, bwd_max, bwd_min,
            all_mean, all_std, all_max, all_min,
            iat_mean, iat_std, iat_max, iat_min,
            flow_pkts_s, flow_bytes_s, fwd_pkts_s, bwd_pkts_s
        ], dtype=np.float32)




def extract_from_pcap(pcap_file, label_name):
    flows = {}

    datasets = {length: [] for length in SEQ_LENGTHS}

    try:
        with PcapReader(pcap_file) as pcap_reader:
            for pkt in tqdm(pcap_reader, desc=f"   ► 同源提取 {os.path.basename(pcap_file)[:20]}"):

                has_ipv4 = IP in pkt
                has_ipv6 = IPv6 in pkt

                if (has_ipv4 or has_ipv6) and (TCP in pkt or UDP in pkt):
                    if has_ipv4:
                        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
                        proto, ip_layer = pkt[IP].proto, IP
                    else:
                        src_ip, dst_ip = pkt[IPv6].src, pkt[IPv6].dst
                        proto, ip_layer = pkt[IPv6].nh, IPv6

                    ts = float(pkt.time)

                    if TCP in pkt: src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
                    else: src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport

                    end1, end2 = f"{src_ip}:{src_port}", f"{dst_ip}:{dst_port}"
                    is_forward = True

                    if end1 > end2: flow_key = f"{end1}-{end2}-{proto}"
                    else:
                        flow_key = f"{end2}-{end1}-{proto}"
                        is_forward = False

                    if flow_key not in flows:
                        flows[flow_key] = FlowTracker(flow_key)


                    if pkt.haslayer("Ether"): pkt["Ether"].src, pkt["Ether"].dst = "00:00:00:00:00:00", "00:00:00:00:00:00"
                    pkt[ip_layer].src = "0.0.0.0" if has_ipv4 else "::"
                    pkt[ip_layer].dst = "0.0.0.0" if has_ipv4 else "::"
                    if TCP in pkt:
                        pkt[TCP].sport, pkt[TCP].dport = 0, 0
                        if hasattr(pkt[TCP], 'chksum'): del pkt[TCP].chksum
                    elif UDP in pkt:
                        pkt[UDP].sport, pkt[UDP].dport = 0, 0
                        if hasattr(pkt[UDP], 'chksum'): del pkt[UDP].chksum
                    if hasattr(pkt[ip_layer], 'chksum'): del pkt[ip_layer].chksum

                    flows[flow_key].add_packet(bytes(pkt), ts, is_forward)


        for flow_key, flow_obj in flows.items():
            if len(flow_obj.timestamps) >= 3:
                stat_feat = flow_obj.compute_statistical_features()
                payload_bytes = bytes(flow_obj.payload)


                for length in SEQ_LENGTHS:
                    seq_feature = np.frombuffer(payload_bytes.ljust(length, b'\x00')[:length], dtype=np.uint8)

                    datasets[length].append({
                        'stat_raw': stat_feat,
                        'seq_feature': seq_feature,
                        'label': label_name
                    })

    except Exception as e:
        print(f"      └─ ⚠️ 读取报错 {pcap_file}: {e}")


    print(f"      └─ 🎯 成功生成 {len(datasets[SEQ_LENGTHS[0]])} 条纯净双模态流 (已内置映射)！")
    return datasets




if __name__ == "__main__":
    final_datasets = {length: [] for length in SEQ_LENGTHS}
    mapped_benign_count = 0

    print("\n" + "="*80)
    print("🚀 启动 USTC-TFC2016 终极提取引擎 (多尺度并行 + 落地自动映射)")
    print("="*80)

    for pcap_dir in PCAP_DIRS:
        if not os.path.exists(pcap_dir): continue

        pcap_files = [f for f in os.listdir(pcap_dir) if f.endswith('.pcap') or f.endswith('.pcapng')]
        for pcap_f in pcap_files:
            original_label = pcap_f.rsplit('.', 1)[0]


            if original_label in USTC_BENIGN_APPS or "Benign" in pcap_dir:
                mapped_label = 'Benign'
            else:
                mapped_label = original_label


            ds_dict = extract_from_pcap(os.path.join(pcap_dir, pcap_f), mapped_label)

            if ds_dict:
                for length in SEQ_LENGTHS:
                    final_datasets[length].extend(ds_dict[length])

                if mapped_label == 'Benign':
                    mapped_benign_count += len(ds_dict[SEQ_LENGTHS[0]])

    print("\n" + "="*60)
    print(f"🎉 USTC-TFC2016 纯净多尺度数据集提取完毕！")
    print(f"🛡️ 成功将 {mapped_benign_count} 条良性应用流完美映射为 'Benign'")
    print(f"📊 最终可用样本数: {len(final_datasets[SEQ_LENGTHS[0]])} 条")
    print(f"📏 统计特征(左塔)维度: 【 25 维 】")
    print("="*60)


    for length in SEQ_LENGTHS:
        out_path = os.path.join(OUTPUT_DIR, f"ustc_multimodal_{length}.pkl")
        print(f"💾 正在保存 {length} 字节版本 -> {out_path}")
        with open(out_path, 'wb') as f:
            pickle.dump(final_datasets[length], f)

    print("\n✅ 三尺度完美收官！你可以直接把这些 .pkl 喂给模型打榜了！")

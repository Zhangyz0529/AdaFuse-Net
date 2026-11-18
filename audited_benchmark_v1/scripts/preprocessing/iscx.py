"""Portable historical dataset extractor; see docs/data_preparation.md limitations."""
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--raw-dir',required=True)
_parser.add_argument('--output-dir',required=True)
_args=_parser.parse_args()
import os
import pandas as pd
import numpy as np
import pickle
from scapy.all import PcapReader, IP, TCP, UDP
from tqdm import tqdm




PCAP_DIR = _args.raw_dir
OUTPUT_DIR = _args.output_dir
SEQ_LENGTHS = [512, 784, 1024]
MAX_LEN = max(SEQ_LENGTHS)




class FlowTracker:
    def __init__(self, flow_key):
        self.flow_key = flow_key
        self.payload = bytearray()
        self.timestamps = []
        self.fwd_sizes = []
        self.bwd_sizes = []
        self.is_finished = False

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
        """实时计算 24 维经典统计特征"""
        def safe_stat(arr, func, default=0.0):
            return float(func(arr)) if len(arr) > 0 else default


        fwd_mean = safe_stat(self.fwd_sizes, np.mean)
        fwd_std = safe_stat(self.fwd_sizes, np.std)
        fwd_max = safe_stat(self.fwd_sizes, np.max)
        fwd_min = safe_stat(self.fwd_sizes, np.min)

        bwd_mean = safe_stat(self.bwd_sizes, np.mean)
        bwd_std = safe_stat(self.bwd_sizes, np.std)
        bwd_max = safe_stat(self.bwd_sizes, np.max)
        bwd_min = safe_stat(self.bwd_sizes, np.min)

        all_sizes = self.fwd_sizes + self.bwd_sizes
        all_mean = safe_stat(all_sizes, np.mean)
        all_std = safe_stat(all_sizes, np.std)
        all_max = safe_stat(all_sizes, np.max)
        all_min = safe_stat(all_sizes, np.min)


        iats = np.diff(self.timestamps) if len(self.timestamps) > 1 else []
        iat_mean = safe_stat(iats, np.mean)
        iat_std = safe_stat(iats, np.std)
        iat_max = safe_stat(iats, np.max)
        iat_min = safe_stat(iats, np.min)


        duration = safe_stat([self.timestamps[-1] - self.timestamps[0]], np.mean) if len(self.timestamps) > 1 else 0.0
        fwd_pkts = len(self.fwd_sizes)
        bwd_pkts = len(self.bwd_sizes)
        fwd_bytes = sum(self.fwd_sizes)
        bwd_bytes = sum(self.bwd_sizes)

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
    datasets = {l: [] for l in SEQ_LENGTHS}

    try:
        with PcapReader(pcap_file) as pcap_reader:
            for pkt in tqdm(pcap_reader, desc=f"   ► 动态提取 {os.path.basename(pcap_file)[:20]}"):
                if IP in pkt:
                    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
                    proto = pkt[IP].proto
                    ts = float(pkt.time)
                    src_port, dst_port = 0, 0

                    if TCP in pkt:
                        src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
                    elif UDP in pkt:
                        src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
                    else:
                        continue


                    end1 = f"{src_ip}:{src_port}"
                    end2 = f"{dst_ip}:{dst_port}"
                    is_forward = True

                    if end1 > end2:
                        flow_key = f"{end1}-{end2}-{proto}"
                    else:
                        flow_key = f"{end2}-{end1}-{proto}"
                        is_forward = False

                    if flow_key not in flows:
                        flows[flow_key] = FlowTracker(flow_key)


                    if pkt.haslayer("Ether"):
                        pkt["Ether"].src = "00:00:00:00:00:00"
                        pkt["Ether"].dst = "00:00:00:00:00:00"

                    pkt[IP].src = "0.0.0.0"
                    pkt[IP].dst = "0.0.0.0"

                    if TCP in pkt:
                        pkt[TCP].sport = 0
                        pkt[TCP].dport = 0
                    elif UDP in pkt:
                        pkt[UDP].sport = 0
                        pkt[UDP].dport = 0


                    if hasattr(pkt[IP], 'chksum'):
                        del pkt[IP].chksum
                    if TCP in pkt and hasattr(pkt[TCP], 'chksum'):
                        del pkt[TCP].chksum
                    elif UDP in pkt and hasattr(pkt[UDP], 'chksum'):
                        del pkt[UDP].chksum


                    flows[flow_key].add_packet(bytes(pkt), ts, is_forward)


        for flow_key, flow_obj in flows.items():
            if len(flow_obj.timestamps) >= 3:
                stat_feat = flow_obj.compute_statistical_features()

                for length in SEQ_LENGTHS:
                    payload_bytes = bytes(flow_obj.payload)[:length]
                    seq_feature = np.frombuffer(payload_bytes.ljust(length, b'\x00'), dtype=np.uint8)

                    datasets[length].append({
                        'stat_raw': stat_feat,
                        'seq_feature': seq_feature,
                        'label': label_name
                    })

    except Exception as e:
        print(f"      └─ ⚠️ 读取报错 {pcap_file}: {e}")

    print(f"      └─ 🎯 成功生成 {len(datasets[SEQ_LENGTHS[0]])} 条纯净双模态流！")
    return datasets




if __name__ == "__main__":
    final_datasets = {l: [] for l in SEQ_LENGTHS}

    pcap_files = [f for f in os.listdir(PCAP_DIR) if f.endswith('.pcap') or f.endswith('.pcapng')]
    print(f"\n🚀 发现 {len(pcap_files)} 个 PCAP 文件，启动实时双模态提取引擎...")

    for pcap_f in pcap_files:
        label_name = pcap_f.rsplit('.', 1)[0].lower()
        pcap_path = os.path.join(PCAP_DIR, pcap_f)

        datasets = extract_from_pcap(pcap_path, label_name)
        if datasets:
            for length in SEQ_LENGTHS:
                final_datasets[length].extend(datasets[length])

    print("\n" + "="*60)
    print(f"🎉 全部提取完成！左塔统计特征已永久固化为: 【 25 维 】")
    print("="*60)

    for length in SEQ_LENGTHS:
        out_path = os.path.join(OUTPUT_DIR, f"iscx_multimodal_{length}.pkl")
        print(f"💾 正在保存 {length} 字节版本 -> {out_path} (含 {len(final_datasets[length])} 条样本)")
        with open(out_path, 'wb') as f:
            pickle.dump(final_datasets[length], f)

    print("✅ 完美收官！请去 dual_tower_model.py 把输入维度改成 25，即可直接训练！")

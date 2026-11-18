"""Portable historical dataset extractor; see docs/data_preparation.md limitations."""
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--raw-dir',required=True)
_parser.add_argument('--output-dir',required=True)
_args=_parser.parse_args()
import os
import numpy as np
import pandas as pd
import pickle
from scapy.all import PcapReader, IP, IPv6, TCP, UDP
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')




BASE_DIR = _args.raw_dir


SESSIONS = [
    {
        "name": "Test_Day_03-11",
        "csv_dir": os.path.join(BASE_DIR, "Test_Day_03-11", "CSVs", "03-11"),
        "pcap_dir": os.path.join(BASE_DIR, "Test_Day_03-11", "PCAPs")
    },
    {
        "name": "Train_Day_01-12",
        "csv_dir": os.path.join(BASE_DIR, "Train_Day_01-12", "CSVs", "01-12"),
        "pcap_dir": os.path.join(BASE_DIR, "Train_Day_01-12", "PCAPs")
    }
]


OUTPUT_DIR = _args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LENGTHS = [512, 784, 1024]
MAX_LEN = max(SEQ_LENGTHS)




def load_csv_features(csv_dir, session_name):
    print("\n" + "="*80)
    print(f"📖 正在从 {session_name} 加载未脱敏标准特征空间...")

    local_dict = {}
    feature_dim = 0
    if not os.path.exists(csv_dir):
        print(f"⚠️ 找不到目录 {csv_dir}，跳过。")
        return local_dict, 0

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    for f in csv_files:
        print(f"   ► 解析文件: {f}")
        df = pd.read_csv(os.path.join(csv_dir, f), low_memory=False)


        df.columns = df.columns.str.strip()


        meta_cols = [
            'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port',
            'Destination IP', 'Destination Port', 'Protocol',
            'Timestamp', 'SimillarHTTP', 'Inbound', 'Label'
        ]

        actual_meta_cols = [c for c in meta_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in actual_meta_cols]
        feature_dim = len(feature_cols)


        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0.0)


        required_cols = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Label']
        if not all(col in df.columns for col in required_cols):
            print(f"      └─ ⚠️ 缺少核心 5-元组列，跳过 {f}。")
            continue

        subset = df[required_cols + feature_cols]

        for _, row in subset.iterrows():
            src_ip, src_port = str(row['Source IP']).strip(), str(row['Source Port']).strip()
            dst_ip, dst_port = str(row['Destination IP']).strip(), str(row['Destination Port']).strip()
            label = str(row['Label']).strip()


            try:
                proto = str(int(float(row['Protocol'])))
            except:
                proto = str(row['Protocol']).strip()


            features = row[feature_cols].values.astype(np.float32)


            end1, end2 = f"{src_ip}:{src_port}", f"{dst_ip}:{dst_port}"
            if end1 > end2: flow_key = f"{end1}-{end2}-{proto}"
            else: flow_key = f"{end2}-{end1}-{proto}"


            if flow_key in local_dict and local_dict[flow_key]['label'] != 'BENIGN' and label == 'BENIGN':
                continue
            local_dict[flow_key] = {'features': features, 'label': label}

    print(f"✅ {session_name} CSV 载入完毕！测定官方特征维度为: 【 {feature_dim} 维 】")
    return local_dict, feature_dim




class PayloadTracker:
    def __init__(self, flow_key):
        self.flow_key = flow_key
        self.payload = bytearray()

    def add_packet(self, pkt_bytes):
        if len(self.payload) < MAX_LEN:
            self.payload.extend(pkt_bytes)

def extract_and_align_pcap(pcap_file, csv_dict):
    flows = {}
    datasets = {length: [] for length in SEQ_LENGTHS}
    matched_count = 0

    try:
        with PcapReader(pcap_file) as pcap_reader:
            for pkt in tqdm(pcap_reader, desc=f"   ► 扫描对齐 {os.path.basename(pcap_file)[:20]}"):
                has_ipv4, has_ipv6 = IP in pkt, IPv6 in pkt

                if (has_ipv4 or has_ipv6) and (TCP in pkt or UDP in pkt):
                    if has_ipv4: src_ip, dst_ip, proto, ip_layer = pkt[IP].src, pkt[IP].dst, pkt[IP].proto, IP
                    else: src_ip, dst_ip, proto, ip_layer = pkt[IPv6].src, pkt[IPv6].dst, pkt[IPv6].nh, IPv6

                    if TCP in pkt: src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
                    else: src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport

                    end1, end2 = f"{src_ip}:{src_port}", f"{dst_ip}:{dst_port}"
                    if end1 > end2: flow_key = f"{end1}-{end2}-{proto}"
                    else: flow_key = f"{end2}-{end1}-{proto}"


                    if flow_key not in csv_dict:
                        continue

                    if flow_key not in flows:
                        flows[flow_key] = PayloadTracker(flow_key)


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

                    flows[flow_key].add_packet(bytes(pkt))


        for flow_key, flow_obj in flows.items():
            payload_bytes = bytes(flow_obj.payload)
            stat_feat = csv_dict[flow_key]['features']
            label = csv_dict[flow_key]['label']

            for length in SEQ_LENGTHS:
                seq_feature = np.frombuffer(payload_bytes.ljust(length, b'\x00')[:length], dtype=np.uint8)
                datasets[length].append({
                    'stat_raw': stat_feat,
                    'seq_feature': seq_feature,
                    'label': label
                })
            matched_count += 1

    except Exception as e:
        print(f"      └─ ⚠️ 读取报错 {pcap_file}: {e}")

    print(f"      └─ 🎯 成功对齐 {matched_count} 条多模态流！")
    return datasets




if __name__ == "__main__":
    final_datasets = {length: [] for length in SEQ_LENGTHS}
    global_stat_dim = 0

    print("🚀 启动 CIC-DDoS-2019 多模态数据集生成器...")


    for session in SESSIONS:
        if not os.path.exists(session["pcap_dir"]):
            continue


        csv_dict, stat_dim = load_csv_features(session["csv_dir"], session["name"])
        global_stat_dim = max(global_stat_dim, stat_dim)

        if not csv_dict:
            continue


        print(f"🚀 阶段二：启动 {session['name']} 的 PCAP 载荷提取与多模态对齐...")
        pcap_files = [f for f in os.listdir(session["pcap_dir"]) if f.endswith('.pcap') or f.endswith('.pcapng')]

        for pcap_f in pcap_files:
            pcap_path = os.path.join(session["pcap_dir"], pcap_f)
            ds_dict = extract_and_align_pcap(pcap_path, csv_dict)

            if ds_dict:
                for length in SEQ_LENGTHS:
                    final_datasets[length].extend(ds_dict[length])


        del csv_dict

    print("\n" + "="*80)
    print(f"🎉 CIC-DDoS-2019 终极多模态数据集构建完毕！")
    print(f"📊 最终完美对齐样本总数: {len(final_datasets[SEQ_LENGTHS[0]])} 条")
    print(f"📏 左塔统计维度: 【 {global_stat_dim} 维 】")
    print(f"📏 右塔载荷维度: 【 512 / 784 / 1024 字节并行切片 】")
    print("="*80)


    for length in SEQ_LENGTHS:
        out_path = os.path.join(OUTPUT_DIR, f"cicddos2019_multimodal_{length}.pkl")
        print(f"💾 正在保存 {length} 字节版本 -> {out_path}")
        with open(out_path, 'wb') as f:
            pickle.dump(final_datasets[length], f)

    print("\n✅ 数据处理大功告成！记得在模型里将 STAT_DIM 更新为上述打印出的维度！")

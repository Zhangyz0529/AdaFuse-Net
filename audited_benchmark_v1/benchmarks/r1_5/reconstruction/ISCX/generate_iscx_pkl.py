# Exact FlowTracker class extracted via AST; historical top-level execution omitted.
import numpy as np
MAX_LEN=512
class FlowTracker:
    def __init__(self, flow_key):
        self.flow_key = flow_key
        self.payload = bytearray()
        self.timestamps = []
        self.fwd_sizes = []
        self.bwd_sizes = []
        self.is_finished = False

    def add_packet(self, pkt_bytes, ts, is_forward):
        # 1. 收集 Payload 字节 (最多 MAX_LEN)
        if len(self.payload) < MAX_LEN:
            self.payload.extend(pkt_bytes)
            
        # 2. 收集统计所需的基础数据
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

        # 报文长度统计
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

        # 时间间隔 (IAT) 统计
        iats = np.diff(self.timestamps) if len(self.timestamps) > 1 else []
        iat_mean = safe_stat(iats, np.mean)
        iat_std = safe_stat(iats, np.std)
        iat_max = safe_stat(iats, np.max)
        iat_min = safe_stat(iats, np.min)
        
        # 宏观流量特征
        duration = safe_stat([self.timestamps[-1] - self.timestamps[0]], np.mean) if len(self.timestamps) > 1 else 0.0
        fwd_pkts = len(self.fwd_sizes)
        bwd_pkts = len(self.bwd_sizes)
        fwd_bytes = sum(self.fwd_sizes)
        bwd_bytes = sum(self.bwd_sizes)
        
        flow_pkts_s = (fwd_pkts + bwd_pkts) / duration if duration > 0 else 0.0
        flow_bytes_s = (fwd_bytes + bwd_bytes) / duration if duration > 0 else 0.0
        fwd_pkts_s = fwd_pkts / duration if duration > 0 else 0.0
        bwd_pkts_s = bwd_pkts / duration if duration > 0 else 0.0

        # 返回严格的 25 维特征向量
        return np.array([
            duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes,
            fwd_mean, fwd_std, fwd_max, fwd_min,
            bwd_mean, bwd_std, bwd_max, bwd_min,
            all_mean, all_std, all_max, all_min,
            iat_mean, iat_std, iat_max, iat_min,
            flow_pkts_s, flow_bytes_s, fwd_pkts_s, bwd_pkts_s
        ], dtype=np.float32)

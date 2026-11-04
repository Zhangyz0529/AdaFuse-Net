"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class STGraph_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(STGraph_Baseline, self).__init__()
        self.side = int(math.ceil(seq_len ** 0.5))
        self.num_nodes = self.side
        self.node_feat_dim = self.side
        self.node_proj = nn.Sequential(nn.Linear(self.node_feat_dim, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.query_W = nn.Linear(64, 64)
        self.key_W = nn.Linear(64, 64)
        self.gcn_W = nn.Linear(64, 64)
        self.tcn = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.AdaptiveMaxPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        batch_size = x_seq.size(0)
        if x_seq.max() > 1.0:
            x_seq = x_seq / 255.0
        current_len = x_seq.size(1)
        target_len = self.side * self.side
        if current_len < target_len:
            x_seq = F.pad(x_seq, (0, target_len - current_len))
        elif current_len > target_len:
            x_seq = x_seq[:, :target_len]
        nodes = x_seq.view(batch_size, self.num_nodes, self.node_feat_dim).float()
        nodes_flat = nodes.view(-1, self.node_feat_dim)
        h_nodes = self.node_proj(nodes_flat).view(batch_size, self.num_nodes, 64)
        Q = self.query_W(h_nodes)
        K = self.key_W(h_nodes)
        A = torch.bmm(Q, K.transpose(1, 2)) / 64 ** 0.5
        A = F.softmax(A, dim=-1)
        h_spatial = torch.bmm(A, h_nodes)
        h_spatial = F.relu(self.gcn_W(h_spatial))
        h_spatial_t = h_spatial.transpose(1, 2)
        h_st = self.tcn(h_spatial_t).squeeze(-1)
        h_final = self.fc(h_st)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

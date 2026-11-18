"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HCMVGAE_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(HCMVGAE_Baseline, self).__init__()
        self.side = int(math.ceil(seq_len ** 0.5))
        self.num_seq_nodes = self.side
        self.node_feat_dim = self.side
        self.hidden_dim = 64
        self.stat_proj = nn.Sequential(nn.Linear(stat_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(inplace=True))
        self.seq_proj = nn.Sequential(nn.Linear(self.node_feat_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(inplace=True))
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn_mu = nn.BatchNorm1d(self.hidden_dim)
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        batch_size = x_seq.size(0)
        h_seq = x_seq.float()
        if h_seq.max() > 1.0:
            h_seq = h_seq / 255.0
        current_len = h_seq.size(1)
        target_len = self.side * self.side
        if current_len < target_len:
            h_seq = F.pad(h_seq, (0, target_len - current_len))
        elif current_len > target_len:
            h_seq = h_seq[:, :target_len]
        seq_nodes_flat = h_seq.view(-1, self.node_feat_dim)
        h_seq_nodes = self.seq_proj(seq_nodes_flat).view(batch_size, self.num_seq_nodes, self.hidden_dim)
        h_stat_node = self.stat_proj(x_stat.float()).unsqueeze(1)
        H_graph = torch.cat([h_stat_node, h_seq_nodes], dim=1)
        Q = self.W_q(H_graph)
        K = self.W_k(H_graph)
        A = torch.bmm(Q, K.transpose(1, 2)) / self.hidden_dim ** 0.5
        A = F.softmax(A, dim=-1)
        H_mu_3d = torch.bmm(A, H_graph)
        H_mu_flat = H_mu_3d.view(-1, self.hidden_dim)
        H_mu_flat = self.bn_mu(H_mu_flat)
        H_mu_3d = H_mu_flat.view(batch_size, self.num_seq_nodes + 1, self.hidden_dim)
        pooled_out = H_mu_3d.mean(dim=1)
        h_final = self.fc(pooled_out)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

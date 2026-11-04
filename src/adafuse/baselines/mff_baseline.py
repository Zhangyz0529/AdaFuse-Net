"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFF_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(MFF_Baseline, self).__init__()
        self.stat_encoder = nn.Sequential(nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        spatial_dim = seq_len // 4
        flattened_dim = 64 * spatial_dim
        self.seq_encoder = nn.Sequential(nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(flattened_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fusion_mlp = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        x_seq = x_seq.unsqueeze(1).float()
        if x_seq.max() > 1.0:
            x_seq = x_seq / 255.0
        h_stat = self.stat_encoder(x_stat.float())
        h_seq = self.seq_encoder(x_seq)
        h_concat = torch.cat([h_stat, h_seq], dim=1)
        h_final = self.fusion_mlp(h_concat)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

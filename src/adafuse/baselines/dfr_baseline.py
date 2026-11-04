"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DFR_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(DFR_Baseline, self).__init__()
        self.seq_encoder = nn.Sequential(nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3), nn.BatchNorm1d(32), nn.ELU(alpha=1.0), nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=3), nn.BatchNorm1d(32), nn.ELU(alpha=1.0), nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.Dropout(0.1), nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3), nn.BatchNorm1d(64), nn.ELU(alpha=1.0), nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=3), nn.BatchNorm1d(64), nn.ELU(alpha=1.0), nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.Dropout(0.1), nn.Flatten())
        spatial_dim = seq_len // 16
        seq_out_dim = 64 * spatial_dim
        self.seq_proj = nn.Sequential(nn.Linear(seq_out_dim, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.stat_encoder = nn.Sequential(nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1))
        self.fc = nn.Sequential(nn.Linear(128 + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        x_seq = x_seq.unsqueeze(1).float()
        raw_seq_feat = self.seq_encoder(x_seq)
        balanced_seq_feat = self.seq_proj(raw_seq_feat)
        stat_feat = self.stat_encoder(x_stat.float())
        combined_feat = torch.cat([stat_feat, balanced_seq_feat], dim=1)
        h_final = self.fc(combined_feat)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

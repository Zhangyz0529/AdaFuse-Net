"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class SmartDetector_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(SmartDetector_Baseline, self).__init__()
        self.seq_embedding = nn.Embedding(257, 128)
        self.seq_encoder = nn.Sequential(ResNetBlock1D(128, 128), ResNetBlock1D(128, 256, stride=2), ResNetBlock1D(256, 256), nn.AdaptiveAvgPool1d(1))
        self.stat_encoder = nn.Sequential(nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        self.projection_head = nn.Sequential(nn.Linear(256 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(512, embed_dim))

    def forward(self, x_stat, x_seq):
        if x_seq.max() <= 1.0:
            x_seq = torch.round(x_seq * 255.0)
        x_seq_safe = torch.clamp(x_seq, 0, 256).long()
        emb_seq = self.seq_embedding(x_seq_safe).transpose(1, 2)
        h_seq = self.seq_encoder(emb_seq).squeeze(-1)
        h_stat = self.stat_encoder(x_stat.float())
        h_concat = torch.cat([h_stat, h_seq], dim=1)
        h_final = self.projection_head(h_concat)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

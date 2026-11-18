"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractionNetwork(nn.Module):

    def __init__(self, stat_dim=25, seq_len=512, embed_dim=256):
        super(FeatureExtractionNetwork, self).__init__()
        self.stat_encoder = nn.Sequential(nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        spatial_dim = seq_len // 4
        flattened_dim = 64 * spatial_dim
        self.seq_encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(flattened_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.naive_fusion = nn.Sequential(nn.Linear(128 + 128, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))

    def forward(self, stat_x, seq_x):
        seq_x = seq_x.unsqueeze(1).float()
        if seq_x.max() > 1.0:
            seq_x = seq_x / 255.0
        stat_feat = self.stat_encoder(stat_x)
        seq_feat = self.seq_encoder(seq_x)
        combined = torch.cat([stat_feat, seq_feat], dim=1)
        out_feat = self.naive_fusion(combined)
        return out_feat

class ComparisonNetwork(nn.Module):

    def __init__(self, embed_dim=256):
        super(ComparisonNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(embed_dim * 2, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.fc(x)

class FCNet_Baseline(nn.Module):

    def __init__(self, stat_dim=25, seq_len=784, embed_dim=256):
        super(FCNet_Baseline, self).__init__()
        self.extractor = FeatureExtractionNetwork(stat_dim, seq_len, embed_dim)
        self.comparator = ComparisonNetwork(embed_dim)

    def extract_feature(self, stat_x, seq_x):
        return self.extractor(stat_x, seq_x)

    def compare(self, support_proto, query_feat):
        combined = torch.cat([support_proto, query_feat], dim=-1)
        (N_query, N_way, feat_dim) = combined.size()
        combined_flat = combined.view(N_query * N_way, feat_dim)
        scores = self.comparator(combined_flat)
        return scores.view(N_query, N_way)

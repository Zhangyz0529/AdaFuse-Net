"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustFeatureExtractor(nn.Module):

    def __init__(self, stat_encoder, seq_encoder, embed_dim=128):
        super().__init__()
        self.stat_encoder = stat_encoder
        self.seq_encoder = seq_encoder
        self.fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.smooth_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, stat, seq, training=True):
        h_stat = self.stat_encoder(stat)
        h_seq = self.seq_encoder(seq)
        combined = torch.cat([h_stat, h_seq], dim=1)
        emb = self.fusion(combined)
        if training:
            noise = torch.randn_like(emb) * self.smooth_scale.abs() * 0.01
            emb = emb + noise
        return emb

class MetaTraffic_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=512, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        from ..models.backbone import DSTA_Encoder_GLU, Sequence_Encoder_LocalGlobal
        self.stat_encoder = DSTA_Encoder_GLU(input_dim=stat_dim, embed_dim=embed_dim)
        self.seq_encoder = Sequence_Encoder_LocalGlobal(embed_dim=embed_dim)
        self.extractor = RobustFeatureExtractor(stat_encoder=self.stat_encoder, seq_encoder=self.seq_encoder, embed_dim=embed_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Identity()

    def encode(self, stat, seq, training=True):
        return self.extractor(stat, seq, training=training)

    def forward(self, x_stat, x_seq):
        h_final = self.encode(x_stat, x_seq, training=self.training)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_tensor = torch.zeros_like(h_final)
        return (h_final, fake_tensor, fake_tensor, None)

    def get_temperature(self):
        return self.temperature

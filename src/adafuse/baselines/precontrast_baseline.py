"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1, eps=1e-08)

class PreContrast_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=512, embed_dim=128, proj_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        from ..models.backbone import DSTA_Encoder_GLU, Sequence_Encoder_LocalGlobal
        self.stat_encoder = DSTA_Encoder_GLU(input_dim=stat_dim, embed_dim=embed_dim)
        self.seq_encoder = Sequence_Encoder_LocalGlobal(embed_dim=embed_dim)
        self.fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.projection_head = ProjectionHead(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=proj_dim)

    def encode(self, stat, seq):
        h_stat = self.stat_encoder(stat)
        h_seq = self.seq_encoder(seq)
        combined = torch.cat([h_stat, h_seq], dim=1)
        emb = self.fusion(combined)
        return emb

    def forward(self, x_stat, x_seq):
        h_final = self.encode(x_stat, x_seq)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_tensor = torch.zeros_like(h_final)
        return (h_final, fake_tensor, fake_tensor, None)

    def get_projection(self, stat, seq):
        emb = self.encode(stat, seq)
        return self.projection_head(emb)

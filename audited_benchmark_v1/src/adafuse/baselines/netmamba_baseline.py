"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSSMBlock(nn.Module):

    def __init__(self, d_model):
        super(SimpleSSMBlock, self).__init__()
        self.proj_in = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, groups=d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.norm(x)
        xz = self.proj_in(x)
        (x_proj, z) = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_proj.transpose(1, 2)).transpose(1, 2)
        x_act = self.act(x_conv)
        out = x_act * self.act(z)
        out = self.proj_out(out)
        return out + res

class NetMamba_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(NetMamba_Baseline, self).__init__()
        self.d_model = 128
        self.seq_embedding = nn.Embedding(257, self.d_model)
        self.stat_proj = nn.Sequential(nn.Linear(stat_dim, self.d_model), nn.LayerNorm(self.d_model), nn.SiLU())
        self.ssm_layers = nn.Sequential(SimpleSSMBlock(self.d_model), SimpleSSMBlock(self.d_model), SimpleSSMBlock(self.d_model))
        self.fc = nn.Sequential(nn.Linear(self.d_model, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        if x_seq.max() <= 1.0:
            x_seq = torch.round(x_seq * 255.0)
        x_seq_safe = torch.clamp(x_seq, 0, 256).long()
        h_seq = self.seq_embedding(x_seq_safe)
        h_stat = self.stat_proj(x_stat.float()).unsqueeze(1)
        h_combined = torch.cat([h_stat, h_seq], dim=1)
        h_ssm = self.ssm_layers(h_combined)
        pooled_out = h_ssm.mean(dim=1)
        h_final = self.fc(pooled_out)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

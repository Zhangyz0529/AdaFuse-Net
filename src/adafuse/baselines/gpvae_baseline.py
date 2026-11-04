"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):

    def __init__(self, stat_dim, seq_dim, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.stat_proj = nn.Sequential(nn.Linear(stat_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.seq_proj = nn.Sequential(nn.Linear(seq_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, stat, seq):
        h_stat = self.stat_proj(stat)
        h_seq = self.seq_proj(seq)
        h = self.fusion(torch.cat([h_stat, h_seq], dim=1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return (z, mu, logvar)

class GPVAE_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=512, embed_dim=128, latent_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        from ..models.backbone import DSTA_Encoder_GLU, Sequence_Encoder_LocalGlobal
        self.stat_encoder = DSTA_Encoder_GLU(input_dim=stat_dim, embed_dim=embed_dim)
        self.seq_encoder = Sequence_Encoder_LocalGlobal(embed_dim=embed_dim)
        self.vae_encoder = VAEEncoder(stat_dim=embed_dim, seq_dim=embed_dim, latent_dim=latent_dim, hidden_dim=256)

    def forward(self, x_stat, x_seq):
        h_stat = self.stat_encoder(x_stat)
        h_seq = self.seq_encoder(x_seq)
        (z, mu, logvar) = self.vae_encoder(h_stat, h_seq)
        h_final = F.normalize(mu, dim=1, eps=1e-08)
        fake_tensor = torch.zeros_like(h_final)
        return (h_final, fake_tensor, fake_tensor, None)

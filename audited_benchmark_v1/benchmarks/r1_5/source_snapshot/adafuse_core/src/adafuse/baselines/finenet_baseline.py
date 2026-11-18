"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FineNet_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128, nhead=4, num_layers=2):
        super(FineNet_Baseline, self).__init__()
        self.d_model = embed_dim if embed_dim % nhead == 0 else 128
        self.embedding = nn.Embedding(257, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=2000)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=self.d_model * 4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(self.d_model, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        if x_seq.max() <= 1.0:
            x_seq = torch.round(x_seq * 255.0)
        x_seq_safe = torch.clamp(x_seq, 0, 256).long()
        x = self.embedding(x_seq_safe) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        transformer_out = self.transformer(x)
        pooled_out = transformer_out.mean(dim=1)
        h_final = self.fc(pooled_out)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

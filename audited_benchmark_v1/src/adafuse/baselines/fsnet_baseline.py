"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FSNet_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128, word_embed_dim=64, hidden_size=128):
        super(FSNet_Baseline, self).__init__()
        self.embedding = nn.Embedding(257, word_embed_dim)
        self.bi_gru = nn.GRU(input_size=word_embed_dim, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x_stat, x_seq):
        if x_seq.max() <= 1.0:
            x_seq = torch.round(x_seq * 255.0)
        x_seq_safe = torch.clamp(x_seq, 0, 256).long()
        embedded = self.embedding(x_seq_safe)
        self.bi_gru.flatten_parameters()
        (out, _) = self.bi_gru(embedded)
        pooled_out = out[:, -1, :]
        h_final = self.fc(pooled_out)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

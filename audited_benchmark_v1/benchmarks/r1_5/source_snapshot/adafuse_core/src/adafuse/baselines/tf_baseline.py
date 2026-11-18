"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TF_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(TF_Baseline, self).__init__()
        self.seq_encoder = nn.Sequential(nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.Dropout(0.1), nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.Dropout(0.1), nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.Dropout(0.1), nn.Flatten())
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_len)
            dummy_out = self.seq_encoder(dummy_input)
            seq_out_dim = dummy_out.shape[1]
        self.fc = nn.Sequential(nn.Linear(seq_out_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        x_seq = x_seq.unsqueeze(1).float()
        if x_seq.max() > 1.0:
            x_seq = x_seq / 255.0
        h_seq = self.seq_encoder(x_seq)
        h_final = self.fc(h_seq)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

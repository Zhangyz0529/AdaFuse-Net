"""Local research approximation; not a certified official author reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlowPic_Baseline(nn.Module):

    def __init__(self, stat_dim=77, seq_len=784, embed_dim=128):
        super(FlowPic_Baseline, self).__init__()
        self.side = int(math.ceil(seq_len ** 0.5))
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveMaxPool2d((7, 7)), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, embed_dim))

    def forward(self, x_stat, x_seq):
        if x_seq.max() > 1.0:
            x_seq = x_seq / 255.0
        (batch_size, current_len) = x_seq.size()
        target_len = self.side * self.side
        if current_len < target_len:
            x_seq = F.pad(x_seq, (0, target_len - current_len))
        elif current_len > target_len:
            x_seq = x_seq[:, :target_len]
        x_2d = x_seq.view(-1, 1, self.side, self.side).float()
        feat_2d = self.conv_layers(x_2d)
        h_final = self.fc(feat_2d)
        h_final = F.normalize(h_final, dim=1, eps=1e-08)
        fake_h_stat = torch.zeros_like(h_final)
        return (h_final, fake_h_stat, h_final, None)

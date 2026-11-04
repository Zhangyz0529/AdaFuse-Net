import torch
import torch.nn as nn
import torch.nn.functional as F




class DSTA_Encoder_GLU(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(DSTA_Encoder_GLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.gate1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, embed_dim)
        self.gate2 = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.fc1(x) * F.silu(self.gate1(x))
        h1 = self.dropout(self.bn1(h1))
        return self.fc2(h1) * F.silu(self.gate2(h1))




class Sequence_Encoder_Transformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, max_len=1024):
        super(Sequence_Encoder_Transformer, self).__init__()
        self.embedding = nn.Embedding(257, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        if x.max() <= 1.0: x = torch.round(x * 255.0)
        x_safe = torch.clamp(x, 0, 256).long()
        seq_len = x_safe.size(1)
        x_emb = self.embedding(x_safe) + self.pos_embedding[:, :seq_len, :]
        out = torch.mean(self.transformer(x_emb), dim=1)
        return self.proj(out)

class Sequence_Encoder_TCN(nn.Module):
    def __init__(self, embed_dim=128):
        super(Sequence_Encoder_TCN, self).__init__()
        self.embedding = nn.Embedding(257, embed_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        if x.max() <= 1.0: x = torch.round(x * 255.0)
        x_safe = torch.clamp(x, 0, 256).long()
        x_emb = self.embedding(x_safe).transpose(1, 2)
        return self.cnn(x_emb).squeeze(-1)




class Sequence_Encoder_LocalGlobal(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, max_len=1024):
        super(Sequence_Encoder_LocalGlobal, self).__init__()
        self.embedding = nn.Embedding(257, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True
        )
        self.global_engine = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.local_engine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        if x.max() <= 1.0: x = torch.round(x * 255.0)
        x_safe = torch.clamp(x, 0, 256).long()
        seq_len = x_safe.size(1)

        x_emb = self.embedding(x_safe)
        x_global = x_emb + self.pos_embedding[:, :seq_len, :]
        out_global = torch.mean(self.global_engine(x_global), dim=1)
        x_local = x_emb.transpose(1, 2)
        out_local = self.local_engine(x_local).squeeze(-1)

        out_combined = torch.cat([out_global, out_local], dim=1)
        return self.fusion_proj(out_combined)




class FeatureRecalibration_SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.BatchNorm1d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(torch.abs(x))

class DynamicSoftThresholding(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 4),
            nn.BatchNorm1d(channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        abs_x = torch.abs(x)
        abs_mean = torch.mean(abs_x, dim=1, keepdim=True)
        threshold = abs_mean * self.fc(abs_x)
        return torch.sign(x) * F.relu(abs_x - threshold)




class Ablation_SE_DST_Fusion(nn.Module):
    def __init__(self, channel, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.se_expert = FeatureRecalibration_SE(channel)
        self.dst_expert = DynamicSoftThresholding(channel)
        self.w_se = nn.Parameter(torch.ones(1))
        self.w_dst = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out_se = self.se_expert(x)
        out_dst = x + self.dst_expert(x)

        if self.ablation_mode == 'base':
            return x
        elif self.ablation_mode == 'se_only':
            return out_se
        elif self.ablation_mode == 'dst_only':
            return out_dst
        elif self.ablation_mode == 'add':
            return out_se + out_dst
        else:
            return (self.w_se * out_se) + (self.w_dst * out_dst)




class UniversalDSTANet(nn.Module):
    def __init__(self, stat_dim, embed_dim=128, tower_mode='dual', enc_type='local_global', dropout=0.1, ablation_mode='full', **kwargs):
        super(UniversalDSTANet, self).__init__()
        self.tower_mode = tower_mode
        self.embed_dim = embed_dim

        self.left_tower = DSTA_Encoder_GLU(input_dim=stat_dim, embed_dim=embed_dim)

        if enc_type == 'local_global':
            self.right_tower = Sequence_Encoder_LocalGlobal(embed_dim=embed_dim)
        elif enc_type == 'transformer':
            self.right_tower = Sequence_Encoder_Transformer(embed_dim=embed_dim)
        else:
            self.right_tower = Sequence_Encoder_TCN(embed_dim=embed_dim)


        self.fusion_block = Ablation_SE_DST_Fusion(channel=embed_dim, ablation_mode=ablation_mode)

        if self.tower_mode == 'dual':
            self.modality_attention = nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Sigmoid()
            )
            proj_in_dim = embed_dim * 2
        else:
            proj_in_dim = embed_dim

        self.projection_head = nn.Sequential(
            nn.Linear(proj_in_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x_stat, x_seq, return_ib_penalty=False):
        ib_penalty = torch.tensor(0.0).to(x_seq.device)

        if self.tower_mode == 'seq_only':
            h_final = self.right_tower(x_seq)
        elif self.tower_mode == 'stat_only':
            h_final = self.left_tower(x_stat)
        else:
            h_stat = self.left_tower(x_stat)
            h_seq_raw = self.right_tower(x_seq)


            h_seq_clean = self.fusion_block(h_seq_raw)

            h_concat = torch.cat([h_stat, h_seq_clean], dim=1)
            weights = self.modality_attention(h_concat)
            w_stat, w_seq = weights[:, 0:1], weights[:, 1:2]

            h_stat_gated = h_stat * w_stat
            h_seq_gated = h_seq_clean * w_seq

            if return_ib_penalty:
                ib_penalty = torch.mean(torch.abs(w_stat - 0.5)) + torch.mean(torch.abs(w_seq - 0.5))

            h_final = torch.cat([h_stat_gated, h_seq_gated], dim=1)

        z = F.normalize(self.projection_head(h_final), dim=1)

        if return_ib_penalty: return z, ib_penalty
        return z

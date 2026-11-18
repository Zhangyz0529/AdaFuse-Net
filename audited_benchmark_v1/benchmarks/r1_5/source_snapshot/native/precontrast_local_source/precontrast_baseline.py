"""
Pre-Contrast Baseline for Encrypted Traffic Classification
Reference: "When Pre-Training Meets Contrast Learning: Few-Shot Encrypted 
            Traffic Classification With Novelty Detection" TON 2026

核心思想：预训练 + 监督对比学习，在 few-shot 场景下用对比损失拉紧同类样本。
由于我们没有预训练阶段，采用端到端的 ProtoNet + SupCon 联合训练。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """对比学习投影头（训练时用于计算对比损失，分类时可用可不用）"""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1, eps=1e-8)


class PreContrast_Baseline(nn.Module):
    """
    Pre-Contrast 基线模型
    双塔结构 + 监督对比学习正则
    """
    def __init__(self, stat_dim=77, seq_len=512, embed_dim=128, proj_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        # 🔥 复用您的编码器
        from universal_model import DSTA_Encoder_GLU, Sequence_Encoder_LocalGlobal
        self.stat_encoder = DSTA_Encoder_GLU(input_dim=stat_dim, embed_dim=embed_dim)
        self.seq_encoder = Sequence_Encoder_LocalGlobal(embed_dim=embed_dim)

        # 多模态融合（拼接 + MLP）
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 投影头（用于对比损失，与分类 embedding 共用主干但不同头）
        self.projection_head = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=proj_dim
        )

    def encode(self, stat, seq):
        """提取融合后的 embedding"""
        h_stat = self.stat_encoder(stat)
        h_seq = self.seq_encoder(seq)
        combined = torch.cat([h_stat, h_seq], dim=1)
        emb = self.fusion(combined)
        return emb

    def forward(self, x_stat, x_seq):
        """
        Returns:
            h_final: [B, embed_dim] 融合后的 embedding（用于原型分类）
            以及三个占位符
        """
        h_final = self.encode(x_stat, x_seq)
        h_final = F.normalize(h_final, dim=1, eps=1e-8)

        fake_tensor = torch.zeros_like(h_final)
        return h_final, fake_tensor, fake_tensor, None

    def get_projection(self, stat, seq):
        """获取投影头输出（用于对比损失计算，在外部训练循环中调用）"""
        emb = self.encode(stat, seq)
        return self.projection_head(emb)
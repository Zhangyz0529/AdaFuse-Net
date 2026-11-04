"""Factorial switches over the frozen V21 backbone, without legacy penalties.

All modules are instantiated in the same order for matching initial weights.
Inactive ADEF/gate modules are never executed and have gradients disabled.
The legacy backbone keeps its actual dropout rates (.3 stat / .1 Transformer);
its historical dropout argument is unused. This is recorded in the manifest.
"""
import torch
import torch.nn.functional as F
from .backbone import UniversalDSTANet


class FactorialNet(UniversalDSTANet):
    def __init__(self, use_adef, use_gating, use_dec, embed_dim=256, stat_dim=77):
        super().__init__(stat_dim=stat_dim, embed_dim=embed_dim, tower_mode='dual', ablation_mode='full')
        self.use_adef = bool(use_adef)
        self.use_gating = bool(use_gating)
        self.use_dec = bool(use_dec)
        self.fusion_block.requires_grad_(self.use_adef)
        self.modality_attention.requires_grad_(self.use_gating)

    def forward(self, x_stat, x_seq, return_parts=False):
        h_stat = self.left_tower(x_stat)
        h_seq_raw = self.right_tower(x_seq)
        h_clean = self.fusion_block(h_seq_raw) if self.use_adef else h_seq_raw
        if self.use_gating:
            w = self.modality_attention(torch.cat([h_stat, h_clean], dim=1))
            fused = torch.cat([h_stat*w[:, :1], h_clean*w[:, 1:]], dim=1)
        else:
            w = torch.ones((len(h_stat), 2), device=h_stat.device, dtype=h_stat.dtype)
            fused = torch.cat([h_stat, h_clean], dim=1)
        z = F.normalize(self.projection_head(fused), dim=1)
        if return_parts:
            return {'z': z, 'h_stat': h_stat, 'h_seq_raw': h_seq_raw,
                    'h_clean': h_clean, 'w_stat': w[:, :1], 'w_seq': w[:, 1:]}
        return z

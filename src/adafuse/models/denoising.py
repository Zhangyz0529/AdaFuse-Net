"""Same noise-training structure for rec0 and rec0.1; decoder excluded at inference."""
from torch import nn
from .fusion_variants import FusionNet
class DenoisingNet(FusionNet):
    def __init__(self,stat_dim=77):
        super().__init__('Concatenation',stat_dim=stat_dim)
        self.reconstruction_head=nn.Sequential(nn.Linear(256,512*16),nn.Unflatten(1,(512,16)),nn.Linear(16,257))

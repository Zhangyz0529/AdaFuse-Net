"""User-authorized original project variant, explicitly not author reproduction."""
import sys
from pathlib import Path
from torch import nn
SOURCE=Path(__file__).resolve().parent/'precontrast_local_source'
sys.path.insert(0,str(SOURCE))
from precontrast_baseline import PreContrast_Baseline
from training_loss import supervised_contrastive_loss
class PreContrastLocal(nn.Module):
 def __init__(self,d,n):
  super().__init__();self.encoder=PreContrast_Baseline(stat_dim=d,seq_len=512,embed_dim=128)
 def features(self,s,q):return self.encoder(s,q.long())[0]
 def episode_loss(self,support,query,labels):return supervised_contrastive_loss(__import__('torch').cat([support,query]),labels,temperature=.07)
def build(name,d,n):
 assert name=='Pre-Contrast-Local'
 return PreContrastLocal(d,n)

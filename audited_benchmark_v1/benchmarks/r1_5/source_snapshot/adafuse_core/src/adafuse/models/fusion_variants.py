"""Six same-backbone fusion variants; no dynamic expert router is invented."""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from .network import FactorialNet
VARIANTS=['Concatenation','FiLM','Cross-Attention','AdaFuse-Net','Residual-MLP','Standard-DST']
class StandardDST(nn.Module):
 def __init__(self,old):
  super().__init__();self.se_expert=old.se_expert;self.dst_expert=old.dst_expert;self.w_se=old.w_se;self.w_dst=old.w_dst
 def forward(self,x):return self.w_se*self.se_expert(x)+self.w_dst*self.dst_expert(x)
class FusionNet(FactorialNet):
 def __init__(self,variant,stat_dim=77,embed_dim=256):
  assert variant in VARIANTS
  if embed_dim!=256:raise ValueError('Alternative fusion implementations require embed_dim256')
  gate=variant in ['AdaFuse-Net','Residual-MLP','Standard-DST'];super().__init__(True,gate,True,embed_dim=embed_dim,stat_dim=stat_dim);self.variant=variant
  if variant=='Residual-MLP':
   # ADEF66434 params; MLP66176 params, difference258 (~0.39%).
   self.fusion_block=nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128,256))
  elif variant=='Standard-DST':self.fusion_block=StandardDST(self.fusion_block)
  elif variant in ['Concatenation','FiLM','Cross-Attention']:
   self.fusion_block=nn.Identity();self.modality_attention=nn.Identity()
   if variant=='FiLM':self.film=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,512))
   if variant=='Cross-Attention':
    self.stat_to_seq=nn.MultiheadAttention(16,4,batch_first=True);self.seq_to_stat=nn.MultiheadAttention(16,4,batch_first=True);self.stat_norm=nn.LayerNorm(16);self.seq_norm=nn.LayerNorm(16)
 def forward(self,xs,xq,return_parts=False):
  hs=self.left_tower(xs);hr=self.right_tower(xq);stat=hs
  if self.variant=='Residual-MLP':hc=hr+self.fusion_block(hr)
  elif self.variant=='FiLM':
   gamma,beta=self.film(hs).chunk(2,dim=1);hc=(1+gamma.tanh())*hr+beta
  elif self.variant=='Cross-Attention':
   #16 channel-group tokens of width16, not packet/time tokens or single-key attention.
   st=hs.reshape(-1,16,16);sq=hr.reshape(-1,16,16);delta_s,_=self.seq_to_stat(st,sq,sq,need_weights=False);delta_q,_=self.stat_to_seq(sq,st,st,need_weights=False);stat=self.stat_norm(st+delta_s).reshape(-1,256);hc=self.seq_norm(sq+delta_q).reshape(-1,256)
  else:hc=self.fusion_block(hr)
  if self.use_gating:w=self.modality_attention(torch.cat([stat,hc],1))
  else:w=torch.ones((len(hs),2),device=hs.device,dtype=hs.dtype)
  z=F.normalize(self.projection_head(torch.cat([stat*w[:,:1],hc*w[:,1:]],1)),dim=1)
  return dict(z=z,h_stat=stat,h_seq_raw=hr,h_clean=hc,w_stat=w[:,:1],w_seq=w[:,1:]) if return_parts else z

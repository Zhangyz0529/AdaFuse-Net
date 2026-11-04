"""Local architecture adapters; no author-reproduction or pretrained claim."""
import importlib
import torch
from torch import nn
REGISTRY={'HC-MVGAE': ('hc_mvgae_baseline', 'HCMVGAE_Baseline'), 'MetaMimetic': ('metamimetic_baseline', 'MetaMimetic_Baseline'), 'MFF': ('mff_baseline', 'MFF_Baseline'), 'NetMamba': ('netmamba_baseline', 'NetMamba_Baseline'), 'SmartDetector': ('smartdetector_baseline', 'SmartDetector_Baseline'), 'ST-Graph': ('stgraph_baseline', 'STGraph_Baseline'), 'TF': ('tf_baseline', 'TF_Baseline'), 'Trafficformer-Light': ('trafficformer_light', 'Trafficformer_Light'), 'DFR': ('dfr_baseline', 'DFR_Baseline'), 'FC-Net': ('fcnet_baseline', 'FCNet_Baseline'), 'FineNet': ('finenet_baseline', 'FineNet_Baseline'), 'FlowPic': ('flowpic_baseline', 'FlowPic_Baseline'), 'FS-Net': ('fsnet_baseline', 'FSNet_Baseline'), 'GP-VAE': ('gpvae_baseline', 'GPVAE_Baseline'), 'Pre-Contrast': ('precontrast_baseline', 'PreContrast_Baseline'), 'MetaTraffic': ('metatraffic_baseline', 'MetaTraffic_Baseline')}
class Baseline(nn.Module):
    def __init__(self,name,stat_dim,embed_dim=128):
        super().__init__();self.name=name;mod,cls=REGISTRY[name]
        self.encoder=getattr(importlib.import_module('.'+mod,__package__),cls)(stat_dim=stat_dim,seq_len=512,embed_dim=embed_dim)
    def forward(self,stat,seq,return_parts=False):
        if self.name=='FC-Net':z=self.encoder.extract_feature(stat,seq.float())
        else:
            z=self.encoder(stat,seq)
            if isinstance(z,(tuple,list)):z=z[0]
        return {'z':z} if return_parts else z

def local_supcon(z,labels,temperature=.07):
    logits=z@z.T/temperature;logits-=logits.max(1,keepdim=True).values.detach()
    off=~torch.eye(len(z),dtype=torch.bool,device=z.device);positive=labels[:,None].eq(labels[None,:])&off
    logp=logits-torch.log((logits.exp()*off).sum(1,keepdim=True)+1e-9)
    return -(logp*positive).sum(1)[positive.any(1)].mean()

"""Explicit native-mechanism adaptations, never shared AdaFuse encoders.

MFF follows published Table5: 1-D SE-ResNet18 + AE + MLP softmax.
GP-VAE imports unmodified official Open-Detect networks/model/loss at pinned commit.
"""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
VENDOR=Path(__file__).resolve().parent.parent/'R1_5_audited_benchmark_v1/vendor'
class SEBlock(nn.Module):
 def __init__(self,c):
  super().__init__();self.fc=nn.Sequential(nn.Linear(c,max(c//16,1)),nn.ReLU(),nn.Linear(max(c//16,1),c),nn.Sigmoid())
 def forward(self,x):return x*self.fc(x.mean(-1)).unsqueeze(-1)
class Residual(nn.Module):
 def __init__(self,a,b,stride):
  super().__init__();self.body=nn.Sequential(nn.Conv1d(a,b,3,stride,1,bias=False),nn.BatchNorm1d(b),nn.ReLU(),nn.Conv1d(b,b,3,1,1,bias=False),nn.BatchNorm1d(b));self.se=SEBlock(b);self.skip=nn.Identity() if a==b and stride==1 else nn.Sequential(nn.Conv1d(a,b,1,stride,bias=False),nn.BatchNorm1d(b))
 def forward(self,x):return F.relu(self.se(self.body(x))+self.skip(x))
class MFF(nn.Module):
 def __init__(self,d,n):
  super().__init__();layers=[nn.Conv1d(1,32,9,padding=4),nn.BatchNorm1d(32),nn.ReLU()];a=32
  for b,stride in [(32,1),(64,2),(128,2),(256,2)]:layers.extend([Residual(a,b,stride),Residual(b,b,1)]);a=b
  self.seq=nn.Sequential(*layers,nn.AdaptiveAvgPool1d(1),nn.Flatten());self.ae=nn.Sequential(nn.Linear(d,40),nn.ReLU(),nn.Linear(40,26));self.decoder=nn.Sequential(nn.Linear(26,40),nn.ReLU(),nn.Linear(40,d));self.head=nn.Sequential(nn.Linear(282,100),nn.ReLU(),nn.Linear(100,n))
 def features(self,s,q):return torch.cat([self.seq(q.float().unsqueeze(1)/255),self.ae(s)],1)
 def loss(self,s,q,y):
  z=self.ae(s);rec=F.mse_loss(self.decoder(z),s);ce=F.cross_entropy(self.head(torch.cat([self.seq(q.float().unsqueeze(1)/255),z],1)),y);return ce+.1*rec,dict(ce=ce,reconstruction=rec)
class GPVAE(nn.Module):
 def __init__(self,d,n):
  super().__init__();sys.path.insert(0,str(VENDOR/'Open-Detect'));from model import OpenDetectNet
  self.net=OpenDetectNet(arch='resnet18',channel=1,latent_dim=128,n_classes=n,temp_inter=1,temp_intra=1)
 def image(self,q):return F.pad(q.float()/255,(0,512)).reshape(-1,1,32,32)
 def features(self,s,q):return self.net.encoder(self.image(q))[0]
 def loss(self,s,q,y):
  _,_,_,p=self.net.loss(self.image(q),y);return .005*(p['rec']+p['kld']+p['ent'])+.995*p['dis'],p
class FlowPic(nn.Module):
 def __init__(self,d,n):
  super().__init__();self.c1=nn.Conv2d(1,10,10,stride=5);self.c2=nn.Conv2d(10,20,10,stride=5);self.fc=nn.Sequential(nn.Flatten(),nn.Linear(4500,64),nn.ReLU(),nn.Dropout(.5));self.head=nn.Linear(64,n)
 def features(self,s,q):
  x=F.max_pool2d(F.relu(self.c1(F.pad(q,(2,3,2,3)))),2);x=F.max_pool2d(F.dropout(F.relu(self.c2(F.pad(x,(2,3,2,3)))),.25,self.training),2);return self.fc(x)
 def loss(self,s,q,y):
  ce=F.cross_entropy(self.head(self.features(s,q)),y);return ce,dict(ce=ce)
def build(name,d,n):return {'MFF':MFF,'GP-VAE':GPVAE,'FlowPic':FlowPic}[name](d,n)

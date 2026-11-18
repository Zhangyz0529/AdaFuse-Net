"""Author-native MetaTraffic trainer and author-UER lightweight TrafficFormer.

Explicit adaptations: synthetic size-pattern domains for MetaTraffic; Train-only
small-model pretraining for Trafficformer-Light. No external pretrained exposure.
"""
import sys,argparse,copy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models import MFF,GPVAE,FlowPic,VENDOR
class MetaTraffic(nn.Module):
 native_optimizer=True
 def __init__(self,d,n):
  super().__init__();sys.path.insert(0,str(VENDOR/'MetaTraffic'));from masf import MASF
  from base_models import DFNet
  args=SimpleNamespace(update_lr=.001,meta_lr=.0002,embedding_lr=.0001,batch_size=48,global_weight=1.,local_weight=.005,clip_value=2.,nb_classes=n,cuda=0,IMP=True,ATS=True,AUG=True,imp_param=.1,ats_param=1.,aug_param=.1)
  self.native=MASF(args,DFNet,(5000,n),(128,64,32),3)
 def train(self,mode=True):
  self.training=mode
  if mode:self.native.train()
  else:self.native.eval()
  return self
 def features(self,s,q):return self.native.featureExtract(q.float())
 def step(self,s,q,y):
  # Same labeled Train records only; domains are explicit synthetic length scaling,
  # not asserted independent capture/network environments.
  x=torch.stack([q,q*.9,q*1.1]);labels=y.repeat(3,1);before=next(self.native.net.parameters()).detach().clone();self.native(x,labels,idx_batch=0)
  with torch.no_grad():ce=F.cross_entropy(self.native.predict(q),y);delta=(next(self.native.net.parameters())-before).norm()
  return dict(native_training_CE=float(ce),native_parameter_update_norm=float(delta))
class TrafficformerLight(nn.Module):
 def __init__(self,d,n):
  super().__init__();sys.path.insert(0,str(VENDOR/'TrafficFormer'));from uer.opts import model_opts
  from uer.layers.embeddings import WordPosSegEmbedding
  from uer.encoders.transformer_encoder import TransformerEncoder
  from uer.targets.bertflow_target import BertFlowTarget
  parser=argparse.ArgumentParser();model_opts(parser);a=parser.parse_args([]);a.hidden_size=128;a.emb_size=128;a.feedforward_size=512;a.heads_num=4;a.layers_num=2;a.dropout=.1;a.hidden_act='gelu';a.max_seq_length=512;a.is_moe=False
  self.embedding=WordPosSegEmbedding(a,65541);self.encoder=TransformerEncoder(a);self.target=BertFlowTarget(a,65541);self.head=nn.Linear(128,n)
 def tokens(self,q):
  q=q.long();big=q[:,:-1]*256+q[:,1:]+5;return torch.cat([torch.full((len(q),1),1,device=q.device,dtype=torch.long),big],1)
 def memory(self,t):
  seg=torch.ones_like(t);return self.encoder(self.embedding(t,seg),seg)
 def features(self,s,q):return self.memory(self.tokens(q))[:,0]
 def loss(self,s,q,y):
  ce=F.cross_entropy(self.head(self.features(s,q)),y);return ce,dict(ce=ce)
 def pretrain(self,q,spans):
  # Five author MSP labels: same-burst forward/reverse, different-flow burst,
  # adjacent-burst forward/reverse. Cases unavailable in a short sample are not
  # invented; fallback uses actual other-flow burst (label2).
  docs=[]
  for i,ss in enumerate(spans):
   doc=[]
   for begin,end in ss:
    b=q[i,begin:end].long()
    if len(b)>=2:doc.append(b[:-1]*256+b[1:]+5)
   if not doc:doc=[self.tokens(q[i:i+1])[0,1:]]
   docs.append(doc)
  tokens=[];labels=[];segments=[]
  for i,doc in enumerate(docs):
   case=int(torch.randint(0,5,()).item());burst=doc[0]
   if case in [0,1]:
    mid=max(1,len(burst)//2);a,b=burst[:mid],burst[mid:]
    if case==1:a,b=b,a
   elif case==2:a,b=burst,docs[(i+1)%len(docs)][0]
   elif len(doc)>1:
    a,b=doc[:2]
    if case==4:a,b=b,a
   else:case=2;a,b=burst,docs[(i+1)%len(docs)][0]
   a=a[:254];b=b[:254];t=torch.cat([torch.tensor([1],device=q.device),a,torch.tensor([2],device=q.device),b,torch.tensor([2],device=q.device)]).long();seg=torch.cat([torch.ones(len(a)+2,device=q.device),torch.full((len(b)+1,),2,device=q.device)]).long();tokens.append(F.pad(t,(0,512-len(t))));segments.append(F.pad(seg,(0,512-len(seg))));labels.append(case)
  src=torch.stack(tokens);seg=torch.stack(segments);mask=(torch.rand(src.shape,device=q.device)<.15)&(src>=5);mask[:,1]=src[:,1]>=5;target=src.masked_fill(~mask,0);masked=src.clone();chance=torch.rand(src.shape,device=q.device);masked[mask&(chance<.8)]=3;randoms=mask&(chance>=.8)&(chance<.9);masked[randoms]=torch.randint(5,65541,(int(randoms.sum()),),device=q.device);memory=self.encoder(self.embedding(masked,seg),seg);lm,ms,*_=self.target(memory,(target,torch.tensor(labels,device=q.device)));return lm+ms,dict(mlm=lm,msp=ms)
def build(name,d,n):return {'MFF':MFF,'GP-VAE':GPVAE,'FlowPic':FlowPic,'MetaTraffic':MetaTraffic,'Trafficformer-Light':TrafficformerLight}[name](d,n)

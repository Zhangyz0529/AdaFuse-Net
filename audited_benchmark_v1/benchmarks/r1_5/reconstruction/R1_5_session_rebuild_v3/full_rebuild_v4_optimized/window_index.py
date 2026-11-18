"""Equivalent exact-window search with one set of prefix sums per key.
Only timestamp-permitted starts are tested; all allowed offsets are exhausted.
Metadata anchor searches have an explicit two-witness ambiguity return: callers
only distinguish zero, one, or multiple windows, never select one of many.
"""
import collections
import numpy as np
class WindowIndex:
 def __init__(self,packets):
  self.ps=packets;self.size=len(packets);self.ts=np.array([p['ts'] for p in packets]);self.origin=tuple(packets[0]['src']) if packets else None
  self.direction=np.array([tuple(p['src'])==self.origin for p in packets],dtype=np.int64);self.f=np.r_[0,np.cumsum(self.direction)];self.prefix={}
  for field in ['payload_len','capture_payload_len']:
   v=np.array([p[field] for p in packets],dtype=np.int64);self.prefix[field]=(np.r_[0,np.cumsum(v*self.direction)],np.r_[0,np.cumsum(v*(1-self.direction))])
  self.cache=collections.OrderedDict()
 def directed(self,r):
  same=(r['Source IP'],r['Source Port'])==self.origin
  return (r['Total Fwd Packets'] if same else r['Total Backward Packets']),same
 def metadata(self,r):
  n=r['n_packets'];fw,same=self.directed(r);key=(n,fw,r['Flow Duration'])
  if key in self.cache:return self.cache[key]
  out=[]
  for begin in range(0,max(0,self.size-n+1),100000):
   a=np.arange(begin,min(begin+100000,self.size-n+1));b=a+n;good=(self.f[b]-self.f[a]==fw)&(np.abs((self.ts[b-1]-self.ts[a])*1e6-r['Flow Duration'])<=2.)
   for x in a[good]:
    out.append((int(x),int(x+n)))
    if len(out)>=2:break
   if len(out)>=2:break
  self.cache[key]=out
  if len(self.cache)>1024:self.cache.popitem(last=False)
  return out
 def windows(self,r,offsets,metric):
  n=r['n_packets']
  if n>self.size:return []
  fw,same=self.directed(r);fb,bb=self.prefix[metric]
  expectedf=r['Total Length of Fwd Packets'] if same else r['Total Length of Bwd Packets'];expectedb=r['Total Length of Bwd Packets'] if same else r['Total Length of Fwd Packets'];out=set()
  for offset in offsets:
   lo=np.searchsorted(self.ts,r['clock']+offset,side='left');hi=min(self.size-n+1,np.searchsorted(self.ts,r['clock']+offset+r['clock_precision_s'],side='left'))
   for begin in range(int(lo),int(hi),100000):
    a=np.arange(begin,min(begin+100000,hi));b=a+n
    good=(self.f[b]-self.f[a]==fw)&(fb[b]-fb[a]==expectedf)&(bb[b]-bb[a]==expectedb)&(np.abs((self.ts[b-1]-self.ts[a])*1e6-r['Flow Duration'])<=2.)
    out.update((int(x),int(x+n)) for x in a[good])
  return sorted(out)

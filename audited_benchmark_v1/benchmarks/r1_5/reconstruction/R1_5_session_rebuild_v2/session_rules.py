"""Observed connection IDs; absence of SYN is recorded, never assumed complete.
A separate canonical-tuple leakage group keeps even reused tuples together.
"""
import hashlib,json
RULES={'tcp_idle_s':120.,'udp_idle_s':30.,'new_syn':'SYN without ACK and a new initial sequence or previous flow midstream','close':'RST or FIN observed in both directions','active_timeout':None,'midstream':'allowed and marked','group':'same bidirectional tuple across all captures in dataset is never split'}
def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def assign_sessions(dataset,key,packets):
 packets.sort(key=lambda p:(p['ts'],p['file_id'],p['index']))
 state=None;ordinal=-1;seen=set();result=[];excluded=0
 for p in packets:
  # Same packet appearing in overlapping capture fragments must not be counted twice.
  ident=(p['ts'],p['raw_sha256'])
  if ident in seen:excluded+=1;continue
  seen.add(ident);syn=p['proto']==6 and bool(p['flags']&2) and not bool(p['flags']&16)
  new=state is None or state['ended'] or p['ts']-state['last']>(120 if p['proto']==6 else 30) or (syn and (state['isn']!=p['isn'] or not state['syn']))
  if new:
   ordinal+=1;state={'last':p['ts'],'syn':syn,'isn':p['isn'],'ended':False,'fin':set(),'id':stable([dataset,key,ordinal,p['ts'],p['raw_sha256']])}
  p['session_id']=state['id'];p['missing_initial_syn']=p['proto']==6 and not state['syn'];state['last']=p['ts']
  if p['flags']&1:state['fin'].add(tuple(p['src']))
  if p['flags']&4 or len(state['fin'])==2:state['ended']=True
  result.append(p)
 return result,excluded

"""Assign indivisible components to class-disjoint task roles; no seed retries."""
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def assign_roles(frame,roles,seed=42):
 allowed={label:{part for part,labels in roles.items() if label in labels} for label in frame.label.unique()};assignments={};flexible=[];conflicts=[]
 for group,labels in frame.groupby('split_group_id',sort=True).label.unique().items():
  possible=set(['Train','Val','Test'])
  for label in labels:possible&=allowed[label]
  if not possible:conflicts.append(group)
  elif len(possible)==1:assignments[group]=next(iter(possible))
  elif possible==set(['Train','Val','Test']):flexible.append(group)
  else:raise RuntimeError('Unspecified partially flexible group role')
 if len(flexible)<3:raise RuntimeError('Reference-only components insufficient for fixed three-way allocation')
 indexes=np.arange(len(flexible));train,rest=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=seed).split(indexes,groups=flexible));v,t=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(rest,groups=np.asarray(flexible)[rest]))
 for part,ids in [('Train',train),('Val',rest[v]),('Test',rest[t])]:
  for i in ids:assignments[flexible[int(i)]]=part
 result=frame[~frame.split_group_id.isin(conflicts)].copy();result['partition']=result.split_group_id.map(assignments)
 if result.partition.isna().any():raise RuntimeError('Unassigned component')
 excluded=frame[frame.split_group_id.isin(conflicts)].copy();excluded['reason']='component_contains_labels_requiring_different_task_roles'
 return result,excluded

"""Expose route-prefix loads and the next visit's actual directed edges.

Primitive state information only: no candidate evaluation or objective ranking.
16 zero-initialized new input columns leave the initial policy unchanged.
"""
import numpy as np
import torch
from torch import nn
from src.ppo_objects_v8 import ObjectState as PreviousState
from src.ppo_objects_v8c import ObjectPolicy as PreviousPolicy


class ObjectState(PreviousState):
    def encode(self,plan,preference,solution=None,metrics=None,age=0.):
        f,links,interaction=super().encode(plan,preference,solution,metrics,age)
        f=np.pad(f,((0,0),(0,16)))
        p=self.p;raw=self._raw
        for slot,(key,pos) in self._members.items():
            n,t=self.pickups[slot];route=plan[key];before,after=route[pos-1],route[pos+1]
            f[slot,64:68]=[raw.get(('F',before,n,s,key[0],t),0.)/p.vehicle_capacity for s in p.waste_types]
            f[slot,76:79]=[p.accident_probability.get(edge,0.)/max(self.max_prob,1e-12)
                            for edge in [(before,n),(n,after),(before,after)]]
            f[slot,79]=slot in self._processing_safe_drops
            if slot in self._next:
                u,next_key,_,_,_=self._next[slot]
                _,next_pos=self._members[self.pickup_slot[n,u]]
                next_route=plan[next_key];left,right=next_route[next_pos-1],next_route[next_pos+1]
                f[slot,68:72]=[raw.get(('F',left,n,s,next_key[0],u),0.)/p.vehicle_capacity for s in p.waste_types]
                f[slot,72:76]=(p.distance[left,n]/self.max_dist,p.distance[n,right]/self.max_dist,
                               p.accident_probability[left,n]/max(self.max_prob,1e-12),
                               p.accident_probability[n,right]/max(self.max_prob,1e-12))
        return f,links,interaction


class ObjectPolicy(PreviousPolicy):
    def __init__(self,dimension=64,layers=1):
        super().__init__(dimension,layers)
        previous=self.features
        self.features=nn.Linear(80,dimension)
        with torch.no_grad():
            self.features.weight[:,:64].copy_(previous.weight)
            self.features.weight[:,64:].zero_()
            self.features.bias.copy_(previous.bias)

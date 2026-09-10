"""v7b: cross-period capacity context and necessary feasibility masks only.

No objective calls, sorting by merit, or heuristic object selection in masks.
The unchanged three independent 512-class neural heads decide among legal slots.
"""
import numpy as np
from hazardous_waste_model import pickup_owner,pickup_type,normalize_pair
from src.solution_utils import route_plan_to_solution
from src.ppo_objects_v7 import ObjectState as BaseState,OPS


class ObjectState(BaseState):
    def prepare(self,plan,solution=None):
        if getattr(self,'_cached_plan',None) is plan:return
        self._cached_plan=plan
        p=self.p;raw=(solution or route_plan_to_solution(p,plan))['raw']
        self._raw=raw;members=super().membership(plan);self._members=members
        self._loads={key:sum(raw.get(('q',n,key[0],key[1]),0.) for n in r[1:-1]) for key,r in plan.items()}
        self._producer_bg={(i,t):sum(raw.get(('BG',n,t),0.) for n in p.pickup_nodes if pickup_owner(n)==i)
                           for i in p.producers for t in p.periods}
        drops=[];self._next={}
        for slot,(key,pos) in members.items():
            n,t=self.pickups[slot]
            if t==p.periods[-1]:continue
            route=plan[key]
            subsequent=next((u for u in p.periods if u>t and self.pickup_slot[n,u] in members),None)
            if subsequent is None:continue
            future_key,_=members[self.pickup_slot[n,subsequent]]
            q=raw.get(('q',n,key[0],t),0.)
            inventory_slack=min(p.producer_capacity[pickup_owner(n)]-self._producer_bg[pickup_owner(n),u]
                                for u in p.periods if t<u<=subsequent)
            route_slack=p.vehicle_capacity-self._loads[future_key]
            self._next[slot]=(subsequent,future_key,route_slack,inventory_slack,q)
            if len(route)>3 and (route[pos-1],route[pos+1]) not in p.distance:continue
            if q>route_slack+1e-8 or q>inventory_slack+1e-8:continue
            drops.append(slot)
        self._drops=set(drops)

    def encode(self,plan,preference,solution=None,metrics=None,age=0.):
        self.prepare(plan,solution)
        f,links,interaction=super().encode(plan,preference,solution,metrics,age)
        p=self.p
        for slot,(n,t) in enumerate(self.pickups):
            risk_scale=self.max_dist*self.max_prob*p.vehicle_capacity/self.refs[1]
            f[slot,15]=p.waste_consequence[pickup_type(n)]*risk_scale
            if slot in self._next:
                future,key,route_slack,inventory_slack,q=self._next[slot]
                f[slot,48:54]=(p.periods.index(future)/max(1,len(p.periods)-1),
                              (self.route_slot[key]+1)/512,route_slack/p.vehicle_capacity,
                              inventory_slack/p.vehicle_capacity,q/p.vehicle_capacity,slot in self._drops)
        # Absolute interaction intensity survives normalization, including max
        # accident-probability/distance scales hidden by relative matrix inputs.
        for i,s in enumerate(p.waste_types):
            for j,u in enumerate(p.waste_types):
                interaction[i*4+j]=p.coload_risk.get(normalize_pair(s,u),0.)*self.max_dist*self.max_prob*p.vehicle_capacity/self.refs[1]
        return f,links,interaction

    def masks(self,plan,chosen=()):
        self.prepare(plan)
        base=super().masks(plan,chosen);p=self.p
        if not chosen:
            base[0]=bool(self._drops);return base
        op=OPS[chosen[0]]
        if op=='drop' and len(chosen)==1:
            return base & self.mask(self._drops)
        if len(chosen)!=2:return base
        a=chosen[1]
        if op=='facility':
            key=self.routes[a-self.r0]
            for slot in np.flatnonzero(base):
                j,_=self.facilities[slot-self.f0]
                if any(not p.technology[j,pickup_type(n)] for n in plan[key][1:-1]):base[slot]=False
        if op in ('relocate','add'):
            n,t=self.pickups[a];waste=pickup_type(n)
            q=self._raw.get(('BG',n,t),0.)
            source=self._members[a][0] if a in self._members else None
            for slot in np.flatnonzero(base):
                target=self.routes[slot-self.r0]
                if target==source:continue
                if self._loads.get(target,0.)+q>p.vehicle_capacity+1e-8:
                    base[slot]=False;continue
                route=plan.get(target)
                if route and (not p.technology[route[0],waste] or any(
                        not p.compatibility[normalize_pair(waste,pickup_type(u))] for u in route[1:-1])):
                    base[slot]=False
        return base

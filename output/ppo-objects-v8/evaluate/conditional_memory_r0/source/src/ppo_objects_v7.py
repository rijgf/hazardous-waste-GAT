"""Experimental PPO with explicit objects and complete directed edge inputs.

The operator and three independent 512-class heads select every move argument.
Masks express object types, identities and feasibility, never objective rankings.
No insertion heuristic, genetic solution or repair-based reassignment is used.
Legacy v4 PPO and its checkpoints are intentionally untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from hazardous_waste_model import pickup_owner, pickup_type, normalize_pair
from src.solution_utils import route_plan_to_solution, evaluate_solution

OPS = ('drop', 'relocate', 'swap', 'two_opt', 'facility', 'add')
OBJECTS = 512
DUMMY = 511
FEATURES = 64
MAX_NODES = 128


@dataclass(frozen=True)
class Action:
    operator: int
    first: int
    second: int = DUMMY
    third: int = DUMMY


class ObjectState:
    """Fixed (entity, period) slots; no modulo aliases or visit-order indexing."""
    def __init__(self, params, refs):
        self.p = params
        self.refs = tuple(float(x) for x in refs)
        if len(self.refs) != 2 or not all(np.isfinite(x) and x > 0 for x in self.refs):
            raise ValueError('Each instance must provide its own fixed positive b')
        self.nodes = list(params.pickup_nodes) + list(params.facilities)
        if len(self.nodes) > MAX_NODES:
            raise ValueError('Node matrix capacity exceeded; never silently truncate')
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.pickups = [(n, t) for t in params.periods for n in params.pickup_nodes]
        self.routes = [(k, t) for t in params.periods for k in params.vehicles]
        self.facilities = [(j, t) for t in params.periods for j in params.facilities]
        self.r0 = len(self.pickups)
        self.f0 = self.r0 + len(self.routes)
        self.size = self.f0 + len(self.facilities)
        if self.size > DUMMY:
            raise ValueError('Object vocabulary exceeds 511 real slots')
        self.pickup_slot = {x: i for i, x in enumerate(self.pickups)}
        self.route_slot = {x: self.r0 + i for i, x in enumerate(self.routes)}
        self.facility_slot = {x: self.f0 + i for i, x in enumerate(self.facilities)}
        self.max_dist = max(params.distance.values())
        self.max_prob = max(params.accident_probability.values())
        self.edges = np.zeros((len(self.nodes), MAX_NODES, 3), dtype=np.float32)
        for (a, b), distance in params.distance.items():
            i, j = self.node_index[a], self.node_index[b]
            self.edges[i, j] = (distance / self.max_dist,
                                params.accident_probability[a, b] / max(self.max_prob, 1e-12), 1.)

    def membership(self, plan):
        return {self.pickup_slot[n, t]: (key, pos)
                for key, r in plan.items() for pos, n in enumerate(r[1:-1], 1)
                for t in [key[1]]}

    def encode(self, plan, preference, solution=None, metrics=None, age=0.):
        p = self.p
        solution = solution or route_plan_to_solution(p, plan)
        metrics = metrics or evaluate_solution(p, solution, preference, self.refs)
        raw = solution['raw']
        f = np.zeros((self.size, FEATURES), dtype=np.float32)
        # Four explicit node links: own node, predecessor, successor, home depot.
        links = np.full((self.size, 4), len(self.nodes), dtype=np.int64)
        membership = self.membership(plan)
        route_loads = {key: [sum(raw.get(('q', n, key[0], key[1]), 0.)
                                for n in r[1:-1] if pickup_type(n) == s)
                            for s in p.waste_types] for key, r in plan.items()}
        for slot, (n, t) in enumerate(self.pickups):
            owner, waste = pickup_owner(n), pickup_type(n)
            f[slot, 0] = 1
            f[slot, 3:7] = (p.periods.index(t) / max(1, len(p.periods)-1),
                            t == p.periods[-1], slot in membership, self.node_index[n] / MAX_NODES)
            f[slot, 7:11] = (p.generation[owner, waste, t] / p.vehicle_capacity,
                             raw.get(('BG', n, t), 0.) / p.vehicle_capacity,
                             raw.get(('IG', n, t), 0.) / p.producer_capacity[owner],
                             p.producer_inventory_risk[owner, waste] / self.refs[1])
            f[slot, 11 + p.waste_types.index(waste)] = 1
            links[slot, 0] = self.node_index[n]
            if slot in membership:
                key, pos = membership[slot]; route = plan[key]
                before, after = route[pos-1], route[pos+1]
                links[slot, 1:] = [self.node_index[before], self.node_index[after], self.node_index[route[0]]]
                f[slot, 16:24] = ((self.route_slot[key]+1)/OBJECTS, pos / len(p.pickup_nodes),
                                  len(route[1:-1])/len(p.pickup_nodes),
                                  sum(route_loads[key])/p.vehicle_capacity,
                                  p.distance.get((before,n),0.)/self.max_dist,
                                  p.distance.get((n,after),0.)/self.max_dist,
                                  p.distance.get((before,after),0.)/self.max_dist,
                                  (before,after) in p.distance or len(route)==3)
                f[slot, 24:24+len(p.waste_types)] = np.array(route_loads[key])/p.vehicle_capacity
                f[slot, 28] = raw.get(('q',n,key[0],t),0.)/p.vehicle_capacity
                f[slot, 29] = self.facility_slot[route[0],t]/OBJECTS
            # All possible depot edges/capacities/processing costs, not only chosen depot.
            for jindex, j in enumerate(p.facilities):
                offset = 30 + 6*jindex
                if offset+6 > 54:
                    raise ValueError('Facility feature capacity exceeded')
                f[slot, offset:offset+6] = (p.distance.get((n,j),0.)/self.max_dist,
                    p.distance.get((j,n),0.)/self.max_dist, p.technology[j,waste],
                    p.processing_capacity[j,waste,t]/p.vehicle_capacity,
                    raw.get(('BD',j,waste,t),0.)/p.facility_capacity[j],
                    p.processing_cost[j,waste]/max(p.processing_cost.values()))
        for index, key in enumerate(self.routes):
            slot = self.r0+index; route = plan.get(key,[]); t=key[1]
            f[slot,1] = 1; f[slot,3] = p.periods.index(t)/max(1,len(p.periods)-1)
            f[slot,5] = bool(route); f[slot,16] = (slot+1)/OBJECTS
            f[slot,17] = p.vehicles.index(key[0])/len(p.vehicles)
            if route:
                f[slot,18:20] = (len(route[1:-1])/len(p.pickup_nodes),sum(route_loads[key])/p.vehicle_capacity)
                f[slot,24:24+len(p.waste_types)] = np.array(route_loads[key])/p.vehicle_capacity
                links[slot] = [self.node_index[route[0]],self.node_index[route[1]],self.node_index[route[-2]],self.node_index[route[0]]]
        for index, (j,t) in enumerate(self.facilities):
            slot = self.f0+index; f[slot,2]=1
            f[slot,3] = p.periods.index(t)/max(1,len(p.periods)-1)
            links[slot,0] = self.node_index[j]
            for si,s in enumerate(p.waste_types):
                offset=7+6*si
                f[slot,offset:offset+6] = (p.technology[j,s],p.processing_capacity[j,s,t]/p.vehicle_capacity,
                    raw.get(('BD',j,s,t),0.)/p.facility_capacity[j],raw.get(('ID',j,s,t),0.)/p.facility_capacity[j],
                    p.processing_cost[j,s]/max(p.processing_cost.values()),p.facility_inventory_risk[j,s]/self.refs[1])
        # Objective context and full type-pair interaction matrix are visible to every object.
        common = [*preference,metrics['cost']/self.refs[0],metrics['risk']/self.refs[1],age]
        f[:,54:59] = common
        f[:,59] = p.vehicle_fixed_cost/self.refs[0]
        f[:,60] = p.distance_cost*self.max_dist/self.refs[0]
        f[:,61] = self.refs[0]/max(self.refs[1],1e-9)
        f[:,62] = len(plan)/(len(p.vehicles)*len(p.periods))
        f[:,63] = len(membership)/len(self.pickups)
        interaction = np.zeros(32,dtype=np.float32)
        for i,s in enumerate(p.waste_types):
            for j,u in enumerate(p.waste_types):
                if i*len(p.waste_types)+j >= 16:
                    raise ValueError('Waste interaction capacity exceeded')
                interaction[i*4+j] = p.coload_risk.get(normalize_pair(s,u),0.)/max(1.,max(p.coload_risk.values()))
                interaction[16+i*4+j] = p.compatibility[normalize_pair(s,u)]
        return f, links, interaction

    @staticmethod
    def mask(slots):
        result = np.zeros(OBJECTS,dtype=bool)
        result[list(slots)] = True
        return result

    def masks(self, plan, chosen=()):
        """Sequential HARD masks; neural logits themselves are independent."""
        p=self.p; members=self.membership(plan)
        optional=[s for s in members if self.pickups[s][1] != p.periods[-1]]
        absent=[s for s,(_,t) in enumerate(self.pickups) if s not in members and t != p.periods[-1]]
        if not chosen:
            return np.array([bool(optional),bool(members),len(members)>1,
                             any(len(r)>=4 for r in plan.values()),len(p.facilities)>1 and bool(plan),bool(absent)])
        op=OPS[chosen[0]]
        if len(chosen)==1:
            if op=='drop': slots=optional
            elif op=='add': slots=absent
            elif op=='facility': slots=[self.route_slot[k] for k in plan]
            elif op=='two_opt': slots=[s for s,(k,_) in members.items() if len(plan[k])>=4]
            else: slots=list(members)
            return self.mask(slots)
        a=chosen[1]
        if len(chosen)==2:
            if op=='drop': return self.mask([DUMMY])
            if op=='facility':
                key=self.routes[a-self.r0]; route=plan[key]
                return self.mask(self.facility_slot[j,key[1]] for j in p.facilities if j!=route[0])
            n,t=self.pickups[a]
            if op=='swap': return self.mask(s for s in members if s!=a and self.pickups[s][1]==t)
            if op=='two_opt': return self.mask(s for s,(k,_) in members.items() if s!=a and k==members[a][0])
            slots=[self.route_slot[k] for k in plan if k[1]==t]
            # Homogeneous unused vehicles differ only by name: expose one representative.
            unused=next((k for k in self.routes if k[1]==t and k not in plan),None)
            if unused is not None: slots.append(self.route_slot[unused])
            return self.mask(slots)
        if op not in ('add','relocate'): return self.mask([DUMMY])
        n,t=self.pickups[a]; key=self.routes[chosen[2]-self.r0]
        route=[x for x in plan.get(key,[]) if x!=n]
        if len(route)<3:
            return self.mask(self.facility_slot[j,t] for j in p.facilities
                             if p.technology[j,pickup_type(n)] and (j,n) in p.distance and (n,j) in p.distance)
        slots=[]
        for pos in range(len(route)-1):
            before,after=route[pos:pos+2]
            if (before,n) in p.distance and (n,after) in p.distance:
                slots.append(self.route_slot[key] if pos==0 else self.pickup_slot[before,t])
        return self.mask(slots)

    def apply(self, plan, action):
        """Execute the exact decoded tuple. Never optimize/repair its arguments."""
        parts=(action.operator,action.first,action.second,action.third)
        for index,choice in enumerate(parts):
            mask=self.masks(plan,parts[:index])
            if choice<0 or choice>=len(mask) or not mask[choice]:
                return None,'masked_or_dead_end'
        op=OPS[action.operator]; out={k:list(r) for k,r in plan.items()}; members=self.membership(plan)
        a,b,c=action.first,action.second,action.third
        if op=='drop':
            key,pos=members[a]; out[key].pop(pos)
        elif op=='facility':
            key=self.routes[a-self.r0]; depot,_=self.facilities[b-self.f0]
            out[key][0]=out[key][-1]=depot
        elif op=='swap':
            key,pos=members[a]; key2,pos2=members[b]
            out[key][pos],out[key2][pos2]=out[key2][pos2],out[key][pos]
        elif op=='two_opt':
            key,pos=members[a]; _,pos2=members[b]; left,right=sorted((pos,pos2))
            out[key][left:right+1]=reversed(out[key][left:right+1])
        else:
            n,t=self.pickups[a]
            if op=='relocate':
                source,pos=members[a];out[source].pop(pos)
            target=self.routes[b-self.r0]
            if c>=self.f0:
                depot,_=self.facilities[c-self.f0];out[target]=[depot,n,depot]
            else:
                route=out[target]
                pos=1 if c==b else route.index(self.pickups[c][0])+1
                route.insert(pos,n)
        out={k:r for k,r in out.items() if len(r)>=3}
        if out==plan: return None,'no_change'
        # Missing edges must not be treated as zero-cost travel.
        if any((a,b) not in self.p.distance for r in out.values() for a,b in zip(r,r[1:])):
            return None,'missing_edge'
        return out,'candidate'


class ObjectPolicy(nn.Module):
    """Three separately parameterized 512-class heads, shared rich state encoder."""
    def __init__(self, dimension=96, layers=1):
        super().__init__()
        self.dimension=dimension; self.layers=layers
        self.edge_encoder=nn.Sequential(nn.Linear(MAX_NODES*3,dimension),nn.GELU(),nn.Linear(dimension,dimension))
        self.features=nn.Linear(FEATURES,dimension)
        self.link_projection=nn.Linear(4*dimension,dimension)
        self.interaction=nn.Linear(32,dimension)
        self.slot_embedding=nn.Embedding(OBJECTS,dimension)
        layer=nn.TransformerEncoderLayer(dimension,4,dimension*2,dropout=0.,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(dimension)
        self.context=nn.Sequential(nn.Linear(dimension+5,dimension),nn.GELU())
        self.operator=nn.Linear(dimension,len(OPS))
        self.heads=nn.ModuleList([nn.Linear(dimension,OBJECTS) for _ in range(3)])
        self.local_heads=nn.ModuleList([nn.Linear(dimension,1) for _ in range(3)])
        self.value=nn.Linear(dimension,1)
        for head in [self.operator,*self.heads,*self.local_heads]:
            nn.init.orthogonal_(head.weight,gain=.01);nn.init.zeros_(head.bias)

    def forward(self, features, links, edges, interaction):
        batch,count,_=features.shape
        nodes=self.edge_encoder(edges.flatten(-2))
        nodes=torch.cat([nodes,torch.zeros(batch,1,self.dimension,device=nodes.device)],dim=1)
        linked=nodes[torch.arange(batch,device=nodes.device)[:,None,None],links]
        h=self.features(features)+self.link_projection(linked.flatten(-2))
        h=h+self.slot_embedding(torch.arange(count,device=h.device))[None]+self.interaction(interaction)[:,None]
        h=self.norm(self.transformer(h))
        global_h=self.context(torch.cat([h.mean(1),features[:,0,54:59]],dim=-1))
        obj=[]
        for head,local in zip(self.heads,self.local_heads):
            local_scores=local(h).squeeze(-1)
            obj.append(head(global_h)+nn.functional.pad(local_scores,(0,OBJECTS-count)))
        return [self.operator(global_h),*obj],self.value(global_h).squeeze(-1)


def distributions(logits, masks):
    return [Categorical(logits=x.masked_fill(~mask,-1e9)) for x,mask in zip(logits,masks)]


def sample_action(state,plan,logits,rng):
    """Numpy sampling makes seeds identical across inference device types."""
    chosen=[];masks=[];logprob=0.
    for scores in logits:
        mask=state.masks(plan,chosen)
        if not mask.any():
            # A dead-end conditional tuple is a counted rejected proposal.
            mask=state.mask([DUMMY])
        scores=np.asarray(scores,dtype=np.float64)
        allowed=np.flatnonzero(mask);v=scores[allowed];v=np.exp(v-v.max());v/=v.sum()
        index=int(rng.choice(len(allowed),p=v));chosen.append(int(allowed[index]))
        logprob+=float(np.log(max(v[index],1e-300)));masks.append(mask)
    return Action(*chosen),masks,logprob


class SearchEnvironment:
    def __init__(self,state,initial_plan,preference=(.5,.5),horizon=384):
        self.state=state;self.initial_plan=initial_plan;self.preference=tuple(preference);self.horizon=horizon
        self.reset()

    def reset(self):
        self.plan={k:list(r) for k,r in self.initial_plan.items()};self.age=0
        self.solution=route_plan_to_solution(self.state.p,self.plan)
        self.metrics=evaluate_solution(self.state.p,self.solution,self.preference,self.state.refs)
        if not self.metrics['feasible']:raise ValueError(self.metrics['violations'])

    def encode(self):
        return self.state.encode(self.plan,self.preference,self.solution,self.metrics,min(self.age/self.horizon,1.))

    def step(self,action):
        candidate,status=self.state.apply(self.plan,action); self.age+=1
        if candidate is None:return (-.01 if status=='no_change' else -.2),status
        solution=route_plan_to_solution(self.state.p,candidate)
        metric=evaluate_solution(self.state.p,solution,self.preference,self.state.refs)
        if not metric['feasible']:return -.2,'infeasible'
        difference=self.metrics['weighted_objective']-metric['weighted_objective']
        if difference>1e-12:
            self.plan,self.solution,self.metrics=candidate,solution,metric;status='accepted'
        else:status='rejected'
        return float(np.clip(100.*difference,-5.,5.)),status

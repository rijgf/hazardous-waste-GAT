"""Six simple route moves, homogeneous-vehicle symmetry reduction, scalar search.

This is a separate experimental hybrid, not the original frozen PPO improve().
Optional PPO guidance uses only shared operator heads; object heads are unused.
All objective calls, including rejected/no-op attempts, count against the budget.
"""
from collections import Counter
import random
import time

from src.operators import repair_plan
from src.solution_utils import route_plan_to_solution, evaluate_solution

FAMILIES=('drop','relocate','swap','two_opt','facility','add')


def canonical_routes(params,plan):
    """Vehicle names are irrelevant; route membership and order are not."""
    result={}
    for period in params.periods:
        routes=sorted(tuple(r) for (k,t),r in plan.items() if t==period and len(r)>2)
        if len(routes)>len(params.vehicles):raise ValueError('Too many routes')
        for vehicle,route in zip(params.vehicles,routes):result[vehicle,period]=list(route)
    return result


def propose(p,plan,family,rng):
    candidate={key:list(route) for key,route in plan.items()}
    keys=list(candidate)
    visits=[(key,i) for key,r in candidate.items() for i in range(1,len(r)-1)]
    if not visits:return None
    if family=='drop':
        optional=[(k,i) for k,i in visits if k[1]!=p.periods[-1]]
        if not optional:return None
        key,i=rng.choice(optional);candidate[key].pop(i)
    elif family=='relocate':
        key,i=rng.choice(visits);node=candidate[key].pop(i)
        targets=[k for k in keys if k[1]==key[1]]
        free=next((v for v in p.vehicles if (v,key[1]) not in plan),None)
        if free is not None:
            new=free,key[1];candidate[new]=[plan[key][0],plan[key][0]];targets.append(new)
        target=rng.choice(targets);route=candidate[target]
        positions=[j for j in range(1,len(route)) if (route[j-1],node) in p.distance and (node,route[j]) in p.distance]
        if not positions:return None
        # Cheapest distance insertion is a construction heuristic, not a full
        # objective evaluation; the full cost+risk evaluator decides acceptance.
        j=min(positions,key=lambda j:p.distance[route[j-1],node]+p.distance[node,route[j]]-p.distance.get((route[j-1],route[j]),0))
        route.insert(j,node)
    elif family=='swap':
        key,i=rng.choice(visits)
        others=[(k,j) for k,j in visits if k[1]==key[1] and (k,j)!=(key,i)]
        if not others:return None
        key2,j=rng.choice(others)
        candidate[key][i],candidate[key2][j]=candidate[key2][j],candidate[key][i]
    elif family=='two_opt':
        choices=[k for k in keys if len(candidate[k])>=4]
        if not choices:return None
        key=rng.choice(choices);i,j=sorted(rng.sample(range(1,len(candidate[key])-1),2))
        candidate[key][i:j+1]=reversed(candidate[key][i:j+1])
    elif family=='facility':
        key=rng.choice(keys);facilities=[j for j in p.facilities if j!=candidate[key][0]]
        if not facilities:return None
        j=rng.choice(facilities);candidate[key][0]=candidate[key][-1]=j
    elif family=='add':
        if len(p.periods)<2:return None
        period=rng.choice(p.periods[:-1]);node=rng.choice(p.pickup_nodes)
        if any(node in r[1:-1] for (k,t),r in plan.items() if t==period):return None
        targets=[k for k in keys if k[1]==period]
        if targets:key=rng.choice(targets)
        else:
            key=p.vehicles[0],period;candidate[key]=[p.facilities[0],p.facilities[0]]
        route=candidate[key]
        positions=[j for j in range(1,len(route)) if (route[j-1],node) in p.distance and (node,route[j]) in p.distance]
        if not positions:return None
        j=min(positions,key=lambda j:p.distance[route[j-1],node]+p.distance[node,route[j]]-p.distance.get((route[j-1],route[j]),0))
        route.insert(j,node)
    else:raise ValueError(family)
    return {k:r for k,r in candidate.items() if len(r)>2}


def search(params,initial_plan,refs,*,budget=8000,seed=0,policy=None,progress=None,
           guidance=0.5,shake_after=600):
    """First-improvement descent with a short perturbation after stagnation."""
    rng=random.Random(seed);start=time.perf_counter();counts=Counter();cache={}
    plan=canonical_routes(params,initial_plan)
    metric=evaluate_solution(params,route_plan_to_solution(params,plan),(.5,.5),refs)
    if not metric['feasible']:raise ValueError(metric['violations'])
    best_plan=plan;best_metric=metric;trace=[];stale=0;shake=0;weights=[.3,.3,.1,.15,.1,.05]
    original_weights=list(weights)
    for attempt in range(budget):
        if policy is not None and attempt%64==0:
            import torch
            t,m,g=policy.encoder.encode(plan,(.5,.5),attempt/max(1,budget),stale/max(1,budget))
            with torch.inference_mode():
                logits,_=policy.policy(t[None].to(policy.device),m[None].to(policy.device),g[None].to(policy.device))
            probs=logits[0][0].softmax(-1).cpu().tolist()
            # drop is new and stays explicitly unlearned. Five shared operation
            # families reuse their matching old operator probabilities only.
            learned=[original_weights[0],probs[0],probs[1],probs[2],probs[3],probs[5]]
            z=sum(learned);weights=[(1-guidance)*a+guidance*b/z for a,b in zip(original_weights,learned)]
        family=rng.choices(FAMILIES,weights)[0];counts[family+'_attempts']+=1
        candidate=propose(params,plan,family,rng);counts['attempts']+=1;stale+=1
        if candidate is None:counts['no_proposal']+=1
        else:
            candidate,ok=repair_plan(params,candidate)
            if not ok:counts['repair_failed']+=1
            else:
                candidate=canonical_routes(params,candidate)
                key=tuple((k,tuple(r)) for k,r in candidate.items())
                if key in cache:counts['cache_hits']+=1;met=cache[key]
                else:
                    met=evaluate_solution(params,route_plan_to_solution(params,candidate),(.5,.5),refs)
                    cache[key]=met;counts['objective_evaluations']+=1
                if not met['feasible']:counts['invalid']+=1
                elif met['weighted_objective']<metric['weighted_objective']-1e-12 or shake>0:
                    plan,metric=candidate,met;counts[family+'_accepted']+=1
                    if shake>0:shake-=1
                    if met['weighted_objective']<best_metric['weighted_objective']-1e-12:
                        best_plan,best_metric=candidate,met;stale=0
                        trace.append({'attempt':attempt+1,'family':family,'C':met['cost'],'R':met['risk'],'J':met['weighted_objective']})
        if shake_after and stale>=shake_after and shake==0:
            plan,metric=best_plan,best_metric;shake=3;stale=0;counts['perturbations']+=1
        if progress is not None and ((attempt+1)%250==0 or attempt+1==budget):
            progress({'attempt':attempt+1,'seconds':time.perf_counter()-start,'best_J':best_metric['weighted_objective'],'counts':dict(counts)})
    return {'plan':best_plan,'metrics':best_metric,'counts':dict(counts),'trace':trace,'seconds':time.perf_counter()-start}

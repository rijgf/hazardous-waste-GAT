"""Exact PPO joint-policy sampling without replacement inside one state pool.

Subtract already sampled complete tuples' probability mass at each prefix.
No rejection-resampling, objective calls, or extra candidate evaluations.
"""
import numpy as np
from src.ppo_objects_v7 import Action,DUMMY


def probabilities(state,plan,scores,chosen,forbidden,prefix_mass):
    mask=state.masks(plan,chosen)
    if not mask.any():mask=state.mask([DUMMY])
    allowed=np.flatnonzero(mask);logits=np.asarray(scores[allowed],dtype=np.float64)
    weights=np.exp(logits-logits.max());weights/=weights.sum()
    remaining=prefix_mass*weights
    lookup={int(v):i for i,v in enumerate(allowed)}
    for parts,mass in forbidden:
        if tuple(parts[:len(chosen)])==tuple(chosen):
            remaining[lookup[parts[len(chosen)]]]-=mass
    # Roundoff cancellation at exhausted leaves is at most floating point eps.
    remaining=np.maximum(remaining,0.)
    remaining[remaining < 1e-14*max(prefix_mass,1e-300)]=0.
    total=remaining.sum()
    if total<=0:raise ValueError('No unseen positive-probability tuple remains')
    return allowed,weights,remaining/total,mask


def sample_unique(state,plan,score_fn,rng,forbidden):
    """Mutates only the pool-local exclusion list; returns the actual draw LP."""
    chosen=[];masks=[];mass=1.;lp=0.
    for head in range(4):
        scores=np.asarray(score_fn(chosen))
        allowed,base,conditional,mask=probabilities(state,plan,scores,chosen,forbidden,mass)
        index=int(rng.choice(len(allowed),p=conditional))
        chosen.append(int(allowed[index]));mass*=base[index]
        lp+=float(np.log(max(conditional[index],1e-300)));masks.append(mask)
    forbidden.append((tuple(chosen),mass))
    return Action(*chosen),masks,lp

"""Frozen, multi-preference frontier adapter for the retained v8 PPO policy.

The external archive is observation only. Each real attempted tuple is charged
once; root policy exhaustion is an explicit early stop, not a padded budget.
"""
from collections import Counter
from dataclasses import asdict
import json
import time

import numpy as np
import torch

from src.pareto_experiment import CandidateLedger
from src.ppo_objects_v7 import SearchEnvironment, OPS
from src.ppo_objects_v8 import ObjectState
from src.ppo_unique_sampling import probabilities, sample_unique
from src.reproducibility import plan_sha256, plan_to_canonical_data
from src.solution_utils import evaluate_solution, route_plan_to_solution


def next_unique(state,plan,score_fn,rng,forbidden):
    """Only exhaustion at the ROOT is a valid stop; prefix errors propagate."""
    try:
        probabilities(state,plan,np.asarray(score_fn([])),[],forbidden,1.)
    except ValueError as error:
        if str(error)!='No unseen positive-probability tuple remains':raise
        return None
    return sample_unique(state,plan,score_fn,rng,forbidden)


def scalar_metrics(point,preference,refs):
    metrics={k:v for k,v in point.items() if k not in ('plan','provenance','solution_id')}
    raw=preference[0]*point['cost']+preference[1]*point['risk']
    normalized=preference[0]*point['cost']/refs[0]+preference[1]*point['risk']/refs[1]
    metrics.update(weighted_objective_raw=raw,weighted_objective_normalized=normalized,
                   weighted_objective=normalized)
    return metrics


def search_preference(params,refs,reference,policy,preference,seed,*,ledger=None,
                      budget=1920,width=8,temperature=2.,horizon=1024,log_handle=None):
    if len(preference)!=2 or min(preference)<0 or abs(sum(preference)-1.)>1e-12:
        raise ValueError('A nonnegative two-objective preference summing to one is required')
    if budget<0 or width<1 or horizon<1 or temperature<=0:raise ValueError('Invalid search budget/config')
    if policy.training:raise ValueError('Frozen inference requires eval mode')
    started=time.perf_counter();state=ObjectState(params,refs)
    env=SearchEnvironment(state,reference,preference,horizon)
    ledger=ledger if ledger is not None else CandidateLedger(params,refs,reference)
    before=dict(ledger.counts);counts=Counter();attempts=0;forbidden=[];memory_key=None;trace=[]
    rng=np.random.default_rng(seed);device=next(policy.parameters()).device
    edge=torch.as_tensor(state.edges,device=device)[None]
    stop='candidate_budget_reached';batches=0
    while attempts<budget:
        features,links,interaction=env.encode()
        with torch.inference_mode():
            context=policy.encode(torch.as_tensor(features,device=device)[None],
                torch.as_tensor(links,device=device)[None],edge,torch.as_tensor(interaction,device=device)[None])
        current_key=(id(env.plan),min(env.age,horizon))
        if current_key!=memory_key:forbidden=[]
        memory_key=current_key;scores={};best=None;exhausted=False;batch_start=attempts;batches+=1
        state_hash=plan_sha256(env.plan)
        def score_fn(chosen):
            key=tuple(chosen)
            if key not in scores:
                with torch.inference_mode():
                    value=policy.head_logits(context,torch.tensor([chosen],dtype=torch.long,device=device))
                scores[key]=value[0].cpu().numpy()/temperature
                if not np.isfinite(scores[key]).all():raise ValueError('Nonfinite policy logits')
            return scores[key]
        for _ in range(min(width,budget-attempts)):
            sampled=next_unique(state,env.plan,score_fn,rng,forbidden)
            if sampled is None:
                exhausted=True;counts['exhausted_batches']+=1;break
            action,_,logprob=sampled;attempts+=1;counts[OPS[action.operator]+'_attempts']+=1
            candidate,status=state.apply(env.plan,action);point=None;metric=None
            provenance={'source':'PPO_candidate','preference':list(preference),'evaluation_seed':seed,
                        'candidate_index':attempts,'operator':OPS[action.operator],'action':asdict(action)}
            if candidate is None:
                ledger.counts['candidate_attempts']+=1
                ledger.counts['operator_failures']+=1
                counts['early_operator_rejections']+=1
            else:
                point=ledger.evaluate(candidate,provenance)
                if not point['feasible']:status='infeasible'
                else:
                    metric=scalar_metrics(point,preference,refs)
                    status='accepted_proposal' if metric['weighted_objective']<env.metrics['weighted_objective']-1e-12 else 'rejected'
                    if best is None or metric['weighted_objective']<best[2]['weighted_objective']-1e-12:
                        best=(candidate,point,metric,action)
            counts[status]+=1
            if log_handle is not None:
                log_handle.write(json.dumps({'event':'candidate','attempt':attempts,'batch':batches,
                    'state_id':state_hash,'state_age':env.age,'action':asdict(action),
                    'log_probability':logprob,'status':status,
                    'candidate_id':point['solution_id'] if point else None,
                    'C':point['cost'] if point else None,'R':point['risk'] if point else None})+'\n')
        accepted=best is not None and best[2]['weighted_objective']<env.metrics['weighted_objective']-1e-12
        if accepted:
            env.plan,point,env.metrics,chosen_action=best
            env.solution=route_plan_to_solution(params,env.plan)
            counts['accepted_transitions']+=1
            trace.append({'attempt':attempts,'solution_id':point['solution_id'],'action':asdict(chosen_action),
                          'C':point['cost'],'R':point['risk'],'J':env.metrics['weighted_objective']})
        env.age=attempts
        if log_handle is not None:
            log_handle.write(json.dumps({'event':'batch_end','batch':batches,'attempt':attempts,
                'batch_attempts':attempts-batch_start,'exhausted':exhausted,'accepted':accepted,
                'result_state_id':plan_sha256(env.plan)})+'\n');log_handle.flush()
        if exhausted and (id(env.plan),min(env.age,horizon))==memory_key:
            stop='positive_probability_action_space_exhausted';break
    metrics=evaluate_solution(params,env.solution,preference,objective_refs=refs)
    assert metrics['feasible'] and abs(metrics['weighted_objective']-env.metrics['weighted_objective'])<1e-10
    delta={key:ledger.counts[key]-before.get(key,0) for key in ledger.counts}
    assert delta['candidate_attempts']==attempts==sum(counts[op+'_attempts'] for op in OPS)
    return {'preference':list(preference),'seed':seed,'seconds':time.perf_counter()-started,
            'metrics':metrics,'plan':plan_to_canonical_data(env.plan),'solution_id':plan_sha256(env.plan),
            'candidate_budget':budget,'actual_candidate_attempts':attempts,'unused_candidate_budget':budget-attempts,
            'stop_reason':stop,'search_counts':dict(counts),'ledger_counts':delta,
            'final_validation_evaluations':1,'preference_initialization_evaluations':1,
            'trace':trace,'batches':batches}

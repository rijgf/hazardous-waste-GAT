"""PPO selects every argument; bounded Metropolis acceptance ablation only.

Exactly 12000 proposals from the reference plan. No restarts, no extra candidate
screening, and no genetic solution injection. Final output is the best visited.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import math
import shutil
import time
import numpy as np
import torch
from run_ppo_objects_v8 import ROOT,OUT,read,write,sha,protect,tensors,forward
from src.ppo_objects_v8 import ObjectState
from src.ppo_objects_v7 import ObjectPolicy,SearchEnvironment,sample_action,OPS
from src.reproducibility import deserialize_plan,plan_to_canonical_data,plan_sha256
from src.solution_utils import route_plan_to_solution,evaluate_solution
from sample_params import params_from_json_data


def temperature(attempt,budget,peak):
    # Fixed cooling schedule; no tuning from the current objective.
    return peak*max(0.,1.-attempt/budget)**2


def evaluate(args):
    protect();torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);accept_rng=np.random.default_rng(args.seed ^ 0x5a5a5a5a)
    dest=OUT/'evaluate'/args.id
    if dest.exists():raise FileExistsError(dest)
    input_path=ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json'
    data=read(input_path);p=params_from_json_data(data['params']);refs=(data['b_C'],data['b_R'])
    state=ObjectState(p,refs);env=SearchEnvironment(state,deserialize_plan(data['reference_plan']),(.5,.5),1024)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert ckpt['format']=='full_objects_v7' and ckpt['layers']==1 and ckpt['config']['args']['variant']=='b'
    device=torch.device(args.device);model=ObjectPolicy(ckpt['dimension'],1).to(device)
    model.load_state_dict(ckpt['model_state_dict']);model.eval();edge=torch.tensor(state.edges,device=device)[None]
    from run_ppo_scalar_exploration import benchmark
    _,target=benchmark()
    config={'args':vars(args),'acceptance':'best feasible of 8, Metropolis if not improving, quadratic cooling',
            'exploratory':True,'checkpoint':{'path':args.checkpoint,'sha256':sha(args.checkpoint)},
            'input_sha256':sha(input_path),'benchmark':target,'object_heads_used':True,
            'sources':{f:sha(ROOT/f) for f in ('run_ppo_objects_v8_anneal.py','run_ppo_objects_v8.py',
                'src/ppo_objects_v8.py','src/ppo_objects_v7b.py','src/ppo_objects_v7.py',
                'src/solution_utils.py','hazardous_waste_model.py','sample_params.py')}}
    write(dest/'config.json',config)
    for f in config['sources']:
        path=dest/'source'/f;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,path)
    best_global=(env.plan,env.solution,env.metrics);counts=Counter();trace=[];decisions=[];start=time.perf_counter()
    for block in range(0,args.budget,args.width):
        with torch.inference_mode():logits,_=forward(model,tensors([env.encode()],device),edge)
        scores=[x[0].cpu().numpy()/args.temperature for x in logits]
        if args.object_ablation=='uniform':scores[1:]=[np.zeros(512) for _ in range(3)]
        candidate_best=None;chosen=None;records=[]
        for offset in range(min(args.width,args.budget-block)):
            action,masks,lp=sample_action(state,env.plan,scores,rng)
            candidate,status=state.apply(env.plan,action)
            metric=None
            if candidate is not None:
                solution=route_plan_to_solution(p,candidate);metric=evaluate_solution(p,solution,(.5,.5),refs)
                if not metric['feasible']:status='infeasible'
                else:
                    status='accepted' if metric['weighted_objective']<env.metrics['weighted_objective']-1e-12 else 'rejected'
                    if candidate_best is None or metric['weighted_objective']<candidate_best[2]['weighted_objective']-1e-12:
                        candidate_best=(candidate,solution,metric);chosen=asdict(action)
            counts[status]+=1;counts[OPS[action.operator]+'_attempts']+=1
            records.append({'attempt':block+offset+1,'action':asdict(action),'status':status,'log_probability':lp})
        end=block+len(records);accepted=False;draw=None;probability=0.
        if candidate_best:
            delta=candidate_best[2]['weighted_objective']-env.metrics['weighted_objective']
            temp=temperature(end,args.budget,args.anneal)
            probability=1. if delta<=1e-12 else (math.exp(-min(745.,delta/temp)) if temp>0 else 0.)
            draw=float(accept_rng.random());accepted=draw<probability
            if accepted:
                env.plan,env.solution,env.metrics=candidate_best
                trace.append({'attempt':end,'action':chosen,'J':env.metrics['weighted_objective'],
                              'C':env.metrics['cost'],'R':env.metrics['risk'],'non_improving':delta>=-1e-12})
                if env.metrics['weighted_objective']<best_global[2]['weighted_objective']-1e-12:
                    best_global=candidate_best
        env.age=end
        decisions.append({'attempt':end,'accepted':accepted,'probability':probability,'draw':draw})
        with (dest/'actions.jsonl').open('a',encoding='utf8') as stream:
            import json
            for item in records:stream.write(json.dumps(item)+'\n')
        if end%400==0:
            progress={'attempts':end,'J':best_global[2]['weighted_objective'],'current_J':env.metrics['weighted_objective'],
                      'seconds':time.perf_counter()-start,'counts':dict(counts)}
            write(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    plan,solution,metrics=best_global
    result={'metrics':metrics,'plan':plan_to_canonical_data(plan),'solution_id':plan_sha256(plan),
            'counts':dict(counts),'trace':trace,'decisions':decisions,'attempts':args.budget,
            'seconds':time.perf_counter()-start,'beats_all_existing_nsga':metrics['weighted_objective']<target['target_J']-1e-8,
            'checkpoint_used':True,'object_heads_used':args.object_ablation=='none','config_sha256':sha(dest/'config.json')}
    write(dest/'result.json',result)
    print(json.dumps({'J':metrics['weighted_objective'],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--checkpoint',default='output/ppo-objects-v7/train/v7b_capacity_context/large_model.pt')
    parser.add_argument('--device',default='cuda');parser.add_argument('--budget',type=int,default=12000)
    parser.add_argument('--width',type=int,default=8);parser.add_argument('--temperature',type=float,default=2.)
    parser.add_argument('--anneal',type=float,default=.002)
    parser.add_argument('--object-ablation',choices=['none','uniform'],default='none')
    args=parser.parse_args()
    if args.budget!=12000 or args.width!=8 or args.anneal<0:raise ValueError('Fixed trial limits')
    if args.id!=__import__('pathlib').Path(args.id).name:raise ValueError('Invalid ID')
    evaluate(args)

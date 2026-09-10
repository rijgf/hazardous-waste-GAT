"""Replay full-object PPO scientific artifacts; no solver or optimizer updates."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import math
import numpy as np
import torch

from run_ppo_objects_v8 import ROOT,OUT,read,sha,write,protect,forward,tensors
from src.ppo_objects_v7 import ObjectState,ObjectPolicy,Action,OPS,SearchEnvironment
from src.ppo_objects_v7b import ObjectState as CapacityState
from src.ppo_objects_v8 import ObjectState as StrictState
from src.ppo_objects_v8c import ObjectPolicy as ConditionalPolicy
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan,plan_sha256
from src.solution_utils import route_plan_to_solution,evaluate_solution


def audit_trial(path,provenance=True):
    result=read(path);folder=path.parent;config=read(folder/'config.json');args=config['args']
    anneal='acceptance' in config
    if anneal:args={**args,'masks':'processing','variant':'b','horizon':1024}
    assert sha(folder/'config.json')==result['config_sha256']
    for file,h in config['sources'].items():
        assert sha(folder/'source'/file)==h
        if file.startswith('src/'):assert sha(ROOT/file)==h, 'Replay core differs from captured source'
    data_path=ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json'
    assert sha(data_path)==config['input_sha256']
    data=read(data_path);p=params_from_json_data(data['params']);refs=(data['b_C'],data['b_R'])
    state_class=StrictState if args.get('masks')=='processing' else (CapacityState if args.get('variant','a')=='b' else ObjectState)
    state=state_class(p,refs)
    assert args['budget']<=12000
    env=SearchEnvironment(state,deserialize_plan(data['reference_plan']),(.5,.5),args['horizon'])
    plan=deserialize_plan(result['plan']);assert plan_sha256(plan)==result['solution_id']
    assert result['metrics']['cost_ref']==refs[0] and result['metrics']['risk_ref']==refs[1]
    assert result['metrics']['feasible'] and not result['metrics']['violations']
    assert result['attempts']==args['budget']==sum(result['counts'].get(op+'_attempts',0) for op in OPS)
    assert result['seconds']>0 and math.isfinite(result['seconds'])
    assert result['beats_all_existing_nsga']==(result['metrics']['weighted_objective']<config['benchmark']['target_J']-1e-8)
    model=None;conditional=False
    if result['checkpoint_used']:
        ckptpath=ROOT/config['checkpoint']['path'];assert sha(ckptpath)==config['checkpoint']['sha256']
        ckpt=torch.load(ckptpath,map_location='cpu',weights_only=False)
        assert ckpt['format'] in ('full_objects_v7','conditional_objects_v8c')
        conditional=ckpt['format']=='conditional_objects_v8c'
        assert ckpt['layers']==1
        assert ckpt['config']['args'].get('variant','a')==args.get('variant','a'),'Silent encoder/checkpoint mismatch'
        for file,h in ckpt['config']['training_inputs'].items():
            assert '/train/Test-4/' in file and sha(ROOT/file)==h
        assert len(ckpt['config']['training_inputs'])==24
        assert all(all(g>0 for g in item['head_gradient_norms']) for item in ckpt['history'])
        model=(ConditionalPolicy if conditional else ObjectPolicy)(ckpt['dimension'],ckpt['layers']);model.load_state_dict(ckpt['model_state_dict']);model.eval()
    actions=[json.loads(line) for line in (folder/'actions.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(actions)==args['budget']
    status_counts=Counter(x['status'] for x in actions)
    op_counts=Counter(OPS[x['action']['operator']]+'_attempts' for x in actions)
    assert all(result['counts'][k]==v for k,v in (status_counts+op_counts).items())
    trace={x['attempt']:x for x in result['trace']};maxlog=0.;maxmetric=0.
    best_global=(env.plan,env.solution,env.metrics)
    accept_rng=np.random.default_rng(args['seed'] ^ 0x5a5a5a5a)
    decisions={x['attempt']:x for x in result.get('decisions',[])}
    edge=torch.tensor(state.edges)[None]
    for start in range(0,len(actions),args['width']):
        block=actions[start:start+args['width']]
        scores=None
        if provenance:
            if model is None:scores=[np.zeros(6),*([np.zeros(512)]*3)]
            else:
                with torch.inference_mode():
                    encoded=tensors([env.encode()],'cpu')
                    if conditional:
                        context=model.encode(encoded[0],encoded[1],edge,encoded[2])
                        width=len(block);context=tuple(x.expand(width,*x.shape[1:]) for x in context)
                        choices=torch.tensor([list(item['action'].values()) for item in block])
                        conditional_scores=[model.head_logits(context,choices[:,:i]).numpy()/args['temperature'] for i in range(4)]
                        logits=None
                    else:logits,_=forward(model,encoded,edge)
                scores=None if conditional else [x[0].numpy()/args['temperature'] for x in logits]
                if args.get('object_temperature') is not None:
                    assert not conditional
                    scores[1:]=[x[0].numpy()/args['object_temperature'] for x in logits[1:]]
                if args.get('object_ablation','none')=='uniform':
                    if conditional:conditional_scores[1:]=[np.zeros((len(block),512)) for _ in range(3)]
                    else:scores[1:]=[np.zeros(512) for _ in range(3)]
        candidate_best=None;selected_action=None
        excluded_mass=0.;seen=set()
        for offset,item in enumerate(block,start+1):
            assert item['attempt']==offset
            action=Action(**item['action']);parts=[action.operator,action.first,action.second,action.third]
            assert 0<=parts[0]<6 and all(0<=x<512 for x in parts[1:])
            if provenance:
                lp=0.
                for head,choice in enumerate(parts):
                    mask=state.masks(env.plan,parts[:head])
                    if not mask.any():mask=state.mask([511])
                    assert mask[choice]
                    allowed=np.flatnonzero(mask)
                    values=conditional_scores[head][offset-start-1] if conditional else scores[head]
                    v=np.asarray(values[allowed],dtype=np.float64)
                    v=np.exp(v-v.max());v/=v.sum();lp+=float(np.log(max(v[list(allowed).index(choice)],1e-300)))
                if args.get('unique'):
                    assert tuple(parts) not in seen
                    seen.add(tuple(parts));original_mass=math.exp(lp)
                    lp-=math.log1p(-excluded_mass);excluded_mass+=original_mass
                error=abs(lp-item['log_probability']);maxlog=max(maxlog,error)
                assert error<5e-4,(folder.name,offset,error)
            candidate,status=state.apply(env.plan,action)
            if candidate is not None:
                solution=route_plan_to_solution(p,candidate)
                metric=evaluate_solution(p,solution,(.5,.5),refs)
                if not metric['feasible']:status='infeasible'
                else:
                    status='accepted' if metric['weighted_objective']<env.metrics['weighted_objective']-1e-12 else 'rejected'
                    if candidate_best is None or metric['weighted_objective']<candidate_best[2]['weighted_objective']-1e-12:
                        candidate_best=(candidate,solution,metric);selected_action=item['action']
            assert status==item['status'],(folder.name,offset,status,item['status'])
        end=start+len(block)
        accepted=False
        if anneal:
            decision=decisions[end];draw=None;probability=0.
            if candidate_best is not None:
                delta=candidate_best[2]['weighted_objective']-env.metrics['weighted_objective']
                temperature=args['anneal']*max(0.,1.-end/args['budget'])**2
                probability=1. if delta<=1e-12 else (math.exp(-min(745.,delta/temperature)) if temperature>0 else 0.)
                draw=float(accept_rng.random());accepted=draw<probability
            assert decision['accepted']==accepted and decision['draw']==draw
            assert abs(decision['probability']-probability)<1e-12
        elif candidate_best is not None:
            accepted=candidate_best[2]['weighted_objective']<env.metrics['weighted_objective']-1e-12
        assert accepted==(end in trace)
        if accepted:
            point=trace[end]
            assert point['action']==selected_action
            env.plan,env.solution,env.metrics=candidate_best
            for key,metric in [('C','cost'),('R','risk'),('J','weighted_objective')]:
                error=abs(env.metrics[metric]-point[key]);maxmetric=max(maxmetric,error);assert error<1e-7
            if env.metrics['weighted_objective']<best_global[2]['weighted_objective']-1e-12:
                best_global=(env.plan,env.solution,env.metrics)
        env.age=end
    assert plan_sha256(best_global[0])==result['solution_id']
    return p,plan,result['metrics'],{'trial':folder.name,'sampled_actions':len(actions),
        'accepted_transitions':len(trace),'max_log_probability_error':maxlog,
        'all_candidate_statuses_and_selection_replayed':True,'acceptance_rng_replayed':anneal,
        'max_trace_metric_error':maxmetric,'J':result['metrics']['weighted_objective'],
        'result_sha256':sha(path),'source_config_sha256':sha(folder/'config.json'),
        'actions_sha256':sha(folder/'actions.jsonl'),'checkpoint_used':result['checkpoint_used'],
        'object_heads_used':result['checkpoint_used'] and args.get('object_ablation','none')=='none'}


def worker(task):
    torch.set_num_threads(1)
    return audit_trial(*task)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--id',action='append')
    parser.add_argument('--skip-provenance',action='store_true');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    torch.set_num_threads(1);protect()
    preserved=read(ROOT/'output/ppo-scalar-advantage-v5/preserved_inputs.json')
    for file,h in preserved.items():assert sha(ROOT/file)==h, file
    paths=sorted((OUT/'evaluate').glob('*/result.json'))
    if args.id:paths=[p for p in paths if p.parent.name in args.id]
    assert paths
    records=[];samples=[];params=None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,(path,not args.skip_provenance)) for path in paths]
        for future in as_completed(futures):
            params,plan,metrics,record=future.result()
            records.append(record);samples.append((plan,metrics));print(json.dumps(record),flush=True)
    records.sort(key=lambda x:x['trial'])
    from audit_scalar_advantage_v5 import full_model_replay
    replay=full_model_replay(params,samples)
    assert replay['violated_rows']==0 and replay['max_bound_residual']<1e-7 and replay['max_integrality_residual']<1e-7
    report={'status':'PASS','preserved_files':len(preserved),'action_provenance':not args.skip_provenance,
            'trials':records,'full_milp':replay,'audit_source_sha256':sha(Path(__file__))}
    filename='audit_selected.json' if args.id else 'audit.json'
    write(OUT/filename,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

"""Replay full-object PPO scientific artifacts; no solver or optimizer updates."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import math
import numpy as np
import torch

from run_ppo_objects_v7 import ROOT,OUT,read,sha,write,protect,forward,tensors
from src.ppo_objects_v7 import ObjectState,ObjectPolicy,Action,OPS,SearchEnvironment
from src.ppo_objects_v7b import ObjectState as CapacityState
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan,plan_sha256


def audit_trial(path,provenance=True):
    result=read(path);folder=path.parent;config=read(folder/'config.json');args=config['args']
    assert sha(folder/'config.json')==result['config_sha256']
    for file,h in config['sources'].items():assert sha(folder/'source'/file)==h
    data_path=ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json'
    assert sha(data_path)==config['input_sha256']
    data=read(data_path);p=params_from_json_data(data['params']);refs=(data['b_C'],data['b_R'])
    state=(CapacityState if args.get('variant','a')=='b' else ObjectState)(p,refs)
    env=SearchEnvironment(state,deserialize_plan(data['reference_plan']),(.5,.5),args['horizon'])
    plan=deserialize_plan(result['plan']);assert plan_sha256(plan)==result['solution_id']
    assert result['metrics']['cost_ref']==refs[0] and result['metrics']['risk_ref']==refs[1]
    assert result['metrics']['feasible'] and not result['metrics']['violations']
    assert result['attempts']==args['budget']==sum(result['counts'].get(op+'_attempts',0) for op in OPS)
    assert result['seconds']>0 and math.isfinite(result['seconds'])
    assert result['beats_all_existing_nsga']==(result['metrics']['weighted_objective']<config['benchmark']['target_J']-1e-8)
    model=None
    if result['checkpoint_used']:
        ckptpath=ROOT/config['checkpoint']['path'];assert sha(ckptpath)==config['checkpoint']['sha256']
        ckpt=torch.load(ckptpath,map_location='cpu',weights_only=False)
        assert ckpt['format']=='full_objects_v7'
        assert ckpt['config']['args'].get('variant','a')==args.get('variant','a'),'Silent encoder/checkpoint mismatch'
        for file,h in ckpt['config']['training_inputs'].items():
            assert '/train/Test-4/' in file and sha(ROOT/file)==h
        assert len(ckpt['config']['training_inputs'])==24
        assert all(all(g>0 for g in item['head_gradient_norms']) for item in ckpt['history'])
        model=ObjectPolicy(ckpt['dimension'],ckpt['layers']);model.load_state_dict(ckpt['model_state_dict']);model.eval()
    actions=[json.loads(line) for line in (folder/'actions.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(actions)==args['budget']
    status_counts=Counter(x['status'] for x in actions)
    op_counts=Counter(OPS[x['action']['operator']]+'_attempts' for x in actions)
    assert all(result['counts'][k]==v for k,v in (status_counts+op_counts).items())
    trace={x['attempt']:x for x in result['trace']};maxlog=0.;maxmetric=0.
    edge=torch.tensor(state.edges)[None]
    for start in range(0,len(actions),args['width']):
        block=actions[start:start+args['width']]
        scores=None
        if provenance:
            if model is None:scores=[np.zeros(6),*([np.zeros(512)]*3)]
            else:
                with torch.inference_mode():logits,_=forward(model,tensors([env.encode()],'cpu'),edge)
                scores=[x[0].numpy()/args['temperature'] for x in logits]
                if args.get('object_ablation','none')=='uniform':scores[1:]=[np.zeros(512) for _ in range(3)]
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
                    allowed=np.flatnonzero(mask);v=np.asarray(scores[head][allowed],dtype=np.float64)
                    v=np.exp(v-v.max());v/=v.sum();lp+=float(np.log(max(v[list(allowed).index(choice)],1e-300)))
                error=abs(lp-item['log_probability']);maxlog=max(maxlog,error)
                assert error<5e-4,(folder.name,offset,error)
        end=start+len(block)
        if end in trace:
            point=trace[end]
            assert any(x['action']==point['action'] and x['status']=='accepted' for x in block)
            reward,status=env.step(Action(**point['action']));assert status=='accepted'
            for key,metric in [('C','cost'),('R','risk'),('J','weighted_objective')]:
                error=abs(env.metrics[metric]-point[key]);maxmetric=max(maxmetric,error);assert error<1e-7
        env.age=end
    assert plan_sha256(env.plan)==result['solution_id']
    return p,plan,result['metrics'],{'trial':folder.name,'sampled_actions':len(actions),
        'accepted_transitions':len(trace),'max_log_probability_error':maxlog,
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

"""Versioned single-point trials. Existing NSGA plans are never search inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/ppo-scalar-exploration-v6'
BASE=ROOT/'output/pareto-parameter-revision-v4'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def benchmark():
    data=read(BASE/'instances/Test-4.json');refs=(data['b_C'],data['b_R']);rows=[]
    for budget in (40320,120960):
        for repeat in range(3):
            path=BASE/f'fronts/NSGA-Test-4-r{repeat}-B{budget}.json';front=read(path)
            assert (front['b_C'],front['b_R'])==refs and front['instance_sha256']==data['instance_sha256']
            p=min(front['points'],key=lambda p:.5*p['cost']/refs[0]+.5*p['risk']/refs[1])
            rows.append({'budget':budget,'repeat':repeat,'cost':p['cost'],'risk':p['risk'],
                         'J':.5*p['cost']/refs[0]+.5*p['risk']/refs[1],
                         'archive_sha256':sha(path),'solution_id':p['solution_id']})
    return data,{'preference':[.5,.5],'b_C':refs[0],'b_R':refs[1],'rows':rows,
                'target_J':min(r['J'] for r in rows),'instance_file_sha256':sha(BASE/'instances/Test-4.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',required=True)
    parser.add_argument('--method',choices=['legacy','compact','ppo_compact'],required=True)
    parser.add_argument('--budget',type=int,default=8000)
    parser.add_argument('--seed',type=int,default=3381415411)
    parser.add_argument('--width',type=int,default=8)
    parser.add_argument('--guidance',type=float,default=.5)
    parser.add_argument('--shake-after',type=int,default=600)
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    if Path(args.id).name!=args.id:raise ValueError('trial id must be a basename')
    dest=OUT/'trials'/args.id
    if dest.exists():raise FileExistsError('Never overwrite a previous trial')
    data,target=benchmark()
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),ROOT/'src/compact_search.py',ROOT/'src/ppo_improver.py',ROOT/'src/operators.py',ROOT/'src/solution_utils.py',ROOT/'hazardous_waste_model.py']}
    config={'args':vars(args),'benchmark':target,'sources':sources,
            'start':'common_reference_plan_only','exploratory':True}
    write(dest/'config.json',config)
    for path in sources:
        saved=dest/'source'/path;saved.parent.mkdir(parents=True,exist_ok=True)
        # Byte-for-byte version artifact, not a generated diagnostic program.
        saved.write_bytes((ROOT/path).read_bytes())
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    from sample_params import params_from_json_data
    from src.reproducibility import deserialize_plan,plan_to_canonical_data,plan_sha256
    from src.ppo_improver import PPOImprover
    from src.solution_utils import evaluate_solution
    from src.compact_search import search
    from src.pareto_experiment import CandidateLedger
    torch.cuda.is_available=lambda:False
    torch.set_num_threads(1)
    params=params_from_json_data(data['params']);plan=deserialize_plan(data['reference_plan']);refs=(data['b_C'],data['b_R'])
    policy=None
    if args.method!='compact':
        reg=read(ROOT/'configs/ppo_models_parameter_revision_v4.json');entry=reg['models']['large']
        checkpoint=ROOT/entry['checkpoint'];assert sha(checkpoint)==entry['sha256']
        policy=PPOImprover.from_frozen_checkpoint(params,checkpoint,objective_refs=refs)
        config['checkpoint']=entry;write(dest/'config.json',config)
    def progress(info):
        write(dest/'progress.json',info)
        print(args.id,info['attempt'],'J',info['best_J'],flush=True)
    start=time.perf_counter()
    if args.method=='legacy':
        assert args.budget%args.width==0
        policy.algorithm_config['ppo']['eval_candidate_samples']=args.width
        ledger=CandidateLedger(params,refs,plan)
        with ledger.observe_ppo((.5,.5),args.seed):
            r=policy.improve((.5,.5),steps=args.budget//args.width,seed=args.seed,initial_plan=plan)
        result={'plan':r.plan,'metrics':evaluate_solution(params,r.solution,(.5,.5),refs),
                'counts':dict(ledger.counts),'trace':r.trace,'seconds':time.perf_counter()-start}
    else:
        result=search(params,plan,refs,budget=args.budget,seed=args.seed,policy=policy,
                      guidance=args.guidance,shake_after=args.shake_after,progress=progress)
    saved_plan=result.pop('plan');result['solution_id']=plan_sha256(saved_plan)
    result['plan']=plan_to_canonical_data(saved_plan)
    result['beats_all_existing_nsga']=result['metrics']['weighted_objective']<target['target_J']-1e-8
    result['config_sha256']=sha(dest/'config.json')
    for path,value in sources.items():assert sha(ROOT/path)==value,'Source changed during trial'
    if args.audit:
        from audit_scalar_advantage_v5 import full_model_replay
        result['full_milp_audit']=full_model_replay(params,[(saved_plan,result['metrics'])])
    write(dest/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('plan','trace')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

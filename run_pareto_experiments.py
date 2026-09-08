"""Frozen single-instance Pareto protocol v3. No training entry point exists.

prepare -> preflight -> (independent audit approval) -> run -> separate report.
Formal records are immutable, source-bound and individually resumable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from sample_params import params_from_json_data, params_to_json_data
from src.heuristics import build_greedy_initial_plan
from src.instance_generator import generate_random_params
from src.pareto_experiment import CandidateLedger, coverage_pair, nsga_fronts
from src.reproducibility import (deserialize_plan, derive_seed, environment_snapshot,
                                file_sha256, json_sha256, plan_sha256, plan_to_canonical_data)
from src.solution_utils import evaluate_solution, route_plan_to_solution

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/pareto-single-instance-v3'
DATA = ROOT / 'datasets/reference_generalization_v2'
REGISTRY = ROOT / 'configs/frozen_ppo_models.json'
SOURCES = ['run_pareto_experiments.py', 'src/pareto_experiment.py', 'src/ppo_improver.py',
           'src/operators.py', 'src/heuristics.py', 'src/solution_utils.py', 'src/reproducibility.py',
           'src/instance_generator.py', 'sample_params.py', 'hazardous_waste_model.py']
SMALL_SEEDS = [[2571384110,1602506532,3447525709], [3229999428,4248913066,2951447865],
               [4084462614,794579791,1716606784], [2790166943,14408268,1540054241],
               [1809039277,4194498594,1142480955]]
SMALL_PREFS = [(1.,0.),(.75,.25),(.5,.5),(.25,.75),(0.,1.)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def source_hashes():
    return {name: file_sha256(ROOT/name) for name in SOURCES}


def verify_models():
    registry = read(REGISTRY)
    for item in registry['models'].values():
        if file_sha256(ROOT/item['checkpoint']) != item['sha256']:
            raise RuntimeError('Frozen checkpoint identity mismatch; stop, never retrain or substitute')
    if file_sha256(DATA/'manifest.json') != registry['dataset_manifest_sha256']:
        raise RuntimeError('Locked dataset manifest changed')
    return registry


def checked_instance(data):
    params = params_from_json_data(data['params'])
    plan = deserialize_plan(data['reference_plan'])
    refs = (data['b_C'], data['b_R'])
    if json_sha256(data['params']) != data['instance_sha256'] or plan_sha256(plan) != data['reference_plan_sha256']:
        raise RuntimeError('Instance/reference plan identity mismatch')
    metrics = evaluate_solution(params, route_plan_to_solution(params, plan), (.5,.5), objective_refs=refs)
    if not metrics['feasible'] or min(refs) <= 0:
        raise RuntimeError('Invalid bound reference')
    if abs(metrics['cost']-refs[0]) > 1e-9 or abs(metrics['risk']-refs[1]) > 1e-9:
        raise RuntimeError('Persisted b does not match its reference plan')
    return params, plan, refs


def bind(instance_id, params, plan, **metadata):
    metrics = evaluate_solution(params, route_plan_to_solution(params, plan), (.5,.5))
    if not metrics['feasible'] or min(metrics['cost'], metrics['risk']) <= 0:
        raise RuntimeError(f'Cannot bind a strictly feasible positive reference: {instance_id}')
    return {'instance_id': instance_id, 'params': params_to_json_data(params),
            'instance_sha256': json_sha256(params_to_json_data(params)),
            'reference_plan': plan_to_canonical_data(plan), 'reference_plan_sha256': plan_sha256(plan),
            'search_initial_plan_sha256': plan_sha256(plan), 'b_C': metrics['cost'], 'b_R': metrics['risk'],
            'reference_metrics': metrics, **metadata}


def prepare():
    registry = verify_models()
    if (OUT/'protocol.json').exists():
        # Before any successful preflight/formal launch, fixes may revise the
        # draft source lock; retain the previous draft for an audit trail.
        if not (OUT/'execution.json').exists() and not (OUT/'completed').exists():
            prior = read(OUT/'protocol.json')
            if prior['sources'] != source_hashes():
                write(OUT/f'protocol-drafts/{file_sha256(OUT/"protocol.json")}.json',prior)
                if (OUT/'preflight.json').exists():
                    write(OUT/f'protocol-drafts/preflight-{file_sha256(OUT/"protocol.json")}.json',read(OUT/'preflight.json'))
                prior['sources'] = source_hashes()
                write(OUT/'protocol.json',prior)
        return load_protocol()
    instances = {}
    data_manifest = read(DATA/'manifest.json')
    for scale in ('Test-1','Test-2','Test-3','Test-4'):
        relative = f'test/{scale}/000.json'
        item = next(record for record in data_manifest['records'] if record['path'] == relative)
        path = DATA/relative
        if file_sha256(path) != item['sha256']:
            raise RuntimeError('Dataset instance file hash mismatch')
        data = read(path)
        checked_instance(data)
        instances[scale] = {**data, 'source_dataset_path': str(path.relative_to(ROOT)).replace('\\','/'),
                            'source_dataset_sha256': item['sha256'], 'base_instance_id': data['instance_id'],
                            'scenario_parameter': 'baseline', 'scenario_multiplier': 1.}
    old_params = generate_random_params(read(ROOT/'configs/model_config.json')['small'],3723524230)
    old_plan = build_greedy_initial_plan(old_params,3949079119)
    old = bind('legacy-small-fixed', old_params, old_plan, instance_seed=3723524230, reference_seed=3949079119)
    assert old['instance_sha256'] == 'c61c8fc4c2d43afb69fb76d26a69ef7a246a77ff11b451396b302849988a803c'
    assert old['reference_plan_sha256'] == '5fdfb1130823c21856db79670ab76cbbbac336b68ac2548cdf8c3b81d1a791f3'
    instances['small-fixed'] = old
    base = instances['Test-4']
    base_params, base_plan, _ = checked_instance(base)
    for field, name in [('processing_capacity','capacity'), ('coload_risk','coload')]:
        for multiplier in (.7,.8,1.2,1.3):
            changed = {key: value*multiplier for key,value in getattr(base_params,field).items()}
            params = replace(base_params, **{field: changed})
            scene = f'{name}-{int(round(multiplier*100))}'
            before, after = params_to_json_data(base_params), params_to_json_data(params)
            differences = [key for key in before if before[key] != after[key]]
            if differences != [field]:
                raise RuntimeError(f'Not a single-parameter variant: {differences}')
            # Reuse the same feasible reference route if valid, but evaluate and
            # bind it exactly once under this scenario's own parameters.
            plan = base_plan
            if not evaluate_solution(params, route_plan_to_solution(params,plan))['feasible']:
                from src.operators import repair_plan
                plan, ok = repair_plan(params,plan)
                if not ok:
                    raise RuntimeError(f'Reference construction failure (not proven infeasible): {scene}')
            instances[scene] = bind(f"{base['instance_id']}--{scene}", params, plan,
                                   base_instance_id=base['instance_id'], scenario_parameter=field,
                                   scenario_multiplier=multiplier, changed_fields=differences,
                                   reference_seed=base['reference_seed'])
    for name,data in instances.items():
        write(OUT/f'instances/{name}.json',data)
    tasks = []
    for scale in ('Test-1','Test-2','Test-3','Test-4'):
        for repeat in range(3):
            tasks.append({'id': f'NSGA-{scale}-r{repeat}', 'kind':'NSGA', 'instance':scale, 'repeat':repeat,
                          'budgets':[40320,120960] if scale=='Test-4' else [40320],
                          'seed': derive_seed(20260908,'pareto-v3',scale,repeat,'NSGA',bits=32)})
            for model in ('small','large'):
                tasks.append({'id':f'PPO-{scale}-{model}-r{repeat}', 'kind':'PPO', 'instance':scale,
                              'model':model,'repeat':repeat,
                              'seed':derive_seed(20260908,'pareto-v3',scale,repeat,model,bits=32)})
    for scene in instances:
        if not scene.startswith(('capacity-','coload-')):
            continue
        for repeat in range(3):
            tasks.append({'id':f'PPO-{scene}-large-r{repeat}', 'kind':'PPO','instance':scene,
                          'model':'large','repeat':repeat,
                          'seed':derive_seed(20260908,'pareto-v3','Test-4',repeat,'large',bits=32)})
    protocol = {'protocol':'pareto-single-instance-v3','sources':source_hashes(),
                'registry_sha256':file_sha256(REGISTRY),'registry':registry,
                'dataset_manifest_sha256':file_sha256(DATA/'manifest.json'),
                'instances':{name:file_sha256(OUT/f'instances/{name}.json') for name in instances},
                'tasks':tasks,'preferences':[(k/20,1-k/20) for k in range(21)],
                'ppo_steps':60,'ppo_candidates_per_step':32,'repeats':3,
                'nsga':{'population':100,'crossover':.90,'mutation':.25,'cache_entries':2048},
                'tolerance_normalized':1e-8,'small_preferences':SMALL_PREFS,'small_seeds':SMALL_SEEDS,
                'milp_time_limit':3600,'cpu_threads_per_worker':1,
                'environment':environment_snapshot(), 'test_instances_per_scale':1,
                'scenario_reference_policy':'each scenario binds own positive b before all searches'}
    write(OUT/'protocol.json',protocol)
    return protocol


def load_protocol():
    protocol = read(OUT/'protocol.json')
    if protocol['sources'] != source_hashes():
        raise RuntimeError('Computation source changed after protocol freeze')
    verify_models()
    for name,expected in protocol['instances'].items():
        if file_sha256(OUT/f'instances/{name}.json') != expected:
            raise RuntimeError('Protocol instance identity changed')
    return protocol


def load_model(params, refs, name, protocol):
    import torch
    from src.ppo_improver import PPOImprover
    torch.set_num_threads(1)
    entry = protocol['registry']['models'][name]
    # Legacy constructor auto-selects CUDA and has no public device argument.
    # Scope the device selection override to construction only; weights, network,
    # inference parameters and CPU sampling remain exactly the frozen versions.
    from unittest.mock import patch
    with patch.object(torch.cuda,'is_available',return_value=False):
        model = PPOImprover.from_frozen_checkpoint(params,ROOT/entry['checkpoint'],objective_refs=refs)
    if model.device.type != 'cpu' or not model.is_frozen or model.optimizer is not None:
        raise RuntimeError('This protocol requires CPU frozen inference without optimizer')
    if model.algorithm_config['ppo']['eval_steps'] != 60 or model.algorithm_config['ppo']['eval_candidate_samples'] != 32:
        raise RuntimeError('Checkpoint inference budget differs from locked protocol')
    return model


def save_front(task, data, snapshot, protocol):
    archive_id = f"{task['id']}-B{snapshot['budget']}"
    points = []
    for point in snapshot['points']:
        solution_id = point['solution_id']
        relative = f'solutions/{data["instance_id"]}/{solution_id}.json'
        target = OUT/relative
        plan_record = {'instance_id':data['instance_id'], 'instance_sha256':data['instance_sha256'],
                       'b_C':data['b_C'],'b_R':data['b_R'],'solution_id':solution_id,'plan':point['plan']}
        if target.exists():
            if read(target) != plan_record:
                raise RuntimeError('Solution hash collision or mismatched existing identity')
        else:
            write(target,plan_record)
        points.append({**{k:v for k,v in point.items() if k!='plan'},'solution_path':relative})
    result = {**snapshot,'points':points,'archive_id':archive_id,'task':task,
              'instance_id':data['instance_id'],'instance_sha256':data['instance_sha256'],
              'b_C':data['b_C'],'b_R':data['b_R'],'reference_plan_sha256':data['reference_plan_sha256'],
              'search_initial_plan_sha256':data['reference_plan_sha256'],
              'protocol_sha256':file_sha256(OUT/'protocol.json'),'strict_feasible_points':len(points),
              'status':'success' if points else 'failed_empty_front',
              'checkpoint':protocol['registry']['models'][task['model']] if task['kind']=='PPO' else None}
    write(OUT/f'fronts/{archive_id}.json',result)
    return result


def run_task(task):
    protocol = load_protocol()
    completion = OUT/f'completed/{task["id"]}.json'
    if completion.exists():
        if read(completion)['protocol_sha256'] != file_sha256(OUT/'protocol.json'):
            raise RuntimeError('Existing completion protocol changed')
        return f"REUSED {task['id']}"
    data = read(OUT/f'instances/{task["instance"]}.json')
    params, plan, refs = checked_instance(data)
    started = time.perf_counter()
    try:
        if task['kind']=='NSGA':
            nsga_fronts(params,refs,plan,task['seed'],tuple(task['budgets']),
                        checkpoint=lambda result: save_front(task,data,result,protocol))
        else:
            model = load_model(params,refs,task['model'],protocol)
            ledger = CandidateLedger(params,refs,plan)
            started_search = time.perf_counter()
            final_solutions = []
            for index,preference in enumerate(protocol['preferences']):
                evaluation_seed = derive_seed(task['seed'],'preference',index,bits=32)
                pref_started = time.perf_counter()
                with ledger.observe_ppo(preference,evaluation_seed):
                    result = model.improve(tuple(preference),seed=evaluation_seed,initial_plan=plan)
                metrics = evaluate_solution(params,result.solution,tuple(preference),objective_refs=refs)
                if not metrics['feasible']:
                    raise RuntimeError('PPO returned infeasible scalar incumbent')
                final_solutions.append({'preference':preference,'seed':evaluation_seed,
                                        'seconds':time.perf_counter()-pref_started,
                                        'plan':plan_to_canonical_data(result.plan),'solution_id':plan_sha256(result.plan),
                                        'metrics':metrics})
                write(OUT/f'progress/{task["id"]}.json',{'preferences_completed':index+1,
                      'seconds':time.perf_counter()-started_search,'counts':dict(ledger.counts)})
            assert ledger.counts['candidate_attempts']==40320
            snapshot = {'budget':40320,'counts':dict(ledger.counts),'seconds':time.perf_counter()-started_search,
                        'preparation_seconds':ledger.preparation_seconds,'points':ledger.archive.sorted_points(),
                        'final_solutions':final_solutions,'stop_reason':'all_21_weight_budgets_completed'}
            save_front(task,data,snapshot,protocol)
        write(completion,{'task':task,'protocol_sha256':file_sha256(OUT/'protocol.json'),
                          'seconds_total_including_load_and_export':time.perf_counter()-started})
        return f"DONE {task['id']} {time.perf_counter()-started:.1f}s"
    except Exception:
        write(OUT/f'failures/{task["id"]}-{time.time_ns()}.json',
              {'task':task,'traceback':traceback.format_exc(),'protocol_sha256':file_sha256(OUT/'protocol.json')})
        raise


def run_small():
    protocol = load_protocol()
    data = read(OUT/'instances/small-fixed.json')
    params,plan,refs = checked_instance(data)
    model = load_model(params,refs,'small',protocol)
    for index,preference in enumerate(SMALL_PREFS):
        for repeat,evaluation_seed in enumerate(SMALL_SEEDS[index]):
            target = OUT/f'small/ppo-p{index}-r{repeat}.json'
            if target.exists():
                if read(target)['protocol_sha256'] != file_sha256(OUT/'protocol.json'):
                    raise RuntimeError('Small result protocol mismatch')
                continue
            started = time.perf_counter()
            result = model.improve(preference,seed=evaluation_seed,initial_plan=plan)
            seconds = time.perf_counter()-started
            metrics = evaluate_solution(params,result.solution,preference,objective_refs=refs)
            if not metrics['feasible']:
                raise RuntimeError('Small fixed-case inference failure')
            write(target,{'preference':preference,'repeat':repeat,'seed':evaluation_seed,'seconds':seconds,
                          'metrics':metrics,'plan':plan_to_canonical_data(result.plan),'solution_id':plan_sha256(result.plan),
                          'instance_id':data['instance_id'],'instance_sha256':data['instance_sha256'],
                          'b_C':refs[0],'b_R':refs[1],'reference_plan_sha256':data['reference_plan_sha256'],
                          'checkpoint':protocol['registry']['models']['small'],
                          'protocol_sha256':file_sha256(OUT/'protocol.json')})
            print(f'SMALL p{index} r{repeat} J={metrics["weighted_objective"]:.6f}',flush=True)


def preflight():
    protocol = load_protocol()
    from src.ppo_improver import StateEncoder
    report = {'protocol_sha256':file_sha256(OUT/'protocol.json'),'instances':{},'full_run_started':False}
    network = read(DATA/'manifest.json')['network']
    training_hashes = {r['instance_sha256'] for r in read(DATA/'manifest.json')['records'] if r['split']=='train'}
    for name in protocol['instances']:
        data=read(OUT/f'instances/{name}.json')
        params,plan,refs=checked_instance(data)
        token_capacity=StateEncoder.required_token_capacity(params,network)
        object_capacity=len(params.pickup_nodes)*len(params.periods)
        assert token_capacity <= 896 and object_capacity <= 512
        if name.startswith('Test-'):
            assert data['instance_sha256'] not in training_hashes
        metrics=evaluate_solution(params,route_plan_to_solution(params,plan))
        solution=route_plan_to_solution(params,plan)
        processing_util=max(solution['raw'].get(('p',j,s,t),0)/params.processing_capacity[j,s,t]
                            for j,s,t in params.processing_capacity)
        vehicle_loads = [sum(solution['raw'].get(('q',node,vehicle,period),0) for node in params.pickup_nodes)
                         for vehicle in params.vehicles for period in params.periods]
        processing_slacks = [params.processing_capacity[j,s,t]*params.technology[j,s]-solution['raw'].get(('p',j,s,t),0)
                             for j,s,t in params.processing_capacity if params.technology[j,s]]
        report['instances'][name]={'instance_id':data['instance_id'],'strict_reference_feasible':metrics['feasible'],
                                  'b_C':refs[0],'b_R':refs[1],'required_tokens':token_capacity,
                                  'objects':object_capacity,'max_processing_utilization':processing_util,
                                  'max_vehicle_utilization':max(vehicle_loads)/params.vehicle_capacity,
                                  'minimum_vehicle_capacity_slack':params.vehicle_capacity-max(vehicle_loads),
                                  'minimum_processing_capacity_slack':min(processing_slacks)}
    data=read(OUT/'instances/Test-4.json')
    params,plan,refs=checked_instance(data)
    model=load_model(params,refs,'large',protocol)
    ledger=CandidateLedger(params,refs,plan)
    start=time.perf_counter()
    with ledger.observe_ppo((.5,.5),123):
        result=model.improve((.5,.5),steps=2,seed=123,initial_plan=plan)
    report['large_two_step_probe']={'seconds':time.perf_counter()-start,'counts':dict(ledger.counts),
                                     'points':len(ledger.archive.points)}
    nsga=nsga_fronts(params,refs,plan,123,budgets=(120,200))
    report['large_nsga_probe']={budget:{k:v for k,v in result.items() if k!='points'} for budget,result in nsga.items()}
    test_task={'id':'preflight','kind':'PPO','model':'large','instance':'Test-4','repeat':-1,'seed':123}
    snapshot={'budget':64,'counts':dict(ledger.counts),'seconds':report['large_two_step_probe']['seconds'],
              'points':ledger.archive.sorted_points(),'stop_reason':'preflight_only'}
    saved=save_front(test_task,data,snapshot,protocol)
    for point in saved['points']:
        record=read(OUT/point['solution_path'])
        metrics=evaluate_solution(params,route_plan_to_solution(params,deserialize_plan(record['plan'])))
        assert metrics['feasible'] and abs(metrics['cost']-point['cost'])<1e-9 and abs(metrics['risk']-point['risk'])<1e-9
    report['replay_points_verified']=len(saved['points'])
    report['passed']=True
    write(OUT/'preflight.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','preflight','small','run'])
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare()
    elif args.stage=='preflight':
        preflight()
    elif args.stage=='small':
        run_small()
    else:
        protocol=load_protocol()
        if not read(OUT/'preflight.json')['passed']:
            raise RuntimeError('Preflight must pass')
        write(OUT/'execution.json',{'workers':args.workers,'device':'cpu','threads_per_worker':1,
                                   'protocol_sha256':file_sha256(OUT/'protocol.json'),
                                   'timing_note':'Concurrent wall-clock search times, not isolated speed benchmark'})
        with ProcessPoolExecutor(args.workers,mp_context=get_context('spawn')) as pool:
            futures=[pool.submit(run_task,task) for task in protocol['tasks']]
            for index,future in enumerate(as_completed(futures)):
                print(f'{index+1}/{len(futures)} {future.result()}',flush=True)
        run_small()


if __name__=='__main__':
    main()

"""Versioned complete-frontier rerun using optimized, preference-conditioned PPO.

NSGA candidate budgets are REQUIRED explicit arguments at protocol preparation;
the reference paper's iteration counts are never silently reinterpreted here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import time
import traceback

import run_pareto_experiments as engine
from src.pareto_experiment import CandidateLedger,nsga_fronts
from src.reproducibility import derive_seed,file_sha256,json_sha256,environment_snapshot
from src.ppo_frontier_v9 import search_preference

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/pareto-ppo-v9'
OLD=ROOT/'output/pareto-parameter-revision-v4'
DATA=ROOT/'datasets/parameter_revision_v4'
MODELS=ROOT/'outputs/ppo_frontier_v9'
REGISTRY=ROOT/'configs/ppo_models_frontier_v9.json'
read,write,checked_instance=engine.read,engine.write,engine.checked_instance
SOURCES=['run_ppo_frontier_v9.py','train_ppo_frontier_v9.py','src/ppo_frontier_v9.py',
    'src/ppo_objects_v7.py','src/ppo_objects_v7b.py','src/ppo_objects_v8.py','src/ppo_objects_v8c.py',
    'src/ppo_unique_sampling.py','src/pareto_experiment.py','src/operators.py','src/heuristics.py',
    'src/solution_utils.py','src/reproducibility.py','sample_params.py','hazardous_waste_model.py',
    'run_pareto_experiments.py']
engine.OUT=OUT;engine.DATA=DATA;engine.REGISTRY=REGISTRY;engine.SOURCES=SOURCES


def model_entry(name):
    config_path=MODELS/'training'/name/'config.json';complete_path=config_path.parent/'completed.json'
    cfg=read(config_path);done=read(complete_path);checkpoint=MODELS/'models'/f'{name}_model.pt'
    assert file_sha256(checkpoint)==done['checkpoint_sha256']
    assert done['source_config_sha256']==file_sha256(config_path)
    assert done['training_actions']==36864 and done['optimizer_steps']==1152
    assert cfg['state_class']=='src.ppo_objects_v8.ObjectState' and cfg['args']['layers']==1
    assert len(cfg['training_inputs'])==24 and len(cfg['preference_grid'])==21
    for source,h in cfg['sources'].items():assert file_sha256(ROOT/source)==h
    for source,h in cfg['training_inputs'].items():assert file_sha256(ROOT/source)==h
    coverage=read(config_path.parent/'preference_coverage.json')
    assert file_sha256(config_path.parent/'preference_coverage.json')==done['coverage_sha256']
    return {'experiment_alias':'New-'+('S' if name=='small' else 'L')+'-v9',
            'training_scale':cfg['scale'],'sha256':done['checkpoint_sha256'],
            'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
            'training_record':config_path.relative_to(ROOT).as_posix(),
            'training_completed':complete_path.relative_to(ROOT).as_posix(),
            'training_record_sha256':file_sha256(config_path),
            'training_completed_sha256':file_sha256(complete_path)}


def register_models():
    """Model provenance can be frozen before the unrelated NSGA budget choice."""
    v4=read(ROOT/'configs/ppo_models_parameter_revision_v4.json')
    assert file_sha256(DATA/'manifest.json')==v4['dataset_manifest_sha256']
    registry={'registry_version':1,'status':'user_requested_v9_two_models_frozen_after_training',
              'objective_normalization':'instance_reference','path_base':'repository_root',
              'dataset_manifest':'datasets/parameter_revision_v4/manifest.json',
              'dataset_manifest_sha256':file_sha256(DATA/'manifest.json'),
              'architecture':'conditional_objects_v8c','models':{name:model_entry(name) for name in ('small','large')}}
    if REGISTRY.exists():assert read(REGISTRY)==registry,'Refuse to replace different registry'
    else:write(REGISTRY,registry)
    return registry


def _relative_file(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Artifact path escapes source/output root')
    return path


def nsga_reuse_manifest(old,tasks,instances,budgets):
    """Lock exact old artifacts; iteration counts are not candidate counts."""
    old_tasks={t['id']:t for t in old['tasks'] if t['kind']=='NSGA'}
    new_tasks={t['id']:t for t in tasks if t['kind']=='NSGA'}
    old_budgets=sorted({budget for t in old_tasks.values() for budget in t['budgets']})
    if list(budgets)!=old_budgets:raise ValueError('Reuse requires unchanged old NSGA budgets')
    if old_tasks!=new_tasks:raise ValueError('NSGA tasks/seeds/budgets differ from source protocol')
    source_hash=file_sha256(OLD/'protocol.json')
    execution=read(OLD/'execution.json')
    if execution['protocol_sha256']!=source_hash:raise ValueError('Old execution belongs to another protocol')
    result={'source_root':OLD.resolve().relative_to(ROOT.resolve()).as_posix(),
            'source_protocol_sha256':source_hash,'source_execution_sha256':file_sha256(OLD/'execution.json'),
            'fronts':{},'solutions':{},'completions':{},
            'semantics':'Byte-identical existing NSGA archives and plans; no new NSGA optimization; old timing retained'}
    for task in new_tasks.values():
        instance_path=f'instances/{task["instance"]}.json'
        if instances[task['instance']]!=old['instances'][task['instance']]:raise ValueError('New instance differs')
        if file_sha256(OLD/instance_path)!=instances[task['instance']]:raise ValueError('Source instance changed')
        data=read(OLD/instance_path)
        done_path=f'completed/{task["id"]}.json';done=read(OLD/done_path)
        if done['task']!=task or done['protocol_sha256']!=source_hash:raise ValueError('Old completion mismatch')
        result['completions'][done_path]=file_sha256(OLD/done_path)
        for budget in task['budgets']:
            relative=f'fronts/{task["id"]}-B{budget}.json';front=read(OLD/relative)
            if front['task']!=task or front['budget']!=budget or front['protocol_sha256']!=source_hash:
                raise ValueError('Old front task/budget/protocol mismatch')
            for key in ('instance_id','instance_sha256','b_C','b_R','reference_plan_sha256'):
                if front[key]!=data[key]:raise ValueError('Old front instance/reference mismatch')
            if front['search_initial_plan_sha256']!=data['reference_plan_sha256']:
                raise ValueError('Old front did not start from shared reference')
            result['fronts'][relative]=file_sha256(OLD/relative)
            for point in front['points']:
                solution_path=point['solution_path'];solution_file=_relative_file(OLD,solution_path)
                solution=read(solution_file)
                for key in ('instance_id','instance_sha256','b_C','b_R'):
                    if solution[key]!=data[key]:raise ValueError('Old solution instance/reference mismatch')
                if solution['solution_id']!=point['solution_id']:raise ValueError('Old solution ID mismatch')
                result['solutions'][solution_path]=file_sha256(solution_file)
    return result


def verify_reused_nsga(protocol,front,path):
    """Verify a copied NSGA archive without relabelling its historical protocol.

    ``path`` is the actual copied front, absolute or relative to OUT. Returns
    True only when source, destination, wrapper completion and all plans match.
    """
    reuse=protocol.get('nsga_reuse')
    if not reuse:raise ValueError('Protocol does not authorize NSGA archive reuse')
    source=_relative_file(ROOT,reuse['source_root'])
    for name,key in [('protocol.json','source_protocol_sha256'),('execution.json','source_execution_sha256')]:
        if file_sha256(source/name)!=reuse[key]:raise ValueError('Historical provenance hash mismatch')
    file=Path(path)
    if not file.is_absolute():file=OUT/file
    relative=file.resolve().relative_to(OUT.resolve()).as_posix()
    expected=reuse['fronts'].get(relative)
    if expected is None or file_sha256(file)!=expected or file_sha256(_relative_file(source,relative))!=expected:
        raise ValueError('Copied or source NSGA front hash mismatch')
    if read(file)!=front:raise ValueError('Supplied front differs from file bytes')
    task=front['task']
    if task['kind']!='NSGA' or task not in protocol['tasks']:raise ValueError('Unregistered NSGA task')
    if front['protocol_sha256']!=reuse['source_protocol_sha256']:raise ValueError('Historical protocol was relabelled')
    if front['budget'] not in task['budgets']:raise ValueError('Unregistered NSGA budget')
    instance=OUT/f'instances/{task["instance"]}.json'
    if file_sha256(instance)!=protocol['instances'][task['instance']]:raise ValueError('New instance hash mismatch')
    data=read(instance)
    for key in ('instance_id','instance_sha256','b_C','b_R','reference_plan_sha256'):
        if front[key]!=data[key]:raise ValueError('Reused front reference mismatch')
    complete_relative=f'completed/{task["id"]}.json'
    old_complete=_relative_file(source,complete_relative)
    if file_sha256(old_complete)!=reuse['completions'][complete_relative]:raise ValueError('Source completion changed')
    original=read(old_complete)
    if original['task']!=task or original['protocol_sha256']!=reuse['source_protocol_sha256']:
        raise ValueError('Source completion task/protocol mismatch')
    wrapper=read(OUT/complete_relative)
    if wrapper['task']!=task or wrapper['protocol_sha256']!=file_sha256(OUT/'protocol.json'):
        raise ValueError('Reuse wrapper does not match new protocol')
    expected_origin={'path':(source/complete_relative).relative_to(ROOT.resolve()).as_posix(),
                     'sha256':reuse['completions'][complete_relative],
                     'protocol_sha256':reuse['source_protocol_sha256'],
                     'execution_sha256':reuse['source_execution_sha256']}
    if wrapper.get('reused_from')!=expected_origin or wrapper.get('solver_rerun') is not False:
        raise ValueError('Missing or incorrect explicit reuse provenance')
    for point in front['points']:
        relative=point['solution_path'];expected=reuse['solutions'][relative]
        if file_sha256(_relative_file(OUT,relative))!=expected or file_sha256(_relative_file(source,relative))!=expected:
            raise ValueError('Copied or source NSGA solution hash mismatch')
    return True


def import_reused_nsga(protocol,task=None):
    reuse=protocol.get('nsga_reuse')
    if not reuse:raise ValueError('NSGA reuse must be frozen during prepare')
    source=_relative_file(ROOT,reuse['source_root'])
    for name,key in [('protocol.json','source_protocol_sha256'),('execution.json','source_execution_sha256')]:
        if file_sha256(source/name)!=reuse[key]:raise ValueError('Historical provenance hash mismatch')
    tasks=[t for t in protocol['tasks'] if t['kind']=='NSGA' and (task is None or t==task)]
    if not tasks:raise ValueError('No registered NSGA reuse tasks')
    for current in tasks:
        fronts=[f'fronts/{current["id"]}-B{budget}.json' for budget in current['budgets']]
        files={relative:reuse['fronts'][relative] for relative in fronts}
        for relative in fronts:
            for point in read(source/relative)['points']:
                files[point['solution_path']]=reuse['solutions'][point['solution_path']]
        for relative,expected in files.items():
            original=_relative_file(source,relative);target=_relative_file(OUT,relative)
            if file_sha256(original)!=expected:raise ValueError('Reuse source artifact changed')
            if target.exists():
                if file_sha256(target)!=expected:raise ValueError('Refuse to overwrite different existing artifact')
            else:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
        complete_relative=f'completed/{current["id"]}.json'
        if file_sha256(source/complete_relative)!=reuse['completions'][complete_relative]:
            raise ValueError('Source completion changed')
        wrapper={'task':current,'protocol_sha256':file_sha256(OUT/'protocol.json'),'solver_rerun':False,
                 'reused_from':{'path':(source/complete_relative).relative_to(ROOT.resolve()).as_posix(),
                    'sha256':reuse['completions'][complete_relative],
                    'protocol_sha256':reuse['source_protocol_sha256'],
                    'execution_sha256':reuse['source_execution_sha256']},
                 'historical_seconds_total_including_load_and_export':read(source/complete_relative)['seconds_total_including_load_and_export'],
                 'timing_note':'Copied front seconds are original v4 search times; no v9 NSGA run occurred'}
        target=OUT/complete_relative
        if target.exists():
            if read(target)!=wrapper:raise ValueError('Refuse to replace an existing different completion')
        else:write(target,wrapper)
        for relative in fronts:verify_reused_nsga(protocol,read(OUT/relative),OUT/relative)
    return len(tasks)


def prepare(budgets,reuse_nsga=False):
    if len(budgets)!=2 or not 100<=budgets[0]<budgets[1]:raise ValueError('Explicit increasing NSGA candidate limits required')
    if (OUT/'protocol.json').exists():
        prior=load_protocol()
        assert prior['nsga']['budgets']==list(budgets),'Cannot alter a frozen experiment budget'
        assert bool(prior.get('nsga_reuse'))==reuse_nsga,'Cannot alter frozen reuse policy'
        if reuse_nsga:import_reused_nsga(prior)
        return prior
    old=read(OLD/'protocol.json');v4=read(ROOT/'configs/ppo_models_parameter_revision_v4.json')
    assert file_sha256(DATA/'manifest.json')==v4['dataset_manifest_sha256']
    for model in v4['models'].values():assert file_sha256(ROOT/model['checkpoint'])==model['sha256']
    instances={}
    for name,h in old['instances'].items():
        source=OLD/f'instances/{name}.json';assert file_sha256(source)==h
        checked_instance(read(source));target=OUT/f'instances/{name}.json'
        if target.exists():assert file_sha256(target)==h
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        instances[name]=h
    registry=register_models()
    tasks=deepcopy(old['tasks'])
    for task in tasks:
        if task['kind']=='NSGA':task['budgets']=list(budgets if task['instance']=='Test-4' else budgets[:1])
        else:task['budget']=40320
    protocol={**old,'protocol':'ppo-frontier-v9','sources':engine.source_hashes(),
        'registry':registry,'registry_sha256':file_sha256(REGISTRY),'instances':instances,'tasks':tasks,
        'ppo_steps':240,'ppo_candidates_per_step':8,'ppo_candidate_budget_per_preference':1920,
        'ppo_frontier_budget':40320,'ppo_temperature':2.,'ppo_horizon':1024,'ppo_memoize':True,
        'ppo_device':'cpu','cpu_threads_per_worker':1,
        'nsga':{**old['nsga'],'budgets':list(budgets),'budget_unit':'candidate_attempts'},
        'budget_source':{'paper_units':'iterations','paper_counts':[1000,3000],
            'equivalent_to_paper':False,'candidate_limits':list(budgets),
            'authorization':'Explicit user choice required before prepare; not inferred from paper'},
        'early_stop_policy':'Only root positive-probability action support exhaustion at unchanged input; retain partial batch and actual counts',
        'algorithm_design_test_informed':True,'training_selection':'predetermined final update 48',
        'environment':environment_snapshot(),
        'parameter_revision_source_protocol_sha256':file_sha256(OLD/'protocol.json')}
    # The inherited v4 parameter calibration is historical data provenance,
    # not a claim that this architecture/learning setup is unchanged from v4.
    protocol['parameter_revision']={**old['parameter_revision'],'reused_unchanged_v4_instances':True}
    if reuse_nsga:
        protocol['nsga_reuse']=nsga_reuse_manifest(old,tasks,instances,budgets)
        protocol['budget_source']['authorization']='User explicitly retained old candidate budgets and authorized byte-identical NSGA archive reuse'
    write(OUT/'protocol.json',protocol)
    for source in SOURCES:
        target=OUT/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
    if reuse_nsga:import_reused_nsga(protocol)
    return protocol


def load_protocol():return engine.load_protocol()


def load_model(name,protocol):
    import torch
    from src.ppo_objects_v8c import ObjectPolicy
    torch.set_num_threads(1)
    record=protocol['registry']['models'][name];path=ROOT/record['checkpoint']
    assert file_sha256(path)==record['sha256']
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    assert ckpt['format']=='conditional_objects_v8c'
    assert ckpt['config']['version']=='ppo-frontier-v9-multipreference'
    model=ObjectPolicy(ckpt['dimension'],ckpt['layers']);model.load_state_dict(ckpt['model_state_dict'])
    model.eval();model.requires_grad_(False)
    return model


def run_task(task):
    protocol=load_protocol();target=OUT/f'completed/{task["id"]}.json'
    if task['kind']=='NSGA' and protocol.get('nsga_reuse'):
        import_reused_nsga(protocol,task)
        return 'REUSED HISTORICAL NSGA '+task['id']
    if target.exists():
        assert read(target)['protocol_sha256']==file_sha256(OUT/'protocol.json')
        return 'REUSED '+task['id']
    data=read(OUT/f'instances/{task["instance"]}.json');params,reference,refs=checked_instance(data)
    started=time.perf_counter()
    try:
        if task['kind']=='NSGA':
            nsga_fronts(params,refs,reference,task['seed'],tuple(task['budgets']),
                population_size=protocol['nsga']['population'],
                checkpoint=lambda snapshot:engine.save_front(task,data,snapshot,protocol))
        else:
            model=load_model(task['model'],protocol);ledger=CandidateLedger(params,refs,reference)
            results=[];search_started=time.perf_counter();log_dir=OUT/'actions'/task['id']
            log_dir.mkdir(parents=True,exist_ok=True)
            for index,preference in enumerate(protocol['preferences']):
                seed=derive_seed(task['seed'],'preference',index,bits=32)
                log_path=log_dir/f'p{index}.jsonl'
                if log_path.exists():raise RuntimeError('Unfinished run has action logs; preserve and investigate, do not overwrite')
                with log_path.open('x',encoding='utf8') as log:
                    result=search_preference(params,refs,reference,model,tuple(preference),seed,ledger=ledger,
                        budget=1920,width=8,temperature=2.,horizon=1024,log_handle=log)
                result['action_log_path']=log_path.relative_to(OUT).as_posix()
                result['action_log_sha256']=file_sha256(log_path);results.append(result)
                write(OUT/f'progress/{task["id"]}.json',{'preferences_completed':index+1,
                    'seconds':time.perf_counter()-search_started,'counts':dict(ledger.counts),
                    'last_stop_reason':result['stop_reason']})
                print(f"{task['id']} preference {index+1}/21 J={result['metrics']['weighted_objective']:.7f}",flush=True)
            actual=sum(r['actual_candidate_attempts'] for r in results)
            assert actual==ledger.counts['candidate_attempts']<=40320
            snapshot={'budget':40320,'counts':dict(ledger.counts),'actual_candidate_attempts':actual,
                'unused_candidate_budget':40320-actual,'seconds':time.perf_counter()-search_started,
                'preparation_seconds':ledger.preparation_seconds,'points':ledger.archive.sorted_points(),
                'final_solutions':results,'stop_reason':'all_21_preferences_completed_with_recorded_stops',
                'device':'cpu','threads':1,'final_validation_evaluations':21,
                'preference_initialization_evaluations':21}
            engine.save_front(task,data,snapshot,protocol)
        load_protocol()
        write(target,{'task':task,'protocol_sha256':file_sha256(OUT/'protocol.json'),
            'seconds_total_including_load_and_export':time.perf_counter()-started})
        return f"DONE {task['id']} {time.perf_counter()-started:.1f}s"
    except Exception:
        write(OUT/f'failures/{task["id"]}-{time.time_ns()}.json',
            {'task':task,'traceback':traceback.format_exc(),'protocol_sha256':file_sha256(OUT/'protocol.json')})
        raise


def run_small():
    protocol=load_protocol();data=read(OUT/'instances/small-fixed.json')
    params,reference,refs=checked_instance(data);model=load_model('small',protocol)
    for index,preference in enumerate(protocol['small_preferences']):
        for repeat,seed in enumerate(protocol['small_seeds'][index]):
            target=OUT/f'small/ppo-p{index}-r{repeat}.json'
            if target.exists():
                assert read(target)['protocol_sha256']==file_sha256(OUT/'protocol.json');continue
            log_path=OUT/f'actions/small/ppo-p{index}-r{repeat}.jsonl';log_path.parent.mkdir(parents=True,exist_ok=True)
            with log_path.open('x',encoding='utf8') as log:
                result=search_preference(params,refs,reference,model,tuple(preference),seed,log_handle=log)
            write(target,{**result,'repeat':repeat,'instance_id':data['instance_id'],
                'instance_sha256':data['instance_sha256'],'b_C':refs[0],'b_R':refs[1],
                'reference_plan_sha256':data['reference_plan_sha256'],
                'checkpoint':protocol['registry']['models']['small'],
                'protocol_sha256':file_sha256(OUT/'protocol.json'),
                'action_log_path':log_path.relative_to(OUT).as_posix(),'action_log_sha256':file_sha256(log_path)})
            print(f"SMALL p{index} r{repeat}: J={result['metrics']['weighted_objective']:.7f}, attempts={result['actual_candidate_attempts']}",flush=True)


def preflight():
    import torch
    from src.ppo_objects_v8c import ObjectPolicy
    from src.ppo_objects_v8 import ObjectState
    torch.set_num_threads(1);old=read(OLD/'protocol.json');entries=[]
    path=MODELS/'models/small_model.pt';complete=read(MODELS/'training/small/completed.json')
    assert file_sha256(path)==complete['checkpoint_sha256']
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=ObjectPolicy(96,1);model.load_state_dict(ckpt['model_state_dict']);model.eval();model.requires_grad_(False)
    for name in ('Test-1','Test-2','Test-3','Test-4','small-fixed'):
        source=OLD/f'instances/{name}.json';assert file_sha256(source)==old['instances'][name]
        params,reference,refs=checked_instance(read(source));state=ObjectState(params,refs)
        result=search_preference(params,refs,reference,model,(.5,.5),derive_seed(20260909,'v9-preflight',name,bits=32),
                                 budget=1920 if name=='small-fixed' else 32)
        assert result['metrics']['feasible'] and result['actual_candidate_attempts']<=result['candidate_budget']
        entries.append({'instance':name,'physical_nodes':len(state.nodes),'object_slots':state.size,
                        'result':result,'phase':'preflight_not_formal_result'})
        print(f"PREFLIGHT {name}: {result['actual_candidate_attempts']}/{result['candidate_budget']} {result['stop_reason']}",flush=True)
    write(OUT/'preflight.json',{'status':'PASS','model_sha256':file_sha256(path),
          'source_sha256':engine.source_hashes(),'entries':entries,'not_used_in_formal_results':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['register','prepare','reuse-nsga','preflight','run','small','normalize'])
    parser.add_argument('--nsga-budgets',type=int,nargs=2)
    parser.add_argument('--reuse-nsga',action='store_true',help='Copy and verify old NSGA bytes, retaining their original protocol identity')
    parser.add_argument('--id');parser.add_argument('--kind',choices=['PPO','NSGA'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.mode=='register':register_models();print('Two completed model identities registered; no search budget selected');return
    if args.mode=='prepare':
        if args.nsga_budgets is None:parser.error('Budget units must be explicitly authorized; supply --nsga-budgets B1 B2')
        prepare(args.nsga_budgets,args.reuse_nsga);print('Protocol and two-model registry frozen');return
    if args.mode=='preflight':preflight();return
    if args.mode=='normalize':
        import run_frontier_v9_milp as wrapper
        import normalize_pareto_milp as cleanup
        wrapper.configure();cleanup.DIRECTORY=OUT/'milp';cleanup.historical_case=wrapper.exact.historical_case
        cleanup.main();return
    protocol=load_protocol()
    if args.mode=='reuse-nsga':print(f'Reused {import_reused_nsga(protocol)} historical NSGA tasks');return
    if args.mode=='small':run_small();return
    tasks=[t for t in protocol['tasks'] if (not args.id or t['id']==args.id) and (not args.kind or t['kind']==args.kind)]
    if not tasks:raise ValueError('No matching tasks')
    write(OUT/f'execution-{time.time_ns()}.json',{'workers':args.workers,'device':'cpu',
          'threads_per_worker':1,'concurrent_milp_may_be_running':True,'tasks':[t['id'] for t in tasks]})
    if args.workers==1:
        for task in tasks:print(run_task(task),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(run_task,task):task for task in tasks}
            for future in as_completed(futures):print(future.result(),flush=True)


if __name__=='__main__':main()

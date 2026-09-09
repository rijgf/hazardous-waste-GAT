"""All PPO experiments at 12000 candidates/preference using frozen v9 PTs.

Only the candidate cap changes. Existing MILP and NSGA results retain bytes,
identities and original timing. Logs are losslessly gzip-compressed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid
import zipfile

import run_ppo_frontier_v9 as v9
from src.pareto_experiment import CandidateLedger
from src.ppo_frontier_v9 import search_preference
from src.reproducibility import derive_seed, file_sha256, environment_snapshot

ROOT=v9.ROOT
OUT=ROOT/'output/pareto-ppo-budget-v11'
OLD=v9.OLD
SOURCE=v9.OUT
REGISTRY=v9.REGISTRY
SOURCES=[*v9.SOURCES,'run_ppo_budget_v11.py']
read,write,checked_instance=v9.read,v9.write,v9.checked_instance
CAP=12000
FRONT_CAP=21*CAP


@contextmanager
def destination():
    previous=v9.OUT
    v9.OUT=OUT
    try:yield
    finally:v9.OUT=previous


def verify_reused_nsga(protocol,front,path):
    with destination():return v9.verify_reused_nsga(protocol,front,path)


def load_model(name,protocol):return v9.load_model(name,protocol)


def load_protocol():
    p=read(OUT/'protocol.json')
    assert p['protocol']=='ppo-budget-v11' and p['ppo_candidate_budget_per_preference']==CAP
    assert p['ppo_frontier_budget']==FRONT_CAP
    assert file_sha256(REGISTRY)==p['registry_sha256'] and read(REGISTRY)==p['registry']
    for source,h in p['sources'].items():assert file_sha256(ROOT/source)==h,source
    for name,h in p['instances'].items():assert file_sha256(OUT/f'instances/{name}.json')==h,name
    for item in p['registry']['models'].values():assert file_sha256(ROOT/item['checkpoint'])==item['sha256']
    for path,h in p['milp_reuse']['files'].items():
        assert file_sha256(SOURCE/path)==file_sha256(OUT/path)==h,path
    return p


def copy_exact(source,target):
    if target.exists():assert file_sha256(source)==file_sha256(target)
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)


def prepare():
    if (OUT/'protocol.json').exists():return load_protocol()
    old=v9.load_protocol();p=deepcopy(old)
    backup=ROOT/'backups/ppo-v9-before-budget-v11';backup.mkdir(parents=True,exist_ok=True)
    paper=ROOT/'供应链管理写作/数值实验与结果分析_论文稿.md'
    copy_exact(paper,backup/paper.name)
    saved=[*v9.SOURCES,'configs/ppo_models_frontier_v9.json',
           *[x['checkpoint'] for x in old['registry']['models'].values()]]
    with zipfile.ZipFile(backup/'source-and-models.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for source in saved:z.write(ROOT/source,source)
    write(backup/'manifest.json',{'files':{s:file_sha256(ROOT/s) for s in saved},
          'paper_sha256':file_sha256(paper),'zip_sha256':file_sha256(backup/'source-and-models.zip')})
    for name in p['instances']:copy_exact(SOURCE/f'instances/{name}.json',OUT/f'instances/{name}.json')
    milp={}
    for src in sorted((SOURCE/'milp').rglob('*')):
        if src.is_file() and src.name!='report_optimality_verdicts.json':
            relative=src.relative_to(SOURCE).as_posix();milp[relative]=file_sha256(src)
            copy_exact(src,OUT/relative)
    p.update(protocol='ppo-budget-v11',sources={s:file_sha256(ROOT/s) for s in SOURCES},
       ppo_candidate_budget_per_preference=CAP,ppo_frontier_budget=FRONT_CAP,ppo_steps=CAP//8,
       models_retrained_this_round=False,
       previous_protocol_sha256=file_sha256(SOURCE/'protocol.json'),
       budget_source={'authorization':'User explicitly chose frozen v9 models, no retraining, all PPO experiments at 12000 per preference',
          'ppo_candidate_limit_per_preference':CAP,'ppo_frontier_candidate_limit':FRONT_CAP,
          'nsga_candidate_limits':[40320,120960],'equal_candidate_budgets':False},
       milp_reuse={'source_root':SOURCE.relative_to(ROOT).as_posix(),'files':milp,'solver_rerun':False},
       action_log_encoding='gzip compressed UTF-8 JSONL; lossless; no action changes',
       environment=environment_snapshot())
    for task in p['tasks']:
        if task['kind']=='PPO':task['budget']=FRONT_CAP
    write(OUT/'protocol.json',p)
    for source in SOURCES:copy_exact(ROOT/source,OUT/'source'/source)
    with destination():v9.import_reused_nsga(p)
    return load_protocol()


def save_front(task,data,snapshot,protocol):
    previous=v9.engine.OUT;v9.engine.OUT=OUT
    try:return v9.engine.save_front(task,data,snapshot,protocol)
    finally:v9.engine.OUT=previous


def solve_preference(params,refs,reference,model,preference,seed,path,ledger=None,budget=CAP):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'xt',encoding='utf8',compresslevel=1) as log:
        result=search_preference(params,refs,reference,model,tuple(preference),seed,ledger=ledger,
               budget=budget,width=8,temperature=2.,horizon=1024,log_handle=log)
    result['action_log_path']=path.relative_to(OUT).as_posix()
    result['action_log_sha256']=file_sha256(path)
    return result


def run_task(task,execution_id):
    p=load_protocol();identity=task['id'];done=OUT/f'completed/{identity}.json'
    if done.exists():
        assert read(done)['protocol_sha256']==file_sha256(OUT/'protocol.json')
        return 'REUSED '+identity
    claim=OUT/f'claims/{identity}.json';claim.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive claim prevents concurrent CLI invocations from overwriting work.
    with claim.open('x',encoding='utf8') as h:h.write(execution_id)
    started=time.perf_counter()
    provenance={'task':task,'execution_id':execution_id,'pid':os.getpid(),
                'start_utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','threads':1}
    write(OUT/f'actual_execution/{identity}.json',provenance)
    try:
        data=read(OUT/f'instances/{task["instance"]}.json');params,reference,refs=checked_instance(data)
        model=load_model(task['model'],p);ledger=CandidateLedger(params,refs,reference)
        results=[];search_started=time.perf_counter()
        for index,pref in enumerate(p['preferences']):
            seed=derive_seed(task['seed'],'preference',index,bits=32)
            result=solve_preference(params,refs,reference,model,pref,seed,
                 OUT/f'actions/{identity}/p{index}.jsonl.gz',ledger)
            results.append(result)
            write(OUT/f'progress/{identity}.json',{'preferences_completed':index+1,
               'seconds':time.perf_counter()-search_started,'last_J':result['metrics']['weighted_objective'],
               'counts':dict(ledger.counts)})
            print(f"{identity} preference {index+1}/21 J={result['metrics']['weighted_objective']:.7f} attempts={result['actual_candidate_attempts']}",flush=True)
        actual=sum(r['actual_candidate_attempts'] for r in results)
        assert actual==ledger.counts['candidate_attempts']<=FRONT_CAP
        snapshot={'budget':FRONT_CAP,'counts':dict(ledger.counts),'actual_candidate_attempts':actual,
           'unused_candidate_budget':FRONT_CAP-actual,'seconds':time.perf_counter()-search_started,
           'preparation_seconds':ledger.preparation_seconds,'points':ledger.archive.sorted_points(),
           'final_solutions':results,'stop_reason':'all_21_preferences_completed_with_recorded_stops',
           'device':'cpu','threads':1,'final_validation_evaluations':21,'preference_initialization_evaluations':21}
        save_front(task,data,snapshot,p);load_protocol()
        write(done,{'task':task,'protocol_sha256':file_sha256(OUT/'protocol.json'),
             'execution_id':execution_id,'seconds_total_including_load_and_export':time.perf_counter()-started})
        write(OUT/f'actual_execution/{identity}.json',{**provenance,'end_utc':datetime.now(timezone.utc).isoformat(),
              'status':'completed','completed_sha256':file_sha256(done)})
        return f'DONE {identity} {time.perf_counter()-started:.1f}s'
    except Exception:
        write(OUT/f'failures/{identity}-{uuid.uuid4().hex}.json',{'provenance':provenance,'traceback':traceback.format_exc()})
        raise


def small():
    p=load_protocol();data=read(OUT/'instances/small-fixed.json');params,reference,refs=checked_instance(data)
    model=load_model('small',p)
    for index,pref in enumerate(p['small_preferences']):
        for repeat,seed in enumerate(p['small_seeds'][index]):
            target=OUT/f'small/ppo-p{index}-r{repeat}.json'
            if target.exists():continue
            result=solve_preference(params,refs,reference,model,pref,seed,OUT/f'actions/small/p{index}-r{repeat}.jsonl.gz')
            write(target,{**result,'repeat':repeat,'instance_id':data['instance_id'],'instance_sha256':data['instance_sha256'],
              'b_C':refs[0],'b_R':refs[1],'reference_plan_sha256':data['reference_plan_sha256'],
              'checkpoint':p['registry']['models']['small'],'protocol_sha256':file_sha256(OUT/'protocol.json')})
            print(f"SMALL p{index} r{repeat} J={result['metrics']['weighted_objective']:.8f} attempts={result['actual_candidate_attempts']}",flush=True)


def preflight():
    p=load_protocol();data=read(OUT/'instances/Test-4.json');params,reference,refs=checked_instance(data)
    model=load_model('large',p)
    old=read(SOURCE/'fronts/PPO-Test-4-large-r0-B40320.json')['final_solutions'][10]
    result=solve_preference(params,refs,reference,model,(.5,.5),old['seed'],OUT/'preflight/p10.jsonl.gz',budget=1920)
    assert result['solution_id']==old['solution_id'] and result['metrics']==old['metrics']
    write(OUT/'preflight.json',{'status':'PASS','same_1920_prefix_solution_id':result['solution_id'],
       'checkpoint_sha256':p['registry']['models']['large']['sha256'],'result':result,
       'not_formal_result':True,'new_budget':CAP})
    print('PASS identical v9 checkpoint/1920-candidate result with gzip log',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','preflight','run','small'])
    parser.add_argument('--workers',type=int,default=16);parser.add_argument('--id')
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='preflight':preflight()
    elif args.mode=='small':small()
    else:
        p=load_protocol();tasks=[t for t in p['tasks'] if t['kind']=='PPO' and (not args.id or t['id']==args.id)]
        if not tasks:raise ValueError('No matching PPO tasks')
        identity=uuid.uuid4().hex
        write(OUT/f'execution-{identity}.json',{'execution_id':identity,'workers':args.workers,'device':'cpu',
           'threads_per_worker':1,'tasks':[t['id'] for t in tasks],'start_utc':datetime.now(timezone.utc).isoformat()})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_task,t,identity) for t in tasks]):print(future.result(),flush=True)

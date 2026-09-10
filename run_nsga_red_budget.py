"""Rerun manuscript NSGA-II budgets with the authorized red-line parameters."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

from sample_params import params_from_json_data
from src import pareto_experiment as core
from src.reproducibility import deserialize_plan, file_sha256, plan_sha256
from run_nsga_parameter_probe import parameterized

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'output/pareto-ppo-budget-v11'
OUT=ROOT/'output/pareto-nsga-red-budget'
PAPER=ROOT/'供应链管理写作/数值实验与结果分析_论文稿.md'
SOURCES=['run_nsga_red_budget.py','run_nsga_parameter_probe.py','src/pareto_experiment.py',
         'src/operators.py','src/solution_utils.py','src/reproducibility.py','sample_params.py','hazardous_waste_model.py']


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)
def utc(): return datetime.now(timezone.utc).isoformat()


def verify():
    p=read(OUT/'protocol.json')
    for f,h in p['sources'].items(): assert file_sha256(ROOT/f)==h,f
    for f,h in p['preserved'].items(): assert file_sha256(ROOT/f)==h,f
    return p


def prepare():
    if (OUT/'protocol.json').exists(): return verify()
    assert file_sha256(ROOT/'src/pareto_experiment.py')=='4ea667b52c5bd01a7b7123cbb358602999c760ae84f7aebee20c4aa1f9f4ee5a'
    tasks=[]
    for scale in range(1,5):
        for repeat in range(3):
            old=read(BASE/f'fronts/NSGA-Test-{scale}-r{repeat}-B40320.json')
            tasks.append(dict(old['task']))
    tasks.sort(key=lambda t:(t['instance']!='Test-4',t['instance'],t['repeat']))
    preserved={}
    for folder in ('fronts','solutions','instances','milp'):
        for path in (BASE/folder).rglob('*'):
            if path.is_file(): preserved[path.relative_to(ROOT).as_posix()]=file_sha256(path)
    registry=read(ROOT/'configs/ppo_models_frontier_v9.json')
    for model in registry['models'].values():
        assert file_sha256(ROOT/model['checkpoint'])==model['sha256']
        preserved[model['checkpoint']]=model['sha256']
    backup=OUT/'before';backup.mkdir(parents=True,exist_ok=True)
    shutil.copy2(PAPER,backup/PAPER.name)
    p=dict(protocol='nsga-red-budget',created_utc=utc(),population=20,crossover=.1,mutation=1.,
           initial_service_probability=.5,threads_per_worker=1,workers=4,
           budgets=[40320,120960],tasks=tasks,ppo_reused=True,milp_reused=True,
           paper_before_sha256=file_sha256(PAPER),registry=registry,
           sources={s:file_sha256(ROOT/s) for s in SOURCES},preserved=preserved,
           selection_history='Test-4 adaptive low-budget spread exploration; parameters fixed before this rerun')
    write(OUT/'protocol.json',p)
    return p


def run_task(task):
    p=verify();task_id=task['id'];done=OUT/f'completed/{task_id}.json'
    if done.exists():
        for path,h in read(done)['files'].items(): assert file_sha256(OUT/path)==h
        return dict(task=task_id,status='REUSED')
    execution=OUT/f'execution/{task_id}.json'
    assert not execution.exists(),'Prior start exists; inspect before any retry'
    meta=dict(task=task,pid=os.getpid(),start_utc=utc(),status='running',threads=1,
              protocol_sha256=file_sha256(OUT/'protocol.json'))
    write(execution,meta)
    d=read(BASE/f"instances/{task['instance']}.json")
    params=params_from_json_data(d['params']);refs=(d['b_C'],d['b_R'])
    plan=deserialize_plan(d['reference_plan'])
    assert plan_sha256(plan)==d['reference_plan_sha256']
    files={};started=time.perf_counter()
    def checkpoint(result):
        archive_id=f"{task_id}-B{result['budget']}"
        points=[]
        for original in result['points']:
            point=dict(original);saved_plan=point.pop('plan')
            path=f"solutions/{d['instance_id']}/{point['solution_id']}.json"
            write(OUT/path,dict(plan=saved_plan,instance_id=d['instance_id'],instance_sha256=d['instance_sha256']))
            point['solution_path']=path;points.append(point)
        archive={**result,'points':points,'archive_id':archive_id,'task':task,
                 **{k:d[k] for k in ('instance_id','instance_sha256','b_C','b_R','reference_plan_sha256','search_initial_plan_sha256')},
                 'protocol_sha256':file_sha256(OUT/'protocol.json'),'strict_feasible_points':len(points),
                 'status':'success','checkpoint':None,'parameters':{k:p[k] for k in ('population','crossover','mutation','initial_service_probability')}}
        path=f'fronts/{archive_id}.json';write(OUT/path,archive);files[path]=file_sha256(OUT/path)
        write(OUT/f'progress/{task_id}.json',dict(budget=result['budget'],points=len(points),seconds=result['seconds'],utc=utc()))
        print(json.dumps(dict(task=task_id,budget=result['budget'],points=len(points),seconds=result['seconds'])),flush=True)
    try:
        with parameterized(.1,1.):
            core.nsga_fronts(params,refs,plan,task['seed'],tuple(task['budgets']),20,checkpoint)
        meta.update(status='completed',end_utc=utc(),task_seconds=time.perf_counter()-started)
        write(done,dict(task=task,files=files,protocol_sha256=meta['protocol_sha256']))
        meta['completed_sha256']=file_sha256(done);write(execution,meta)
        return dict(task=task_id,status='completed')
    except BaseException:
        meta.update(status='failed',end_utc=utc(),error=traceback.format_exc());write(execution,meta)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run'])
    a=parser.parse_args();p=prepare()
    if a.stage=='run':
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            futures=[pool.submit(run_task,t) for t in p['tasks']]
            for f in as_completed(futures):print(json.dumps(f.result()),flush=True)

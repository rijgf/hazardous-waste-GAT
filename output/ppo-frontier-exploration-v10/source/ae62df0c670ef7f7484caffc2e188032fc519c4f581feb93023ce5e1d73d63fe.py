"""Versioned, fixed-budget search exploration; never overwrites the v9 study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np
import run_ppo_frontier_v9 as base
from src.pareto_experiment import CandidateLedger, coverage_pair
from src.ppo_frontier_v9 import search_preference
from src.reproducibility import derive_seed, file_sha256, deserialize_plan

ROOT=base.ROOT
OUT=ROOT/'output/ppo-frontier-exploration-v10'
BACKUP=ROOT/'backups/ppo-v9-before-frontier-exploration-v10'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def prepare():
    protocol=base.load_protocol()
    if (OUT/'protocol.json').exists():
        check_protected();return
    files=set(base.SOURCES)
    files.update(['configs/ppo_models_frontier_v9.json','output/pareto-ppo-v9/protocol.json',
                  '供应链管理写作/数值实验与结果分析_论文稿.md',
                  'output/pareto-ppo-v9/report_verification.json',
                  'output/pareto-ppo-v9/audit/numerics-with-report.json'])
    files.update(m['checkpoint'] for m in protocol['registry']['models'].values())
    files.update(p.relative_to(ROOT).as_posix() for p in (base.OUT/'fronts').glob('*.json'))
    files.update(p.relative_to(ROOT).as_posix() for p in (base.OUT/'instances').glob('*.json'))
    protected={p:file_sha256(ROOT/p) for p in sorted(files)}
    BACKUP.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(BACKUP/'code-models-and-fronts.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(ROOT/p,p)
    save(BACKUP/'manifest.json',{'files':protected,'archive_sha256':file_sha256(BACKUP/'code-models-and-fronts.zip')})
    save(OUT/'protocol.json',{'version':'v10-fixed-budget-frontier-exploration','protected':protected,
         'instance':'Test-4','budget_per_preference':1920,'full_frontier_budget':40320,
         'screen_preferences':[0,10,20],'screen_repeat':0,
         'confirmation_preferences':list(range(21)),'confirmation_repeats':[0,1,2],
         'initial_hypotheses':['width 8 limits trajectory depth','sparse per-instance training preference coverage',
                               'temperature 2 overexplores late search','mask/progress/memory inefficiency'],
         'policy':'v9 frozen large model; PPO selects operator and all three objects',
         'restrictions':'no extra candidate budget, Transformer layers, heuristic objects, new instances, NSGA warm start',
         'design_test_informed':True,'formal_report_modified':False})
    check_protected()


def check_protected():
    base.load_protocol()
    for p,h in base.read(OUT/'protocol.json')['protected'].items():
        assert file_sha256(ROOT/p)==h, p


def task(args):
    width,temperature,repeat,index,label=args
    check_protected();protocol=base.load_protocol()
    target=OUT/'runs'/label/f'r{repeat}'/f'p{index}.json'
    if target.exists():return 'REUSED '+str(target.relative_to(OUT))
    config_path=target.with_suffix('.config.json')
    source_hash=file_sha256(Path(__file__))
    config={'width':width,'temperature':temperature,'repeat':repeat,'preference_index':index,
            'label':label,'runner_sha256':source_hash,'budget':1920,'horizon':1024,
            'protocol_sha256':file_sha256(OUT/'protocol.json'),'device':'cpu','threads':1}
    save(config_path,config)
    try:
        old=base.read(base.OUT/f'fronts/PPO-Test-4-large-r{repeat}-B40320.json')
        old_point=old['final_solutions'][index]
        seed=derive_seed(old['task']['seed'],'preference',index,bits=32)
        assert old_point['seed']==seed
        data=base.read(base.OUT/'instances/Test-4.json');params,reference,refs=base.checked_instance(data)
        model=base.load_model('large',protocol);ledger=CandidateLedger(params,refs,reference)
        log_path=target.with_suffix('.actions.jsonl')
        with log_path.open('x',encoding='utf8') as log:
            result=search_preference(params,refs,reference,model,tuple(protocol['preferences'][index]),seed,
                 ledger=ledger,budget=1920,width=width,temperature=temperature,horizon=1024,log_handle=log)
        points=ledger.archive.sorted_points()
        best_nsga={}
        for budget in (40320,120960):
            n=base.read(base.OUT/f'fronts/NSGA-Test-4-r{repeat}-B{budget}.json')
            w=result['preference']
            best_nsga[str(budget)]=min(w[0]*p['cost']/refs[0]+w[1]*p['risk']/refs[1] for p in n['points'])
        assert file_sha256(Path(__file__))==source_hash
        save(target,{'config':config,'result':result,'points':points,
             'action_log_path':log_path.relative_to(ROOT).as_posix(),'action_log_sha256':file_sha256(log_path),
             'baseline_J':old_point['metrics']['weighted_objective'],
             'baseline_transitions':old_point['search_counts'].get('accepted_transitions',0),
             'nsga_best_scalar_by_budget':best_nsga,'instance_sha256':data['instance_sha256'],
             'b_C':refs[0],'b_R':refs[1],'checkpoint':protocol['registry']['models']['large']})
        check_protected()
        return f"DONE {label} r{repeat} p{index}: J={result['metrics']['weighted_objective']:.8f} vs v9={old_point['metrics']['weighted_objective']:.8f}, transitions={result['search_counts'].get('accepted_transitions',0)}, time={result['seconds']:.1f}s"
    except Exception:
        save(target.with_suffix('.failure.json'),{'config':config,'traceback':traceback.format_exc()})
        raise


def summarize():
    for path in sorted((OUT/'runs').glob('*/*/p*.json')):
        if path.name.endswith(('.config.json','.failure.json')):continue
        d=base.read(path);r=d['result'];j=r['metrics']['weighted_objective']
        print(path.relative_to(OUT),f'J={j:.8f}',f'v9={d["baseline_J"]:.8f}',
              f'improvement={(d["baseline_J"]-j)/d["baseline_J"]:.2%}',
              f'NSGA1={d["nsga_best_scalar_by_budget"]["40320"]:.8f}',
              f'accepted={r["search_counts"].get("accepted_transitions",0)}/{d["baseline_transitions"]}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','summarize'])
    p.add_argument('--width',type=int,choices=[1,2,4,8],default=8);p.add_argument('--temperature',type=float,default=2.)
    p.add_argument('--indices',type=int,nargs='+',default=[0,10,20]);p.add_argument('--repeats',type=int,nargs='+',default=[0])
    p.add_argument('--label');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='summarize':summarize()
    else:
        if not a.label or any(i not in range(21) for i in a.indices) or any(r not in range(3) for r in a.repeats):
            p.error('Explicit label, preference indices 0..20, repeats 0..2 required')
        jobs=[(a.width,a.temperature,r,i,a.label) for r in a.repeats for i in a.indices]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(task,t) for t in jobs]):print(f.result(),flush=True)

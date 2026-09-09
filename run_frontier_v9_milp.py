"""Fresh 3600-second small-case MILPs; original solver and v4 case preserved."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import run_pareto_milp as exact
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan, file_sha256, json_sha256, plan_sha256
from src.solution_utils import evaluate_solution, route_plan_to_solution

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/pareto-ppo-v9/milp'
SOURCE_CASE=ROOT/'output/pareto-parameter-revision-v4/instances/small-fixed.json'


def configure():
    data=json.loads(SOURCE_CASE.read_text(encoding='utf8'))
    old=json.loads((SOURCE_CASE.parent.parent/'protocol.json').read_text(encoding='utf8'))
    assert file_sha256(SOURCE_CASE)==old['instances']['small-fixed']
    params=params_from_json_data(data['params']);plan=deserialize_plan(data['reference_plan'])
    assert json_sha256(data['params'])==data['instance_sha256']
    assert plan_sha256(plan)==data['reference_plan_sha256']
    refs=(data['b_C'],data['b_R'])
    metric=evaluate_solution(params,route_plan_to_solution(params,plan),(.5,.5),objective_refs=refs)
    assert metric['feasible'] and abs(metric['cost']-refs[0])<1e-9 and abs(metric['risk']-refs[1])<1e-9
    exact.INSTANCE_HASH=data['instance_sha256'];exact.REFERENCE_HASH=data['reference_plan_sha256']
    exact.REFS=refs;exact.DEFAULT_OUTPUT=OUT;exact.historical_case=lambda:(params,plan)
    exact.SOURCE_FILES=(*exact.SOURCE_FILES,'run_frontier_v9_milp.py')
    original=exact.protocol
    def protocol(limit):
        value=original(limit)
        value.update(schema='ppo-frontier-v9-fresh-milp',instance_id=data['instance_id'],
                     source_instance_path=str(SOURCE_CASE.relative_to(ROOT)).replace('\\','/'),
                     source_instance_sha256=file_sha256(SOURCE_CASE),
                     independent_of_ppo_and_nsga_budgets=True)
        return value
    exact.protocol=protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,choices=range(5));parser.add_argument('--precheck',action='store_true')
    args=parser.parse_args();configure()
    if args.precheck:
        print('PASS: same v4 small-fixed instance, reference, b and unmodified MILP formulation')
        return
    if args.index is not None:
        exact.run_one(args.index,OUT,3600.);return
    exact.bind_protocol(OUT,3600.)
    processes=[]
    for index in range(5):
        if (OUT/f'p{index}.json').exists():
            exact.run_one(index,OUT,3600.);continue
        with (OUT/f'p{index}.log').open('a',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--index',str(index)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        processes.append((index,child));print(f'Fresh MILP p{index}: PID {child.pid}',flush=True)
    codes=[(index,child.wait()) for index,child in processes]
    if any(code for _,code in codes):raise RuntimeError(str(codes))
    print('All five fresh MILP runs completed',flush=True)


if __name__=='__main__':main()

"""Isolated NSGA-II parameter exploration; historical source and archives stay intact."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import statistics

from sample_params import params_from_json_data
from src import pareto_experiment as core
from src.reproducibility import deserialize_plan, file_sha256, plan_sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/nsga-parameter-probe'
INSTANCE = ROOT / 'output/pareto-parameter-revision-v4/instances/Test-4.json'
SOURCE_HASH = '4ea667b52c5bd01a7b7123cbb358602999c760ae84f7aebee20c4aa1f9f4ee5a'
CONFIGS = {
    'baseline': (100, .9, .25),
    'mutation50': (100, .9, .5),
    'mutation100': (100, .9, 1.),
    'crossover50': (100, .5, .25),
    'population40': (40, .9, .25),
    'population200': (200, .9, .25),
    'combined40': (40, .9, 1.),
    'combined40cross50': (40, .5, 1.),
    'combined20': (20, .9, 1.),
}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def hv(points, refs):
    # Fixed before experiments: rectangle [0,1.1]^2 in instance-normalized objectives.
    previous, area = 1.1, 0.
    for point in sorted(points, key=lambda p: p['cost']):
        x, y = point['cost']/refs[0], point['risk']/refs[1]
        if x < 1.1 and y < previous:
            area += (1.1-x)*(previous-y)
            previous = y
    return area


@contextmanager
def parameterized(crossover, mutation):
    original = core.MultiPeriodEncoding
    counts = Counter()

    class Encoding(original):
        def child(self, left, right, rng):
            genes = super().child(left, right, rng, crossover=crossover, mutation=mutation)
            counts['children'] += 1
            counts['genome_identical_to_a_parent'] += int(genes == left or genes == right)
            return genes

    core.MultiPeriodEncoding = Encoding
    try:
        yield counts
    finally:
        core.MultiPeriodEncoding = original


def run(name, seed, budget):
    target = OUT / f'{name}-s{seed}-B{budget}.json'
    assert not target.exists(), f'Refuse overwrite: {target}'
    assert file_sha256(ROOT/'src/pareto_experiment.py') == SOURCE_HASH
    data = read(INSTANCE)
    params = params_from_json_data(data['params'])
    refs = (data['b_C'], data['b_R'])
    plan = deserialize_plan(data['reference_plan'])
    n, cross, mutation = CONFIGS[name]
    meta = dict(name=name, seed=seed, budget=budget, population=n, crossover=cross,
                mutation=mutation, instance_sha256=file_sha256(INSTANCE), refs=refs,
                core_sha256=SOURCE_HASH, driver_sha256=file_sha256(Path(__file__)),
                pid=os.getpid(), hv_reference=[1.1,1.1])
    write(target.with_suffix('.started.json'), meta)
    with parameterized(cross, mutation) as counters:
        result = core.nsga_fronts(params, refs, plan, seed, (budget,), n)[budget]
    points = result['points']
    assert result['counts']['candidate_attempts'] == budget
    assert all(p['feasible'] and plan_sha256(deserialize_plan(p['plan'])) == p['solution_id'] for p in points)
    summary = dict(hv=hv(points,refs), points=len(points),
                   min_cost=min(p['cost'] for p in points), max_cost=max(p['cost'] for p in points),
                   min_risk=min(p['risk'] for p in points), max_risk=max(p['risk'] for p in points),
                   best_J=min(.5*p['cost']/refs[0]+.5*p['risk']/refs[1] for p in points))
    write(target, dict(**meta, result=result, diagnostics=dict(counters), summary=summary))
    print(json.dumps(dict(name=name,seed=seed,budget=budget,seconds=result['seconds'],**summary),ensure_ascii=False),flush=True)


def audit():
    from audit_scalar_advantage_v5 import full_model_replay
    from src.solution_utils import evaluate_solution, route_plan_to_solution
    import math
    data = read(INSTANCE)
    params = params_from_json_data(data['params'])
    samples = {}
    paths = sorted(p for p in OUT.glob('*-B*.json') if '.started.' not in p.name)
    for path in paths:
        d = read(path)
        assert d['instance_sha256'] == file_sha256(INSTANCE)
        assert d['core_sha256'] == file_sha256(ROOT/'src/pareto_experiment.py')
        assert d['result']['counts']['candidate_attempts'] == d['budget']
        assert math.isclose(hv(d['result']['points'],d['refs']),d['summary']['hv'],abs_tol=1e-12)
        for p in d['result']['points']:
            plan = deserialize_plan(p['plan'])
            assert plan_sha256(plan) == p['solution_id']
            m = evaluate_solution(params,route_plan_to_solution(params,plan),(.5,.5),d['refs'])
            assert m['feasible'],m['violations']
            assert all(math.isclose(m[k],p[k],rel_tol=1e-11,abs_tol=1e-8) for k in ('cost','risk'))
            samples[p['solution_id']] = (plan,p)
    print('MILP replay',len(samples),'unique plans',flush=True)
    result = full_model_replay(params,list(samples.values()))
    write(OUT/'audit.json',dict(status='PASS',runs=len(paths),unique_plans=len(samples),milp=result,
                              hashes={p.name:file_sha256(p) for p in paths}))
    print('PASS',len(paths),'runs',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name',choices=[*CONFIGS,'audit'])
    parser.add_argument('--seed',type=int,default=20260910)
    parser.add_argument('--budget',type=int,default=2000)
    args = parser.parse_args()
    if args.name == 'audit':
        audit()
    else:
        run(args.name,args.seed,args.budget)

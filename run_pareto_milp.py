"""Rebuild the historical small case and retain auditable 3600-second MILPs.

The existing MILP formulation and decoder are used unchanged.  SciPy's complete
termination evidence and the solver vector are retained in addition to decoded
raw variables, since a summary cannot establish feasibility or optimality.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from hazardous_waste_model import HazardousWasteMILP
from sample_params import params_to_json_data
from src.heuristics import build_greedy_initial_plan
from src.instance_generator import generate_random_params
from src.reproducibility import (environment_snapshot, file_sha256, json_sha256,
                                 plan_sha256, plan_to_canonical_data)
from src.solution_utils import evaluate_solution, route_plan_to_solution
from src.supplementary_experiment import atomic_write_json, serialize_solution

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'output' / 'pareto-single-instance-v3' / 'milp'
PREFERENCES = ((1., 0.), (.75, .25), (.5, .5), (.25, .75), (0., 1.))
INSTANCE_HASH = 'c61c8fc4c2d43afb69fb76d26a69ef7a246a77ff11b451396b302849988a803c'
REFERENCE_HASH = '5fdfb1130823c21856db79670ab76cbbbac336b68ac2548cdf8c3b81d1a791f3'
REFS = (1384.991951, 10.876517565143931)
SOURCE_FILES = ('run_pareto_milp.py', 'hazardous_waste_model.py', 'sample_params.py',
                'configs/model_config.json', 'src/heuristics.py',
                'src/instance_generator.py', 'src/reproducibility.py',
                'src/solution_utils.py', 'src/supplementary_experiment.py')


def historical_case():
    config = json.loads((ROOT / 'configs/model_config.json').read_text(encoding='utf-8'))
    params = generate_random_params(config['small'], 3723524230)
    reference = build_greedy_initial_plan(params, seed=3949079119)
    metrics = evaluate_solution(params, route_plan_to_solution(params, reference),
                                (.5, .5), objective_refs=REFS)
    if json_sha256(params_to_json_data(params)) != INSTANCE_HASH:
        raise ValueError('Historical instance hash mismatch')
    if plan_sha256(reference) != REFERENCE_HASH:
        raise ValueError('Historical reference hash mismatch')
    if not metrics['feasible'] or (metrics['cost'], metrics['risk']) != REFS:
        raise ValueError('Historical reference feasibility or b mismatch')
    return params, reference


def protocol(time_limit):
    return {'schema': 'pareto-historical-milp-v1', 'instance_sha256': INSTANCE_HASH,
            'reference_sha256': REFERENCE_HASH, 'instance_seed': 3723524230,
            'reference_seed': 3949079119, 'objective_refs': list(REFS),
            'preferences': [list(p) for p in PREFERENCES],
            'solver_options': {'time_limit': time_limit, 'mip_rel_gap': 0.,
                               'threads': 1, 'random_seed': 0, 'disp': True},
            'sources': {name: file_sha256(ROOT / name) for name in SOURCE_FILES},
            'time_definition': 'Model build, MILP solve, decode and strict validation; excludes shared case preparation and file serialization.',
            'parallel_jobs': 5}


def bind_protocol(output, time_limit):
    params, reference = historical_case()
    expected = protocol(time_limit)
    path = output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Existing MILP protocol differs; refuse stale resume or overwrite')
    else:
        atomic_write_json(path, expected)
        atomic_write_json(output / 'case.json', {
            'params': params_to_json_data(params),
            'reference_plan': plan_to_canonical_data(reference),
            'objective_refs': list(REFS), 'instance_sha256': INSTANCE_HASH,
            'reference_sha256': REFERENCE_HASH})
        atomic_write_json(output / 'environment.json', environment_snapshot())
    return expected, params


def finite_or_none(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def run_one(index, output, time_limit):
    locked, params = bind_protocol(output, time_limit)
    destination = output / f'p{index}.json'
    if destination.exists():
        record = json.loads(destination.read_text(encoding='utf-8'))
        if record['protocol_sha256'] != json_sha256(locked):
            raise ValueError('Completed MILP has a different protocol')
        print(f'p{index}: retained completed result', flush=True)
        return
    preference = PREFERENCES[index]
    optimizer_params = replace(params, cost_weight=preference[0] / REFS[0],
                               risk_weight=preference[1] / REFS[1])
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    model = HazardousWasteMILP(optimizer_params)
    model._build()
    matrix = lil_matrix((len(model.rows), len(model.names)), dtype=float)
    for row, coefficients in enumerate(model.rows):
        for col, value in coefficients.items():
            matrix[row, col] = value
    constraint = LinearConstraint(matrix.tocsr(), np.array(model.row_lbs), np.array(model.row_ubs))
    build_seconds = time.perf_counter() - started
    solver_started = time.perf_counter()
    result = milp(c=np.array(model.costs, dtype=float),
                  integrality=np.array(model.integrality, dtype=int),
                  bounds=Bounds(np.array(model.lbs), np.array(model.ubs)),
                  constraints=constraint, options=locked['solver_options'].copy())
    solver_seconds = time.perf_counter() - solver_started
    solution = model._decode(result.x) if result.x is not None else None
    metrics = evaluate_solution(params, solution, preference, objective_refs=REFS) if solution else None
    feasible = bool(metrics and metrics['feasible'])
    objective_consistent = bool(metrics and math.isclose(metrics['weighted_objective'],
                                                         float(result.fun), abs_tol=1e-7, rel_tol=1e-7))
    proven_optimal = bool(result.success and result.status == 0 and feasible and objective_consistent)
    elapsed = time.perf_counter() - started
    record = {'schema': 'pareto-historical-milp-result-v1',
              'protocol_sha256': json_sha256(locked), 'preference_index': index,
              'preference': list(preference), 'objective_refs': list(REFS),
              'instance_sha256': INSTANCE_HASH, 'reference_sha256': REFERENCE_HASH,
              'started_at_utc': started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'optimal' if proven_optimal else f'solver_status_{result.status}',
              'strict_feasible': feasible, 'objective_consistent': objective_consistent,
              'proven_optimal': proven_optimal, 'metrics': metrics,
              'elapsed_seconds': elapsed, 'build_seconds': build_seconds,
              'solver_seconds': solver_seconds,
              'solver': {'status': int(result.status), 'success': bool(result.success),
                         'message': str(result.message), 'fun': finite_or_none(result.fun),
                         'mip_gap': finite_or_none(getattr(result, 'mip_gap', None)),
                         'mip_dual_bound': finite_or_none(getattr(result, 'mip_dual_bound', None)),
                         'mip_node_count': finite_or_none(getattr(result, 'mip_node_count', None))},
              'model_dimensions': {'variables': len(model.names), 'constraints': len(model.rows)},
              'diagnostics': model.diagnostics,
              'solution': serialize_solution(solution) if solution else None,
              'full_solver_vector': result.x.tolist() if result.x is not None else None}
    if protocol(time_limit) != locked:
        raise ValueError('MILP sources changed during solve; refusing untraceable result')
    atomic_write_json(destination, record)
    print(json.dumps({key: record[key] for key in ('preference_index', 'status',
                     'strict_feasible', 'proven_optimal', 'elapsed_seconds')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--index', type=int, choices=range(5))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--time-limit', type=float, default=3600.)
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('time limit must be positive and finite')
    args.output = args.output.resolve()
    if args.time_limit != 3600. and args.output == DEFAULT_OUTPUT.resolve():
        parser.error('Pilot runs require a separate --output directory')
    if args.precheck:
        historical_case()
        print('PASS: exact historical instance, reference plan, strict feasibility, and b')
        return
    if args.index is not None:
        run_one(args.index, args.output, args.time_limit)
        return
    bind_protocol(args.output, args.time_limit)
    processes = []
    for index in range(5):
        if (args.output / f'p{index}.json').exists():
            run_one(index, args.output, args.time_limit)
            continue
        log_path = args.output / f'p{index}.log'
        with log_path.open('a', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--index', str(index), '--output', str(args.output),
                                      '--time-limit', str(args.time_limit)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        processes.append((index, child))
        print(f'p{index}: started, PID {child.pid}, log {log_path.relative_to(ROOT)}', flush=True)
    failures = [(index, child.wait()) for index, child in processes]
    if any(code for _, code in failures):
        raise RuntimeError(f'MILP child return codes: {failures}')
    print('All five MILP result records complete', flush=True)


if __name__ == '__main__':
    main()

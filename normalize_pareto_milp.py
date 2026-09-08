"""Validate tolerance-normalized MILP vectors without any additional solving.

Original result JSONs are immutable. This separate handoff boundary reconciles
HiGHS integer feasibility tolerance with the common route evaluator's exact
presence predicates. All five preferences use the same declared rule.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np

from hazardous_waste_model import HazardousWasteMILP
from run_pareto_milp import historical_case
from sample_params import params_to_json_data
from src.reproducibility import file_sha256, json_sha256
from src.solution_utils import evaluate_solution
from src.supplementary_experiment import atomic_write_json, serialize_solution

ROOT=Path(__file__).resolve().parent
DIRECTORY=ROOT/'output/pareto-single-instance-v3/milp'
FEASIBILITY_TOLERANCE=1e-5
OBJECTIVE_TOLERANCE=1e-7


def normalize_record(record, params):
    started=time.perf_counter()
    if json_sha256(params_to_json_data(params))!=record['instance_sha256']:
        raise ValueError('Cannot normalize a vector for a different instance')
    preference=tuple(record['preference'])
    refs=tuple(record['objective_refs'])
    optimizer_params=replace(params,cost_weight=preference[0]/refs[0],risk_weight=preference[1]/refs[1])
    model=HazardousWasteMILP(optimizer_params)
    model._build()
    original=np.asarray(record['full_solver_vector'],dtype=float)
    if len(original)!=len(model.names) or not np.isfinite(original).all():
        raise ValueError('Invalid original complete vector')
    solver_objective=record['solver']['fun']
    original_linear=float(np.dot(model.costs,original))
    if not math.isclose(original_linear,solver_objective,rel_tol=OBJECTIVE_TOLERANCE,abs_tol=OBJECTIVE_TOLERANCE):
        raise ValueError('Original vector does not reproduce the solver objective')
    cleaned=original.copy()
    integer_mask=np.asarray(model.integrality,dtype=bool)
    integer_distance=np.abs(cleaned[integer_mask]-np.round(cleaned[integer_mask]))
    if np.max(integer_distance,initial=0)>FEASIBILITY_TOLERANCE:
        raise ValueError('Integer value outside the declared cleanup tolerance')
    cleaned[integer_mask]=np.round(cleaned[integer_mask])
    continuous_small=(~integer_mask)&(np.abs(cleaned)<FEASIBILITY_TOLERANCE)
    cleaned[continuous_small]=0.
    activities=np.array([sum(coefficient*cleaned[index] for index,coefficient in row.items()) for row in model.rows])
    matrix_residual=float(max(0.,np.max(np.asarray(model.row_lbs)-activities),np.max(activities-np.asarray(model.row_ubs))))
    bound_residual=float(max(0.,np.max(np.asarray(model.lbs)-cleaned),np.max(cleaned-np.asarray(model.ubs))))
    integer_residual=float(np.max(np.abs(cleaned[integer_mask]-np.round(cleaned[integer_mask])),initial=0))
    if max(matrix_residual,bound_residual,integer_residual)>FEASIBILITY_TOLERANCE:
        raise ValueError('Normalized vector violates the original complete model')
    solution=model._decode(cleaned)
    metrics=evaluate_solution(params,solution,preference,objective_refs=refs)
    cleaned_linear=float(np.dot(model.costs,cleaned))
    consistent=all(math.isclose(metrics['weighted_objective'],value,rel_tol=OBJECTIVE_TOLERANCE,abs_tol=OBJECTIVE_TOLERANCE)
                   for value in (solver_objective,cleaned_linear))
    if not metrics['feasible'] or not consistent:
        raise ValueError('Normalized common objective/strict feasibility mismatch')
    changes=[{'index':index,'name':list(model.names[index]),'before':float(before),'after':float(after)}
             for index,(before,after) in enumerate(zip(original,cleaned)) if before!=after]
    return {**record,'schema':'pareto-tolerance-normalized-milp-v1',
            'original_objective_consistent':record['objective_consistent'],
            'original_metrics':record['metrics'],'original_full_solver_vector':record['full_solver_vector'],
            'full_solver_vector':cleaned.tolist(),'full_cleaned_solver_vector':cleaned.tolist(),
            'solution':serialize_solution(solution),'metrics':metrics,'strict_feasible':True,
            'objective_consistent':consistent,'proven_optimal':record['proven_optimal'],
            'cleanup_rule':'Round all integer variables within 1e-5 of an integer; set continuous abs(value)<1e-5 to zero; do not change other continuous entries.',
            'feasibility_tolerance':FEASIBILITY_TOLERANCE,'objective_tolerance':OBJECTIVE_TOLERANCE,
            'changed_variables':changes,'max_matrix_residual':matrix_residual,
            'max_bound_residual':bound_residual,'max_integer_residual':integer_residual,
            'original_linear_objective':original_linear,'cleaned_linear_objective':cleaned_linear,
            'solver_objective_difference':metrics['weighted_objective']-solver_objective,
            'cleanup_seconds':time.perf_counter()-started,'additional_solver_calls':0}


def main():
    params,_=historical_case()
    protocol=json.loads((DIRECTORY/'protocol.json').read_text(encoding='utf8'))
    for name,expected in protocol['sources'].items():
        if file_sha256(ROOT/name)!=expected:
            raise RuntimeError('Original MILP computation source changed')
    for index in range(5):
        original_path=DIRECTORY/f'p{index}.json'
        record=json.loads(original_path.read_text(encoding='utf8'))
        original_hash=file_sha256(original_path)
        cleaned=normalize_record(record,params)
        cleaned.update(original_result_path=str(original_path.relative_to(ROOT)).replace('\\','/'),
                       original_result_sha256=original_hash,
                       cleanup_source_sha256=file_sha256(Path(__file__)),
                       original_protocol_file_sha256=file_sha256(DIRECTORY/'protocol.json'))
        output=DIRECTORY/f'validated/p{index}.json'
        if output.exists():
            prior=json.loads(output.read_text(encoding='utf8'))
            if prior['original_result_sha256']!=original_hash or prior['cleanup_source_sha256']!=cleaned['cleanup_source_sha256']:
                raise RuntimeError('Refuse to overwrite a changed normalization lineage')
        else:
            atomic_write_json(output,cleaned)
        if file_sha256(original_path)!=original_hash:
            raise RuntimeError('Original solver record changed during normalization')
        print(index,'PASS',metrics_text(cleaned),flush=True)


def metrics_text(record):
    return {'J':record['metrics']['weighted_objective'],'objective_difference':record['solver_objective_difference'],
            'matrix_residual':record['max_matrix_residual'],'changed_variables':len(record['changed_variables'])}


if __name__=='__main__':
    main()

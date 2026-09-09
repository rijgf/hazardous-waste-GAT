"""Regression for late-stage wasted actions in the fixed-budget PPO search."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan
from src.ppo_objects_v8 import ObjectState
from src.ppo_objects_v7b import ObjectState as PreviousState
from src.ppo_objects_v7 import Action
from src.solution_utils import route_plan_to_solution, evaluate_solution

ROOT=Path(__file__).resolve().parents[1]


def case():
    data=json.loads((ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json').read_text(encoding='utf8'))
    final=json.loads((ROOT/'output/ppo-objects-v7/evaluate/v7b_r0_temperature2_B12000/result.json').read_text(encoding='utf8'))
    return params_from_json_data(data['params']),deserialize_plan(final['plan']),(data['b_C'],data['b_R'])


def test_known_terminal_processing_dead_end_is_masked():
    p,plan,refs=case();state=ObjectState(p,refs)
    assert state.pickups[50]==('n::G13::S3',1)
    assert not state.masks(plan,(0,))[50], 'Slot 50 certainly leaves terminal facility inventory: D2,S3'


def test_drop_mask_agrees_with_independent_full_evaluation():
    p,plan,refs=case()
    data=json.loads((ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json').read_text(encoding='utf8'))
    for plan in [plan,deserialize_plan(data['reference_plan'])]:
        before=PreviousState(p,refs);after=ObjectState(p,refs)
        mask=after.masks(plan,(0,))
        for slot in np.flatnonzero(before.masks(plan,(0,))):
            candidate,status=before.apply(plan,Action(0,int(slot)))
            assert status=='candidate'
            metric=evaluate_solution(p,route_plan_to_solution(p,candidate),(.5,.5),refs)
            assert bool(mask[slot])==metric['feasible'], (slot,metric['violations'])


def test_mask_only_change_preserves_trained_encoding():
    p,plan,refs=case()
    for old,new in zip(PreviousState(p,refs).encode(plan,(.5,.5)),ObjectState(p,refs).encode(plan,(.5,.5))):
        np.testing.assert_array_equal(old,new)


def test_feasibility_is_independent_of_objective_coefficients():
    p,plan,refs=case();original=ObjectState(p,refs).masks(plan,(0,))
    changed=replace(p,distance_cost=p.distance_cost*10,
                    coload_risk={key:value*10 for key,value in p.coload_risk.items()})
    np.testing.assert_array_equal(original,ObjectState(changed,refs).masks(plan,(0,)))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))

"""The new delivery audit must detect errors the shared checker can miss."""
from dataclasses import replace
import unittest

from audit_scalar_advantage_v5 import full_model_replay, compare_source_semantics
from run_scalar_advantage_v5 import BASE, read
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan
from src.solution_utils import route_plan_to_solution, evaluate_solution


class ScalarAdvantageAuditTests(unittest.TestCase):
    def setUp(self):
        data = read(BASE/'instances/Test-1.json')
        self.params = params_from_json_data(data['params'])
        self.plan = deserialize_plan(data['reference_plan'])
        self.solution = route_plan_to_solution(self.params,self.plan)
        self.point = evaluate_solution(self.params,self.solution)

    def test_preserved_source_diff_is_only_the_authorized_actor_change(self):
        compare_source_semantics()

    def test_complete_milp_accepts_reference_and_matches_separate_objectives(self):
        result=full_model_replay(self.params,[(self.plan,self.point)])
        self.assertEqual(result['violated_rows'],0)
        self.assertLess(result['max_cost_difference'],1e-8)
        self.assertLess(result['max_risk_difference'],1e-8)

    def test_complete_milp_rejects_before_processing_inventory_overflow(self):
        params,raw=self.params,self.solution['raw']
        totals={j:max(sum(raw.get(('BD',j,s,t),0.) for s in params.waste_types) for t in params.periods)
                for j in params.facilities}
        facility=max(totals,key=totals.get)
        altered=replace(params,facility_capacity={**params.facility_capacity,facility:.99*totals[facility]})
        with self.assertRaises(AssertionError):
            full_model_replay(altered,[(self.plan,self.point)])

    def test_complete_milp_rejects_falsified_front_objective(self):
        with self.assertRaises(AssertionError):
            full_model_replay(self.params,[(self.plan,{**self.point,'risk':self.point['risk']*.5})])

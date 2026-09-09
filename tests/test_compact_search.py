import json
from pathlib import Path
from dataclasses import replace
import random
import unittest

from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan
from src.solution_utils import route_plan_to_solution, evaluate_solution
from src.compact_search import canonical_routes, propose, FAMILIES


class CompactSearchTests(unittest.TestCase):
    def setUp(self):
        d=json.loads(Path('output/pareto-parameter-revision-v4/instances/Test-1.json').read_text(encoding='utf8'))
        self.p=params_from_json_data(d['params'])
        self.plan=deserialize_plan(d['reference_plan'])

    def test_homogeneous_vehicle_renaming_has_identical_canonical_plan(self):
        p=self.p
        mapping=dict(zip(p.vehicles,reversed(p.vehicles)))
        renamed={(mapping[k],t):list(r) for (k,t),r in self.plan.items()}
        self.assertEqual(canonical_routes(p,self.plan),canonical_routes(p,renamed))
        a=evaluate_solution(p,route_plan_to_solution(p,self.plan))
        b=evaluate_solution(p,route_plan_to_solution(p,renamed))
        self.assertAlmostEqual(a['cost'],b['cost'])
        self.assertAlmostEqual(a['risk'],b['risk'])

    def test_drop_is_simple_optional_service_removal(self):
        candidate=propose(self.p,self.plan,'drop',random.Random(7))
        self.assertIsNotNone(candidate)
        old=sum(len(r)-2 for r in self.plan.values())
        self.assertEqual(sum(len(r)-2 for r in candidate.values()),old-1)
        terminal=lambda plan:{n for (k,t),r in plan.items() if t==self.p.periods[-1] for n in r[1:-1]}
        self.assertEqual(terminal(candidate),terminal(self.plan))

    def test_no_more_than_six_simple_families(self):
        self.assertLessEqual(len(FAMILIES),6)

    def test_before_processing_capacity_is_enforced(self):
        solution=route_plan_to_solution(self.p,self.plan)
        raw=solution['raw']
        totals={j:max(sum(raw['BD',j,s,t] for s in self.p.waste_types) for t in self.p.periods) for j in self.p.facilities}
        j=max(totals,key=totals.get)
        p=replace(self.p,facility_capacity={**self.p.facility_capacity,j:.99*totals[j]})
        self.assertFalse(evaluate_solution(p,solution)['feasible'])


if __name__=='__main__':unittest.main()

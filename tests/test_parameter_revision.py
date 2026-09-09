import json
from dataclasses import replace
from pathlib import Path
import unittest

from src.instance_generator import generate_random_params
from src.parameter_revision import construct_reference, revised_parameters
from src.solution_utils import evaluate_solution, route_plan_to_solution
from sample_params import params_to_json_data


class ParameterRevisionTests(unittest.TestCase):
    def setUp(self):
        config = json.loads(Path('configs/model_config.json').read_text(encoding='utf8'))['small']
        base = generate_random_params(config, 123)
        self.params = replace(base,
            generation={k: 1. for k in base.generation},
            technology={('D1', 'S1'): 1, ('D2', 'S1'): 0, ('D1', 'S2'): 1, ('D2', 'S2'): 1},
            waste_consequence={'S1': 2., 'S2': 4.})

    def test_capacity_uses_period_average_and_only_eligible_facilities(self):
        revised, _ = revised_parameters(self.params, seed=42)
        # Three producers each generate one unit per type each period. The
        # system gets six units per period, not six per facility or per horizon.
        for period in [1, 2]:
            self.assertEqual(revised.processing_capacity['D1', 'S1', period], 6.)
            self.assertEqual(revised.processing_capacity['D1', 'S2', period], 3.)
            self.assertEqual(revised.processing_capacity['D2', 'S2', period], 3.)
            self.assertEqual(sum(revised.processing_capacity[j, 'S1', period] * revised.technology[j, 'S1']
                                 for j in revised.facilities), 6.)

    def test_risk_is_seeded_pair_random_not_exact_maximum_calibration(self):
        a, metadata = revised_parameters(self.params, seed=42)
        b, _ = revised_parameters(self.params, seed=42)
        c, _ = revised_parameters(self.params, seed=43)
        self.assertEqual(a.coload_risk, b.coload_risk)
        self.assertNotEqual(a.coload_risk, c.coload_risk)
        gamma = a.coload_risk['S1', 'S2']
        # Unit equal loads with consequences 2 and 4: ordinary risk is 6;
        # the existing evaluator adds twice gamma on that same exposure.
        ratio = (6. + 2. * gamma) / 6.
        self.assertGreaterEqual(ratio, 3.)
        self.assertLessEqual(ratio, 6.)
        self.assertNotEqual(ratio, 5.)
        self.assertEqual(a.coload_risk['S1', 'S1'], 0.)
        self.assertAlmostEqual(metadata['pairs'][0]['equal_load_total_transport_risk_ratio'], ratio)

    def test_only_two_parameter_fields_change(self):
        revised, _ = revised_parameters(self.params, seed=42)
        before, after = params_to_json_data(self.params), params_to_json_data(revised)
        self.assertEqual({key for key in before if before[key] != after[key]},
                         {'processing_capacity', 'coload_risk'})

    def test_reference_is_strictly_feasible_with_every_period_serviced(self):
        revised, _ = revised_parameters(self.params, seed=42)
        plan, evidence = construct_reference(revised)
        metrics = evaluate_solution(revised, route_plan_to_solution(revised, plan))
        self.assertTrue(metrics['feasible'], metrics['violations'])
        for period in revised.periods:
            visits = [n for (v, t), route in plan.items() if t == period for n in route[1:-1]]
            self.assertCountEqual(visits, revised.pickup_nodes)
        self.assertGreaterEqual(evidence['seconds'], 0.)


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import unittest

from normalize_pareto_milp import normalize_record
from run_pareto_milp import historical_case


class MILPNormalizationTests(unittest.TestCase):
    def test_near_zero_binary_arc_does_not_become_a_full_cost_arc(self):
        record=json.loads(Path('output/pareto-single-instance-v3/milp/p3.json').read_text(encoding='utf8'))
        params,_=historical_case()
        self.assertFalse(record['objective_consistent'])
        self.assertGreater(record['metrics']['weighted_objective']-record['solver']['fun'],.01)
        normalized=normalize_record(record,params)
        self.assertTrue(normalized['objective_consistent'])
        self.assertTrue(normalized['strict_feasible'])
        self.assertFalse(normalized['original_objective_consistent'])
        self.assertAlmostEqual(normalized['metrics']['cost'],1909.944,places=3)
        self.assertLessEqual(normalized['max_matrix_residual'],1e-5)
        self.assertFalse(record['objective_consistent'])

    def test_near_zero_second_waste_flow_does_not_activate_coload_risk(self):
        record=json.loads(Path('output/pareto-single-instance-v3/milp/p4.json').read_text(encoding='utf8'))
        params,_=historical_case()
        self.assertFalse(record['objective_consistent'])
        self.assertGreater(record['metrics']['weighted_objective']-record['solver']['fun'],.006)
        normalized=normalize_record(record,params)
        tiny_flows=[change for change in normalized['changed_variables']
                    if change['name'][0]=='F' and 0<abs(change['before'])<1e-5 and change['after']==0]
        self.assertTrue(tiny_flows)
        self.assertGreater(record['metrics']['coload_risk']-normalized['metrics']['coload_risk'],.01)
        self.assertTrue(normalized['objective_consistent'])
        self.assertTrue(normalized['strict_feasible'])
        self.assertLessEqual(abs(normalized['solver_objective_difference']),1e-7)
        self.assertLessEqual(normalized['max_matrix_residual'],1e-5)
        self.assertFalse(record['objective_consistent'])


if __name__=='__main__':
    unittest.main()

import unittest
from sample_params import params_from_json_data
from run_nsga_red_budget import BASE,read
from run_nsga_parameter_probe import parameterized
from src.pareto_experiment import nsga_fronts
from src.reproducibility import deserialize_plan


class BudgetTests(unittest.TestCase):
    def test_prefix_budget_is_not_restarted(self):
        d=read(BASE/'instances/Test-1.json');params=params_from_json_data(d['params'])
        refs=(d['b_C'],d['b_R']);plan=deserialize_plan(d['reference_plan'])
        with parameterized(.1,1.):
            both=nsga_fronts(params,refs,plan,123,(40,60),20)
            first=nsga_fronts(params,refs,plan,123,(40,),20)
        self.assertEqual(both[40]['counts'],first[40]['counts'])
        self.assertEqual([p['solution_id'] for p in both[40]['points']],[p['solution_id'] for p in first[40]['points']])
        self.assertEqual(both[40]['counts']['candidate_attempts'],40)
        self.assertEqual(both[60]['counts']['candidate_attempts'],60)
        self.assertGreater(both[60]['seconds'],both[40]['seconds'])

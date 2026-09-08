"""Report-level contracts for the user-authorized parameter revision."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

import report_pareto_experiments as report
from sample_params import build_sample_params


def revised_fixture():
    original = build_sample_params()
    params = replace(original,
        processing_capacity={key: 5.5 if key[1] == 'S1' else 4. for key in original.processing_capacity},
        coload_risk={**original.coload_risk, ('S1', 'S2'): 3.75})
    instances = {name: params for name in ['Test-1', 'Test-2', 'Test-3', 'Test-4', 'small-fixed']}
    for field, name in [('processing_capacity', 'capacity'), ('coload_risk', 'coload')]:
        for multiplier in [.7, .8, 1.2, 1.3]:
            instances[f'{name}-{round(multiplier * 100)}'] = replace(params,
                **{field: {key: value * multiplier for key, value in getattr(params, field).items()}})
    return {'parameter_revision': {'capacity_factor': 2., 'risk_total_ratio_range': [3., 6.],
        'risk_strength_range': [2., 5.], 'models_retrained': True,
        'training_hyperparameters_changed': False, 'small_case_revised': True}}, instances


class ParameterRevisionReportTests(unittest.TestCase):
    def test_risk_endpoint_description_contains_all_four_additive_components(self):
        point={'minimum_risk':10.,'transport_risk_at_minimum_risk':1.,
               'coload_risk_at_minimum_risk':2.,'producer_inventory_risk_at_minimum_risk':3.,
               'facility_inventory_risk_at_minimum_risk':4.}
        self.assertEqual(report.risk_component_description(point),
            '运输、共载、产废端库存、设施库存风险分别为1.0000、2.0000、3.0000、4.0000')
        with self.assertRaises(ValueError):
            report.risk_component_description({**point,'minimum_risk':9.})

    def test_capacity_description_uses_five_actual_endpoints_and_baseline_coload_share(self):
        scenes={name:{'minimum_cost':cost,'minimum_risk':risk,'coload_risk_at_minimum_risk':12.}
                for name,cost,risk in [('capacity-70',80.,50.),('capacity-80',90.,40.),
                    ('baseline',100.,30.),('capacity-120',110.,20.),('capacity-130',120.,10.)]}
        text=report.capacity_endpoint_interpretation(scenes)
        self.assertIn('80.000、90.000、100.000、110.000、120.000',text)
        self.assertIn('50.0000、40.0000、30.0000、20.0000、10.0000',text)
        self.assertIn('40.00%',text)
        self.assertIn('+66.67%和−66.67%',text)
        self.assertIn('通常不属于同一方案',text)
        self.assertIn('不是精确最优响应或因果效应',text)

    def test_complete_large_nsga_coverage_is_stated_directly_not_as_hypothetical_ppo_superiority(self):
        rows=[{'instance':'Test-4','model':'large','repeat':repeat,'nsga_budget':budget,
               'status':'valid','N_covers_P':1.,'P_covers_N':0.}
              for repeat in range(3) for budget in (40320,120960)]
        text=report.large_coverage_interpretation(rows)
        self.assertIn('NSGA-II档案均覆盖对应PPO档案的全部点',text)
        self.assertIn('PPO均未覆盖NSGA-II档案中的任何点',text)
        self.assertIn('不证明NSGA-II已获得真实完整帕累托前沿',text)
        self.assertNotIn('若PPO',text)
        rows[0]['N_covers_P']=.9
        self.assertNotIn('NSGA-II档案均覆盖对应PPO档案的全部点',report.large_coverage_interpretation(rows))

    def test_nsga_diagnostics_distinguish_public_reference_from_successful_search(self):
        records=[]
        for repeat, invalid, points in [(0,10,['ref']),(1,8,['ref','new']),(2,0,['new','other'])]:
            records.append({'archive_id':f'NSGA-Test-4-r{repeat}-B10','instance_id':'case',
                'task':{'kind':'NSGA','repeat':repeat,'instance':'Test-4'},'budget':10,'reference_plan_sha256':'ref',
                'points':[{'solution_id':identity} for identity in points],
                'counts':{'candidate_attempts':10,'invalid_candidate_attempts':invalid,
                    'candidate_objective_evaluations':10,'invalid_candidates':invalid,'operator_failures':invalid}})
        rows=report.nsga_search_diagnostics(records, {'Test-4':{'instance_id':'case','reference_plan_sha256':'ref'}})
        self.assertEqual(len(rows),1)
        row=rows[0]
        self.assertEqual(row['candidate_attempts'],30)
        self.assertEqual(row['invalid_candidate_attempts'],18)
        self.assertEqual(row['invalid_attempt_rate'],.6)
        self.assertEqual(row['feasible_candidate_attempts'],12)
        self.assertEqual(row['reference_only_run_count'],1)
        self.assertEqual(row['reference_only_archive_ids'],['NSGA-Test-4-r0-B10'])
        self.assertEqual(row['nonreference_front_point_occurrences'],3)
        self.assertEqual(row['nonreference_unique_front_solutions'],2)

    def test_zero_b2_coverage_alone_does_not_claim_a_budget_improvement(self):
        rows=[{'instance':'Test-4','model':'large','repeat':repeat,'nsga_budget':budget,
               'status':'valid','P_covers_N':0.,'N_covers_P':.5}
              for repeat in range(3) for budget in (40320,120960)]
        text=report.budget_response_interpretation(rows)
        self.assertIn('未观察到追加预算的覆盖收益',text)
        changed=[{**row,'N_covers_P':.75} if row['nsga_budget']==120960 else row for row in rows]
        self.assertIn('至少一项严格改善',report.budget_response_interpretation(changed))
        mixed=[{**row,'N_covers_P':.25} if row['repeat']==0 and row['nsga_budget']==120960 else row for row in changed]
        self.assertIn('不能概括为一致改善',report.budget_response_interpretation(mixed))

    def test_report_validates_new_capacity_and_random_risk_without_old_redundancy_claim(self):
        protocol, instances = revised_fixture()
        evidence = report.parameter_revision_evidence(protocol, instances)
        self.assertEqual(evidence['capacity_by_instance']['Test-4'][0]['average_period_generation'], 2.75)
        self.assertEqual(evidence['capacity_by_instance']['Test-4'][0]['effective_capacity_each_period'], [5.5, 5.5])
        self.assertEqual(evidence['risk_by_instance']['Test-4']['observed_total_ratio_range'], [4., 4.])
        self.assertEqual(evidence['capacity_scenario_factors'], [.7, .8, 1., 1.2, 1.3])
        self.assertIsNone(report.parameter_revision_evidence({}, {}))

    def test_report_rejects_wrong_effective_capacity_risk_and_sensitivity(self):
        for violation in ['capacity','risk','multiplier','extra_field']:
            with self.subTest(violation=violation):
                protocol, instances = revised_fixture()
                params = instances['Test-4']
                if violation == 'capacity':
                    instances['Test-4'] = replace(params, processing_capacity={key: 20. for key in params.processing_capacity})
                elif violation == 'risk':
                    instances['Test-4'] = replace(params, coload_risk={**params.coload_risk, ('S1','S2'): .2})
                elif violation == 'multiplier':
                    instances['capacity-70'] = instances['capacity-80']
                else:
                    instances['capacity-70'] = replace(instances['capacity-70'], distance_cost=4.)
                with self.assertRaises(ValueError):
                    report.parameter_revision_evidence(protocol, instances)

    def test_revision_entry_routes_all_report_outputs_without_building_or_training(self):
        code = '''
import run_pareto_experiments as engine
old = engine.OUT
import report_parameter_revision as entry
assert engine.OUT == old
entry.configure()
import report_pareto_experiments as report
import plot_pareto_sensitivity as plot
assert engine.OUT == report.OUT == plot.OUT
assert report.OUT.name == 'pareto-parameter-revision-v4'
assert report.ENTRY_SCRIPT.name == 'report_parameter_revision.py'
assert 'run_parameter_revision.py' in engine.SOURCES
assert 'src/parameter_revision.py' in engine.SOURCES
'''
        result = subprocess.run([sys.executable, '-B', '-c', code],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_solver_status_disclosure_does_not_call_every_nonoptimal_result_time_limited(self):
        optimal = {'proven_optimal': True, 'strict_feasible': True,
                   'solver': {'status': 0, 'message': 'Optimal'}}
        time_limit = {'proven_optimal': False, 'strict_feasible': True,
                      'solver': {'status': 1, 'message': 'Time limit reached'}}
        other = {'proven_optimal': False, 'strict_feasible': True,
                 'solver': {'status': 4, 'message': 'Other stopping reason'}}
        self.assertEqual(report.milp_status_label(optimal), '已证最优')
        self.assertEqual(report.milp_status_label(time_limit), '限时可行（未证最优）')
        self.assertEqual(report.milp_status_label(other), '可行（未证最优；状态4）')


if __name__ == '__main__':
    unittest.main()

"""Reporting contracts: dynamic NSGA counts and honest incomplete PPO budgets."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import report_ppo_frontier_v9 as report
from plot_ppo_frontier_v9 import visual_curve


def protocol_fixture(b1=1000,b2=3000):
    return {'tasks':[
        {'kind':'NSGA','instance':name,'budgets':[b1,b2] if name=='Test-4' else [b1]}
        for name in ('Test-1','Test-2','Test-3','Test-4')]+[
        {'kind':'PPO','instance':'Test-4','budget':40320}]}


def preference_fixture(i,actual=1920):
    return {'preference':[i/20,1-i/20],'candidate_budget':1920,
        'actual_candidate_attempts':actual,'unused_candidate_budget':1920-actual,
        'stop_reason':'candidate_budget_reached' if actual==1920 else 'positive_probability_action_space_exhausted',
        'ledger_counts':{'candidate_attempts':actual},'batches':240,'seconds':1.,
        'final_validation_evaluations':1,'preference_initialization_evaluations':1}


def normalized_milp_fixture(source_optimal=False):
    solver={'status':0,'success':True,'message':'Optimal','fun':.3,'mip_gap':0.,'mip_dual_bound':.3}
    original={'solver':solver,'proven_optimal':source_optimal,'objective_consistent':source_optimal}
    normalized={'solver':copy.deepcopy(solver),'proven_optimal':source_optimal,
        'original_objective_consistent':source_optimal,'objective_consistent':True,
        'strict_feasible':True,'metrics':{'weighted_objective':.3-1e-8,'feasible':True},
        'feasibility_tolerance':1e-5,'objective_tolerance':1e-7,
        'max_matrix_residual':7.1e-6,'max_bound_residual':0.,'max_integer_residual':0.,
        'original_linear_objective':.3,'cleaned_linear_objective':.3-1e-8,
        'solver_objective_difference':-1e-8,'additional_solver_calls':0}
    return original,normalized


class FrontierV9ReportTests(unittest.TestCase):
    def test_import_does_not_target_formal_paper(self):
        self.assertEqual(report.PAPER.parent,report.OUT)
        self.assertNotEqual(report.PAPER,report.FORMAL_PAPER)

    def test_partial_b1_and_complete_b2_coverage_has_explicit_current_conclusion(self):
        rows=[{'instance':'Test-4','model':'large','repeat':repeat,'nsga_budget':budget,
               'status':'valid','N_covers_P':.7781 if budget==40320 else 1.,'P_covers_N':0.}
              for repeat in range(3) for budget in (40320,120960)]
        with patch.object(report,'B1',40320),patch.object(report,'B2',120960):
            text=report.large_coverage_interpretation(rows)
        self.assertIn('仍落后于NSGA-II',text)
        self.assertIn('22.2%',text)
        self.assertIn('反向覆盖仍为0',text)
        self.assertIn('每次12000候选',text)
        self.assertIn('每个偏好1920候选',text)
        self.assertIn('尚不能将差距定量归因于其中某一项',text)

    def test_nsga_budget_is_dynamic_and_distinct_from_ppo(self):
        for pair in ((1000,3000),(999,4321),(40320,120960)):
            self.assertEqual(report.budget_levels(protocol_fixture(*pair)),pair)
        bad=protocol_fixture();bad['tasks'][0]['budgets']=[1001]
        with self.assertRaises(ValueError):report.budget_levels(bad)
        bad=protocol_fixture();bad['tasks'][-1]['budget']=12000*21
        with self.assertRaises(ValueError):report.budget_levels(bad)

    def test_paper_iterations_not_labeled_equivalent_candidate_evaluations(self):
        text=report.budget_reference_text(protocol_fixture(1500,4500))
        self.assertIn('iterations（迭代次数）',text)
        self.assertIn('B1=1500、B2=4500',text)
        self.assertIn('不能宣称',text)

    def test_confirmed_nsga_reuse_is_not_reported_as_new_search(self):
        protocol=protocol_fixture(40320,120960)
        protocol['nsga_reuse']={'source_root':'output/pareto-parameter-revision-v4'}
        text=report.budget_reference_text(protocol)
        self.assertIn('B1=40320、B2=120960',text)
        self.assertIn('用户最终确认预算不变',text)
        self.assertIn('复用v4原12次演化的15份预算档案',text)
        self.assertIn('不将它们写成新运行',text)

    def test_different_concurrency_is_disclosed_without_fair_speed_claim(self):
        execution={'workers':'3','nsga_workers':16}
        training={'small':{'training_seconds':1.25},'large':{'training_seconds':9.75}}
        text=report.timing_description(execution,training)
        self.assertIn('单入口workers配置为3',text)
        self.assertIn('不直接等于全局同时求解任务数',text)
        self.assertIn('原批次并发数为16',text)
        self.assertIn('不能直接用时间比值声称公平加速倍数',text)
        self.assertIn('不因本轮复制档案或离线回放而追加时间',text)
        self.assertIn('1.250 s和9.750 s',text)

    def test_parallel_entries_are_not_mistaken_for_global_concurrency(self):
        execution={'workers':'1/3','orchestration':{'initial_worker_limit':3,
            'declared_concurrent_task_upper_bound':15,'verified_peak_concurrent_tasks':15}}
        text=report.concurrency_description(execution)
        self.assertIn('先采用3任务并发',text)
        self.assertIn('上界为15',text)
        self.assertIn('实际共同求解峰值为15个任务',text)
        self.assertIn('不是全局并发数',text)
        execution['orchestration']['verified_peak_concurrent_tasks']=None
        text=report.concurrency_description(execution)
        self.assertIn('不将调度上界冒充实测峰值',text)
        self.assertNotIn('实际共同求解峰值为15',text)

    def test_missing_execution_metadata_is_disclosed_not_backfilled(self):
        execution={'orchestration':{'source_path':'orchestration/manifest.json',
            'execution_links_path':'orchestration/execution_links.json',
            'metadata_warnings':[{'task_id':'PPO-coload-130-large-r2'}]}}
        text=report.orchestration_provenance_text(execution,'..')
        self.assertIn('有1份辅助入口原始execution记录缺失',text)
        self.assertIn('PPO-coload-130-large-r2',text)
        self.assertIn('也没有补造该文件',text)
        self.assertIn('主入口预排任务清单不被当作辅助入口实际启动的证据',text)
        self.assertIn('方案、种子、预算和动作完整性仍须通过独立核验',text)

    def test_legitimate_early_stop_keeps_actual_count_without_padding(self):
        record={'final_solutions':[preference_fixture(i,1123 if i==10 else 1920) for i in range(21)],
            'budget':40320,'archive_id':'PPO-example','task':{'instance':'Test-4','model':'large','repeat':0},
            'counts':{'candidate_attempts':20*1920+1123}}
        rows=report.verify_preference_budget(record)
        self.assertEqual(rows[10]['actual_candidate_attempts'],1123)
        self.assertEqual(rows[10]['unused_candidate_budget'],797)
        self.assertEqual(sum(r['actual_candidate_attempts'] for r in rows),39523)
        wrong=copy.deepcopy(record);wrong['counts']['candidate_attempts']=40320
        with self.assertRaises(ValueError):report.verify_preference_budget(wrong)
        wrong=copy.deepcopy(record);wrong['final_solutions'][10]['stop_reason']='unknown_error'
        with self.assertRaises(ValueError):report.verify_preference_budget(wrong)

    def test_incomplete_or_duplicate_preference_grid_is_rejected(self):
        record={'final_solutions':[preference_fixture(i) for i in range(21)],
            'budget':40320,'archive_id':'PPO-example','task':{'instance':'Test-4','model':'large','repeat':0},
            'counts':{'candidate_attempts':40320}}
        record['final_solutions'][1]['preference']=[0.,1.]
        with self.assertRaises(ValueError):report.verify_preference_budget(record)

    def test_singleton_sensitivity_does_not_invent_a_second_point(self):
        knots,curve=visual_curve([(3.,2.)])
        self.assertEqual(knots.tolist(),[[3.,2.]])
        self.assertEqual(curve.tolist(),[[3.,2.]])
        knots,curve=visual_curve([(1.,4.),(2.,3.),(4.,1.)])
        self.assertEqual(knots.tolist(),[[1.,4.],[2.,3.],[4.,1.]])
        self.assertEqual(curve[0].tolist(),[1.,4.])
        self.assertEqual(curve[-1].tolist(),[4.,1.])

    def test_normalized_handoff_can_reconcile_false_source_flag_without_mutating_it(self):
        original,normalized=normalized_milp_fixture()
        before=copy.deepcopy((original,normalized))
        verdict=report.milp_optimality_verdict(original,normalized)
        self.assertTrue(verdict['report_proven_optimal'])
        self.assertTrue(verdict['reconciled_source_flag'])
        self.assertFalse(verdict['source_flags']['raw_proven_optimal'])
        self.assertFalse(verdict['source_flags']['normalized_proven_optimal'])
        self.assertEqual(verdict['failed_checks'],[])
        self.assertEqual((original,normalized),before)
        self.assertEqual(report.milp_status_label(normalized,verdict),'已证最优（规范化核验）')

    def test_handoff_requires_solver_proof_not_elapsed_time_or_close_objective(self):
        variants=[('status',1),('success',False),('mip_gap',1e-5),('mip_gap',None),
                  ('mip_dual_bound',.299),('fun',float('nan'))]
        for key,value in variants:
            with self.subTest(key=key,value=value):
                original,normalized=normalized_milp_fixture()
                original['solver'][key]=value;normalized['solver'][key]=value
                original['elapsed_seconds']=normalized['elapsed_seconds']=1.
                verdict=report.milp_optimality_verdict(original,normalized)
                self.assertFalse(verdict['report_proven_optimal'])
                self.assertNotIn('已证最优',report.milp_status_label(normalized,verdict))

    def test_handoff_rejects_failed_matrix_bounds_integrality_or_untrusted_tolerances(self):
        variants=[('max_matrix_residual',1.1e-5),('max_bound_residual',1.1e-5),
            ('max_integer_residual',1.1e-5),('max_matrix_residual',float('nan')),
            ('objective_consistent',False),('strict_feasible',False),
            ('feasibility_tolerance',.1),('objective_tolerance',.1),
            ('original_linear_objective',.31),('cleaned_linear_objective',.31),
            ('solver_objective_difference',1e-4),('additional_solver_calls',1)]
        for key,value in variants:
            with self.subTest(key=key,value=value):
                original,normalized=normalized_milp_fixture(source_optimal=True)
                normalized[key]=value
                verdict=report.milp_optimality_verdict(original,normalized)
                self.assertFalse(verdict['report_proven_optimal'])
                # A source True flag cannot bypass the stricter report verdict.
                self.assertNotIn('已证最优',report.milp_status_label(normalized,verdict))

    def test_actual_v9_p3_is_optimal_handoff_not_time_limited(self):
        directory=report.OUT/'milp'
        original=report.read(directory/'p3.json')
        normalized=report.read(directory/'validated/p3.json')
        verdict=report.milp_optimality_verdict(original,normalized)
        self.assertFalse(original['proven_optimal'])
        self.assertFalse(normalized['proven_optimal'])
        self.assertTrue(verdict['report_proven_optimal'])
        self.assertTrue(verdict['reconciled_source_flag'])
        self.assertEqual(verdict['failed_checks'],[])
        self.assertLess(original['elapsed_seconds'],3600.)
        self.assertNotIn('限时',report.milp_status_label(normalized,verdict))

    def test_all_optimal_table_omits_unused_dagger_footnote(self):
        text=report.milp_reference_note(0)
        self.assertNotIn('†',text)
        self.assertIn('五组MILP参照均满足',text)
        self.assertIn('没有限时可行参照',text)
        self.assertIn('†',report.milp_reference_note(1))
        self.assertIn('不是最优性差距',report.milp_reference_note(1))


if __name__=='__main__':unittest.main()

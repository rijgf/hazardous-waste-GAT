"""Frozen V9 inference budget/report adapter regression contracts."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import run_ppo_budget_v11 as run
import report_ppo_budget_v11 as report
import audit_ppo_budget_v11 as audit


class BudgetV11Tests(unittest.TestCase):
    def test_frozen_contract_and_unchanged_models(self):
        p=run.load_protocol()
        old=run.read(run.SOURCE/'protocol.json')
        self.assertFalse(p['models_retrained_this_round'])
        self.assertEqual(p['registry'],old['registry'])
        self.assertEqual(p['instances'],old['instances'])
        self.assertEqual(p['preferences'],old['preferences'])
        self.assertEqual(p['small_seeds'],old['small_seeds'])
        self.assertEqual(report.budget_levels(p),(40320,120960))
        self.assertEqual(p['ppo_candidate_budget_per_preference'],12000)
        self.assertEqual(p['ppo_frontier_budget'],252000)
        for current,previous in zip(p['tasks'],old['tasks']):
            normalized=copy.deepcopy(current)
            if current['kind']=='PPO':
                self.assertEqual(current['budget'],252000)
                normalized['budget']=previous['budget']
            self.assertEqual(normalized,previous)

    def test_staged_report_is_default_and_budget_history_explicit(self):
        self.assertEqual(report.PAPER.parent,run.OUT)
        self.assertNotEqual(report.PAPER,report.FORMAL_PAPER)
        text=report.budget_reference_text(run.load_protocol())
        self.assertIn('每偏好12000、完整前沿252000',text)
        self.assertIn('15份NSGA-II预算档案',text)
        self.assertIn('不将它们写成新运行',text)
        self.assertNotIn('用户最终确认预算不变',text)

    def test_small_log_complete_gzip_and_false_cap_rejected(self):
        record=run.read(run.OUT/'small/ppo-p0-r0.json')
        checked=audit.check_log(record)
        self.assertEqual(checked['cap'],12000)
        self.assertLess(checked['attempts'],12000)
        bad=copy.deepcopy(record);bad['candidate_budget']=1920
        with self.assertRaises(AssertionError):audit.check_log(bad)

    def test_budget_row_does_not_inflate_exhausted_work(self):
        record=run.read(run.OUT/'small/ppo-p0-r0.json')
        task={'instance':'small-fixed','repeat':0,'model':'small'}
        row=report.preference_budget_row(record,'small-p0-r0',task,0)
        self.assertEqual(row['actual_candidate_attempts'],record['actual_candidate_attempts'])
        self.assertEqual(row['unused_candidate_budget'],12000-row['actual_candidate_attempts'])
        bad=copy.deepcopy(record);bad['stop_reason']='candidate_budget_reached'
        with self.assertRaises(ValueError):report.preference_budget_row(bad,'bad',task,0)

    def test_old_training_and_milp_times_not_claimed_new(self):
        text=report.timing_description({'workers':'16','nsga_workers':16},
            {'small':{'training_seconds':1.},'large':{'training_seconds':2.}})
        self.assertIn('历史训练耗时',text)
        self.assertIn('本轮没有重新运行MILP',text)
        self.assertIn('不能直接用时间比值声称公平加速倍数',text)

    def test_no_truncated_template_or_image_formulas(self):
        source=Path(report.__file__).read_text(encoding='utf8')
        self.assertNotIn('tokens truncated',source)
        self.assertNotIn('Warning: truncated output',source)
        self.assertIn('公式采用可直接编辑的LaTeX数学块',source)

    def test_actual_preflight_schema_and_checkpoint(self):
        p=run.load_protocol()
        self.assertTrue(report.verify_preflight(p)['not_formal_result'])
        bad=copy.deepcopy(p)
        bad['registry']['models']['large']['sha256']='bad'
        with self.assertRaises(AssertionError):report.verify_preflight(bad)

    def test_small_metadata_each_field_is_checked(self):
        p=run.load_protocol();data=run.read(run.OUT/'instances/small-fixed.json')
        record=run.read(run.OUT/'small/ppo-p0-r0.json')
        audit.check_small_metadata(record,data,p,0,0)
        for field in ('instance_id','instance_sha256','b_C','b_R','reference_plan_sha256','repeat','seed'):
            bad=copy.deepcopy(record);bad[field]='wrong'
            with self.assertRaises(AssertionError,msg=field):audit.check_small_metadata(bad,data,p,0,0)

    def test_front_seed_is_derived_from_original_task(self):
        p=run.load_protocol();task=next(t for t in p['tasks'] if t['kind']=='PPO')
        point={'seed':audit.derive_seed(task['seed'],'preference',0,bits=32)}
        audit.check_front_seed(point,task,0)
        with self.assertRaises(AssertionError):audit.check_front_seed(point,task,1)


if __name__=='__main__':unittest.main()

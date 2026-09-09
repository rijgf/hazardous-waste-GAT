import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_ppo_frontier_v9 as runner


class NSGAReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.old=root/'old';self.out=root/'new'
        self.old.mkdir();self.out.mkdir()
        for name,value in [('ROOT',root),('OLD',self.old),('OUT',self.out)]:
            p=patch.object(runner,name,value);p.start();self.addCleanup(p.stop)
        self.task={'id':'NSGA-Test-4-r0','kind':'NSGA','instance':'Test-4',
                   'repeat':0,'budgets':[40320,120960],'seed':55}
        self.data={'instance_id':'one','instance_sha256':'instance-digest','b_C':10.,'b_R':20.,
                   'reference_plan_sha256':'reference-digest'}
        for directory in [self.old,self.out]:runner.write(directory/'instances/Test-4.json',self.data)
        self.instances={'Test-4':runner.file_sha256(self.old/'instances/Test-4.json')}
        # The historical v4 schema stores budgets in tasks, not nsga.budgets.
        old={'tasks':[self.task],'instances':self.instances,'nsga':{'population':100}}
        runner.write(self.old/'protocol.json',old)
        old_hash=runner.file_sha256(self.old/'protocol.json')
        runner.write(self.old/'execution.json',{'protocol_sha256':old_hash,'workers':16})
        runner.write(self.old/'completed/NSGA-Test-4-r0.json',
                     {'task':self.task,'protocol_sha256':old_hash,'seconds_total_including_load_and_export':12.})
        runner.write(self.old/'solutions/one/sol.json',
                     {**{k:self.data[k] for k in ['instance_id','instance_sha256','b_C','b_R']},
                      'solution_id':'sol','plan':[]})
        for budget in self.task['budgets']:
            runner.write(self.old/f'fronts/{self.task["id"]}-B{budget}.json',
                         {**self.data,'task':self.task,'budget':budget,'protocol_sha256':old_hash,
                          'search_initial_plan_sha256':self.data['reference_plan_sha256'],
                          'seconds':12.,'points':[{'solution_id':'sol','solution_path':'solutions/one/sol.json'}]})
        self.old_protocol=old
        self.protocol={'tasks':[self.task],'instances':self.instances,
                       'nsga_reuse':runner.nsga_reuse_manifest(old,[self.task],self.instances,[40320,120960])}
        runner.write(self.out/'protocol.json',self.protocol)

    def test_copy_preserves_bytes_and_original_protocol(self):
        self.assertEqual(runner.import_reused_nsga(self.protocol),1)
        self.assertEqual(runner.import_reused_nsga(self.protocol),1)
        for relative in self.protocol['nsga_reuse']['fronts']:
            self.assertEqual((self.old/relative).read_bytes(),(self.out/relative).read_bytes())
            self.assertTrue(runner.verify_reused_nsga(self.protocol,runner.read(self.out/relative),relative))
        complete=runner.read(self.out/'completed/NSGA-Test-4-r0.json')
        self.assertFalse(complete['solver_rerun'])
        self.assertEqual(complete['historical_seconds_total_including_load_and_export'],12.)

    def test_changed_budget_or_task_cannot_reuse(self):
        with self.assertRaises(ValueError):
            runner.nsga_reuse_manifest(self.old_protocol,[self.task],self.instances,[1000,3000])
        with self.assertRaises(ValueError):
            runner.nsga_reuse_manifest(self.old_protocol,[{**self.task,'seed':99}],self.instances,[40320,120960])

    def test_tampered_solution_and_front_are_rejected(self):
        runner.import_reused_nsga(self.protocol)
        relative=next(iter(self.protocol['nsga_reuse']['fronts']))
        front=runner.read(self.out/relative)
        runner.write(self.out/'solutions/one/sol.json',{'fake':True})
        with self.assertRaises(ValueError):runner.verify_reused_nsga(self.protocol,front,relative)
        with self.assertRaises(ValueError):runner.import_reused_nsga(self.protocol)

    def test_reuse_run_task_never_calls_optimizer(self):
        with patch.object(runner,'load_protocol',return_value=self.protocol),patch.object(runner,'nsga_fronts') as solve:
            self.assertIn('HISTORICAL',runner.run_task(self.task))
            self.assertIn('HISTORICAL',runner.run_task(self.task))
            solve.assert_not_called()

    def test_reference_and_source_execution_changes_rejected(self):
        runner.import_reused_nsga(self.protocol)
        runner.write(self.old/'execution.json',{'changed':True})
        relative=next(iter(self.protocol['nsga_reuse']['fronts']))
        with self.assertRaises(ValueError):
            runner.verify_reused_nsga(self.protocol,runner.read(self.out/relative),relative)


if __name__=='__main__':unittest.main()

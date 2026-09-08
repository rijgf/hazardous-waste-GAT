import unittest

from src.pareto_experiment import ParetoArchive, coverage_pair, MultiPeriodEncoding, nsga_fronts, rank_and_crowding, Individual, CandidateLedger
from pathlib import Path
import json
import random
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan, plan_sha256, file_sha256


class ParetoTests(unittest.TestCase):
    def test_weak_coverage_and_duplicate_archive(self):
        archive = ParetoArchive((10, 10))
        archive.add({'cost': 1., 'risk': 2., 'solution_id': 'a'})
        archive.add({'cost': 2., 'risk': 1., 'solution_id': 'b'})
        archive.add({'cost': 1., 'risk': 2., 'solution_id': 'c'})
        archive.add({'cost': 3., 'risk': 3., 'solution_id': 'd'})
        self.assertEqual(len(archive.points), 2)
        result = coverage_pair(archive.points, archive.points, (10, 10))
        self.assertEqual((result['A_covers_B'], result['B_covers_A']), (1., 1.))
        dominated = [{'cost': 5., 'risk': 5., 'solution_id': 'e'}]
        result = coverage_pair(archive.points, dominated, (10, 10))
        self.assertEqual((result['A_covers_B'], result['B_covers_A']), (1., 0.))
        self.assertIsNone(coverage_pair([], archive.points, (10, 10))['A_covers_B'])

    def test_equal_representative_replacement_removes_newly_dominated_points(self):
        archive = ParetoArchive((1,1))
        for label,c,r in [('b',10,10),('c',10+2e-8,10-1.5e-8),('a',10+.9e-8,10-.9e-8)]:
            archive.add({'solution_id':label,'cost':c,'risk':r})
        self.assertEqual([p['solution_id'] for p in archive.sorted_points()], ['a'])

    def test_nsga_budget_and_multiperiod_expression(self):
        data = json.loads(Path('datasets/reference_generalization_v2/test/Test-1/000.json').read_text(encoding='utf8'))
        params = params_from_json_data(data['params'])
        plan = deserialize_plan(data['reference_plan'])
        refs = (data['b_C'], data['b_R'])
        encoding = MultiPeriodEncoding(params)
        rebuilt, ok = encoding.decode(encoding.encode(plan))
        self.assertTrue(ok)
        self.assertEqual(plan_sha256(rebuilt), plan_sha256(plan))
        genome = [(1, 0, 0, index / len(encoding.slots)) for index, _ in enumerate(encoding.slots)]
        rebuilt, ok = encoding.decode(genome)
        self.assertTrue(ok)
        self.assertEqual(sum(len(route)-2 for route in rebuilt.values()), len(params.pickup_nodes)*len(params.periods))
        results = nsga_fronts(params, refs, plan, 123, budgets=(13, 27), population_size=10)
        self.assertEqual(results[13]['counts']['candidate_attempts'], 13)
        self.assertEqual(results[27]['counts']['candidate_attempts'], 27)
        self.assertTrue(all(p['feasible'] and 'plan' in p for p in results[27]['points']))

    def test_all_four_genome_components_can_change_feasible_decisions(self):
        data = json.loads(Path('datasets/reference_generalization_v2/test/Test-1/000.json').read_text(encoding='utf8'))
        params = params_from_json_data(data['params'])
        plan = deserialize_plan(data['reference_plan'])
        encoding = MultiPeriodEncoding(params)
        original = encoding.encode(plan)
        original_hash = plan_sha256(plan)
        observed = set()
        for component in range(4):
            for index in range(len(original)):
                genes = list(original)
                changed = list(genes[index])
                if component == 0:
                    if encoding.slots[index][1] == params.periods[-1]:
                        continue
                    changed[0] = 1-changed[0]
                elif component == 1:
                    changed[1] = (changed[1]+1)%len(params.vehicles)
                elif component == 2:
                    changed[2] = (changed[2]+1)%len(params.facilities)
                else:
                    changed[3] = -1
                genes[index] = tuple(changed)
                result, ok = encoding.decode(genes)
                if ok and plan_sha256(result) != original_hash:
                    observed.add(component)
                    break
        self.assertEqual(observed, {0,1,2,3})

    def test_nsga_sort_front_and_crowding(self):
        points = [Individual([], {'cost': c, 'risk': r, 'feasible': True}) for c, r in [(1, 4), (2, 2), (4, 1), (5, 5)]]
        rank_and_crowding(points, (1, 1))
        self.assertEqual([p.rank for p in points], [0, 0, 0, 1])
        self.assertEqual(points[1].crowding, 2.)
        self.assertEqual(points[0].crowding, float('inf'))

    def test_observer_keeps_frozen_search_identical(self):
        import torch
        from src.ppo_improver import PPOImprover
        torch.set_num_threads(1)
        data = json.loads(Path('datasets/reference_generalization_v2/test/Test-1/000.json').read_text(encoding='utf8'))
        params, plan = params_from_json_data(data['params']), deserialize_plan(data['reference_plan'])
        refs = (data['b_C'], data['b_R'])
        registry = json.loads(Path('configs/frozen_ppo_models.json').read_text(encoding='utf8'))
        record = registry['models']['small']
        self.assertEqual(file_sha256(record['checkpoint']),record['sha256'])
        model = PPOImprover.from_frozen_checkpoint(params, record['checkpoint'], objective_refs=refs)
        model.policy.to('cpu')
        model.device = torch.device('cpu')
        before = model.improve((.5, .5), steps=2, seed=123, initial_plan=plan)
        ledger = CandidateLedger(params, refs, plan)
        with ledger.observe_ppo((.5, .5), 123):
            after = model.improve((.5, .5), steps=2, seed=123, initial_plan=plan)
        self.assertEqual(plan_sha256(before.plan), plan_sha256(after.plan))
        self.assertEqual(before.trace, after.trace)
        self.assertEqual(ledger.counts['candidate_attempts'], 64)
        self.assertTrue(all(p['feasible'] for p in ledger.archive.points))


if __name__ == '__main__':
    unittest.main()

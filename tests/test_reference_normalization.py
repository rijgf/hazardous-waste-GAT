from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from sample_params import build_sample_params
from src.heuristics import build_greedy_initial_plan
from src.ppo_improver import PPOImprover, PPOTrainingInstance, StateEncoder
from src.solution_utils import evaluate_solution, route_plan_to_solution
from test_frozen_ppo import _algorithm_config, _network_config


class ReferenceNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.params = build_sample_params()
        self.plan = build_greedy_initial_plan(self.params)
        self.metrics = evaluate_solution(self.params, route_plan_to_solution(self.params, self.plan), (.5,.5))
        self.refs = (self.metrics["cost"], self.metrics["risk"])
        self.network = {**_network_config(), "objective_normalization": "instance_reference"}

    def test_reference_input_and_fixed_denominator(self):
        refs = (self.refs[0] * 2, self.refs[1] * 4)
        enc = StateEncoder(self.params, self.network, refs)
        for pref in [(1.,0.),(.5,.5),(0.,1.)]:
            tokens, mask, global_vec = enc.encode(self.plan, pref, 0, 0)
            self.assertAlmostEqual(float(tokens[0,3]), .5)
            self.assertAlmostEqual(float(tokens[0,4]), .25)
            self.assertAlmostEqual(float(tokens[0,5]), pref[0]*.5+pref[1]*.25)
            torch.testing.assert_close(tokens[0], global_vec)
            self.assertEqual(enc.objective_refs, refs)
        with self.assertRaises(ValueError): StateEncoder(self.params, self.network)
        for bad in [(0,1),(-1,2),(1,float('nan'))]:
            with self.assertRaises(ValueError): StateEncoder(self.params, self.network, bad)

    def test_search_does_not_reset_refs_to_starting_solution(self):
        algo = _algorithm_config()
        refs = (self.refs[0]*2, self.refs[1]*4)
        model = PPOImprover(self.params, algo, self.network, seed=91, objective_refs=refs)
        result = model.improve((.5,.5), steps=1, seed=5, initial_plan=self.plan)
        self.assertEqual(model.encoder.objective_refs, refs)
        for row in result.trace:
            if "incumbent_objective_before" in row:
                self.assertAlmostEqual(row["incumbent_objective_before"], .375)

    def test_multi_instance_training_and_checkpoint_semantics(self):
        other = replace(self.params, vehicle_fixed_cost=self.params.vehicle_fixed_cost*2)
        other_plan = build_greedy_initial_plan(other)
        met = evaluate_solution(other, route_plan_to_solution(other, other_plan), (.5,.5))
        samples = [PPOTrainingInstance("a", self.params, self.plan, self.refs),
                   PPOTrainingInstance("b", other, other_plan, (met["cost"],met["risk"]))]
        algo = _algorithm_config()
        algo["ppo"].update(train_iterations=2,num_parallel_episodes=2,episode_steps=2)
        model = PPOImprover(self.params, algo, self.network, seed=93, objective_refs=self.refs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            history = model.train(path, training_instances=samples)
            self.assertEqual(model.training_instance_counts, {"a":2,"b":2})
            self.assertEqual(len(history["mean_best_objective"]),2)
            loaded = PPOImprover.from_frozen_checkpoint(other,path,objective_refs=samples[1].objective_refs)
            self.assertEqual(loaded.encoder.objective_refs,samples[1].objective_refs)
            with self.assertRaisesRegex(ValueError,"objective_normalization"):
                PPOImprover.from_frozen_checkpoint(other,path,network_config=_network_config())
            with self.assertRaises(ValueError): PPOImprover.from_frozen_checkpoint(other,path)

    def test_padding_trim_preserves_policy_output(self):
        model = PPOImprover(self.params,_algorithm_config(),self.network,objective_refs=self.refs)
        model.policy.train()
        tok, mask, glob = model.encoder.encode(self.plan,(.5,.5),0,0)
        n = int(mask.sum())
        with torch.no_grad():
            full = model.policy(tok[None].to(model.device),mask[None].to(model.device),glob[None].to(model.device))
            trimmed = model.policy(tok[None,:n].to(model.device),mask[None,:n].to(model.device),glob[None].to(model.device))
        for x,y in zip([*full[0],full[1]],[*trimmed[0],trimmed[1]]):
            torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)


if __name__ == '__main__': unittest.main()

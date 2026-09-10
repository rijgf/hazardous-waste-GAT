"""Regression: fixed preference weights must precede advantage whitening."""
import inspect
import unittest
import torch
from src.ppo_improver import PPOImprover


class ScalarAdvantageTests(unittest.TestCase):
    def test_cost_risk_tradeoff_is_not_reweighted_by_component_variance(self):
        vector = torch.tensor([[1., -2.], [-1., 2.], [.1, 20.], [-.1, -20.]])
        preferences = torch.full_like(vector, .5)
        weighted = (vector * preferences).sum(-1)
        expected = (weighted - weighted.mean()) / weighted.std().clamp_min(1e-6)
        actual = PPOImprover._scalarize_advantages(vector, preferences)
        torch.testing.assert_close(actual, expected)
        self.assertLess(float(actual[0]), 0.)

    def test_each_sample_uses_its_own_preference_before_global_normalization(self):
        vector = torch.tensor([[1., 30.], [10., -1.], [-4., 2.], [3., -6.]])
        preferences = torch.tensor([[1., 0.], [0., 1.], [.25, .75], [.8, .2]])
        weighted = (vector * preferences).sum(-1)
        actual = PPOImprover._scalarize_advantages(vector, preferences)
        torch.testing.assert_close(actual, (weighted-weighted.mean())/weighted.std())

    def test_constant_batch_is_finite_and_zero(self):
        actual = PPOImprover._scalarize_advantages(torch.ones(4, 2), torch.full((4, 2), .5))
        torch.testing.assert_close(actual, torch.zeros(4))

    def test_actual_training_calls_corrected_scalarization(self):
        source = inspect.getsource(PPOImprover.train)
        self.assertIn('self._scalarize_advantages(advantages_vector, preferences_b)', source)
        self.assertNotIn('advantages_vector.std(dim=0', source)

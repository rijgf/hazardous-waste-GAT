from __future__ import annotations

import unittest

from sample_params import build_sample_params
from src.heuristics import build_greedy_initial_plan
from src.operators import OPERATORS
from src.ppo_improver import PPOImprover
from src.solution_utils import evaluate_solution


class PPOSmokeTests(unittest.TestCase):
    def test_vector_critic_training_and_inference_return_feasible_solution(self) -> None:
        params = build_sample_params()
        algorithm = {
            "ppo": {
                "train_iterations": 1,
                "num_parallel_episodes": 2,
                "episode_steps": 2,
                "update_epochs": 1,
                "gamma": 0.95,
                "gae_lambda": 0.9,
                "clip_ratio": 0.2,
                "learning_rate": 0.0003,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "repair_failure_penalty": 0.2,
                "no_change_penalty": 0.01,
                "eval_steps": 2,
                "eval_candidate_samples": 2,
                "preference_sampling": {
                    "endpoint_probability_each": 0.1,
                    "beta_concentration": 0.5,
                },
            }
        }
        network = {
            "network_type": "full_transformer",
            "embedding_dim": 16,
            "transformer_layers": 1,
            "attention_heads": 4,
            "ff_hidden_dim": 32,
            "dropout": 0.0,
            "use_preference_token": True,
            "use_preference_in_global": False,
            "use_route_arc_tokens": True,
            "use_facility_tokens": True,
            "max_tokens": 128,
            "operator_count": len(OPERATORS),
            "object_count": 32,
        }
        improver = PPOImprover(params, algorithm, network, seed=17)

        history = improver.train()
        result = improver.improve(
            (0.0, 1.0),
            seed=19,
            initial_plan=build_greedy_initial_plan(params, seed=23),
        )

        self.assertEqual(len(history["episode_reward"]), 1)
        metrics = evaluate_solution(params, result.solution, (0.0, 1.0))
        self.assertTrue(metrics["feasible"], metrics["violations"])
        self.assertEqual(len(result.trace), 2)


if __name__ == "__main__":
    unittest.main()

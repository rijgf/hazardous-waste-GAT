from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from sample_params import build_sample_params
from src.config import load_model_config
from src.heuristics import build_greedy_initial_plan
from src.instance_generator import generate_random_params
from src.operators import OPERATORS
from src.ppo_improver import PPOImprover


def _algorithm_config() -> dict:
    return {
        "ppo": {
            "train_iterations": 0,
            "num_parallel_episodes": 1,
            "episode_steps": 1,
            "update_epochs": 1,
            "gamma": 0.95,
            "gae_lambda": 0.9,
            "clip_ratio": 0.2,
            "learning_rate": 0.0003,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "repair_failure_penalty": 0.2,
            "no_change_penalty": 0.01,
            "eval_steps": 1,
            "eval_candidate_samples": 1,
        }
    }


def _network_config() -> dict:
    return {
        "network_type": "full_transformer",
        "embedding_dim": 8,
        "transformer_layers": 1,
        "attention_heads": 2,
        "ff_hidden_dim": 16,
        "dropout": 0.0,
        "use_preference_token": True,
        "use_preference_in_global": False,
        "use_route_arc_tokens": True,
        "use_facility_tokens": True,
        "max_tokens": 64,
        "operator_count": len(OPERATORS),
        "object_count": 32,
    }


class FrozenPPOTests(unittest.TestCase):
    def test_frozen_checkpoint_binds_new_instance_and_cannot_train(self) -> None:
        training_params = build_sample_params()
        target_params = replace(
            build_sample_params(),
            generation={
                key: value + 0.125
                for key, value in build_sample_params().generation.items()
            },
        )
        algorithm = _algorithm_config()
        network = _network_config()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(training_params, algorithm, network, seed=11).train(
                save_path=checkpoint_path
            )

            frozen = PPOImprover.from_frozen_checkpoint(
                target_params,
                checkpoint_path,
                seed=13,
            )

        self.assertIs(frozen.params, target_params)
        self.assertIs(frozen.encoder.params, target_params)
        self.assertTrue(frozen.is_frozen)
        self.assertFalse(frozen.policy.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen.policy.parameters()))
        self.assertIsNone(frozen.optimizer)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            frozen.train()

    def test_frozen_inference_uses_inference_mode_without_changing_state(self) -> None:
        params = build_sample_params()
        algorithm = _algorithm_config()
        network = _network_config()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(params, algorithm, network, seed=17).train(
                save_path=checkpoint_path
            )
            frozen = PPOImprover.from_frozen_checkpoint(
                build_sample_params(), checkpoint_path, seed=19
            )

            inference_mode_seen: list[bool] = []
            hook = frozen.policy.register_forward_pre_hook(
                lambda _module, _inputs: inference_mode_seen.append(
                    torch.is_inference_mode_enabled()
                )
            )
            before = frozen.policy_state_sha256()
            result = frozen.improve(
                (0.5, 0.5),
                steps=1,
                seed=23,
                initial_plan=build_greedy_initial_plan(frozen.params, seed=29),
            )
            after = frozen.policy_state_sha256()
            hook.remove()

        self.assertEqual(len(result.trace), 1)
        self.assertTrue(inference_mode_seen)
        self.assertTrue(all(inference_mode_seen))
        self.assertEqual(before, after)

    def test_frozen_inference_detects_policy_state_mutation(self) -> None:
        params = build_sample_params()
        algorithm = _algorithm_config()
        network = _network_config()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(params, algorithm, network, seed=101).train(
                save_path=checkpoint_path
            )
            frozen = PPOImprover.from_frozen_checkpoint(
                build_sample_params(), checkpoint_path, seed=103
            )

            parameter = next(frozen.policy.parameters())

            def mutate_parameter(_module: torch.nn.Module, _inputs: tuple) -> None:
                parameter.add_(1.0)

            hook = frozen.policy.register_forward_pre_hook(mutate_parameter)
            try:
                with self.assertRaisesRegex(RuntimeError, "changed during inference"):
                    frozen.improve(
                        (0.5, 0.5),
                        steps=1,
                        seed=107,
                        initial_plan=build_greedy_initial_plan(frozen.params, seed=109),
                    )
            finally:
                hook.remove()

    def test_frozen_load_rejects_network_semantics_mismatch(self) -> None:
        params = build_sample_params()
        algorithm = _algorithm_config()
        network = _network_config()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(params, algorithm, network, seed=31).train(
                save_path=checkpoint_path
            )
            incompatible_network = dict(network)
            incompatible_network["use_preference_token"] = False

            with self.assertRaisesRegex(ValueError, "use_preference_token"):
                PPOImprover.from_frozen_checkpoint(
                    build_sample_params(),
                    checkpoint_path,
                    algorithm_config=algorithm,
                    network_config=incompatible_network,
                    seed=37,
                )

    def test_frozen_load_rejects_operator_vocabulary_mismatch(self) -> None:
        params = build_sample_params()
        algorithm = _algorithm_config()
        network = _network_config()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(params, algorithm, network, seed=79).train(
                save_path=checkpoint_path
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint["operators"] = list(reversed(checkpoint["operators"]))
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "operator vocabulary"):
                PPOImprover.from_frozen_checkpoint(
                    build_sample_params(), checkpoint_path, seed=83
                )

    def test_current_checkpoint_requires_operator_vocabulary_metadata(self) -> None:
        params = build_sample_params()
        algorithm = _algorithm_config()
        network = _network_config()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(params, algorithm, network, seed=131).train(
                save_path=checkpoint_path
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint.pop("operators")
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "missing operator vocabulary"):
                PPOImprover.from_frozen_checkpoint(
                    build_sample_params(), checkpoint_path, seed=137
                )

    def test_legacy_checkpoint_loads_when_configs_are_supplied(self) -> None:
        params = build_sample_params()
        algorithm = _algorithm_config()
        network = _network_config()
        source = PPOImprover(params, algorithm, network, seed=89)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "legacy.pt"
            torch.save({"state_dict": source.policy.state_dict()}, checkpoint_path)

            frozen = PPOImprover.from_frozen_checkpoint(
                build_sample_params(),
                checkpoint_path,
                algorithm_config=algorithm,
                network_config=network,
                seed=97,
            )

        self.assertTrue(frozen.is_frozen)
        self.assertEqual(source.policy_state_sha256(), frozen.policy_state_sha256())

    def test_frozen_object_always_rejects_optimizer_loading(self) -> None:
        params = build_sample_params()
        algorithm = _algorithm_config()
        network = _network_config()
        source = PPOImprover(params, algorithm, network, seed=113)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "legacy-without-optimizer.pt"
            torch.save({"state_dict": source.policy.state_dict()}, checkpoint_path)
            frozen = PPOImprover.from_frozen_checkpoint(
                build_sample_params(),
                checkpoint_path,
                algorithm_config=algorithm,
                network_config=network,
                seed=127,
            )

            with self.assertRaisesRegex(RuntimeError, "frozen.*optimizer"):
                frozen.load_checkpoint(checkpoint_path, load_optimizer=True)

    def test_frozen_load_rejects_target_over_object_capacity(self) -> None:
        training_params = build_sample_params()
        target_params = generate_random_params(load_model_config("small"), seed=41)
        algorithm = _algorithm_config()
        network = _network_config()
        network["object_count"] = 8

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(training_params, algorithm, network, seed=43).train(
                save_path=checkpoint_path
            )

            with self.assertRaisesRegex(ValueError, "object_count.*12"):
                PPOImprover.from_frozen_checkpoint(
                    target_params, checkpoint_path, seed=47
                )

    def test_frozen_load_counts_facilities_in_action_capacity(self) -> None:
        training_params = build_sample_params()
        target_config = load_model_config("small")
        target_config.update(
            {
                "producers_count": 1,
                "waste_types_count": 1,
                "facilities_count": 12,
                "vehicles_count": 1,
                "periods_count": 1,
            }
        )
        target_params = generate_random_params(target_config, seed=67)
        algorithm = _algorithm_config()
        network = _network_config()
        network["object_count"] = 8

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(training_params, algorithm, network, seed=71).train(
                save_path=checkpoint_path
            )

            with self.assertRaisesRegex(ValueError, "object_count.*12"):
                PPOImprover.from_frozen_checkpoint(
                    target_params, checkpoint_path, seed=73
                )

    def test_frozen_load_rejects_target_over_token_capacity(self) -> None:
        training_params = build_sample_params()
        target_params = generate_random_params(load_model_config("small"), seed=53)
        algorithm = _algorithm_config()
        network = _network_config()
        network["max_tokens"] = 40

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            PPOImprover(training_params, algorithm, network, seed=59).train(
                save_path=checkpoint_path
            )

            with self.assertRaisesRegex(ValueError, "max_tokens=40.*48"):
                PPOImprover.from_frozen_checkpoint(
                    target_params, checkpoint_path, seed=61
                )


if __name__ == "__main__":
    unittest.main()

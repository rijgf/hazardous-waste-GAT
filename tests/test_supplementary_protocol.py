from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from math import sqrt
from pathlib import Path

from src.supplementary_protocol import (
    ExperimentCell,
    aggregate_restarts_then_instances,
    build_cell_id,
    capacity_preflight,
    derive_protocol_seed,
    enumerate_cells,
    generator_config_for_scale,
    load_protocol_config,
    milp_gap_summary,
    protocol_cell_counts,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "supplementary_experiment.json"


class SupplementaryProtocolTests(unittest.TestCase):
    def test_frozen_config_describes_exact_execution_grid(self) -> None:
        config = load_protocol_config(CONFIG_PATH)

        validate_protocol(config)

        self.assertEqual(
            {
                "ppo": 6000,
                "ga": 3000,
                "heuristic": 500,
                "milp": 250,
                "total": 9750,
            },
            protocol_cell_counts(config),
        )

    def test_all_training_and_test_scales_share_one_generator_profile(self) -> None:
        config = load_protocol_config(CONFIG_PATH)
        expected_scale_fields = {
            "producers_count",
            "waste_types_count",
            "facilities_count",
            "vehicles_count",
            "periods_count",
        }

        self.assertEqual(expected_scale_fields, set(config["scale_fields"]))
        self.assertTrue(expected_scale_fields.isdisjoint(config["generator_profile"]))
        for scale_name in ("Test-1", "Test-2", "Test-3", "Test-4"):
            merged = generator_config_for_scale(config, scale_name)
            self.assertEqual(
                config["generator_profile"],
                {key: value for key, value in merged.items() if key not in expected_scale_fields},
            )

        models = {item["id"]: item["train_scale"] for item in config["models"]}
        self.assertEqual({"Train-S": "Test-1", "Train-L": "Test-4"}, models)

    def test_capacity_preflight_matches_encoder_upper_bounds(self) -> None:
        report = capacity_preflight(load_protocol_config(CONFIG_PATH))

        self.assertEqual(
            {
                "Test-1": (48, 12),
                "Test-2": (113, 36),
                "Test-3": (242, 90),
                "Test-4": (758, 320),
            },
            {
                name: (item["required_tokens"], item["required_object_slots"])
                for name, item in report.items()
            },
        )
        self.assertTrue(all(item["fits"] for item in report.values()))

    def test_seed_namespaces_are_deterministic_and_mutually_exclusive(self) -> None:
        config = load_protocol_config(CONFIG_PATH)
        seeds = {
            namespace: derive_protocol_seed(config, namespace, "Test-2", 17)
            for namespace in config["seed_namespaces"]
        }

        self.assertEqual(len(seeds), len(set(seeds.values())))
        self.assertTrue(all(0 <= seed < 2**32 for seed in seeds.values()))
        self.assertEqual(
            seeds["test_instance"],
            derive_protocol_seed(config, "test_instance", "Test-2", 17),
        )

    def test_cell_ids_are_stable_complete_and_collision_free(self) -> None:
        config = load_protocol_config(CONFIG_PATH)
        cells = enumerate_cells(config)
        cell_ids = [build_cell_id(cell) for cell in cells]

        self.assertEqual(9750, len(cells))
        self.assertEqual(9750, len(set(cell_ids)))
        self.assertEqual(cell_ids, [build_cell_id(cell) for cell in enumerate_cells(config)])

        example = ExperimentCell(
            protocol_id="protocol-demo",
            method="ppo",
            model_id="Train-S",
            test_scale="Test-2",
            instance_index=7,
            preference_id="C050-R050",
            cost_weight=0.5,
            risk_weight=0.5,
            restart_index=2,
            initial_solution_id="initial-plan-sha256-deadbeef",
            seed=1234,
        )
        example_id = build_cell_id(example)
        for required_identity in (
            "protocol-demo",
            "Train-S",
            "Test-2",
            "initial-plan-sha256-deadbeef",
        ):
            self.assertIn(required_identity, example_id)
        self.assertEqual(example_id, build_cell_id(example))
        self.assertNotEqual(
            example_id,
            build_cell_id(replace(example, initial_solution_id="a-different-initial-plan")),
        )

        shared_initial_ids = {
            cell.initial_solution_id
            for cell in cells
            if cell.test_scale == "Test-1" and cell.instance_index == 0
        }
        self.assertEqual(1, len(shared_initial_ids))

    def test_cell_ledger_can_be_locked_to_actual_initial_plan_hashes(self) -> None:
        config = load_protocol_config(CONFIG_PATH)
        identities = {
            (scale_name, instance_index): f"plan-sha256-{scale_name}-{instance_index:02d}"
            for scale_name in config["test_scales"]
            for instance_index in range(50)
        }

        cells = enumerate_cells(config, initial_solution_ids=identities)

        self.assertEqual(
            "plan-sha256-Test-3-11",
            next(
                cell.initial_solution_id
                for cell in cells
                if cell.test_scale == "Test-3" and cell.instance_index == 11
            ),
        )

    def test_summary_averages_restarts_before_using_instance_level_ddof_one(self) -> None:
        rows = [
            {"instance_index": 0, "restart_index": 0, "objective": 1.0},
            {"instance_index": 0, "restart_index": 1, "objective": 3.0},
            {"instance_index": 0, "restart_index": 2, "objective": 5.0},
            {"instance_index": 1, "restart_index": 0, "objective": 2.0},
            {"instance_index": 1, "restart_index": 1, "objective": 4.0},
            {"instance_index": 1, "restart_index": 2, "objective": 6.0},
            {"instance_index": 2, "restart_index": 0, "objective": 7.0},
            {"instance_index": 2, "restart_index": 1, "objective": 8.0},
            {"instance_index": 2, "restart_index": 2, "objective": 9.0},
        ]

        summary = aggregate_restarts_then_instances(rows, "objective")

        self.assertEqual({0: 3.0, 1: 4.0, 2: 8.0}, summary["instance_means"])
        self.assertEqual(3, summary["n_instances"])
        self.assertEqual(5.0, summary["mean"])
        self.assertAlmostEqual(sqrt(7.0), summary["sample_std"])
        self.assertEqual(1, summary["ddof"])

    def test_milp_gap_uses_only_proven_optima_and_reports_valid_n(self) -> None:
        rows = []
        for restart, objective in enumerate((90.0, 100.0, 110.0)):
            rows.append(
                {
                    "instance_index": 0,
                    "restart_index": restart,
                    "objective": objective,
                    "milp_objective": 80.0,
                    "milp_status": "Optimal",
                }
            )
        for restart, objective in enumerate((1.0, 1.0, 1.0)):
            rows.append(
                {
                    "instance_index": 1,
                    "restart_index": restart,
                    "objective": objective,
                    "milp_objective": 1.0,
                    "milp_status": "time_limit_feasible",
                }
            )
        for restart, objective in enumerate((45.0, 50.0, 55.0)):
            rows.append(
                {
                    "instance_index": 2,
                    "restart_index": restart,
                    "objective": objective,
                    "milp_objective": 50.0,
                    "milp_status": "optimal",
                }
            )

        summary = milp_gap_summary(rows)

        self.assertEqual(2, summary["valid_n"])
        self.assertEqual({0: 25.0, 2: 0.0}, summary["gaps_by_instance"])
        self.assertEqual(12.5, summary["mean_gap_percent"])
        self.assertAlmostEqual(25.0 / sqrt(2.0), summary["sample_std_gap_percent"])
        self.assertEqual(1, summary["ddof"])

    def test_validator_locks_algorithm_and_encoder_configuration(self) -> None:
        config = load_protocol_config(CONFIG_PATH)

        self.assertEqual("outputs/supplementary_experiment_v1", config["output_dir"])
        self.assertEqual(
            {
                "network_type": "full_transformer",
                "embedding_dim": 48,
                "transformer_layers": 1,
                "attention_heads": 4,
                "ff_hidden_dim": 96,
                "dropout": 0.0,
                "use_preference_token": True,
                "use_preference_in_global": False,
                "use_route_arc_tokens": True,
                "use_facility_tokens": True,
                "max_tokens": 896,
                "operator_count": 8,
                "object_count": 512,
            },
            config["network"],
        )
        self.assertEqual(
            {
                "train_iterations",
                "num_parallel_episodes",
                "episode_steps",
                "update_epochs",
                "gamma",
                "gae_lambda",
                "clip_ratio",
                "learning_rate",
                "entropy_coef",
                "value_coef",
                "repair_failure_penalty",
                "no_change_penalty",
                "eval_steps",
                "eval_candidate_samples",
                "evaluation_restarts",
                "preference_sampling",
            },
            set(config["algorithm"]["ppo"]),
        )
        ga_by_scale = config["algorithm"]["ga_by_scale"]
        self.assertEqual({"Test-1", "Test-2", "Test-3", "Test-4"}, set(ga_by_scale))
        self.assertEqual((30, 60), (ga_by_scale["Test-1"]["population_size"], ga_by_scale["Test-1"]["generations"]))
        self.assertEqual((30, 60), (ga_by_scale["Test-2"]["population_size"], ga_by_scale["Test-2"]["generations"]))
        self.assertEqual((20, 40), (ga_by_scale["Test-3"]["population_size"], ga_by_scale["Test-3"]["generations"]))
        self.assertEqual((20, 40), (ga_by_scale["Test-4"]["population_size"], ga_by_scale["Test-4"]["generations"]))
        self.assertEqual(180.0, config["algorithm"]["milp_time_limit_seconds"])

        invalid_cases = {}
        missing_ga_scale = copy.deepcopy(config)
        del missing_ga_scale["algorithm"]["ga_by_scale"]["Test-4"]
        invalid_cases["missing GA scale"] = missing_ga_scale
        missing_ppo_key = copy.deepcopy(config)
        del missing_ppo_key["algorithm"]["ppo"]["eval_steps"]
        invalid_cases["missing PPO key"] = missing_ppo_key
        mismatched_capacity = copy.deepcopy(config)
        mismatched_capacity["network"]["max_tokens"] = 895
        invalid_cases["mismatched network capacity"] = mismatched_capacity
        leaked_scale_field = copy.deepcopy(config)
        leaked_scale_field["generator_profile"]["producers_count"] = 3
        invalid_cases["scale field in shared profile"] = leaked_scale_field
        invalid_gamma = copy.deepcopy(config)
        invalid_gamma["algorithm"]["ppo"]["gamma"] = 99
        invalid_cases["out-of-protocol PPO gamma"] = invalid_gamma
        invalid_learning_rate = copy.deepcopy(config)
        invalid_learning_rate["algorithm"]["ppo"]["learning_rate"] = -1
        invalid_cases["negative PPO learning rate"] = invalid_learning_rate
        invalid_network_type = copy.deepcopy(config)
        invalid_network_type["network"]["network_type"] = "typo"
        invalid_cases["unknown network type"] = invalid_network_type
        invalid_ga_rate = copy.deepcopy(config)
        invalid_ga_rate["algorithm"]["ga_by_scale"]["Test-1"]["mutation_rate"] = 2.0
        invalid_cases["out-of-protocol GA mutation rate"] = invalid_ga_rate

        for label, invalid in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                validate_protocol(invalid)


if __name__ == "__main__":
    unittest.main()

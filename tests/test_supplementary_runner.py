from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sample_params import build_sample_params
from src.heuristics import build_greedy_initial_plan
from src.reproducibility import file_sha256, plan_sha256
from src.solution_utils import route_plan_to_solution
from src.supplementary_experiment import (
    ArtifactIntegrityError,
    atomic_write_json,
    commit_cell,
    deserialize_solution,
    _instance_level_frame,
    _summary_by_preference,
    initialize_run,
    load_valid_completed_cell,
    serialize_solution,
    terminal_inventory_clear,
    train_models,
)


class SupplementaryRunnerTests(unittest.TestCase):
    def test_solution_round_trip_preserves_plan_and_raw_tuple_keys(self) -> None:
        params = build_sample_params()
        plan = build_greedy_initial_plan(params, seed=101)
        solution = route_plan_to_solution(params, plan)

        restored = deserialize_solution(serialize_solution(solution))

        self.assertEqual(plan_sha256(plan), plan_sha256(restored["plan"]))
        self.assertEqual(solution["raw"], restored["raw"])
        self.assertTrue(terminal_inventory_clear(params, restored))

    def test_empty_solution_is_not_reported_as_terminally_clear(self) -> None:
        self.assertFalse(terminal_inventory_clear(build_sample_params(), {}))

    def test_atomic_write_replaces_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "record.json"
            atomic_write_json(path, {"value": 1})
            atomic_write_json(path, {"value": 2})

            self.assertEqual({"value": 2}, json.loads(path.read_text(encoding="utf-8")))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_completed_cell_is_skipped_only_while_all_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            input_path = run_dir / "instance.json"
            solution_path = run_dir / "solutions" / "abc.json"
            cell_path = run_dir / "cells" / "abc.json"
            atomic_write_json(input_path, {"instance": 1})
            atomic_write_json(solution_path, {"solution": 1})
            identity = {"instance_file_sha256": file_sha256(input_path)}
            record = {
                "schema_version": "supplementary-cell-v1",
                "cell_id": "abc",
                "status": "complete",
                "input_hashes": identity,
                "instance_file": "instance.json",
                "solution_file": "solutions/abc.json",
                "solution_file_sha256": file_sha256(solution_path),
            }
            commit_cell(cell_path, record)

            loaded = load_valid_completed_cell(run_dir, cell_path, identity)
            self.assertEqual("abc", loaded["cell_id"])

            atomic_write_json(solution_path, {"solution": "tampered"})
            with self.assertRaises(ArtifactIntegrityError):
                load_valid_completed_cell(run_dir, cell_path, identity)

    def test_completed_cell_detects_locked_input_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            instance_path = run_dir / "instance.json"
            solution_path = run_dir / "solution.json"
            cell_path = run_dir / "cell.json"
            atomic_write_json(instance_path, {"instance": 1})
            atomic_write_json(solution_path, {"solution": 1})
            identity = {"instance_file_sha256": file_sha256(instance_path)}
            commit_cell(
                cell_path,
                {
                    "schema_version": "supplementary-cell-v1",
                    "cell_id": "abc",
                    "status": "complete",
                    "input_hashes": identity,
                    "instance_file": "instance.json",
                    "solution_file": "solution.json",
                    "solution_file_sha256": file_sha256(solution_path),
                },
            )
            atomic_write_json(instance_path, {"instance": "tampered"})

            with self.assertRaises(ArtifactIntegrityError):
                load_valid_completed_cell(run_dir, cell_path, identity)

    def test_orphan_solution_is_not_a_completed_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            atomic_write_json(run_dir / "solutions" / "orphan.json", {"solution": 1})

            self.assertIsNone(
                load_valid_completed_cell(
                    run_dir,
                    run_dir / "cells" / "orphan.json",
                    {"instance_sha256": "expected"},
                )
            )

    def test_prepared_checkpoint_recovers_across_atomic_replace_window(self) -> None:
        for checkpoint_already_replaced in (False, True):
            with self.subTest(checkpoint_already_replaced=checkpoint_already_replaced):
                with tempfile.TemporaryDirectory() as tmp:
                    run_dir = initialize_run(
                        run_dir=Path(tmp) / "run",
                        smoke=True,
                    )
                    manifest_path = run_dir / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    checkpoint = run_dir / "models" / "small_model.pt"
                    temporary = checkpoint.with_name(f".{checkpoint.name}.training.tmp")
                    artifact = checkpoint if checkpoint_already_replaced else temporary
                    artifact.write_bytes(b"complete-checkpoint")
                    history = run_dir / "training_history" / "Train-S.csv"
                    history.write_text("iteration,objective\n0,1.0\n", encoding="utf-8")
                    training = manifest["training_instances"]["Train-S"]
                    prepared = {
                        "protocol_sha256": manifest["protocol_sha256"],
                        "model_id": "Train-S",
                        "train_scale": "Test-1",
                        "training_instance_file_sha256": training["file_sha256"],
                        "training_instance_sha256": training["instance_sha256"],
                        "training_seed": 1,
                        "checkpoint_file": "models/small_model.pt",
                        "checkpoint_sha256": file_sha256(artifact),
                        "policy_state_sha256": "state-hash",
                        "source_bundle_sha256": manifest[
                            "source_bundle_sha256_at_init"
                        ],
                        "training_seconds": 1.0,
                        "history_file": "training_history/Train-S.csv",
                        "history_sha256": file_sha256(history),
                        "frozen_at": "2026-09-04T00:00:00+08:00",
                    }
                    prepared_path = run_dir / "models" / ".Train-S.prepared.json"
                    atomic_write_json(prepared_path, prepared)

                    recovered = train_models(run_dir, ["Train-S"])

                    self.assertEqual(prepared["checkpoint_sha256"], recovered["Train-S"]["checkpoint_sha256"])
                    self.assertTrue(checkpoint.is_file())
                    self.assertFalse(temporary.exists())
                    self.assertFalse(prepared_path.exists())
                    self.assertTrue((run_dir / "models" / "Train-S.record.json").is_file())

    def test_failed_restart_remains_in_feasibility_denominator_and_suppresses_primary_mean(self) -> None:
        rows = []
        for instance_index in (0, 1):
            for restart_index in range(3):
                failed = instance_index == 1 and restart_index == 2
                row = {
                    "status": "failed" if failed else "complete",
                    "method": "ppo",
                    "model_id": "Train-S",
                    "test_scale": "Test-1",
                    "instance_index": instance_index,
                    "preference_id": "C050-R050",
                    "cost_weight": 0.5,
                    "risk_weight": 0.5,
                    "restart_index": restart_index,
                    "feasible": None if failed else True,
                    "terminal_inventory_clear": None if failed else True,
                    "runtime_seconds": None if failed else 1.0,
                }
                for field in (
                    "cost",
                    "risk",
                    "normalized_cost",
                    "normalized_risk",
                    "weighted_objective_raw",
                    "weighted_objective_normalized",
                    "fixed_cost",
                    "distance_cost",
                    "processing_cost",
                    "transport_risk",
                    "coload_risk",
                    "producer_inventory_risk",
                    "facility_inventory_risk",
                ):
                    row[field] = None if failed else 1.0
                rows.append(row)
        config = {"methods": {"ppo": {"restarts": 3}}}

        instance = _instance_level_frame(pd.DataFrame(rows), config)
        summary = _summary_by_preference(instance).iloc[0]

        self.assertAlmostEqual(5.0 / 6.0, summary["strict_feasible_rate"])
        self.assertAlmostEqual(1.0 / 6.0, summary["technical_failure_rate"])
        self.assertTrue(pd.isna(summary["weighted_objective_normalized_mean"]))
        self.assertTrue(pd.isna(summary["weighted_objective_normalized_median"]))
        self.assertEqual(1.0, summary["weighted_objective_normalized_conditional_mean"])
        self.assertEqual(1.0, summary["weighted_objective_normalized_conditional_median"])


if __name__ == "__main__":
    unittest.main()

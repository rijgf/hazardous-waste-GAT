from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "output" / "supplementary-experiments" / "scripts" / "build_paper_tables.py"
SPEC = importlib.util.spec_from_file_location("build_paper_tables_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
paper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paper
SPEC.loader.exec_module(paper)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _smoke_protocol() -> dict:
    protocol = json.loads((ROOT / "configs" / "supplementary_experiment.json").read_text(encoding="utf-8"))
    protocol["protocol_id"] = "supplementary-experiment-v1-smoke"
    for item in protocol["test_scales"].values():
        item["instances"] = 1
    protocol["preferences"] = [
        {"id": "C050-R050", "cost_weight": 0.5, "risk_weight": 0.5}
    ]
    protocol["methods"]["ppo"]["restarts"] = 1
    protocol["methods"]["ga"]["restarts"] = 1
    return protocol


def _build_synthetic_smoke(root: Path) -> None:
    protocol = _smoke_protocol()
    protocol_hash = paper._canonical_json_sha256(protocol)
    _write_json(root / "protocol_config.json", protocol)

    schedule_cells = []
    instance_rows = []
    summary_rows = []
    index = 0
    for method, scope in protocol["methods"].items():
        for model_id in scope["models"]:
            for scale in scope["scales"]:
                index += 1
                cell_id = f"cell-{index:03d}"
                objective = 0.6 if method == "ga" else 0.5
                schedule_cells.append(
                    {
                        "cell_id": cell_id,
                        "logical_cell_id": f"logical-{index:03d}",
                        "cell": {
                            "protocol_id": protocol["protocol_id"],
                            "method": method,
                            "model_id": model_id,
                            "test_scale": scale,
                            "instance_index": 0,
                            "preference_id": "C050-R050",
                            "cost_weight": 0.5,
                            "risk_weight": 0.5,
                            "restart_index": 0,
                        },
                    }
                )
                solver_status = "optimal" if method == "milp" else "completed"
                instance_rows.append(
                    {
                        "method": method,
                        "model_id": model_id,
                        "test_scale": scale,
                        "instance_index": 0,
                        "preference_id": "C050-R050",
                        "cost_weight": 0.5,
                        "risk_weight": 0.5,
                        "feasible_rate": 1.0,
                        "all_restarts_feasible": True,
                        "terminal_clear_rate": 1.0,
                        "technical_failure_rate": 0.0,
                        "missing_rate": 0.0,
                        "runtime_seconds": 0.1,
                        "solver_status": solver_status,
                        "cost": 100.0,
                        "risk": 10.0,
                        "normalized_cost": 0.5,
                        "normalized_risk": 0.5,
                        "weighted_objective_raw": 55.0,
                        "weighted_objective_normalized": objective,
                        "fixed_cost": 20.0,
                        "distance_cost": 30.0,
                        "processing_cost": 50.0,
                        "transport_risk": 4.0,
                        "coload_risk": 2.0,
                        "producer_inventory_risk": 3.0,
                        "facility_inventory_risk": 1.0,
                    }
                )
                summary_rows.append(
                    {
                        "method": method,
                        "model_id": model_id,
                        "test_scale": scale,
                        "preference_id": "C050-R050",
                        "cost_weight": 0.5,
                        "risk_weight": 0.5,
                        "n_instances_total": 1,
                        "n_instances_quality": 1,
                        "strict_feasible_rate": 1.0,
                        "all_restarts_feasible_rate": 1.0,
                        "terminal_clear_rate": 1.0,
                        "technical_failure_rate": 0.0,
                        "missing_rate": 0.0,
                        "cost_mean": 100.0,
                        "cost_sd": None,
                        "risk_mean": 10.0,
                        "risk_sd": None,
                        "normalized_cost_mean": 0.5,
                        "normalized_cost_sd": None,
                        "normalized_risk_mean": 0.5,
                        "normalized_risk_sd": None,
                        "weighted_objective_raw_mean": 55.0,
                        "weighted_objective_raw_sd": None,
                        "weighted_objective_normalized_mean": objective,
                        "weighted_objective_normalized_sd": None,
                        "fixed_cost_mean": 20.0,
                        "fixed_cost_sd": None,
                        "distance_cost_mean": 30.0,
                        "distance_cost_sd": None,
                        "processing_cost_mean": 50.0,
                        "processing_cost_sd": None,
                        "runtime_seconds_mean": 0.1,
                        "runtime_seconds_sd": None,
                        "transport_risk_mean": 4.0,
                        "transport_risk_sd": None,
                        "coload_risk_mean": 2.0,
                        "coload_risk_sd": None,
                        "producer_inventory_risk_mean": 3.0,
                        "producer_inventory_risk_sd": None,
                        "facility_inventory_risk_mean": 1.0,
                        "facility_inventory_risk_sd": None,
                    }
                )
    schedule = {
        "schema_version": "supplementary-schedule-v1",
        "protocol_sha256": protocol_hash,
        "cells": schedule_cells,
    }
    _write_json(root / "schedule.json", schedule)
    schedule_hash = _file_hash(root / "schedule.json")

    ledger_rows = [
        {"cell_id": item["cell_id"], "status": "complete", "cell_file_sha256": "a" * 64}
        for item in schedule_cells
    ]
    ledger_hash = paper._canonical_json_sha256(ledger_rows)
    instance = pd.DataFrame(instance_rows)
    summary = pd.DataFrame(summary_rows)
    delta = (0.6 - 0.5) / 0.6 * 100.0
    paired_rows = [
        {
            "model_id": model,
            "test_scale": scale,
            "instance_index": 0,
            "preference_id": "C050-R050",
            "ppo_j": 0.5,
            "ga_j": 0.6,
            "delta_j_percent": delta,
        }
        for model in ("Train-S", "Train-L")
        for scale in paper.FORMAL_SCALE_SHAPES
    ]
    paired = pd.DataFrame(paired_rows)
    paired_summary = pd.DataFrame(
        [
            {
                "model_id": row["model_id"],
                "test_scale": row["test_scale"],
                "preference_id": "C050-R050",
                "expected_paired_n": 1,
                "paired_n": 1,
                "mean_delta_j_percent": delta,
                "sample_sd_delta_j_percent": None,
            }
            for row in paired_rows
        ]
    )
    gap_values = {("ppo", "Train-S"): 0.0, ("ga", "GA"): 20.0, ("heuristic", "Heuristic"): 0.0}
    gap_summary = pd.DataFrame(
        [
            {
                "method": method,
                "model_id": model,
                "preference_id": "C050-R050",
                "optimal_reference_n": 1,
                "valid_n": 1,
                "mean_gap_percent": gap,
                "sample_sd_gap_percent": None,
            }
            for (method, model), gap in gap_values.items()
        ]
    )
    operational_rows = []
    for model, macro_j in (("Train-S", 0.5), ("Train-L", 0.55)):
        for scale in paper.FORMAL_SCALE_SHAPES:
            operational_rows.append(
                {
                    "model_id": model,
                    "test_scale": scale,
                    "independent_instances": 1,
                    "instance_preference_cells": 1,
                    "quality_cells": 1,
                    "macro_j": 0.5,
                    "strict_feasible_rate": 1.0,
                    "terminal_clear_rate": 1.0,
                    "mean_inference_seconds": 0.1,
                    "relative_ga_improvement_percent": delta,
                    "expected_paired_ga_n": 1,
                    "paired_ga_n": 1,
                    "two_model_relative_gap_percent": 0.0,
                }
            )
    # Both instance-level model objectives are 0.5; keep verified matrix consistent.
    matrix = pd.DataFrame(
        [
            {
                "model_id": model,
                **{f"macro_j_{scale}": 0.5 for scale in paper.FORMAL_SCALE_SHAPES},
                **{
                    f"two_model_relative_gap_percent_{scale}": 0.0
                    for scale in paper.FORMAL_SCALE_SHAPES
                },
            }
            for model in ("Train-S", "Train-L")
        ]
    )
    tables = {
        "raw_results.csv": pd.DataFrame({"cell_id": [item["cell_id"] for item in schedule_cells]}),
        "instance_level_results.csv": instance,
        "summary_by_preference.csv": summary,
        "paired_ppo_vs_ga.csv": paired,
        "paired_ppo_vs_ga_summary.csv": paired_summary,
        "milp_gap_by_instance.csv": pd.DataFrame(
            [
                {
                    "method": method,
                    "model_id": model,
                    "instance_index": 0,
                    "preference_id": "C050-R050",
                    "weighted_objective_normalized": 0.6 if method == "ga" else 0.5,
                    "milp_j": 0.5,
                    "milp_status": "optimal",
                    "gap_percent": gap,
                }
                for (method, model), gap in gap_values.items()
            ]
        ),
        "milp_gap_summary.csv": gap_summary,
        "generalization_matrix.csv": matrix,
        "generalization_operational.csv": pd.DataFrame(operational_rows),
        "cell_file_ledger.csv": pd.DataFrame(ledger_rows),
    }
    table_hashes = {}
    table_records = {}
    for name, frame in tables.items():
        path = root / "tables" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")
        digest = _file_hash(path)
        table_hashes[name] = digest
        table_records[name] = {"file": f"tables/{name}", "file_sha256": digest}

    counts = paper._protocol_counts(protocol)
    verification = {
        "schema_version": "supplementary-verification-v1",
        "protocol_id": protocol["protocol_id"],
        "schedule_file_sha256": schedule_hash,
        "summary_input_cell_ledger_sha256": ledger_hash,
        "summary_ledger_match": True,
        "table_files_sha256": table_hashes,
        "expected_cell_counts": counts,
        "observed_complete_counts": {
            key: counts[key] for key in ("ppo", "ga", "heuristic", "milp")
        },
        "count_match": True,
        "model_count": 2,
        "checkpoint_file_count": 2,
        "test_instance_count": 4,
        "missing_count": 0,
        "failed_count": 0,
        "integrity_error_count": 0,
        "artifact_error_count": 0,
        "artifact_rechecked_count": counts["total"],
        "solver_replayed_count": counts["total"],
        "solver_replay_error_count": 0,
        "source_changes_since_init": {},
        "passed": True,
    }
    verification_path = root / "verification" / "verification_report.json"
    _write_json(verification_path, verification)
    verification_hash = _file_hash(verification_path)
    manifest = {
        "schema_version": "supplementary-manifest-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "smoke": True,
        "stage": "verified",
        "models": {
            "Train-S": {"training_seconds": 1.0},
            "Train-L": {"training_seconds": 2.0},
        },
        "test_sets": {scale: [{"index": 0}] for scale in paper.FORMAL_SCALE_SHAPES},
        "schedule": {
            "file": "schedule.json",
            "file_sha256": schedule_hash,
            "cell_count": counts["total"],
        },
        "tables": table_records,
        "summary_input_cell_ledger_sha256": ledger_hash,
        "verification": {
            "file": "verification/verification_report.json",
            "file_sha256": verification_hash,
            "passed": True,
        },
        "evaluation_runs": [
            {
                "methods": "all",
                "models": "all",
                "scales": "all",
                "workers": 1,
                "completed_cells": 15,
                "wall_seconds": 3.0,
                "superseded_for_timing": True,
            },
            {
                "methods": ["ppo"],
                "models": "all",
                "scales": "all",
                "workers": 2,
                "completed_cells": 8,
                "wall_seconds": 1.0,
                "allow_concurrent_ppo": True,
            },
        ],
    }
    _write_json(root / "manifest.json", manifest)


def _refresh_verified_table_hashes(root: Path, *table_names: str) -> None:
    manifest_path = root / "manifest.json"
    verification_path = root / "verification" / "verification_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    for name in table_names:
        digest = _file_hash(root / "tables" / name)
        manifest["tables"][name]["file_sha256"] = digest
        verification["table_files_sha256"][name] = digest
    _write_json(verification_path, verification)
    manifest["verification"]["file_sha256"] = _file_hash(verification_path)
    _write_json(manifest_path, manifest)


class PaperTableGateTests(unittest.TestCase):
    def test_formal_contract_is_exact(self) -> None:
        formal = json.loads(
            (ROOT / "configs" / "supplementary_experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(paper._canonical_json_sha256(formal), paper.FORMAL_PROTOCOL_SHA256)
        self.assertEqual(paper._validate_protocol_shape(formal, smoke=False), paper.FORMAL_COUNTS)
        changed = copy.deepcopy(formal)
        changed["test_scales"]["Test-4"]["instances"] = 49
        with self.assertRaises(paper.PaperTableError):
            paper._validate_protocol_shape(changed, smoke=False)
        changed = copy.deepcopy(formal)
        changed["seed_root"] = 999
        with self.assertRaisesRegex(paper.PaperTableError, "canonical hash"):
            paper._validate_protocol_shape(changed, smoke=False)

    def test_formal_verification_uses_stratified_replay_minima(self) -> None:
        formal = json.loads(
            (ROOT / "configs" / "supplementary_experiment.json").read_text(encoding="utf-8")
        )
        expected_counts = paper._validate_protocol_shape(formal, smoke=False)
        minimum_artifacts, minimum_solvers = paper._verification_stratum_minima(formal)
        self.assertEqual((minimum_artifacts, minimum_solvers), (75, 15))

        manifest = {
            "models": {"Train-S": {}, "Train-L": {}},
            "test_sets": {
                scale: [{"index": index} for index in range(50)]
                for scale in paper.FORMAL_SCALE_SHAPES
            },
        }
        verification = {
            "expected_cell_counts": expected_counts,
            "observed_complete_counts": {
                key: expected_counts[key] for key in ("ppo", "ga", "heuristic", "milp")
            },
            "count_match": True,
            "summary_ledger_match": True,
            "missing_count": 0,
            "failed_count": 0,
            "integrity_error_count": 0,
            "artifact_error_count": 0,
            "solver_replay_error_count": 0,
            "artifact_rechecked_count": minimum_artifacts,
            "solver_replayed_count": minimum_solvers,
            "source_changes_since_init": {},
            "model_count": 2,
            "checkpoint_file_count": 2,
            "test_instance_count": 200,
        }
        paper._validate_report_counts(
            verification,
            manifest=manifest,
            protocol=formal,
            expected_counts=expected_counts,
            smoke=False,
        )

        too_few_artifacts = dict(verification)
        too_few_artifacts["artifact_rechecked_count"] = minimum_artifacts - 1
        with self.assertRaisesRegex(paper.PaperTableError, "artifact"):
            paper._validate_report_counts(
                too_few_artifacts,
                manifest=manifest,
                protocol=formal,
                expected_counts=expected_counts,
                smoke=False,
            )

        too_few_solvers = dict(verification)
        too_few_solvers["solver_replayed_count"] = minimum_solvers - 1
        with self.assertRaisesRegex(paper.PaperTableError, "solver"):
            paper._validate_report_counts(
                too_few_solvers,
                manifest=manifest,
                protocol=formal,
                expected_counts=expected_counts,
                smoke=False,
            )

    def test_smoke_requires_flag_and_build_inventory_has_matching_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            output = Path(temporary) / "paper"
            _build_synthetic_smoke(root)
            with self.assertRaisesRegex(paper.PaperTableError, "--allow-smoke"):
                paper.build_paper_tables(root, output)
            paths = paper.build_paper_tables(root, output, allow_smoke=True)
            path_names = {path.name for path in paths}
            self.assertIn("paper-table-inventory.json", path_names)
            self.assertIn("table5_scale_design.csv", path_names)
            self.assertIn("table5_scale_design.md", path_names)
            self.assertEqual(len(paths), 19)
            registry = pd.read_csv(output / "results-registry.csv")
            adjudication = pd.read_csv(output / "adjudication-log.csv")
            self.assertEqual(set(registry["hypothesis_id"]), paper.ALLOWED_HYPOTHESIS_IDS)
            self.assertEqual(set(adjudication["hypothesis_id"]), paper.ALLOWED_HYPOTHESIS_IDS)
            scale_design = pd.read_csv(output / "table5_scale_design.csv")
            self.assertEqual(
                list(scale_design.columns),
                [
                    "object_type",
                    "object_type_label",
                    "design_id",
                    "source_test_scale",
                    *paper.SCALE_FIELDS,
                    "scale_signature",
                    "quantity",
                    "quantity_unit",
                    "quantity_display",
                    "description",
                ],
            )
            self.assertEqual(
                list(scale_design["design_id"]),
                ["Train-S", "Train-L", "Test-1", "Test-2", "Test-3", "Test-4"],
            )
            smoke_tests = scale_design[scale_design["object_type"] == "test"]
            self.assertEqual(len(scale_design), 6)
            self.assertTrue(smoke_tests["quantity"].eq(1).all())
            inventory = json.loads((output / "paper-table-inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["mode"], "smoke")
            self.assertEqual(inventory["expected_cell_counts"]["total"], 15)
            self.assertEqual(inventory["source_run_dir_path_kind"], "absolute-external")
            self.assertEqual(len(inventory["output_files_sha256"]), 18)
            self.assertEqual(inventory["output_row_counts"]["table5_scale_design.csv"], 6)
            for name, digest in inventory["output_files_sha256"].items():
                self.assertEqual(_file_hash(output / name), digest)

            for name, expected_title in (
                ("table3_small_scale.md", "表3  小规模1个未见实例"),
                ("table4_large_scale.md", "表4  大规模1个未见实例"),
            ):
                self.assertIn(expected_title, (output / name).read_text(encoding="utf-8"))
            for name in (
                "table3_small_scale.md",
                "table4_large_scale.md",
                "table7_generalization_operational.md",
            ):
                markdown = (output / name).read_text(encoding="utf-8")
                for forbidden in ("batch", "workers=", "max_cells=", "整批墙钟="):
                    self.assertNotIn(forbidden, markdown)
                self.assertIn("manifest.evaluation_runs", markdown)
                self.assertIn("不能据此作隔离运行下纯算法速度的因果比较", markdown)
            table3_markdown = (output / "table3_small_scale.md").read_text(encoding="utf-8")
            self.assertIn(
                "任何未证明optimal的incumbent（包括限时或其他中止）均不进入Gap",
                table3_markdown,
            )
            self.assertNotIn("非最优限时解不进入Gap", table3_markdown)

    def test_table5_is_derived_from_the_locked_formal_protocol(self) -> None:
        protocol = json.loads(
            (ROOT / "configs" / "supplementary_experiment.json").read_text(encoding="utf-8")
        )
        frame, markdown = paper._table5_scale_design(protocol)
        self.assertEqual(
            list(frame["design_id"]),
            ["Train-S", "Train-L", "Test-1", "Test-2", "Test-3", "Test-4"],
        )
        tests = frame[frame["object_type"] == "test"].set_index("design_id")
        self.assertEqual(
            tests["scale_signature"].to_dict(),
            {
                "Test-1": "3/2/2/3/2",
                "Test-2": "6/2/2/4/3",
                "Test-3": "10/3/3/5/3",
                "Test-4": "20/4/3/8/4",
            },
        )
        self.assertTrue(tests["quantity"].eq(50).all())
        for model_id, scale_id in (("Train-S", "Test-1"), ("Train-L", "Test-4")):
            training_row = frame[frame["design_id"] == model_id].iloc[0]
            test_row = frame[frame["design_id"] == scale_id].iloc[0]
            self.assertEqual(
                tuple(training_row[field] for field in paper.SCALE_FIELDS),
                tuple(test_row[field] for field in paper.SCALE_FIELDS),
            )
        self.assertIn("表5  两种训练规模与四种测试规模设置", markdown)
        self.assertIn("50个实例", markdown)

        changed = copy.deepcopy(protocol)
        changed["scale_fields"] = list(reversed(changed["scale_fields"]))
        with self.assertRaisesRegex(paper.PaperTableError, "scale_fields"):
            paper._table5_scale_design(changed)

    def test_inventory_source_run_path_is_portable_inside_repository(self) -> None:
        value, kind = paper._portable_source_run_dir(
            ROOT / "outputs" / "supplementary_experiment_v1"
        )
        self.assertEqual(value, "outputs/supplementary_experiment_v1")
        self.assertEqual(kind, "repository-relative")

    def test_heuristic_time_is_test_lock_initialization_not_evaluation_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            output = Path(temporary) / "paper"
            _build_synthetic_smoke(root)
            paper.build_paper_tables(root, output, allow_smoke=True)

            appendix = pd.read_csv(output / "appendix_all_methods.csv")
            heuristic = appendix[appendix["method_id"] == "heuristic"]
            solve_methods = appendix[appendix["method_id"].isin(["ppo", "ga", "milp"])]
            self.assertEqual(len(heuristic), 2)
            self.assertTrue(heuristic["initialization_seconds_mean"].eq(0.1).all())
            self.assertTrue(heuristic["solve_segment_seconds_mean"].isna().all())
            self.assertTrue(heuristic["solve_segment_concurrency_scope"].isna().all())
            self.assertEqual(
                set(heuristic["time_measurement"]),
                {"测试集锁定阶段初始解构造时间(initialization_seconds)"},
            )
            self.assertTrue(solve_methods["initialization_seconds_mean"].isna().all())
            self.assertTrue(solve_methods["solve_segment_seconds_mean"].eq(0.1).all())
            self.assertTrue(solve_methods["solve_segment_concurrency_scope"].notna().all())
            self.assertEqual(
                set(solve_methods["time_measurement"]),
                {"并发条件下求解段观测时间(runtime_seconds)"},
            )

            markdown = (output / "appendix_all_methods.md").read_text(encoding="utf-8")
            self.assertIn("PPO/GA/MILP求解段；启发式初始化", markdown)
            self.assertIn(
                "启发式时间为测试集锁定阶段初始解构造时间(initialization_seconds)",
                markdown,
            )
            self.assertIn("不绑定启发式evaluation并发批次", markdown)

            registry = pd.read_csv(output / "results-registry.csv")
            heuristic_time = registry[
                registry["metric_name"].str.startswith("initialization_seconds.")
            ]
            self.assertEqual(len(heuristic_time), 2)
            self.assertTrue(heuristic_time["metric_value"].eq(0.1).all())
            self.assertTrue(
                heuristic_time["notes"].str.contains("initialization_seconds", regex=False).all()
            )
            self.assertTrue(
                heuristic_time["notes"]
                .str.contains("不绑定启发式evaluation并发批次", regex=False)
                .all()
            )
            self.assertFalse(
                registry["metric_name"].str.startswith("solve_segment_seconds.Heuristic").any()
            )

    def test_inventory_is_published_last_as_complete_output_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            output = Path(temporary) / "paper"
            _build_synthetic_smoke(root)
            _write_json(output / "paper-table-inventory.json", {"stale": True})
            failed_destination = (output / "results-registry.csv").resolve()
            real_replace = paper.os.replace

            def fail_during_output_publish(source: object, destination: object) -> None:
                if Path(destination).resolve() == failed_destination:
                    raise OSError("injected publish failure")
                real_replace(source, destination)

            with mock.patch.object(
                paper.os, "replace", side_effect=fail_during_output_publish
            ):
                with self.assertRaisesRegex(OSError, "injected publish failure"):
                    paper.build_paper_tables(root, output, allow_smoke=True)

            inventory_path = output / "paper-table-inventory.json"
            self.assertFalse(
                inventory_path.exists(),
                "inventory must not be visible after an interrupted output publish",
            )

            paper.build_paper_tables(root, output, allow_smoke=True)
            self.assertTrue(inventory_path.is_file())
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            declared = inventory["output_files_sha256"]
            self.assertEqual(len(declared), 18)
            self.assertIn("table5_scale_design.csv", declared)
            self.assertIn("table5_scale_design.md", declared)
            self.assertEqual(inventory["output_row_counts"]["table5_scale_design.csv"], 6)
            for name, digest in declared.items():
                declared_path = output / name
                self.assertTrue(declared_path.is_file(), name)
                self.assertEqual(_file_hash(declared_path), digest, name)

    def test_protocol_and_verification_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            protocol_path = root / "protocol_config.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["seed_root"] = 999
            _write_json(protocol_path, protocol)
            with self.assertRaisesRegex(paper.PaperTableError, "canonical protocol hash"):
                paper._load_inputs(root, allow_smoke=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            verification_path = root / "verification" / "verification_report.json"
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            verification["protocol_id"] = "tampered-id"
            _write_json(verification_path, verification)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["verification"]["file_sha256"] = _file_hash(verification_path)
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(paper.PaperTableError, "protocol_id differs"):
                paper._load_inputs(root, allow_smoke=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["verification"]["file"] = "../verification_report.json"
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(paper.PaperTableError, "path must be"):
                paper._load_inputs(root, allow_smoke=True)

    def test_failed_primary_value_suppresses_macro(self) -> None:
        frame = pd.DataFrame(
            [
                {"instance_index": 0, "preference_id": "A", "value": 1.0},
                {"instance_index": 0, "preference_id": "B", "value": 3.0},
                {"instance_index": 1, "preference_id": "A", "value": 2.0},
                {"instance_index": 1, "preference_id": "B", "value": None},
            ]
        )
        stats = paper._primary_instance_macro(
            frame,
            value_column="value",
            expected_preferences=["A", "B"],
            expected_instances=2,
        )
        self.assertIsNone(stats.mean)
        self.assertIsNone(stats.sd)
        self.assertEqual(stats.valid_n, 1)

        outcomes = paper._outcome_counts(
            pd.DataFrame(
                [
                    {
                        "instance_index": 0,
                        "technical_failure_rate": 1.0,
                        "missing_rate": 0.0,
                        "feasible_rate": 0.0,
                        "weighted_objective_normalized": None,
                    },
                    {
                        "instance_index": 1,
                        "technical_failure_rate": 0.0,
                        "missing_rate": 0.0,
                        "feasible_rate": 0.0,
                        "weighted_objective_normalized": None,
                    },
                    {
                        "instance_index": 2,
                        "technical_failure_rate": 0.0,
                        "missing_rate": 1.0,
                        "feasible_rate": 0.0,
                        "weighted_objective_normalized": None,
                    },
                ]
            )
        )
        self.assertEqual(outcomes["technical_failure_instances"], 1)
        self.assertEqual(outcomes["infeasible_instances"], 1)
        self.assertEqual(outcomes["missing_instances"], 1)
        self.assertEqual(outcomes["incomplete_quality_instances"], 3)

    def test_incomplete_quality_and_feasibility_cannot_leak_from_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            instance_path = root / "tables" / "instance_level_results.csv"
            summary_path = root / "tables" / "summary_by_preference.csv"
            instance = pd.read_csv(instance_path)
            summary = pd.read_csv(summary_path)
            instance_mask = (
                (instance["method"] == "ppo")
                & (instance["model_id"] == "Train-S")
                & (instance["test_scale"] == "Test-1")
            )
            summary_mask = (
                (summary["method"] == "ppo")
                & (summary["model_id"] == "Train-S")
                & (summary["test_scale"] == "Test-1")
            )
            instance.loc[instance_mask, list(paper.PAPER_QUALITY_FIELDS)] = None
            instance.loc[instance_mask, "feasible_rate"] = 0.0
            instance.loc[instance_mask, "all_restarts_feasible"] = False
            summary.loc[summary_mask, "n_instances_quality"] = 0
            # Deliberately retain the old finite primary means and the false 100% rate.
            instance.to_csv(instance_path, index=False, encoding="utf-8")
            summary.to_csv(summary_path, index=False, encoding="utf-8")
            _refresh_verified_table_hashes(
                root, "instance_level_results.csv", "summary_by_preference.csv"
            )
            with self.assertRaisesRegex(paper.PaperTableError, "cost_mean"):
                paper._load_inputs(root, allow_smoke=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            instance_path = root / "tables" / "instance_level_results.csv"
            instance = pd.read_csv(instance_path)
            mask = (
                (instance["method"] == "ppo")
                & (instance["model_id"] == "Train-S")
                & (instance["test_scale"] == "Test-1")
            )
            instance.loc[mask, "feasible_rate"] = 0.0
            instance.to_csv(instance_path, index=False, encoding="utf-8")
            _refresh_verified_table_hashes(root, "instance_level_results.csv")
            with self.assertRaisesRegex(paper.PaperTableError, "finite primary quality"):
                paper._load_inputs(root, allow_smoke=True)

    def test_heuristic_gap_is_gated_by_milp_optimal_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            instance_path = root / "tables" / "instance_level_results.csv"
            gaps_path = root / "tables" / "milp_gap_by_instance.csv"
            gap_summary_path = root / "tables" / "milp_gap_summary.csv"
            instance = pd.read_csv(instance_path)
            gaps = pd.read_csv(gaps_path)
            gap_summary = pd.read_csv(gap_summary_path)
            instance.loc[instance["method"] == "milp", "solver_status"] = "time_limit"
            gaps["milp_status"] = "time_limit"
            gaps["gap_percent"] = None
            gap_summary["optimal_reference_n"] = 0
            gap_summary["valid_n"] = 0
            gap_summary["mean_gap_percent"] = None
            gap_summary["sample_sd_gap_percent"] = None
            heuristic = gap_summary["method"] == "heuristic"
            gap_summary.loc[heuristic, "valid_n"] = 1
            gap_summary.loc[heuristic, "mean_gap_percent"] = 123.0
            instance.to_csv(instance_path, index=False, encoding="utf-8")
            gaps.to_csv(gaps_path, index=False, encoding="utf-8")
            gap_summary.to_csv(gap_summary_path, index=False, encoding="utf-8")
            _refresh_verified_table_hashes(
                root,
                "instance_level_results.csv",
                "milp_gap_by_instance.csv",
                "milp_gap_summary.csv",
            )
            with self.assertRaisesRegex(paper.PaperTableError, "valid Gap count"):
                paper._load_inputs(root, allow_smoke=True)

    def test_paired_primary_delta_must_match_instance_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            path = root / "tables" / "paired_ppo_vs_ga_summary.csv"
            summary = pd.read_csv(path)
            mask = (
                (summary["model_id"] == "Train-S")
                & (summary["test_scale"] == "Test-1")
            )
            summary.loc[mask, "paired_n"] = 0
            summary.to_csv(path, index=False, encoding="utf-8")
            _refresh_verified_table_hashes(root, "paired_ppo_vs_ga_summary.csv")
            with self.assertRaisesRegex(paper.PaperTableError, "paired_n"):
                paper._load_inputs(root, allow_smoke=True)

    def test_schedule_ledger_and_table_bindings_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_synthetic_smoke(root)
            ledger_path = root / "tables" / "cell_file_ledger.csv"
            ledger = pd.read_csv(ledger_path)
            ledger.loc[0, "cell_file_sha256"] = "b" * 64
            ledger.to_csv(ledger_path, index=False, encoding="utf-8")
            ledger_file_hash = _file_hash(ledger_path)

            verification_path = root / "verification" / "verification_report.json"
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            verification["table_files_sha256"]["cell_file_ledger.csv"] = ledger_file_hash
            _write_json(verification_path, verification)

            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tables"]["cell_file_ledger.csv"]["file_sha256"] = ledger_file_hash
            manifest["verification"]["file_sha256"] = _file_hash(verification_path)
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(paper.PaperTableError, "canonical cell-ledger hash"):
                paper._load_inputs(root, allow_smoke=True)

    def test_superseded_and_all_method_batches_are_disclosed_without_misattribution(self) -> None:
        manifest = {
            "evaluation_runs": [
                {
                    "methods": "all",
                    "models": "all",
                    "scales": "all",
                    "workers": 1,
                    "completed_cells": 15,
                    "wall_seconds": 3.0,
                    "superseded_for_timing": True,
                },
                {
                    "methods": ["ppo"],
                    "models": ["Train-S", "Train-L"],
                    "scales": ["Test-1", "Test-2"],
                    "workers": 2,
                    "completed_cells": 8,
                    "wall_seconds": 1.0,
                    "allow_concurrent_ppo": True,
                },
            ]
        }
        disclosure = paper._concurrency_scope(manifest, "ppo")
        self.assertIn("整批总完成数=15", disclosure)
        self.assertIn("该方法完成数=8", disclosure)
        self.assertIn("已排除superseded_for_timing历史批次=1", disclosure)
        self.assertIn("methods=all", disclosure)
        self.assertIn("methods=ppo", disclosure)
        self.assertNotIn("该方法完成数=15", disclosure)


if __name__ == "__main__":
    unittest.main()

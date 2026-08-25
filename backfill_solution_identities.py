from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from math import isclose
from pathlib import Path
from typing import Any

import pandas as pd

from hazardous_waste_model import pickup_node
from run_experiments import (
    ROOT,
    _artifact_inventory,
    _plot_results,
    _solution_record,
    _write_json,
    _write_result_tables,
    _write_summary,
)
from sample_params import params_from_json_data
from src.reproducibility import (
    deserialize_plan,
    file_sha256,
    plan_sha256,
)
from src.solution_utils import evaluate_solution, route_plan_to_solution


UPDATE_TYPE = "solution-identity-backfill-v1"
SOURCE_FILES = (
    "backfill_solution_identities.py",
    "hazardous_waste_model.py",
    "run_experiments.py",
    "sample_params.py",
    "src/reproducibility.py",
    "src/solution_utils.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _solution_file(
    row: pd.Series,
    manifest: dict[str, Any],
) -> str:
    method = str(row["method"])
    instance_id = str(row["instance_id"])
    preference = str(row["preference"])
    if method == "PPO-Transformer":
        restart = int(row["restart"])
        entry = next(
            item
            for item in manifest["evaluations"]
            if item["instance_id"] == instance_id
            and item["method"] == method
            and f"{float(item['preference'][0]):.2f}_{float(item['preference'][1]):.2f}"
            == preference
            and int(item["restart"]) == restart
        )
        return str(entry["solution_file"]).replace("\\", "/")

    scale = str(row["scale"])
    instance_index = int(row["instance"])
    suffix = {
        "Heuristic": "heuristic",
        "GA": "ga",
        "MILP": "milp",
    }[method]
    return f"solutions/{scale}_{instance_index}_{preference}_{suffix}.json"


def _display_routes_to_plan(
    routes: dict[str, list[str]],
    params: Any,
) -> dict[tuple[str, int], list[str]]:
    def internal_node(node: str) -> str:
        if node in params.facilities:
            return node
        owner, waste = node.split(":", 1)
        candidate = pickup_node(owner, waste)
        if candidate not in params.pickup_nodes:
            raise ValueError(f"unknown displayed pickup node: {node}")
        return candidate

    plan: dict[tuple[str, int], list[str]] = {}
    for raw_key, route in routes.items():
        key = ast.literal_eval(raw_key)
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"invalid saved route key: {raw_key}")
        plan[str(key[0]), int(key[1])] = [internal_node(node) for node in route]
    return plan


def backfill_solution_identities(run_dir: Path, apply: bool = False) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _load_json(run_dir / "manifest.json")
    if any(
        update.get("type") == UPDATE_TYPE
        for update in manifest.get("posthoc_updates", [])
    ):
        raise RuntimeError("solution identities have already been backfilled")

    results = pd.read_csv(run_dir / "results.csv")
    for column in ("solution_file", "result_plan_sha256"):
        if column not in results:
            results[column] = pd.Series([None] * len(results), dtype="object")
        else:
            results[column] = results[column].astype("object")

    instance_entries = {
        item["instance_id"]: item
        for item in manifest["instances"]
    }
    params_by_instance = {
        instance_id: params_from_json_data(
            _load_json(run_dir / entry["instance_file"])
        )
        for instance_id, entry in instance_entries.items()
    }
    updates: list[dict[str, Any]] = []
    pending_milp_records: list[tuple[Path, dict[str, Any]]] = []

    for row_index, row in results.iterrows():
        instance_id = str(row["instance_id"])
        params = params_by_instance[instance_id]
        instance_entry = instance_entries[instance_id]
        preference = (
            float(row["preference_cost"]),
            float(row["preference_risk"]),
        )
        objective_refs = (
            float(instance_entry["cost_ref"]),
            float(instance_entry["risk_ref"]),
        )
        relative_solution_file = _solution_file(row, manifest)
        solution_path = run_dir / relative_solution_file
        saved_record = _load_json(solution_path)
        if "plan" in saved_record:
            plan = deserialize_plan(saved_record["plan"])
        else:
            plan = _display_routes_to_plan(saved_record["routes"], params)

        rebuilt_solution = route_plan_to_solution(params, plan)
        metrics = evaluate_solution(
            params,
            rebuilt_solution,
            preference,
            objective_refs=objective_refs,
        )
        if not metrics["feasible"]:
            raise RuntimeError(
                f"saved plan is infeasible for row {row_index}: {metrics['violations']}"
            )
        for column in ("cost", "risk", "weighted_objective_normalized"):
            if not isclose(
                float(row[column]),
                float(metrics[column]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"saved plan metric mismatch for row {row_index}/{column}: "
                    f"csv={row[column]}, rebuilt={metrics[column]}"
                )

        result_plan_hash = plan_sha256(plan)
        results.at[row_index, "solution_file"] = relative_solution_file
        results.at[row_index, "result_plan_sha256"] = result_plan_hash
        if "plan" not in saved_record:
            pending_milp_records.append(
                (solution_path, _solution_record(rebuilt_solution, metrics))
            )
        updates.append(
            {
                "row": int(row_index),
                "instance_id": instance_id,
                "method": str(row["method"]),
                "preference": list(preference),
                "restart": int(row["restart"]),
                "solution_file": relative_solution_file,
                "result_plan_sha256": result_plan_hash,
                "milp_plan_reconstructed": "plan" not in saved_record,
            }
        )

    report = {
        "run_id": manifest["run_id"],
        "type": UPDATE_TYPE,
        "applied": apply,
        "cells": updates,
    }
    if not apply:
        return report

    for path, record in pending_milp_records:
        _write_json(path, record)
    results = _write_result_tables(run_dir, results)
    _write_summary(run_dir, results)
    _plot_results(run_dir, results)

    solution_artifacts = _artifact_inventory(run_dir, "solutions")
    solution_hash_by_file = {
        artifact["file"]: artifact["sha256"]
        for artifact in solution_artifacts
    }
    plan_hash_by_cell = {
        (
            update["instance_id"],
            tuple(update["preference"]),
            update["restart"],
        ): update["result_plan_sha256"]
        for update in updates
        if update["method"] == "PPO-Transformer"
    }
    for evaluation in manifest["evaluations"]:
        relative = Path(evaluation["solution_file"]).as_posix()
        evaluation["solution_sha256"] = solution_hash_by_file[relative]
        evaluation["result_plan_sha256"] = plan_hash_by_cell[
            (
                evaluation["instance_id"],
                tuple(float(value) for value in evaluation["preference"]),
                int(evaluation["restart"]),
            )
        ]

    applied_at = datetime.now().astimezone().isoformat()
    update_record = {
        **report,
        "applied": True,
        "applied_at": applied_at,
        "milp_plans_reconstructed": len(pending_milp_records),
        "source_files_sha256": {
            relative: file_sha256(ROOT / relative)
            for relative in SOURCE_FILES
        },
    }
    manifest.setdefault("posthoc_updates", []).append(update_record)
    manifest["last_updated_at"] = applied_at
    manifest["solution_artifacts"] = solution_artifacts
    manifest["trace_artifacts"] = _artifact_inventory(run_dir, "traces")
    manifest["artifacts"] = {
        "results.csv": file_sha256(run_dir / "results.csv"),
        "results_with_gap.csv": file_sha256(
            run_dir / "results_with_gap.csv"
        ),
        "summary.md": file_sha256(run_dir / "summary.md"),
    }
    _write_json(run_dir / "manifest.json", manifest)
    return update_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill solution paths and canonical plan hashes in a saved run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            backfill_solution_identities(args.run_dir, apply=args.apply),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

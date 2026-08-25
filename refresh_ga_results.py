from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from run_experiments import (
    ROOT,
    _artifact_inventory,
    _plot_results,
    _row,
    _solution_record,
    _with_normalized_objective,
    _write_json,
    _write_result_tables,
    _write_summary,
    run_ga,
)
from sample_params import params_from_json_data
from src.reproducibility import derive_seed, file_sha256, plan_sha256
from src.solution_utils import evaluate_solution


SOURCE_FILES = (
    "hazardous_waste_model.py",
    "sample_params.py",
    "run_experiments.py",
    "replay_experiment.py",
    "refresh_ga_results.py",
    "configs/algorithm_config.json",
    "configs/network_config.json",
    "src/config.py",
    "src/heuristics.py",
    "src/instance_generator.py",
    "src/operators.py",
    "src/ppo_improver.py",
    "src/reproducibility.py",
    "src/solution_utils.py",
    "遗传算法/genetic_algorithm.py",
)
UPDATE_TYPE = "ga-decoder-shared-repair-v1"
UPDATE_REASON = (
    "refresh GA cells after replacing the fixed-first-facility decoder "
    "with deterministic shared repair"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_ga_results(
    run_dir: Path,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _load_json(run_dir / "manifest.json")
    already_applied = any(
        update.get("type") == UPDATE_TYPE
        or update.get("reason") == UPDATE_REASON
        for update in manifest.get("posthoc_updates", [])
    )
    if apply and already_applied and not force:
        raise RuntimeError(
            "this GA refresh is already recorded in the manifest; "
            "use --force to append a superseding deterministic rerun"
        )
    algorithm_config = _load_json(run_dir / "algorithm_config.json")
    results = pd.read_csv(run_dir / "results.csv")
    for text_column in (
        "fallback_reason",
        "solution_file",
        "result_plan_sha256",
    ):
        if text_column not in results:
            results[text_column] = pd.Series(
                [None] * len(results),
                dtype="object",
            )
        else:
            results[text_column] = results[text_column].astype("object")
    updates: list[dict[str, Any]] = []
    pending_solutions: list[tuple[Path, dict[str, Any]]] = []

    for instance_entry in manifest["instances"]:
        instance_id = str(instance_entry["instance_id"])
        scale = str(instance_entry["scale"])
        instance_index = int(instance_entry["index"])
        params = params_from_json_data(
            _load_json(run_dir / instance_entry["instance_file"])
        )
        objective_refs = (
            float(instance_entry["cost_ref"]),
            float(instance_entry["risk_ref"]),
        )

        for raw_preference in manifest["preferences"]:
            preference = (float(raw_preference[0]), float(raw_preference[1]))
            preference_label = f"{preference[0]:.2f}_{preference[1]:.2f}"
            mask = (
                (results["instance_id"] == instance_id)
                & (results["preference"] == preference_label)
                & (results["method"] == "GA")
            )
            matching = results.index[mask].tolist()
            if len(matching) != 1:
                raise ValueError(
                    f"expected one GA row for {instance_id}/{preference_label}, "
                    f"found {len(matching)}"
                )
            row_index = matching[0]
            saved_seed = int(results.at[row_index, "evaluation_seed"])
            expected_seed = derive_seed(
                int(manifest["master_seed"]),
                "evaluation",
                "GA",
                instance_id,
                preference_label,
                bits=32,
            )
            if saved_seed != expected_seed:
                raise ValueError(
                    f"GA seed mismatch for {instance_id}/{preference_label}: "
                    f"saved={saved_seed}, expected={expected_seed}"
                )

            optimizer_params = _with_normalized_objective(
                params,
                preference,
                objective_refs,
            )
            started = time.perf_counter()
            ga_result = run_ga(
                optimizer_params,
                algorithm_config["ga"][scale],
                saved_seed,
            )
            runtime = time.perf_counter() - started
            metrics = evaluate_solution(
                params,
                ga_result.best_solution,
                preference,
                objective_refs=objective_refs,
            )
            if not metrics["feasible"]:
                raise RuntimeError(
                    f"repaired GA remained infeasible for {instance_id}/{preference_label}: "
                    f"{metrics['violations']}"
                )
            result_plan_hash = plan_sha256(ga_result.best_solution["plan"])
            solution_file = (
                f"solutions/{scale}_{instance_index}_{preference_label}_ga.json"
            )
            pending_solutions.append(
                (
                    run_dir / solution_file,
                    _solution_record(ga_result.best_solution, metrics),
                )
            )

            identity = {
                "run_id": manifest["run_id"],
                "instance_id": instance_id,
                "instance_seed": int(instance_entry["instance_seed"]),
                "initial_solution_seed": int(
                    instance_entry["initial_solution_seed"]
                ),
                "initial_solution_sha256": instance_entry[
                    "initial_solution_sha256"
                ],
                "preference_cost": preference[0],
                "preference_risk": preference[1],
                "solution_file": solution_file,
                "result_plan_sha256": result_plan_hash,
            }
            refreshed_row = _row(
                scale,
                instance_index,
                preference_label,
                "GA",
                metrics,
                runtime,
                restart=0,
                evaluation_seed=saved_seed,
                **identity,
            )
            for column, value in refreshed_row.items():
                results.at[row_index, column] = value

            updates.append(
                {
                    "instance_id": instance_id,
                    "preference": list(preference),
                    "evaluation_seed": saved_seed,
                    "solution_file": solution_file,
                    "solution_plan_sha256": result_plan_hash,
                    "runtime_seconds": runtime,
                    "cost": metrics["cost"],
                    "risk": metrics["risk"],
                    "weighted_objective_normalized": metrics[
                        "weighted_objective_normalized"
                    ],
                }
            )

    report = {
        "run_id": manifest["run_id"],
        "type": UPDATE_TYPE,
        "applied": apply,
        "reason": UPDATE_REASON,
        "cells": updates,
    }
    if not apply:
        return report

    for path, payload in pending_solutions:
        _write_json(path, payload)
    results = _write_result_tables(run_dir, results)
    _write_summary(run_dir, results)
    _plot_results(run_dir, results)

    update_record = {
        **report,
        "applied": True,
        "applied_at": datetime.now().astimezone().isoformat(),
        "source_files_sha256": {
            relative: file_sha256(ROOT / relative)
            for relative in SOURCE_FILES
        },
    }
    posthoc_updates = manifest.setdefault("posthoc_updates", [])
    if force:
        previous_indexes = [
            index
            for index, update in enumerate(posthoc_updates)
            if update.get("type") == UPDATE_TYPE
            or update.get("reason") == UPDATE_REASON
        ]
        if previous_indexes:
            previous_index = previous_indexes[-1]
            previous_update = posthoc_updates[previous_index]
            update_record["supersedes_posthoc_update"] = {
                "index": previous_index,
                "type": previous_update.get("type"),
                "applied_at": previous_update.get("applied_at"),
            }
    posthoc_updates.append(update_record)
    manifest["last_updated_at"] = update_record["applied_at"]
    manifest["solution_artifacts"] = _artifact_inventory(run_dir, "solutions")
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
        description="Recompute only GA cells in a saved experiment run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write refreshed GA rows, solutions, summaries, plots, and manifest metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Append a deterministic rerun that supersedes the previous refresh.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            refresh_ga_results(
                args.run_dir,
                apply=args.apply,
                force=args.force,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

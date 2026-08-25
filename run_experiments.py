from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple

import matplotlib.pyplot as plt
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent
GA_DIR = ROOT / "遗传算法"
if str(GA_DIR) not in sys.path:
    sys.path.insert(0, str(GA_DIR))

EXPERIMENT_SOURCE_FILES = (
    "backfill_solution_identities.py",
    "hazardous_waste_model.py",
    "sample_params.py",
    "refresh_ga_results.py",
    "run_experiments.py",
    "replay_experiment.py",
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

from genetic_algorithm import ClassicGeneticAlgorithm, GAConfig
from hazardous_waste_model import solve_with_milp
from sample_params import params_to_json_data
from src.config import ensure_dir, load_algorithm_config, load_model_config, load_network_config
from src.heuristics import build_greedy_initial_plan
from src.instance_generator import generate_random_params
from src.ppo_improver import PPOImprover
from src.reproducibility import (
    derive_seed,
    environment_snapshot,
    file_sha256,
    git_status_snapshot,
    json_sha256,
    plan_sha256,
    plan_to_canonical_data,
)
from src.solution_utils import evaluate_solution, route_plan_to_solution


@dataclass
class ExperimentRunResult:
    """Standard result returned by a learned-method experiment adapter."""

    solution: Dict[str, Any]
    inference_seconds: float
    metadata: Dict[str, Any] | None = None


class GATExperimentAdapter(Protocol):
    """Contract between this experiment entry point and the future GAT method.

    The GAT implementation should own graph construction, model training/loading,
    checkpoint management, and inference. The experiment entry point only passes
    problem data in and evaluates the returned solution with the common evaluator.
    """

    def prepare(
        self,
        base_params: Any,
        initial_solution: Dict[str, Any],
        objective_refs: Tuple[float, float],
        seed: int,
        artifact_dir: Path,
    ) -> float:
        """Train or load the model and return preparation time in seconds."""
        ...

    def solve(
        self,
        params: Any,
        initial_solution: Dict[str, Any],
        preference: Tuple[float, float],
        objective_refs: Tuple[float, float],
        seed: int,
    ) -> ExperimentRunResult:
        """Return a complete candidate solution for one preference vector."""
        ...


def build_gat_adapter(
    algorithm_config: Dict[str, Any],
    scale: str,
) -> GATExperimentAdapter | None:
    """Future GAT registration point; disabled until the GAT code is implemented.

    TODO(GAT): import the concrete adapter here and return an instance. Keeping
    this hook optional lets the MILP/GA baselines run while the GAT implementation
    is being developed. Legacy learned-method compatibility is intentionally omitted.
    """
    del algorithm_config, scale
    return None


class PPOTransformerExperimentAdapter:
    """Executable, repaired PPO-Transformer baseline behind the common runner seam."""

    def __init__(
        self,
        algorithm_config: Dict[str, Any],
        network_config: Dict[str, Any],
        train_seed: int,
    ) -> None:
        self.algorithm_config = algorithm_config
        self.network_config = network_config
        self.train_seed = train_seed
        self.improver: PPOImprover | None = None
        self.initial_plan = None
        self.checkpoint_path: Path | None = None

    def prepare(
        self,
        base_params: Any,
        initial_solution: Dict[str, Any],
        objective_refs: Tuple[float, float],
        seed: int,
        artifact_dir: Path,
    ) -> float:
        del objective_refs, seed
        self.initial_plan = {
            key: list(route) for key, route in initial_solution["plan"].items()
        }
        self.improver = PPOImprover(
            base_params,
            self.algorithm_config,
            self.network_config,
            seed=self.train_seed,
        )
        self.checkpoint_path = artifact_dir / "checkpoint.pt"
        started = time.perf_counter()
        history = self.improver.train(save_path=self.checkpoint_path)
        if self.improver.device.type == "cuda":
            import torch

            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        pd.DataFrame(history).to_csv(
            artifact_dir / "training_history.csv",
            index_label="iteration",
            encoding="utf-8-sig",
        )
        _write_json(artifact_dir / "network_config.json", self.network_config)
        return elapsed

    def solve(
        self,
        params: Any,
        initial_solution: Dict[str, Any],
        preference: Tuple[float, float],
        objective_refs: Tuple[float, float],
        seed: int,
    ) -> ExperimentRunResult:
        del params, initial_solution, objective_refs
        if self.improver is None or self.initial_plan is None:
            raise RuntimeError("PPO-Transformer adapter must be prepared before solve")
        if self.improver.device.type == "cuda":
            import torch

            torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.improver.improve(
            preference,
            seed=seed,
            initial_plan=self.initial_plan,
        )
        if self.improver.device.type == "cuda":
            import torch

            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return ExperimentRunResult(
            solution=result.solution,
            inference_seconds=elapsed,
            metadata={"trace": result.trace, "evaluation_seed": seed},
        )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_inventory(run_dir: Path, folder: str) -> List[Dict[str, str]]:
    directory = run_dir / folder
    return [
        {
            "file": path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _normalize_result_dtypes(results: pd.DataFrame) -> pd.DataFrame:
    normalized = results.copy()
    for column in (
        "evaluation_seed",
        "instance_seed",
        "initial_solution_seed",
        "training_seed",
    ):
        if column in normalized:
            normalized[column] = pd.array(
                pd.to_numeric(normalized[column], errors="raise"),
                dtype="Int64",
            )
    return normalized


def _write_result_tables(run_dir: Path, results: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_result_dtypes(results)
    normalized.to_csv(run_dir / "results.csv", index=False, encoding="utf-8-sig")
    _add_gaps(normalized).to_csv(
        run_dir / "results_with_gap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return normalized


def _with_preference(params, preference: Tuple[float, float]):
    return replace(params, cost_weight=float(preference[0]), risk_weight=float(preference[1]))


def _with_normalized_objective(params, preference: Tuple[float, float], objective_refs: Tuple[float, float]):
    return replace(
        params,
        cost_weight=float(preference[0]) / max(float(objective_refs[0]), 1e-9),
        risk_weight=float(preference[1]) / max(float(objective_refs[1]), 1e-9),
    )


def run_ga(params, ga_cfg: Dict[str, Any], seed: int):
    config = GAConfig(random_seed=seed, **ga_cfg)
    solver = ClassicGeneticAlgorithm(params, config)
    return solver.run()


def run_experiments(quick: bool = False) -> Path:
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    algorithm_config = load_algorithm_config()
    network_config = load_network_config()
    master_seed = int(algorithm_config["random_seed"])

    if quick:
        algorithm_config["small_instance_count"] = 1
        algorithm_config["large_instance_count"] = 1
        algorithm_config["milp_time_limit"] = 5.0
        algorithm_config["preferences"] = [[0.5, 0.5], [0.0, 1.0]]
        algorithm_config["ppo"]["train_iterations"] = 2
        algorithm_config["ppo"]["num_parallel_episodes"] = 4
        algorithm_config["ppo"]["episode_steps"] = 4
        algorithm_config["ppo"]["update_epochs"] = 2
        algorithm_config["ppo"]["eval_steps"] = 8
        algorithm_config["ppo"]["eval_candidate_samples"] = 4
        algorithm_config["ppo"]["evaluation_restarts"] = 1
        algorithm_config["ga"]["small"]["generations"] = 5
        algorithm_config["ga"]["large"]["generations"] = 2
        algorithm_config["ga"]["small"]["population_size"] = 8
        algorithm_config["ga"]["large"]["population_size"] = 6
        network_config["embedding_dim"] = 32
        network_config["attention_heads"] = 4
        network_config["ff_hidden_dim"] = 64

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ensure_dir(ROOT / algorithm_config["output_dir"] / run_id)
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "instances")
    ensure_dir(run_dir / "initial_solutions")
    ensure_dir(run_dir / "solutions")
    ensure_dir(run_dir / "ppo_transformer")
    ensure_dir(run_dir / "traces")

    _write_json(run_dir / "algorithm_config.json", algorithm_config)
    _write_json(run_dir / "network_config.json", network_config)

    manifest: Dict[str, Any] = {
        "schema_version": "hazardous-waste-experiment-v2",
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "quick_mode": quick,
        "command": " ".join(sys.argv),
        "master_seed": master_seed,
        "seed_scheme": "sha256-seed-v1",
        "training_scope": "instance-specific",
        "preferences": algorithm_config["preferences"],
        "algorithm_config_sha256": json_sha256(algorithm_config),
        "network_config_sha256": json_sha256(network_config),
        "source_files_sha256": {
            relative: file_sha256(ROOT / relative)
            for relative in EXPERIMENT_SOURCE_FILES
        },
        "git": git_status_snapshot(ROOT),
        "environment": environment_snapshot(),
        "instances": [],
        "training": [],
        "evaluations": [],
    }
    _write_json(run_dir / "manifest.json", manifest)

    rows: List[Dict[str, Any]] = []
    preferences = [tuple(item) for item in algorithm_config["preferences"]]

    for scale in ["small", "large"]:
        model_config = load_model_config(scale)
        if quick and scale == "large":
            model_config["producers_count"] = 3
            model_config["waste_types_count"] = 2
            model_config["facilities_count"] = 2
            model_config["vehicles_count"] = 3
            model_config["periods_count"] = 3
        instance_count = int(algorithm_config[f"{scale}_instance_count"])
        _write_json(run_dir / f"model_config_{scale}.json", model_config)
        for instance_idx in range(instance_count):
            instance_id = f"{scale}_{instance_idx}"
            print(f"[{scale}] instance {instance_idx} start", flush=True)
            instance_seed = derive_seed(master_seed, "instance", scale, instance_idx, bits=32)
            initial_seed = derive_seed(master_seed, "initial_solution", instance_id, bits=32)
            train_seed = derive_seed(master_seed, "training", "ppo_transformer", instance_id, bits=32)
            base_params = generate_random_params(model_config, instance_seed)
            instance_path = run_dir / "instances" / f"{instance_id}.json"
            _write_json(instance_path, params_to_json_data(base_params))

            initial_started = time.perf_counter()
            ref_plan = build_greedy_initial_plan(base_params, seed=initial_seed)
            ref_solution = route_plan_to_solution(base_params, ref_plan)
            initialization_seconds = time.perf_counter() - initial_started
            ref_metrics = evaluate_solution(base_params, ref_solution, (0.5, 0.5))
            if not ref_metrics["feasible"]:
                raise RuntimeError(f"initial solution is infeasible: {ref_metrics['violations']}")
            objective_refs = (max(ref_metrics["cost"], 1e-9), max(ref_metrics["risk"], 1e-9))

            initial_hash = plan_sha256(ref_plan)
            initial_path = run_dir / "initial_solutions" / f"{instance_id}.json"
            _write_json(
                initial_path,
                {
                    "schema": "initial-solution-v1",
                    "instance_id": instance_id,
                    "seed": initial_seed,
                    "plan": plan_to_canonical_data(ref_plan),
                    "plan_sha256": initial_hash,
                    "objective_refs": {"cost": objective_refs[0], "risk": objective_refs[1]},
                    "metrics": ref_metrics,
                },
            )

            ppo_dir = ensure_dir(run_dir / "ppo_transformer" / instance_id)
            ppo_adapter = PPOTransformerExperimentAdapter(
                algorithm_config,
                network_config,
                train_seed=train_seed,
            )
            print(f"[{scale}] instance {instance_idx} train PPO-Transformer", flush=True)
            ppo_train_time = ppo_adapter.prepare(
                base_params,
                ref_solution,
                objective_refs,
                train_seed,
                ppo_dir,
            )
            checkpoint_hash = file_sha256(ppo_adapter.checkpoint_path) if ppo_adapter.checkpoint_path else None

            instance_manifest = {
                "instance_id": instance_id,
                "scale": scale,
                "index": instance_idx,
                "dimensions": {
                    "producers": len(base_params.producers),
                    "waste_types": len(base_params.waste_types),
                    "facilities": len(base_params.facilities),
                    "vehicles": len(base_params.vehicles),
                    "periods": len(base_params.periods),
                },
                "instance_seed": instance_seed,
                "instance_file": str(instance_path.relative_to(run_dir)),
                "instance_sha256": file_sha256(instance_path),
                "initial_solution_seed": initial_seed,
                "initial_solution_file": str(initial_path.relative_to(run_dir)),
                "initial_solution_sha256": initial_hash,
                "initialization_seconds": initialization_seconds,
                "cost_ref": objective_refs[0],
                "risk_ref": objective_refs[1],
            }
            manifest["instances"].append(instance_manifest)
            manifest["training"].append(
                {
                    "method": "PPO-Transformer",
                    "instance_id": instance_id,
                    "training_scope": "instance-specific",
                    "train_seed": train_seed,
                    "checkpoint_file": str(ppo_adapter.checkpoint_path.relative_to(run_dir)),
                    "checkpoint_sha256": checkpoint_hash,
                    "training_seconds": ppo_train_time,
                }
            )
            _write_json(run_dir / "manifest.json", manifest)

            for preference in preferences:
                print(f"[{scale}] instance {instance_idx} preference {preference}", flush=True)
                params = _with_preference(base_params, preference)
                optimizer_params = _with_normalized_objective(base_params, preference, objective_refs)
                pref_label = f"{preference[0]:.2f}_{preference[1]:.2f}"
                identity = {
                    "run_id": run_id,
                    "instance_id": instance_id,
                    "instance_seed": instance_seed,
                    "initial_solution_seed": initial_seed,
                    "initial_solution_sha256": initial_hash,
                    "preference_cost": preference[0],
                    "preference_risk": preference[1],
                }

                print(f"[{scale}] instance {instance_idx} run heuristic", flush=True)
                heuristic_solution = ref_solution
                heuristic_eval = evaluate_solution(params, heuristic_solution, preference, objective_refs=objective_refs)
                heuristic_solution_name = f"{scale}_{instance_idx}_{pref_label}_heuristic.json"
                rows.append(_row(
                    scale, instance_idx, pref_label, "Heuristic", heuristic_eval,
                    initialization_seconds, restart=0, evaluation_seed=initial_seed,
                    solution_file=f"solutions/{heuristic_solution_name}",
                    result_plan_sha256=plan_sha256(heuristic_solution["plan"]),
                    **identity,
                ))
                _write_json(
                    run_dir / "solutions" / heuristic_solution_name,
                    _solution_record(heuristic_solution, heuristic_eval),
                )

                start = time.perf_counter()
                print(f"[{scale}] instance {instance_idx} run GA", flush=True)
                ga_seed = derive_seed(master_seed, "evaluation", "GA", instance_id, pref_label, bits=32)
                ga_result = run_ga(optimizer_params, algorithm_config["ga"][scale], ga_seed)
                ga_time = time.perf_counter() - start
                ga_solution = ga_result.best_solution
                ga_eval = evaluate_solution(params, ga_solution, preference, objective_refs=objective_refs)
                ga_fallback_reason = ""
                if not ga_eval["feasible"]:
                    ga_fallback_reason = "ga_infeasible_fallback_to_heuristic"
                    ga_solution = heuristic_solution
                    ga_eval = evaluate_solution(params, ga_solution, preference, objective_refs=objective_refs)
                ga_solution_name = f"{scale}_{instance_idx}_{pref_label}_ga.json"
                rows.append(_row(
                    scale, instance_idx, pref_label, "GA", ga_eval, ga_time,
                    fallback_reason=ga_fallback_reason, restart=0, evaluation_seed=ga_seed,
                    solution_file=f"solutions/{ga_solution_name}",
                    result_plan_sha256=(
                        plan_sha256(ga_solution["plan"])
                        if "plan" in ga_solution else None
                    ),
                    **identity,
                ))
                _write_json(
                    run_dir / "solutions" / ga_solution_name,
                    _solution_record(ga_solution, ga_eval),
                )

                for restart in range(int(algorithm_config["ppo"].get("evaluation_restarts", 1))):
                    eval_seed = derive_seed(
                        master_seed,
                        "evaluation",
                        "PPO-Transformer",
                        instance_id,
                        preference[0],
                        preference[1],
                        restart,
                        bits=32,
                    )
                    print(
                        f"[{scale}] instance {instance_idx} run PPO-Transformer restart {restart}",
                        flush=True,
                    )
                    ppo_result = ppo_adapter.solve(
                        params,
                        ref_solution,
                        preference,
                        objective_refs,
                        eval_seed,
                    )
                    ppo_eval = evaluate_solution(
                        params,
                        ppo_result.solution,
                        preference,
                        objective_refs=objective_refs,
                    )
                    solution_name = f"{instance_id}_{pref_label}_ppo_restart_{restart}.json"
                    solution_path = run_dir / "solutions" / solution_name
                    trace_path = run_dir / "traces" / instance_id / pref_label / f"restart_{restart}.json"
                    row = _row(
                        scale,
                        instance_idx,
                        pref_label,
                        "PPO-Transformer",
                        ppo_eval,
                        ppo_result.inference_seconds,
                        restart=restart,
                        evaluation_seed=eval_seed,
                        training_seed=train_seed,
                        checkpoint_sha256=checkpoint_hash,
                        preparation_seconds=ppo_train_time,
                        training_scope="instance-specific",
                        solution_file=f"solutions/{solution_name}",
                        result_plan_sha256=plan_sha256(ppo_result.solution["plan"]),
                        **identity,
                    )
                    rows.append(row)
                    _write_json(
                        solution_path,
                        _solution_record(ppo_result.solution, ppo_eval),
                    )
                    _write_json(
                        trace_path,
                        {
                            "instance_id": instance_id,
                            "preference": list(preference),
                            "restart": restart,
                            "evaluation_seed": eval_seed,
                            "initial_solution_sha256": initial_hash,
                            "checkpoint_sha256": checkpoint_hash,
                            "trace": (ppo_result.metadata or {}).get("trace", []),
                        },
                    )
                    manifest["evaluations"].append(
                        {
                            "instance_id": instance_id,
                            "method": "PPO-Transformer",
                            "preference": list(preference),
                            "restart": restart,
                            "evaluation_seed": eval_seed,
                            "trace_file": str(trace_path.relative_to(run_dir)),
                            "solution_file": f"solutions/{solution_name}",
                            "trace_sha256": file_sha256(trace_path),
                            "solution_sha256": file_sha256(solution_path),
                            "result_plan_sha256": plan_sha256(ppo_result.solution["plan"]),
                            "checkpoint_sha256": checkpoint_hash,
                        }
                    )

                if scale == "small":
                    start = time.perf_counter()
                    print(f"[{scale}] instance {instance_idx} run MILP", flush=True)
                    milp_result = solve_with_milp(optimizer_params, time_limit=float(algorithm_config["milp_time_limit"]))
                    milp_time = time.perf_counter() - start
                    if milp_result.solution:
                        milp_eval = evaluate_solution(params, milp_result.solution, preference, objective_refs=objective_refs)
                        milp_solution_name = f"{scale}_{instance_idx}_{pref_label}_milp.json"
                        row = _row(
                            scale, instance_idx, pref_label, "MILP", milp_eval, milp_time,
                            restart=0, evaluation_seed=None,
                            solution_file=f"solutions/{milp_solution_name}",
                            result_plan_sha256=None,
                            **identity,
                        )
                        row["solver_status"] = milp_result.status
                        rows.append(row)
                        _write_json(
                            run_dir / "solutions" / milp_solution_name,
                            _solution_record(milp_result.solution, milp_eval),
                        )
                # Persist completed cells immediately so an interrupted long run
                # can be inspected or resumed without losing prior preferences.
                partial_results = pd.DataFrame(rows)
                _write_result_tables(run_dir, partial_results)
                _write_json(run_dir / "manifest.json", manifest)

    results = _write_result_tables(run_dir, pd.DataFrame(rows))
    _write_summary(run_dir, results)
    _plot_results(run_dir, results)
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["artifacts"] = {
        "results.csv": file_sha256(run_dir / "results.csv"),
        "results_with_gap.csv": file_sha256(run_dir / "results_with_gap.csv"),
        "summary.md": file_sha256(run_dir / "summary.md"),
    }
    manifest["solution_artifacts"] = _artifact_inventory(run_dir, "solutions")
    manifest["trace_artifacts"] = _artifact_inventory(run_dir, "traces")
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _row(
    scale: str,
    instance_idx: int,
    pref_label: str,
    method: str,
    metrics: Dict[str, Any],
    runtime: float,
    fallback_reason: str = "",
    restart: int = 0,
    evaluation_seed: int | None = None,
    **metadata: Any,
) -> Dict[str, Any]:
    row = {
        "scale": scale,
        "instance": instance_idx,
        "preference": pref_label,
        "method": method,
        "cost": metrics["cost"],
        "risk": metrics["risk"],
        "normalized_cost": metrics["normalized_cost"],
        "normalized_risk": metrics["normalized_risk"],
        "weighted_objective_raw": metrics["weighted_objective_raw"],
        "weighted_objective_normalized": metrics["weighted_objective_normalized"],
        "weighted_objective": metrics["weighted_objective"],
        "cost_ref": metrics["cost_ref"],
        "risk_ref": metrics["risk_ref"],
        "fixed_cost": metrics["fixed_cost"],
        "distance_cost": metrics["distance_cost"],
        "processing_cost": metrics["processing_cost"],
        "transport_risk": metrics["transport_risk"],
        "coload_risk": metrics["coload_risk"],
        "producer_inventory_risk": metrics["producer_inventory_risk"],
        "facility_inventory_risk": metrics["facility_inventory_risk"],
        "feasible": metrics["feasible"],
        "violation_count": len(metrics["violations"]),
        "fallback_used": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "runtime_seconds": runtime,
        "restart": restart,
        "evaluation_seed": evaluation_seed,
    }
    row.update(metadata)
    return row


def _solution_record(solution: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "routes": {str(k): v for k, v in solution.get("routes", {}).items()},
        "summary": solution.get("summary", {}),
        "metrics": metrics,
    }
    if "plan" in solution:
        record["plan"] = plan_to_canonical_data(solution["plan"])
        record["plan_sha256"] = plan_sha256(solution["plan"])
    return record


def _add_gaps(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["gap_percent"] = None
    out["gap_baseline"] = None
    milp_mask = out["method"] == "MILP"
    if "solver_status" in out.columns:
        milp_mask = milp_mask & (out["solver_status"].fillna("") == "optimal")
    milp = out[milp_mask][["scale", "instance", "preference", "weighted_objective_normalized"]]
    for _, row in milp.iterrows():
        mask = (out["scale"] == row["scale"]) & (out["instance"] == row["instance"]) & (out["preference"] == row["preference"])
        best = float(row["weighted_objective_normalized"])
        out.loc[mask, "gap_percent"] = (out.loc[mask, "weighted_objective_normalized"] - best) / max(abs(best), 1e-9) * 100.0
        out.loc[mask, "gap_baseline"] = "MILP"
    return out


def _write_summary(run_dir: Path, results: pd.DataFrame) -> None:
    gap = _add_gaps(results)
    method_summary = results.groupby(["scale", "method"]).agg(
        weighted_objective_normalized_mean=("weighted_objective_normalized", "mean"),
        weighted_objective_raw_mean=("weighted_objective_raw", "mean"),
        cost_mean=("cost", "mean"),
        risk_mean=("risk", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
        feasible_rate=("feasible", "mean"),
    ).reset_index()
    gap_summary = gap[gap["gap_percent"].notna()].groupby(["scale", "gap_baseline", "method"]).agg(
        gap_percent_mean=("gap_percent", "mean")
    ).reset_index()
    text = [
        "# Experiment Summary",
        "",
        "## Method Summary",
        "",
        _markdown_table(method_summary),
        "",
        "## Gap Summary",
        "",
        _markdown_table(gap_summary) if not gap_summary.empty else "No gap data.",
        "",
        "## Output Files",
        "",
        "- `results.csv`: raw method metrics.",
        "- `results_with_gap.csv`: metrics plus gaps for instances with an optimal MILP baseline.",
        "- `manifest.json`: seeds, hashes, environment, Git snapshot, and artifact checksums.",
        "- `initial_solutions/`: canonical initial plans and objective references.",
        "- `ppo_transformer/`: instance-specific checkpoints and training histories.",
        "- `traces/`: per-preference, per-restart local-search traces.",
        "- `solutions/`: saved method solutions and final plan hashes.",
        "- `figures/`: generated result figures.",
    ]
    (run_dir / "summary.md").write_text("\n".join(text), encoding="utf-8")


def _plot_results(run_dir: Path, results: pd.DataFrame) -> None:
    figures = run_dir / "figures"
    summary = results.groupby(["scale", "method"])["weighted_objective_normalized"].mean().reset_index()
    for scale in summary["scale"].unique():
        data = summary[summary["scale"] == scale]
        plt.figure(figsize=(8, 5))
        plt.bar(data["method"], data["weighted_objective_normalized"])
        plt.title(f"{scale.capitalize()} mean normalized objective")
        plt.ylabel("Normalized weighted objective")
        plt.tight_layout()
        plt.savefig(figures / f"{scale}_normalized_objective.png", dpi=180)
        plt.close()

    gap = _add_gaps(results)
    gap_data = gap[(gap["scale"] == "small") & (gap["method"] != "MILP") & gap["gap_percent"].notna()]
    if not gap_data.empty:
        plot_data = gap_data.groupby("method")["gap_percent"].mean().reset_index()
        plt.figure(figsize=(8, 5))
        plt.bar(plot_data["method"], plot_data["gap_percent"])
        plt.title("Small-scale mean gap to MILP")
        plt.ylabel("Gap (%)")
        plt.tight_layout()
        plt.savefig(figures / "small_gap_to_milp.png", dpi=180)
        plt.close()

    runtime = results.groupby(["scale", "method"])["runtime_seconds"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    for method in runtime["method"].unique():
        data = runtime[runtime["method"] == method]
        plt.plot(data["scale"], data["runtime_seconds"], marker="o", label=method)
    plt.title("Mean runtime by scale")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "runtime_by_scale.png", dpi=180)
    plt.close()

def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small smoke experiment.")
    args = parser.parse_args()
    run_dir = run_experiments(quick=args.quick)
    print(f"outputs: {run_dir}")


if __name__ == "__main__":
    main()

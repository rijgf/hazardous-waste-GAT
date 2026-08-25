from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from sample_params import params_from_json_data
from src.ppo_improver import PPOImprover
from src.reproducibility import (
    deserialize_plan,
    file_sha256,
    json_sha256,
    plan_sha256,
)
from src.solution_utils import evaluate_solution


def _require_hash(actual: str, expected: str | None, label: str) -> None:
    if expected is None:
        raise ValueError(f"manifest does not record {label} SHA-256")
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: actual={actual}, expected={expected}"
        )


def replay(
    run_dir: Path,
    instance_id: str,
    preference: tuple[float, float],
    restart: int,
    steps: int | None = None,
) -> dict:
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    algorithm_config = json.loads((run_dir / "algorithm_config.json").read_text(encoding="utf-8"))
    network_config = json.loads((run_dir / "network_config.json").read_text(encoding="utf-8"))
    _require_hash(
        json_sha256(algorithm_config),
        manifest.get("algorithm_config_sha256"),
        "algorithm config",
    )
    _require_hash(
        json_sha256(network_config),
        manifest.get("network_config_sha256"),
        "network config",
    )
    instance_entry = next(item for item in manifest["instances"] if item["instance_id"] == instance_id)
    training_entry = next(
        item
        for item in manifest["training"]
        if item["instance_id"] == instance_id and item["method"] == "PPO-Transformer"
    )
    evaluation_entry = next(
        item
        for item in manifest["evaluations"]
        if item["instance_id"] == instance_id
        and item["method"] == "PPO-Transformer"
        and tuple(item["preference"]) == preference
        and int(item["restart"]) == restart
    )

    instance_path = run_dir / instance_entry["instance_file"]
    _require_hash(
        file_sha256(instance_path),
        instance_entry.get("instance_sha256"),
        "instance file",
    )
    checkpoint_path = run_dir / training_entry["checkpoint_file"]
    checkpoint_hash = file_sha256(checkpoint_path)
    _require_hash(
        checkpoint_hash,
        training_entry.get("checkpoint_sha256"),
        "training checkpoint",
    )
    _require_hash(
        checkpoint_hash,
        evaluation_entry.get("checkpoint_sha256"),
        "evaluation checkpoint",
    )

    params_data = json.loads(instance_path.read_text(encoding="utf-8"))
    params = params_from_json_data(params_data)
    initial_data = json.loads(
        (run_dir / instance_entry["initial_solution_file"]).read_text(encoding="utf-8")
    )
    initial_plan = deserialize_plan(initial_data["plan"])
    if plan_sha256(initial_plan) != instance_entry["initial_solution_sha256"]:
        raise ValueError("initial plan hash does not match manifest")

    improver = PPOImprover(
        params,
        algorithm_config,
        network_config,
        seed=int(training_entry["train_seed"]),
    )
    improver.load_checkpoint(checkpoint_path)
    result = improver.improve(
        preference,
        steps=steps,
        seed=int(evaluation_entry["evaluation_seed"]),
        initial_plan=initial_plan,
    )
    refs = (float(instance_entry["cost_ref"]), float(instance_entry["risk_ref"]))
    metrics = evaluate_solution(params, result.solution, preference, objective_refs=refs)
    result_plan_hash = plan_sha256(result.plan)
    saved_solution_path = run_dir / evaluation_entry["solution_file"]
    saved_solution_relative = Path(evaluation_entry["solution_file"]).as_posix()
    saved_solution_artifact = next(
        (
            artifact
            for artifact in manifest.get("solution_artifacts", [])
            if Path(artifact["file"]).as_posix() == saved_solution_relative
        ),
        None,
    )
    if saved_solution_artifact is None:
        raise ValueError("manifest does not inventory the saved PPO solution")
    _require_hash(
        file_sha256(saved_solution_path),
        saved_solution_artifact.get("sha256"),
        "saved PPO solution",
    )
    saved_solution = json.loads(saved_solution_path.read_text(encoding="utf-8"))
    expected_result_plan_hash = saved_solution.get("plan_sha256")
    if expected_result_plan_hash is None:
        raise ValueError("saved PPO solution does not record plan_sha256")
    return {
        "run_id": manifest["run_id"],
        "instance_id": instance_id,
        "preference": list(preference),
        "restart": restart,
        "evaluation_seed": evaluation_entry["evaluation_seed"],
        "initial_solution_sha256": instance_entry["initial_solution_sha256"],
        "checkpoint_sha256": checkpoint_hash,
        "expected_result_plan_sha256": expected_result_plan_hash,
        "result_plan_sha256": result_plan_hash,
        "matches_saved_plan": result_plan_hash == expected_result_plan_hash,
        "metrics": metrics,
        "trace": result.trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one saved PPO-Transformer evaluation cell.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--instance", required=True, dest="instance_id")
    parser.add_argument("--cost-weight", type=float, required=True)
    parser.add_argument("--risk-weight", type=float, required=True)
    parser.add_argument("--restart", type=int, default=0)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    payload = replay(
        args.run_dir,
        args.instance_id,
        (args.cost_weight, args.risk_weight),
        args.restart,
        args.steps,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

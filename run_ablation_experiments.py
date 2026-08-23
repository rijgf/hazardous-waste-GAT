from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from sample_params import params_to_json_data
from src.config import ROOT, ensure_dir, load_algorithm_config, load_model_config, load_network_config
from src.heuristics import build_greedy_initial_solution
from src.instance_generator import generate_random_params
from src.ppo_improver import PPOImprover
from src.solution_utils import evaluate_solution


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _variants(base_network: Dict[str, Any], quick: bool = False) -> Dict[str, Dict[str, Any]]:
    variants = {
        "full_transformer": dict(base_network, network_type="full_transformer", use_preference_token=True, use_route_arc_tokens=True, use_facility_tokens=True),
        "hybrid_transformer_fc": dict(base_network, network_type="hybrid_transformer_fc", use_preference_token=True, use_route_arc_tokens=True, use_facility_tokens=True),
        "no_preference_token": dict(base_network, network_type="full_transformer", use_preference_token=False, use_preference_in_global=False, use_route_arc_tokens=True, use_facility_tokens=True),
        "no_route_arc_tokens": dict(base_network, network_type="full_transformer", use_preference_token=True, use_route_arc_tokens=False, use_facility_tokens=True),
    }
    if quick:
        return {key: variants[key] for key in ["full_transformer", "no_preference_token"]}
    return variants


def run_ablation(quick: bool = False) -> Path:
    algorithm_config = load_algorithm_config()
    network_config = load_network_config()
    seed = int(algorithm_config["random_seed"]) + 77
    run_dir = ensure_dir(ROOT / algorithm_config["output_dir"] / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(run_dir / "models")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "instances")

    if quick:
        algorithm_config["preferences"] = [[0.5, 0.5]]
        algorithm_config["ppo"]["train_iterations"] = 2
        algorithm_config["ppo"]["episode_steps"] = 4
        algorithm_config["ppo"]["eval_steps"] = 5
        network_config["embedding_dim"] = 32
        network_config["ff_hidden_dim"] = 64
        network_config["max_tokens"] = 128

    _write_json(run_dir / "algorithm_config.json", algorithm_config)
    _write_json(run_dir / "base_network_config.json", network_config)

    preferences = [tuple(item) for item in algorithm_config["preferences"]]
    rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    for scale in ["small", "large"]:
        model_config = load_model_config(scale)
        if quick and scale == "large":
            model_config["producers_count"] = 4
            model_config["waste_types_count"] = 2
            model_config["facilities_count"] = 2
            model_config["vehicles_count"] = 3
            model_config["periods_count"] = 2
        params = generate_random_params(model_config, seed + (0 if scale == "small" else 10_000))
        _write_json(run_dir / "instances" / f"{scale}.json", params_to_json_data(params))

        heuristic_solution = build_greedy_initial_solution(params, seed=seed)
        ref_metrics = evaluate_solution(params, heuristic_solution, (0.5, 0.5))
        objective_refs = (max(ref_metrics["cost"], 1e-9), max(ref_metrics["risk"], 1e-9))
        for preference in preferences:
            metrics = evaluate_solution(params, heuristic_solution, preference, objective_refs=objective_refs)
            rows.append(_row(scale, "heuristic", "Heuristic", preference, metrics, 0.0, None))

        for variant_name, variant_config in _variants(network_config, quick=quick).items():
            variant_dir = ensure_dir(run_dir / "models" / variant_name)
            model_path = variant_dir / f"ppo_{scale}.pt"
            start = time.perf_counter()
            improver = PPOImprover(params, algorithm_config, variant_config, seed=seed)
            history = improver.train(save_path=model_path)
            train_time = time.perf_counter() - start
            _write_json(run_dir / f"network_config_{variant_name}.json", variant_config)
            for iteration, (reward, best_objective) in enumerate(zip(history["episode_reward"], history["best_objective"])):
                train_rows.append(
                    {
                        "scale": scale,
                        "variant": variant_name,
                        "iteration": iteration,
                        "episode_reward": reward,
                        "best_objective": best_objective,
                    }
                )
            for preference in preferences:
                start = time.perf_counter()
                result = improver.improve(preference, seed=seed)
                eval_time = time.perf_counter() - start
                metrics = evaluate_solution(params, result.solution, preference, objective_refs=objective_refs)
                rows.append(_row(scale, variant_name, "PPO", preference, metrics, eval_time, train_time))

    results = pd.DataFrame(rows)
    train_results = pd.DataFrame(train_rows)
    results.to_csv(run_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")
    train_results.to_csv(run_dir / "ablation_training_history.csv", index=False, encoding="utf-8-sig")
    _write_summary(run_dir, results)
    _plot(run_dir, results, train_results)
    return run_dir


def _row(scale: str, variant: str, method: str, preference: Tuple[float, float], metrics: Dict[str, Any], runtime: float, train_time: float | None) -> Dict[str, Any]:
    return {
        "scale": scale,
        "variant": variant,
        "method": method,
        "preference": f"{preference[0]:.2f}_{preference[1]:.2f}",
        "cost": metrics["cost"],
        "risk": metrics["risk"],
        "normalized_cost": metrics["normalized_cost"],
        "normalized_risk": metrics["normalized_risk"],
        "weighted_objective_raw": metrics["weighted_objective_raw"],
        "weighted_objective_normalized": metrics["weighted_objective_normalized"],
        "weighted_objective": metrics["weighted_objective"],
        "feasible": metrics["feasible"],
        "violation_count": len(metrics["violations"]),
        "runtime_seconds": runtime,
        "train_time_seconds": train_time,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_summary(run_dir: Path, results: pd.DataFrame) -> None:
    summary = results.groupby(["scale", "variant", "method"]).agg(
        weighted_objective_normalized_mean=("weighted_objective_normalized", "mean"),
        weighted_objective_raw_mean=("weighted_objective_raw", "mean"),
        cost_mean=("cost", "mean"),
        risk_mean=("risk", "mean"),
        feasible_rate=("feasible", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
        train_time_mean=("train_time_seconds", "mean"),
    ).reset_index()
    text = [
        "# Ablation Experiment Summary",
        "",
        "## Variant Summary",
        "",
        _markdown_table(summary),
        "",
        "## Variants",
        "",
        "- `full_transformer`: full Transformer with preference token and route-arc tokens.",
        "- `hybrid_transformer_fc`: Transformer token branch plus FC global branch.",
        "- `no_preference_token`: removes preference token.",
        "- `no_route_arc_tokens`: removes route-arc tokens.",
    ]
    (run_dir / "ablation_summary.md").write_text("\n".join(text), encoding="utf-8")


def _plot(run_dir: Path, results: pd.DataFrame, train_results: pd.DataFrame) -> None:
    figures = run_dir / "figures"
    ppo = results[results["method"] == "PPO"]
    for scale in ppo["scale"].unique():
        data = ppo[ppo["scale"] == scale].groupby("variant")["weighted_objective_normalized"].mean().reset_index()
        plt.figure(figsize=(10, 5))
        plt.bar(data["variant"], data["weighted_objective_normalized"])
        plt.xticks(rotation=20, ha="right")
        plt.title(f"{scale.capitalize()} PPO ablation normalized objective")
        plt.ylabel("Mean normalized weighted objective")
        plt.tight_layout()
        plt.savefig(figures / f"{scale}_ablation_objective.png", dpi=180)
        plt.close()

    if not train_results.empty:
        plt.figure(figsize=(9, 5))
        for variant in train_results["variant"].unique():
            data = train_results[train_results["variant"] == variant].groupby("iteration")["episode_reward"].mean().reset_index()
            plt.plot(data["iteration"], data["episode_reward"], label=variant)
        plt.title("Ablation PPO training reward")
        plt.xlabel("Iteration")
        plt.ylabel("Episode reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "ablation_training_reward.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small ablation smoke test.")
    args = parser.parse_args()
    run_dir = run_ablation(quick=args.quick)
    print(f"outputs: {run_dir}")


if __name__ == "__main__":
    main()

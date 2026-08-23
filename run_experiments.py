from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
GA_DIR = ROOT / "遗传算法"
if str(GA_DIR) not in sys.path:
    sys.path.insert(0, str(GA_DIR))

from genetic_algorithm import ClassicGeneticAlgorithm, GAConfig
from hazardous_waste_model import solve_with_milp
from sample_params import params_to_json_data
from src.config import ensure_dir, load_algorithm_config, load_model_config, load_network_config
from src.heuristics import build_greedy_initial_solution
from src.instance_generator import generate_random_params
from src.ppo_improver import PPOImprover
from src.solution_utils import evaluate_solution


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    algorithm_config = load_algorithm_config()
    network_config = load_network_config()
    seed = int(algorithm_config["random_seed"])
    run_dir = ensure_dir(ROOT / algorithm_config["output_dir"] / time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir / "models")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "instances")
    ensure_dir(run_dir / "solutions")

    if quick:
        algorithm_config["small_instance_count"] = 1
        algorithm_config["large_instance_count"] = 1
        algorithm_config["milp_time_limit"] = 5.0
        algorithm_config["preferences"] = [[0.5, 0.5], [1.0, 0.0]]
        algorithm_config["ppo"]["train_iterations"] = 5
        algorithm_config["ppo"]["episode_steps"] = 8
        algorithm_config["ppo"]["eval_steps"] = 10
        algorithm_config["ga"]["small"]["generations"] = 20
        algorithm_config["ga"]["large"]["generations"] = 1
        algorithm_config["ga"]["small"]["population_size"] = 12
        algorithm_config["ga"]["large"]["population_size"] = 4
        network_config["embedding_dim"] = 32
        network_config["transformer_layers"] = 1
        network_config["attention_heads"] = 4
        network_config["ff_hidden_dim"] = 64
        network_config["max_tokens"] = 160

    _write_json(run_dir / "algorithm_config.json", algorithm_config)
    _write_json(run_dir / "network_config.json", network_config)

    rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
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
            print(f"[{scale}] instance {instance_idx} start", flush=True)
            instance_seed = seed + (0 if scale == "small" else 10_000) + instance_idx
            base_params = generate_random_params(model_config, instance_seed)
            _write_json(run_dir / "instances" / f"{scale}_{instance_idx}.json", params_to_json_data(base_params))
            ref_solution = build_greedy_initial_solution(base_params, seed=instance_seed)
            ref_metrics = evaluate_solution(base_params, ref_solution, (0.5, 0.5))
            objective_refs = (max(ref_metrics["cost"], 1e-9), max(ref_metrics["risk"], 1e-9))

            ppo = PPOImprover(base_params, algorithm_config, network_config, seed=instance_seed)
            model_path = run_dir / "models" / f"ppo_{scale}_{instance_idx}.pt"
            start = time.perf_counter()
            print(f"[{scale}] instance {instance_idx} train PPO", flush=True)
            train_history = ppo.train(save_path=model_path)
            ppo_train_time = time.perf_counter() - start
            for step, (reward, best_obj) in enumerate(zip(train_history["episode_reward"], train_history["best_objective"])):
                train_rows.append({"scale": scale, "instance": instance_idx, "iteration": step, "episode_reward": reward, "best_objective": best_obj})

            for preference in preferences:
                print(f"[{scale}] instance {instance_idx} preference {preference}", flush=True)
                params = _with_preference(base_params, preference)
                optimizer_params = _with_normalized_objective(base_params, preference, objective_refs)
                pref_label = f"{preference[0]:.2f}_{preference[1]:.2f}"

                start = time.perf_counter()
                print(f"[{scale}] instance {instance_idx} run heuristic", flush=True)
                heuristic_solution = ref_solution
                heuristic_time = time.perf_counter() - start
                heuristic_eval = evaluate_solution(params, heuristic_solution, preference, objective_refs=objective_refs)
                rows.append(_row(scale, instance_idx, pref_label, "Heuristic", heuristic_eval, heuristic_time))
                _write_json(run_dir / "solutions" / f"{scale}_{instance_idx}_{pref_label}_heuristic.json", _solution_record(heuristic_solution, heuristic_eval))

                start = time.perf_counter()
                print(f"[{scale}] instance {instance_idx} run GA", flush=True)
                ga_result = run_ga(optimizer_params, algorithm_config["ga"][scale], instance_seed)
                ga_time = time.perf_counter() - start
                ga_solution = ga_result.best_solution
                ga_eval = evaluate_solution(params, ga_solution, preference, objective_refs=objective_refs)
                ga_fallback_reason = ""
                if not ga_eval["feasible"]:
                    ga_fallback_reason = "ga_infeasible_fallback_to_heuristic"
                    ga_solution = heuristic_solution
                    ga_eval = evaluate_solution(params, ga_solution, preference, objective_refs=objective_refs)
                rows.append(_row(scale, instance_idx, pref_label, "GA", ga_eval, ga_time, fallback_reason=ga_fallback_reason))
                _write_json(run_dir / "solutions" / f"{scale}_{instance_idx}_{pref_label}_ga.json", _solution_record(ga_solution, ga_eval))

                start = time.perf_counter()
                print(f"[{scale}] instance {instance_idx} run PPO eval", flush=True)
                ppo_result = ppo.improve(preference, seed=instance_seed)
                ppo_time = time.perf_counter() - start
                ppo_eval = evaluate_solution(params, ppo_result.solution, preference, objective_refs=objective_refs)
                row = _row(scale, instance_idx, pref_label, "PPO", ppo_eval, ppo_time)
                row["ppo_train_time"] = ppo_train_time
                rows.append(row)
                _write_json(run_dir / "solutions" / f"{scale}_{instance_idx}_{pref_label}_ppo.json", _solution_record(ppo_result.solution, ppo_eval))

                if scale == "small":
                    start = time.perf_counter()
                    print(f"[{scale}] instance {instance_idx} run MILP", flush=True)
                    milp_result = solve_with_milp(optimizer_params, time_limit=float(algorithm_config["milp_time_limit"]))
                    milp_time = time.perf_counter() - start
                    if milp_result.solution:
                        milp_eval = evaluate_solution(params, milp_result.solution, preference, objective_refs=objective_refs)
                        row = _row(scale, instance_idx, pref_label, "MILP", milp_eval, milp_time)
                        row["solver_status"] = milp_result.status
                        rows.append(row)
                        _write_json(run_dir / "solutions" / f"{scale}_{instance_idx}_{pref_label}_milp.json", _solution_record(milp_result.solution, milp_eval))

    results = pd.DataFrame(rows)
    train_results = pd.DataFrame(train_rows)
    results.to_csv(run_dir / "results.csv", index=False, encoding="utf-8-sig")
    train_results.to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    _add_gaps(results).to_csv(run_dir / "results_with_gap.csv", index=False, encoding="utf-8-sig")
    _write_summary(run_dir, results)
    _plot_results(run_dir, results, train_results)
    return run_dir


def _row(
    scale: str,
    instance_idx: int,
    pref_label: str,
    method: str,
    metrics: Dict[str, Any],
    runtime: float,
    fallback_reason: str = "",
) -> Dict[str, Any]:
    return {
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
    }


def _solution_record(solution: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"routes": {str(k): v for k, v in solution.get("routes", {}).items()}, "summary": solution.get("summary", {}), "metrics": metrics}


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
    ppo = out[out["method"] == "PPO"][["scale", "instance", "preference", "weighted_objective_normalized"]]
    for _, row in ppo.iterrows():
        mask = (
            (out["scale"] == "large")
            & (out["instance"] == row["instance"])
            & (out["preference"] == row["preference"])
        )
        base = float(row["weighted_objective_normalized"])
        out.loc[mask, "gap_percent"] = (out.loc[mask, "weighted_objective_normalized"] - base) / max(abs(base), 1e-9) * 100.0
        out.loc[mask, "gap_baseline"] = "PPO"
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
        "- `results_with_gap.csv`: metrics plus unified gap column. Small-scale gaps use MILP as baseline; large-scale gaps use PPO as baseline.",
        "- `training_history.csv`: PPO training curves.",
        "- `models/`: saved PPO model checkpoints.",
        "- `figures/`: generated result figures.",
    ]
    (run_dir / "summary.md").write_text("\n".join(text), encoding="utf-8")


def _plot_results(run_dir: Path, results: pd.DataFrame, train_results: pd.DataFrame) -> None:
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

    large_gap = gap[(gap["scale"] == "large") & (gap["method"] != "PPO") & gap["gap_percent"].notna()]
    if not large_gap.empty:
        plot_data = large_gap.groupby("method")["gap_percent"].mean().reset_index()
        plt.figure(figsize=(8, 5))
        plt.bar(plot_data["method"], plot_data["gap_percent"])
        plt.title("Large-scale mean gap to PPO")
        plt.ylabel("Gap to PPO (%)")
        plt.tight_layout()
        plt.savefig(figures / "large_gap_to_ppo.png", dpi=180)
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

    if not train_results.empty:
        data = train_results.groupby("iteration")["episode_reward"].mean().reset_index()
        plt.figure(figsize=(8, 5))
        plt.plot(data["iteration"], data["episode_reward"])
        plt.title("PPO training reward")
        plt.xlabel("Iteration")
        plt.ylabel("Episode reward")
        plt.tight_layout()
        plt.savefig(figures / "ppo_training_reward.png", dpi=180)
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

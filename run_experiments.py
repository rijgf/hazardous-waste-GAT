from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
GA_DIR = ROOT / "遗传算法"
if str(GA_DIR) not in sys.path:
    sys.path.insert(0, str(GA_DIR))

from genetic_algorithm import ClassicGeneticAlgorithm, GAConfig
from hazardous_waste_model import solve_with_milp
from sample_params import params_to_json_data
from src.config import ensure_dir, load_algorithm_config, load_model_config
from src.heuristics import build_greedy_initial_solution
from src.instance_generator import generate_random_params
from src.solution_utils import evaluate_solution


@dataclass
class GATRunResult:
    """Standard result returned by the future GAT experiment adapter."""

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
    ) -> GATRunResult:
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
    seed = int(algorithm_config["random_seed"])
    run_dir = ensure_dir(ROOT / algorithm_config["output_dir"] / time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "instances")
    ensure_dir(run_dir / "solutions")

    if quick:
        algorithm_config["small_instance_count"] = 1
        algorithm_config["large_instance_count"] = 1
        algorithm_config["milp_time_limit"] = 5.0
        algorithm_config["preferences"] = [[0.5, 0.5], [1.0, 0.0]]
        algorithm_config["ga"]["small"]["generations"] = 20
        algorithm_config["ga"]["large"]["generations"] = 1
        algorithm_config["ga"]["small"]["population_size"] = 12
        algorithm_config["ga"]["large"]["population_size"] = 4

    _write_json(run_dir / "algorithm_config.json", algorithm_config)

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
            print(f"[{scale}] instance {instance_idx} start", flush=True)
            instance_seed = seed + (0 if scale == "small" else 10_000) + instance_idx
            base_params = generate_random_params(model_config, instance_seed)
            _write_json(run_dir / "instances" / f"{scale}_{instance_idx}.json", params_to_json_data(base_params))
            ref_solution = build_greedy_initial_solution(base_params, seed=instance_seed)
            ref_metrics = evaluate_solution(base_params, ref_solution, (0.5, 0.5))
            objective_refs = (max(ref_metrics["cost"], 1e-9), max(ref_metrics["risk"], 1e-9))

            # Optional future method hook. Once a GAT adapter is registered,
            # it is prepared once per instance and evaluated under every
            # preference using the same objective references as MILP and GA.
            gat_adapter = build_gat_adapter(algorithm_config, scale)
            gat_prepare_time: float | None = None
            if gat_adapter is not None:
                gat_dir = ensure_dir(run_dir / "gat" / f"{scale}_{instance_idx}")
                gat_prepare_time = gat_adapter.prepare(
                    base_params,
                    ref_solution,
                    objective_refs,
                    instance_seed,
                    gat_dir,
                )

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

                if gat_adapter is not None:
                    print(f"[{scale}] instance {instance_idx} run GAT", flush=True)
                    gat_result = gat_adapter.solve(
                        params,
                        ref_solution,
                        preference,
                        objective_refs,
                        instance_seed,
                    )
                    gat_eval = evaluate_solution(
                        params,
                        gat_result.solution,
                        preference,
                        objective_refs=objective_refs,
                    )
                    row = _row(scale, instance_idx, pref_label, "GAT", gat_eval, gat_result.inference_seconds)
                    row["gat_prepare_time_seconds"] = gat_prepare_time
                    row["gat_metadata"] = json.dumps(gat_result.metadata or {}, ensure_ascii=False)
                    rows.append(row)
                    _write_json(
                        run_dir / "solutions" / f"{scale}_{instance_idx}_{pref_label}_gat.json",
                        _solution_record(gat_result.solution, gat_eval),
                    )

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
    results.to_csv(run_dir / "results.csv", index=False, encoding="utf-8-sig")
    _add_gaps(results).to_csv(run_dir / "results_with_gap.csv", index=False, encoding="utf-8-sig")
    _write_summary(run_dir, results)
    _plot_results(run_dir, results)
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
        "- `gat/`: reserved for future GAT checkpoints and training artifacts; created only when a GAT adapter is registered.",
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

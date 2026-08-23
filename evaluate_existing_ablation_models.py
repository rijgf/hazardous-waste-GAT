from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from sample_params import params_from_json_data
from src.config import load_algorithm_config
from src.heuristics import build_greedy_initial_solution
from src.ppo_improver import PPOImprover
from src.solution_utils import evaluate_solution


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "outputs" / "ablation_20260708_001417"

VARIANT_LABELS = {
    "full_transformer": "完整Transformer（偏好+路径弧token）",
    "hybrid_transformer_fc": "Transformer+全局FC融合",
    "no_preference_token": "去除偏好token",
    "no_route_arc_tokens": "去除路径弧token",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(df: pd.DataFrame) -> str:
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
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    algorithm_config = load_algorithm_config()
    preferences = [tuple(item) for item in algorithm_config["preferences"]]
    seed = int(algorithm_config["random_seed"]) + 77
    params = params_from_json_data(load_json(RUN_DIR / "instances" / "large.json"))

    heuristic_solution = build_greedy_initial_solution(params, seed=seed)
    ref_metrics = evaluate_solution(params, heuristic_solution, (0.5, 0.5))
    objective_refs = (max(ref_metrics["cost"], 1e-9), max(ref_metrics["risk"], 1e-9))

    old_results = pd.read_csv(RUN_DIR / "ablation_results.csv")
    train_times = (
        old_results[(old_results["scale"] == "large") & (old_results["method"] == "PPO")]
        .groupby("variant")["train_time_seconds"]
        .first()
        .to_dict()
    )

    rows = []
    for preference in preferences:
        heuristic_eval = evaluate_solution(params, heuristic_solution, preference, objective_refs=objective_refs)
        rows.append(
            {
                "scale": "large",
                "variant": "heuristic",
                "model_name": "贪心初始解",
                "method": "Heuristic",
                "preference": f"{preference[0]:.2f}_{preference[1]:.2f}",
                "cost": heuristic_eval["cost"],
                "risk": heuristic_eval["risk"],
                "normalized_cost": heuristic_eval["normalized_cost"],
                "normalized_risk": heuristic_eval["normalized_risk"],
                "weighted_objective_raw": heuristic_eval["weighted_objective_raw"],
                "weighted_objective_normalized": heuristic_eval["weighted_objective_normalized"],
                "runtime_seconds": 0.0,
                "train_time_seconds": None,
                "feasible": heuristic_eval["feasible"],
                "violation_count": len(heuristic_eval["violations"]),
            }
        )

    for variant, label in VARIANT_LABELS.items():
        network_config = load_json(RUN_DIR / f"network_config_{variant}.json")
        improver = PPOImprover(params, algorithm_config, network_config, seed=seed)
        checkpoint = torch.load(RUN_DIR / "models" / variant / "ppo_large.pt", map_location=improver.device)
        improver.policy.load_state_dict(checkpoint["state_dict"])
        for preference in preferences:
            start = time.perf_counter()
            result = improver.improve(preference, seed=seed)
            runtime = time.perf_counter() - start
            metrics = evaluate_solution(params, result.solution, preference, objective_refs=objective_refs)
            rows.append(
                {
                    "scale": "large",
                    "variant": variant,
                    "model_name": label,
                    "method": "PPO",
                    "preference": f"{preference[0]:.2f}_{preference[1]:.2f}",
                    "cost": metrics["cost"],
                    "risk": metrics["risk"],
                    "normalized_cost": metrics["normalized_cost"],
                    "normalized_risk": metrics["normalized_risk"],
                    "weighted_objective_raw": metrics["weighted_objective_raw"],
                    "weighted_objective_normalized": metrics["weighted_objective_normalized"],
                    "runtime_seconds": runtime,
                    "train_time_seconds": train_times.get(variant),
                    "feasible": metrics["feasible"],
                    "violation_count": len(metrics["violations"]),
                }
            )
            print(variant, f"{preference[0]:.2f}_{preference[1]:.2f}", metrics["weighted_objective_normalized"], runtime, flush=True)

    detail = pd.DataFrame(rows)
    ppo = detail[detail["method"] == "PPO"].copy()
    pref_best = ppo.groupby("preference")["weighted_objective_normalized"].transform("min")
    ppo["gap_to_best_percent"] = (
        (ppo["weighted_objective_normalized"] - pref_best)
        / pref_best.clip(lower=1e-12)
        * 100.0
    )
    heuristic = detail[detail["method"] == "Heuristic"].copy()
    heuristic["gap_to_best_percent"] = None
    detail = pd.concat([heuristic, ppo], ignore_index=True).sort_values(
        ["preference", "method", "weighted_objective_normalized", "runtime_seconds"]
    )

    summary = (
        ppo.groupby(["variant", "model_name"], as_index=False)
        .agg(
            mean_weighted_objective_normalized=("weighted_objective_normalized", "mean"),
            mean_gap_to_best_percent=("gap_to_best_percent", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            train_time_seconds=("train_time_seconds", "first"),
            feasible_rate=("feasible", "mean"),
            violation_count_total=("violation_count", "sum"),
        )
        .sort_values(
            ["mean_weighted_objective_normalized", "mean_gap_to_best_percent", "mean_runtime_seconds"],
            ascending=[True, True, True],
        )
    )
    best = summary["mean_weighted_objective_normalized"].min()
    summary["is_objective_best"] = summary["mean_weighted_objective_normalized"].sub(best).abs() < 1e-12

    out_csv = RUN_DIR / "large_ablation_gap_table_recomputed.csv"
    out_xlsx = RUN_DIR / "large_ablation_gap_table_recomputed.xlsx"
    out_md = RUN_DIR / "large_ablation_gap_table_recomputed.md"
    detail.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_xlsx) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        detail.to_excel(writer, sheet_name="preference_detail", index=False)

    md_summary = summary.rename(
        columns={
            "variant": "消融模型",
            "model_name": "模型含义",
            "mean_weighted_objective_normalized": "平均归一化偏好目标",
            "mean_gap_to_best_percent": "平均gap_to_best(%)",
            "mean_runtime_seconds": "平均求解时间(s)",
            "train_time_seconds": "训练时间(s)",
            "feasible_rate": "可行率",
            "violation_count_total": "违约数",
            "is_objective_best": "是否数值最优",
        }
    )
    md_detail = detail.rename(
        columns={
            "preference": "偏好",
            "variant": "消融模型",
            "model_name": "模型含义",
            "method": "方法",
            "cost": "成本",
            "risk": "风险",
            "normalized_cost": "归一化成本",
            "normalized_risk": "归一化风险",
            "weighted_objective_normalized": "归一化偏好目标",
            "gap_to_best_percent": "gap_to_best(%)",
            "runtime_seconds": "求解时间(s)",
            "train_time_seconds": "训练时间(s)",
            "feasible": "可行",
            "violation_count": "违约数",
        }
    )
    out_md.write_text(
        "\n".join(
            [
                "# 大规模消融实验归一化偏好目标与Gap（修复算子后重算）",
                "",
                "说明：本表加载已有 `.pt` 模型重新求解 large 消融实例，未重新训练。gap 以每个偏好下 PPO 消融模型中的最小归一化偏好目标为基准。",
                "",
                "## 汇总表",
                "",
                markdown_table(md_summary),
                "",
                "## 分偏好明细",
                "",
                markdown_table(md_detail),
                "",
            ]
        ),
        encoding="utf-8-sig",
    )

    print(f"csv={out_csv}")
    print(f"xlsx={out_xlsx}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()

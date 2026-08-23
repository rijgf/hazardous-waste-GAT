from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "outputs" / "ablation_20260708_001417"
RESULTS = RUN_DIR / "ablation_results.csv"


VARIANT_LABELS = {
    "full_transformer": "完整Transformer（偏好+路径弧token）",
    "hybrid_transformer_fc": "Transformer+全局FC融合",
    "no_preference_token": "去除偏好token",
    "no_route_arc_tokens": "去除路径弧token",
}


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
    df = pd.read_csv(RESULTS)
    large = df[(df["scale"] == "large") & (df["method"] == "PPO")].copy()
    large["model_name"] = large["variant"].map(VARIANT_LABELS).fillna(large["variant"])

    # Gap is computed within each preference against the best ablation model
    # on weighted normalized objective. Lower is better.
    pref_best = large.groupby("preference")["weighted_objective_normalized"].transform("min")
    large["gap_to_best_percent"] = (
        (large["weighted_objective_normalized"] - pref_best)
        / pref_best.clip(lower=1e-12)
        * 100.0
    )

    detail = large[
        [
            "preference",
            "variant",
            "model_name",
            "cost",
            "risk",
            "normalized_cost",
            "normalized_risk",
            "weighted_objective_normalized",
            "gap_to_best_percent",
            "runtime_seconds",
            "train_time_seconds",
            "feasible",
            "violation_count",
        ]
    ].sort_values(["preference", "weighted_objective_normalized", "runtime_seconds"])

    summary = (
        large.groupby(["variant", "model_name"], as_index=False)
        .agg(
            mean_weighted_objective_normalized=("weighted_objective_normalized", "mean"),
            mean_gap_to_best_percent=("gap_to_best_percent", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            train_time_seconds=("train_time_seconds", "first"),
            feasible_rate=("feasible", "mean"),
            violation_count_total=("violation_count", "sum"),
        )
        .sort_values(
            [
                "mean_weighted_objective_normalized",
                "mean_gap_to_best_percent",
                "mean_runtime_seconds",
            ],
            ascending=[True, True, True],
        )
    )
    best_value = summary["mean_weighted_objective_normalized"].min()
    summary["is_objective_best"] = (
        summary["mean_weighted_objective_normalized"].sub(best_value).abs() < 1e-12
    )

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

    out_csv = RUN_DIR / "large_ablation_gap_table.csv"
    out_xlsx = RUN_DIR / "large_ablation_gap_table.xlsx"
    out_md = RUN_DIR / "large_ablation_gap_table.md"

    detail.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_xlsx) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        detail.to_excel(writer, sheet_name="preference_detail", index=False)

    note = (
        "说明：本表只读取已有 `ablation_results.csv`，未重新训练模型。"
        "大规模场景下四个消融模型在所有偏好上的归一化偏好目标均为 1.0，"
        "因此按数值表现并列最优，gap_to_best 均为 0。"
    )
    out_md.write_text(
        "\n".join(
            [
                "# 大规模消融实验归一化偏好目标与Gap",
                "",
                note,
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

    print(f"summary_rows={len(summary)}")
    print(f"detail_rows={len(detail)}")
    print(f"csv={out_csv}")
    print(f"xlsx={out_xlsx}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()

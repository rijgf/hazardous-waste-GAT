# Script Index

## Run Order

| # | Script | Purpose | Input | Output | Paper Element |
|---|---|---|---|---|---|
| 1 | `run_supplementary_experiments.py init` | 锁定协议与两组训练实例 | 补充实验配置 | manifest、训练实例 | 实验设置 |
| 2 | `run_supplementary_experiments.py train` | 各训练一次并冻结两份模型 | Train-S/Train-L 实例 | small_model.pt、large_model.pt、训练历史 | 模型训练说明 |
| 3 | `run_supplementary_experiments.py lock-tests` | 两份模型冻结后锁定四组测试集及共同初始解 | 冻结模型、协议 | 200 个测试实例及初始解 | 样本外测试设置 |
| 4 | `run_supplementary_experiments.py schedule` | 构建并锁定去重后的 9,750 个 cell | 测试集、模型哈希 | schedule.json | 完整实验账本 |
| 5 | `run_supplementary_experiments.py evaluate` | 可恢复执行去重后的 9,750 个 cell | 冻结模型、测试集、调度表 | cell、solution、trace 记录 | E1、E2、G |
| 6 | `run_supplementary_experiments.py summarize` | 实例内先聚合重启，再跨实例汇总 | 完整 cell ledger | raw/instance/summary CSV、论文表 | 表 3—表 7 |
| 7 | `run_supplementary_experiments.py verify` | 核对数量、哈希、冻结性与抽样计算回放 | 全部实验产物 | verification report | 审计与复现说明 |
| 8 | `run_supplementary_experiments.py export` | 导出紧凑复现包 | 经核验的正式运行目录 | 模型、表格、manifest、schedule | 复现材料 |
| 9 | `output/supplementary-experiments/scripts/build_paper_tables.py` | 从经核验汇总生成论文表、结果登记、裁决日志与完整性清单 | verification通过且哈希闭环的正式运行 | 表3、表4、表6、表7、附表、registry、inventory | 论文结果与证据登记 |

## Dependencies

正式入口依赖 `src/supplementary_protocol.py`、`src/ppo_improver.py`、`src/instance_generator.py`、`src/solution_utils.py`、`hazardous_waste_model.py` 与 `遗传算法/genetic_algorithm.py`。论文表生成器只接受完整冻结协议哈希匹配、正式验证通过且汇总—cell ledger—输入表三方哈希闭环的运行目录；smoke必须显式使用`--allow-smoke`且不得进入正式论文表目录。

## Seeds

所有训练、实例生成和求解种子均由 master seed 通过确定性哈希派生；具体值写入实验 manifest。

## Paper-Element Correspondence

待正式结果生成后补充。

## Coding Decisions

见 [coding-decisions-log.md](coding-decisions-log.md)。

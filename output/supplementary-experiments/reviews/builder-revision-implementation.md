# Reviewer 02 / Reviewer 05：builder 修订实施与验证报告

## 实施结论

已按 `builder-revision-plan.md` 完成并验证以下修订，未修改正式初稿、正式目标稿或 `outputs/supplementary_experiment_v1` 中的锁定协议、manifest、verification及原始结果：

1. `build_paper_tables.py` 新增 `_table5_scale_design(protocol)`，只从 `_load_inputs()` 已验证并哈希绑定的 run 内协议生成 `table5_scale_design.csv` 与 `table5_scale_design.md`。
2. 表5共6行：2个训练模型行和4个测试规模行；五项规模、训练规模映射、检查点名及测试实例数均来自协议。训练行只写协议可证明的“1份模型”，未额外推断训练实例数量。
3. 表3、表4标题改为由 `expected_n` 动态生成；正式输出分别为“表3  小规模50个未见实例……”和“表4  大规模50个未见实例……”，smoke输出相应为“1个”。
4. 表3、表4、表7的主表Markdown表注已压缩为统计口径、比较指标、训练/推理边界和并发解释边界；逐批 `batch/workers/max_cells/整批墙钟` 不再写入主表。
   表3的Gap规则表述为“任何未证明optimal的incumbent（包括限时或其他中止）均不进入Gap”，没有把全部非optimal状态误归为限时。
5. 并发审计信息没有删除：表3、表4、表7 CSV 的 `*_runtime_concurrency_scope` 均保留，registry时间记录仍保留逐批说明，manifest `evaluation_runs` 未改动。
6. inventory新增表5 CSV/Markdown；`output_files_sha256` 从16项增至18项，`output_row_counts` 从9项增至10项。
7. Reviewer 05 的路径隐私轻微项已处理：正式 `source_run_dir` 写为 `outputs/supplementary_experiment_v1`，并新增 `source_run_dir_path_kind=repository-relative`。当run不在仓库内时，安全回退为解析后的绝对路径并标记 `absolute-external`，避免产生含义不明的伪相对路径。

## 自动化测试

执行命令：

```text
py -m unittest tests.test_build_paper_tables -v
```

结果：14项测试全部通过，`Ran 14 tests ... OK`。新增/扩展覆盖包括：

- 表5正式协议的6行顺序、四个规模签名、50个实例及训练—测试端点一致性；
- smoke协议的测试数量动态为1，证明没有硬编码50；
- 表5 CSV字段顺序、CSV/Markdown进入返回路径与inventory；
- 表3/4 smoke动态标题与表3/4/7主表禁用逐批长注；
- 表3明确覆盖所有未证明optimal的incumbent，且不再出现把非optimal一概称为限时解的旧表述；
- 主表仍指向 `manifest.evaluation_runs`，并保留不能作隔离速度因果比较的边界；
- 仓库内run路径为repository-relative、仓库外smoke路径为absolute-external；
- inventory中断发布时完成标记不可见，成功发布后18项成员全部存在且哈希匹配。

## 正式 builder 重建

执行命令：

```text
py output/supplementary-experiments/scripts/build_paper_tables.py outputs/supplementary_experiment_v1 --output-dir output/supplementary-experiments/tables
```

命令成功返回19条输出路径，其中18个成员文件先发布，`paper-table-inventory.json` 最后发布。独立复核结果：

| 检查项 | 结果 |
|---|---:|
| 目录内文件总数（含inventory） | 19 |
| inventory声明的非自身输出 | 18 |
| 重算SHA-256不匹配 | 0 |
| inventory登记的CSV | 10 |
| 重算CSV行数不匹配 | 0 |
| `table5_scale_design.csv` | 6行 |
| `results-registry.csv` | 521行 |
| `adjudication-log.csv` | 4行 |
| inventory时间不早于全部成员 | 是 |
| 正式 `source_run_dir` | `outputs/supplementary_experiment_v1` |
| 正式路径类型 | `repository-relative` |

表5正式行依次为：Train-S（3/2/2/3/2，1份模型）、Train-L（20/4/3/8/4，1份模型）、Test-1（3/2/2/3/2，50个实例）、Test-2（6/2/2/4/3，50个实例）、Test-3（10/3/3/5/3，50个实例）、Test-4（20/4/3/8/4，50个实例）。

表3、表4、表7主表Markdown均未检出 `batch`、`workers=`、`max_cells=` 或 `整批墙钟=`；表3/表4/表7 CSV的并发审计字段非空行分别为5/5/8，registry仍有33条时间指标记录包含完整batch说明。

## 哈希锚点

| 对象 | SHA-256 |
|---|---|
| 锁定协议canonical hash | `24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba` |
| 正式manifest输入 | `43f9d52058a7b338b1081bfd5e617aed3f505000531e3cdee2aaf88a62244d9e` |
| 正式verification输入 | `e7799970170b264ac0f5e55aec9c4ab45228008f69d5f7fb485180adc3d50f88` |
| 修订后builder | `17c1bc2f9c30a76daddb3021c215070f5e81988de9352a9ad17f8b7852ce3d91` |
| 表5 CSV | `13bc0240ba7d92b73b5717ba092e68fa3a32bd9c5609890d189ddad0661c2e67` |
| 表5 Markdown | `d10fcc7bee2a4bcd686918c71483a7d0512699761656f4033171ae0a86acf1a1` |
| 新inventory | `76edf5c7f05035676b97c6d1dc12f09195b2d2000f87aa5af3788b9abf16e0f3` |

正式锁定输入哈希与审查锚点一致，说明本次只重建下游派生表，没有改写9750个求解单元或其验证证据。

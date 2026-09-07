# 正式结果产物独立复现审计

- 审计日期：2026-09-05（Asia/Shanghai）
- 审计对象：`outputs/supplementary_experiment_v1`、`output/supplementary-experiments/tables`、`output/supplementary-experiments/replication`
- 审计方式：独立解析 JSON/CSV，使用 SHA-256 重算文件哈希，重算 CSV 数据行数，比较目录实际成员与清单声明，并逐行核对表5与锁定协议。
- 总体结论：**PASS**
- 阻断性问题：0
- 非阻断性偏差：0

## 1. 审计锚点

| 对象 | 实际 SHA-256 |
|---|---|
| `paper-table-inventory.json` | `76edf5c7f05035676b97c6d1dc12f09195b2d2000f87aa5af3788b9abf16e0f3` |
| 正式运行 `manifest.json` | `43f9d52058a7b338b1081bfd5e617aed3f505000531e3cdee2aaf88a62244d9e` |
| 正式运行 `verification/verification_report.json` | `e7799970170b264ac0f5e55aec9c4ab45228008f69d5f7fb485180adc3d50f88` |
| 复现包 `inventory.json` | `7e1d236784b1558fb682f6d98e8c33a72ff5f27f3033c3457f1247b17bc0cfdc` |
| 协议规范哈希（清单与 manifest） | `24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba` |

## 2. 论文表产物清单

结论：**PASS**。

- `output_files_sha256` 声明 18 项非自身输出；18/18 文件存在，18/18 重算 SHA-256 与声明一致。
- 表目录实际共有 19 个文件：18 个声明成员及 `paper-table-inventory.json` 自身；无未声明文件、无声明缺失文件。
- 清单文件的修改时间不早于全部 18 个成员，符合“成员先写入、清单最后发布”的原子发布顺序。
- `output_row_counts` 声明的 10 个 CSV 均逐文件重算通过：

| 文件 | 声明行数 | 重算行数 | 结果 |
|---|---:|---:|---|
| `adjudication-log.csv` | 4 | 4 | PASS |
| `appendix_all_methods.csv` | 35 | 35 | PASS |
| `appendix_milp_status.csv` | 9 | 9 | PASS |
| `appendix_risk_components.csv` | 35 | 35 | PASS |
| `results-registry.csv` | 521 | 521 | PASS |
| `table3_small_scale.csv` | 5 | 5 | PASS |
| `table4_large_scale.csv` | 5 | 5 | PASS |
| `table5_scale_design.csv` | 6 | 6 | PASS |
| `table6_generalization_matrix.csv` | 2 | 2 | PASS |
| `table7_generalization_operational.csv` | 8 | 8 | PASS |

- `input_files_sha256` 声明 14 项；14/14 源输入存在且哈希一致，覆盖 manifest、协议配置、schedule、10 张源结果表和 verification report。
- `source_run_dir=outputs/supplementary_experiment_v1`，`source_run_dir_path_kind=repository-relative`；从仓库根目录解析后恰好指向实际正式run，不再包含本机用户目录绝对路径。
- 表构建脚本实际 SHA-256 为 `17c1bc2f9c30a76daddb3021c215070f5e81988de9352a9ad17f8b7852ce3d91`，与清单声明一致。

### 2.1 表5的协议派生闭环

结论：**PASS**。

- `table5_scale_design.csv` 与 `table5_scale_design.md` 均存在、均被纳入 `output_files_sha256`，重算哈希与声明一致；CSV为6行，Markdown标题为“表5  两种训练规模与四种测试规模设置”。
- 逐行按已哈希绑定的正式 `protocol_config.json` 重建并比较，6/6行完全一致，无手工漂移：Train-S与Train-L分别引用Test-1和Test-4的规模向量；Test-1至Test-4的规模签名依次为 `3/2/2/3/2`、`6/2/2/4/3`、`10/3/3/5/3`、`20/4/3/8/4`，测试实例数均为50。
- 五项规模字段顺序与锁定协议的 `scale_fields` 一致：`producers_count`、`waste_types_count`、`facilities_count`、`vehicles_count`、`periods_count`。
- inventory 对 `protocol_config.json` 的输入哈希与正式run实际文件哈希一致，故表5的派生源已进入同一哈希链。

## 3. manifest—verification—源表哈希链

结论：**PASS**。

- manifest、verification report 与论文表清单中的 `protocol_id` 均为 `supplementary-experiment-v1`；论文表清单与 manifest 的协议规范哈希一致。
- 正式运行 `manifest.json` 和 `verification/verification_report.json` 的实际哈希仍分别为 `43f9d52058a7b338b1081bfd5e617aed3f505000531e3cdee2aaf88a62244d9e` 与 `e7799970170b264ac0f5e55aec9c4ab45228008f69d5f7fb485180adc3d50f88`；协议文件实际哈希为 `74edd93e5866d583eef7ffa17a004c9e92070a165260712761d549a691253257`，协议规范哈希仍为 `24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba`。上述正式run锚点均未随论文表重建而变化。
- manifest 声明的 10 张源表、verification report 的 `table_files_sha256` 及论文表清单的 10 项 `tables/...` 输入哈希，文件名集合完全相同，逐项哈希完全一致。
- manifest、论文表清单与实际文件所对应的 schedule 哈希一致：`7312dae1eaccf33b8f28b1421822be32a3e7f6310dc2751a90696db69d3c286a`；verification report 中的 schedule 哈希亦相同。
- manifest 的 verification report 哈希、论文表清单输入哈希和 verification report 实际哈希三者一致。
- manifest 与 verification report 的 `summary_input_cell_ledger_sha256` 相同，且 `summary_ledger_match=true`。
- 正式性标记闭合：论文表清单 `mode=formal`、`allow_smoke_flag=false`，manifest `smoke=false`。

## 4. 紧凑复现包

结论：**PASS**。

- 复现包清单声明 18 个 payload；实际目录共 19 个文件，其中 18 个为声明 payload，另 1 个为 `inventory.json` 自身。
- 18/18 payload 均满足：文件存在、SHA-256 与清单一致、字节数与清单一致、正式源运行中的对应文件存在且哈希相同。
- 未发现重复路径、绝对路径、`..` 路径穿越、非法 SHA-256、未列文件或漏列文件。
- 复现包未携带 `cells`、`traces`、`test_instances`、`initial_solutions`、`training_instances` 目录；以 `cell-*` 命名的逐单元文件为 0。
- 顶层仅含：`inventory.json`、`manifest.json`、`models`、`protocol_config.json`、`schedule.json`、`tables`、`training_history`、`verification`。
- 复现包实际大小为 27,799,907 字节、19 个文件；正式运行目录为 1,099,073,679 字节、29,672 个文件。文件数减少 99.936%，字节数减少 97.471%，同时保留协议、调度、模型、训练历史、完整汇总表与验证报告，符合紧凑性要求。
- 复现包中的 manifest 仍为 `stage=verified`，且与正式运行 manifest 逐字节同哈希。

## 5. 主表—results registry—adjudication ID 闭合

结论：**PASS**。

- 论文表清单声明 ID 集合：`E1`、`E2`、`G-Train-L`、`G-Train-S`。
- `results-registry.csv` 与 `adjudication-log.csv` 的去重 ID 集合均与上述集合精确相等；两文件均无空 ID。
- adjudication 每个 ID 恰有一行，无重复；registry 中 `(hypothesis_id, metric_name)` 无重复。
- registry 行数按 ID 分解为：E1 254 行、E2 185 行、G-Train-L 41 行、G-Train-S 41 行，合计 521 行。
- registry 引用的 7 张源表和 adjudication 引用的 4 张主表全部存在且均纳入 `output_files_sha256`。
- 表5是协议派生的设计表而非统计结果指标，因此不进入 registry 或 adjudication；其CSV/Markdown通过inventory哈希和协议输入哈希单独闭合，不改变registry的521行或adjudication的4行。
- 四张主表的登记覆盖关系闭合：

| 主表 | registry ID | adjudication ID |
|---|---|---|
| `table3_small_scale.csv` | E1；G-Train-S（一次训练时间登记） | E1 |
| `table4_large_scale.csv` | E2；G-Train-L（一次训练时间登记） | E2 |
| `table6_generalization_matrix.csv` | G-Train-S；G-Train-L | G-Train-S；G-Train-L |
| `table7_generalization_operational.csv` | G-Train-S；G-Train-L | G-Train-S；G-Train-L |

## 6. 正式验证门与实际计数

结论：**PASS**。

- 实际解析 `schedule.json` 得 9,750 个单元，`cell_id` 和 `logical_cell_id` 均为 9,750 个唯一值；方法分解为 PPO 6,000、GA 3,000、启发式 500、MILP 250。
- 实际解析 `raw_results.csv` 得 9,750 行和 9,750 个唯一 `cell_id`；全部为 `status=complete`、`feasible=True`、`quality_included=True`。
- 实际解析 `cell_file_ledger.csv` 得 9,750 行和 9,750 个唯一 `cell_id`。
- manifest 中测试实例总数为 200，Test-1 至 Test-4 各 50；磁盘实际测试实例文件亦为 200。
- verification report 的关键门：期望总数 9,750，观察总数 9,750，`count_match=true`；模型数 2，checkpoint 文件数 2，测试实例数 200。
- verification report 记录 artifact 重检 75 项、错误 0；solver replay 15 项、错误 0。
- 缺失单元 0、失败单元 0、完整性错误 0；`source_changes_since_init={}`；`passed=true`。
- manifest 最终阶段为 `verified`，且其 verification 节点记录 `passed=true`。

## 7. 精确问题与范围边界

- 精确问题：**无**。
- 本审计独立重算了文件哈希、文件集合、字节数、CSV 行数、表5协议派生关系、schedule/raw/ledger 计数及 ID 闭合关系；没有重新执行全部 9,750 个求解单元。
- artifact 75 项与 solver 15 项的重放结果来自已哈希锁定的 verification report；本审计验证了该报告与 manifest、论文表清单和复现包之间的哈希闭合，并确认报告中的两类错误计数均为 0。

## 最终判定

**PASS：含协议派生表5在内的18项论文表输出、登记/裁决 ID、验证报告、正式run与紧凑复现包之间形成完整且无偏差的可追溯链；清单使用仓库相对源路径，未保留本机绝对路径。**

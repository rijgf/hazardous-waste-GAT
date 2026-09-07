# 待回填稿占位符与论文表构建器一致性审计

## 1. 审计范围与快照

- 被审稿件：`output/supplementary-experiments/drafts/数值实验与结果分析_待回填稿.md`
- 实际完成全量逐项审计的稿件 SHA256：`2dc01cdcc1eaed3bf2163fe7afaad032dd51e993a6e9db3b5700a38af377310c`
- 本日志落盘前检测到的后续稿件 SHA256：`36caeaaaacf22b12b0bb91e839cf87b4688f04f23d7857a06949974c1753d230`
- 构建器：`output/supplementary-experiments/scripts/build_paper_tables.py`
- 构建器 SHA256：`2e5b1ba71250441b8aafbe006e5f517c07dea91cba2a58ae5cdccf9391c1caa2`
- 审计方式：只读检查稿件占位符、构建器生成字段和标准运行清单 schema；未读取或改动正在运行的正式 cell、solution、trace、manifest 或其他正式输出。

由于稿件在审计完成后已有后续修改，本报告只对 SHA256 为 `2dc01...310c` 的快照给出结论。后续稿件不得自动继承本报告的通过项，必须再次运行占位符审计。

## 2. 总体结论

结论为：**具体 RID 键名闭合通过，但尚不能无歧义地全自动回填。存在三类阻断，修正后必须再次审计。**

被审快照中：

- 122 个 `RID=` 块共包含 148 个唯一、具体的 `hypothesis_id/metric_name` 对；逐一按构建器生成规则核对，未发现无法生成或拼写错误的具体 RID。
- 表3、表4、表6、表7中的具体 RID 均能定位到 `results-registry.csv`。
- `RID-FAMILY` 的E1/E2偏好族、G模型—规模族、风险分项族和失败类型族可在应用下述受限展开规则后定位。
- registry 与 adjudication 的假设编号集合均为 `E1`、`E2`、`G-Train-S`、`G-Train-L`；adjudication 每个编号一行。
- `paper-table-inventory.json` 所称的16项非inventory产物与构建器一致：9个CSV和7个Markdown。

但是，“RID存在”不等于“表格展示口径可直接由该RID替换”。以下三项在正式回填前必须处理。

## 3. 阻断一：registry的SE不能替代表格要求的样本SD和有效n

### 3.1 问题

稿件的表3、表4、表6、表7要求报告独立实例层面的“均值±样本SD”及有效 `n`。构建器的 `results-registry.csv` 每行只保存：

- `metric_value`；
- `metric_se`；
- `component`、`pct_share`、来源和备注。

其中 `metric_se` 是样本SD除以有效样本量平方根后的标准误，不是样本SD；registry也没有统一的 `metric_n` 字段。因此，任何把 `metric_value ± metric_se` 直接写成“均值±样本SD”的回填都会造成统计口径错误。

该问题覆盖：

- 表3的PPO/GA目标、Gap、配对差异、可行率和各方法时间；
- 表4的成本、风险、目标、配对差异、可行率和时间；
- 表6的宏平均目标、样本SD、完整宏实例数和缺失质量单元；
- 表7的可行率、清零率、时间、相对GA改善率、质量单元数和异常计数；
- 风险附表中需要样本SD与有效 `n` 的展示。

### 3.2 必须采用的机械规则

1. 表3、表4、表6、表7及三份附表原则上直接插入构建器生成并经inventory核验的Markdown，不逐个RID重建表体。
2. 如必须从CSV构表：
   - 使用相应的 `*_mean_sd`、`display_Test-*`、`*_sd` 与 `*_n` 字段；
   - 表3的MILP最优展示使用 `milp_optimal_n/milp_total_n (milp_optimal_percent%)`；
   - 表6优先使用 `display_Test-1` 至 `display_Test-4`；
   - 表7的质量单元展示拼接 `quality_cells/expected_instance_preference_cells`。
3. RID仅用于唯一定位、交叉核验及正文中的单值引用。
4. 正文若使用registry的 `metric_se`，必须明确标为SE；需要SD时必须回到 `source_table`。
5. `metric_value` 为空时写“—”或“主结果不可用”，并报告有效 `n` 与失败边界；不得用conditional字段或剩余可行子集替代。

## 4. 阻断二：`all_restarts_feasible`未进入论文级构建产物

### 4.1 问题

被审稿件承诺把“全部3次重启均严格可行”的补充比例另列于附录A.4，并给出从正式run上游表临时汇总的 `FIELD` 占位。但是当前构建器：

- 不把该指标写入 `appendix_all_methods.csv/.md`；
- 不把该指标写入 `results-registry.csv`；
- 不把该指标写入 `adjudication-log.csv`。

虽然 `tables/summary_by_preference.csv:all_restarts_feasible_rate` 和 `tables/instance_level_results.csv:all_restarts_feasible` 含有上游字段，但在写稿阶段临时计算会绕过论文表、registry和inventory的一对一发布链路。

此外，G中“先在实例内跨五偏好平均全重启可行指示，再跨实例平均”得到的是实例—偏好层补充比例，不是“一个实例的所有五个偏好均三重启可行”的实例比例，名称必须避免混淆。

### 4.2 最小修正建议

最小且稳妥的处理是删除稿件中以下承诺：

- 指标将“另列于附录A.4”的说明；
- 表7注释中要求另行汇总该指标的说明；
- A.4中整段 `all_restarts_feasible` 待回填说明。

如研究者坚持保留该补充指标，必须先扩展构建器，使其进入附表、registry、inventory和自动测试，再进行回填；不得由写稿流程临时计算。

## 5. 阻断三：A.6的 `FINAL-MANIFEST` 占位要求了manifest未记录的字段

### 5.1 manifest可直接支持的字段

标准 `evaluation_runs` 记录包含：

- `started_at`、`completed_at`；
- `methods`、`models`、`scales`；
- `workers`、`allow_concurrent_ppo`；
- `max_cells`；
- `wall_seconds`、`completed_cells`、`failed_cells`。

### 5.2 manifest不能直接支持的字段

标准记录不包含：

- 持久化的batch/run ID；
- 批次设备字段；
- solver线程数；
- 真实计划/选中待运行cell数；
- 本批次cell ID清单；
- archive路径、旧/新文件哈希或完整audit元数据。

因此，稿件A.6中未字段化的36处 `FINAL-MANIFEST` 不能按列名自动替换，且当前按PPO、GA、启发式、MILP预置四行的布局无法正确表示 `methods=all` 的批次。

### 5.3 必须采用的机械规则

1. 遍历最终 `manifest.evaluation_runs`，保持数组原序，一条run生成一行；不要预先按方法生成四行。
2. 方法列保留run中的完整 `methods` 范围。若为 `all`，`completed_cells` 是整批总完成数，绝不能复制为每种方法的完成数。
3. manifest没有batch ID；只能写 `evaluation_runs[i]` 或“第i批”，不得伪称正式ID。
4. `workers` 直接读取。
5. `max_cells` 只能标为“上限”，不能写成计划数；null写“未设上限”。
6. 完成数、失败数、开始时间、结束时间与墙钟直接读取。
7. 吞吐机械计算为 `completed_cells * 3600 / wall_seconds`；墙钟非正或字段缺失时保持空白。
8. PPO、GA和启发式的solver线程写“不适用”；MILP线程写“未记录”。workers不得冒充solver线程。
9. device写“批次未记录”；不能由初始化环境中的单GPU事实倒推出每个批次的设备占用。
10. 只有 `superseded_for_timing is true` 的批次从最终时间口径排除；字段缺失或非true时按构建器现行规则纳入。
11. archive/audit字段不存在时明确写“未记录”，不得补造路径或哈希。
12. 可在批次审计表中保留零完成run；构建器的并发说明会忽略完成数不大于零的run，两者用途不同。

### 5.4 额外限制

`evaluation_runs` 没有保存本批次的cell ID列表，因此仅凭manifest不能在重试、混合方法或复杂续跑条件下建立每个 `runtime_seconds` 与某个批次的一一归属。A.6可以披露批次聚合信息，但不得宣称manifest本身提供了逐cell批次绑定。

## 6. 其他占位符与粒度问题

### 6.1 `RID-FAMILY`展开

- 偏好必须按冻结顺序展开：`C100-R000`、`C075-R025`、`C050-R050`、`C025-R075`、`C000-R100`。
- 规模必须按Test-1、Test-2、Test-3、Test-4展开。
- 风险族不能对实验和模型作无约束笛卡尔积：
  - E1只允许Train-S、GA、Heuristic、MILP；
  - E2只允许Train-L、GA、Heuristic。
- `milp_status_count.{preference_id}.{solver_status}` 的状态集合必须遍历 `appendix_milp_status.csv` 的实际状态行，不能预设状态名称。
- 每个展开后的 `(hypothesis_id, metric_name)` 必须在registry中恰好出现一次；若重复或缺失，回填立即终止。

### 6.2 派生量

- E1中“证明optimal的实例—偏好单元总数”应直接求和 `table3_small_scale.csv:milp_optimal_n`，不应从已格式化百分比反推。
- 偏好范围、最小值和最大值只有在五个主 `metric_value` 全部非空时才能计算；有任一空值时整体写“不可判定”，不得只在剩余偏好中取范围。
- 风险分项的占比使用registry的 `pct_share` 或风险附表的相应字段；不得把 `metric_se` 当构成占比。

### 6.3 `FIELD`占位

- `table3_small_scale.csv:outcome_disclosure` 与 `table4_large_scale.csv:outcome_disclosure` 按 `preference_id` 定位。
- `table7_generalization_operational.csv:outcome_disclosure` 按稿件当前行的 `model_id/test_scale` 定位。
- `quality_cells/expected_instance_preference_cells` 是两个字段的显示拼接，不是CSV中的单一字段名。若不整体插入生成Markdown，回填器必须显式执行拼接。
- 任何直接引用正式run上游表、但未进入builder论文产物的 `FIELD` 都不能临时加入论文；应先进入builder和inventory，或从稿件删除。

## 7. adjudication与需人工裁决项

### 7.1 adjudication能支持的内容

- E1/E2各有一行总体方向、幅度、解释和来源。
- G-Train-S/G-Train-L各有一行跨四种规模的方向、幅度、解释和来源。
- adjudication可用于核对总体描述是否与表格方向相容。

### 7.2 adjudication不能自动替代的判断

- E1/E2的adjudication方向是在每个实例内等权平均五个偏好后的总体方向，不能替代逐偏好方向。如果五个偏好正负并存，正文必须逐组披露。
- G的方向主要依据四规模相对GA改善率，并在幅度中报告最低可行率；它没有联合期末清零率、两模型相对差距、失败和缺失边界。
- “主导风险分项”需要结合相关方法、偏好及 `pct_share` 判断。
- “在哪个规模优先采用Train-S或Train-L”及回退阈值没有冻结判定规则。
- “同规模跨实例/特定方向经验迁移/四规模经验迁移/未观察到稳定迁移”的最终类别不能仅凭adjudication自动选择。

因此，下列内容必须人工裁决并由数字逐句复核：

1. 偏好间方向是否一致及如何陈述反向偏好；
2. 风险主导渠道及其边界；
3. 模型选择和回退规则；
4. 最终迁移结论强度；
5. 任何管理含义和部署建议。

## 8. A.7哈希占位的可回填边界

- 测试实例的 `file_sha256` 与 `instance_sha256` 可从 `manifest.test_sets[scale][index]` 逐项取得，但没有一个现成的“200实例总哈希”。论文宜写“200/200经verification核验”并指向机器可读manifest，而不是在单个表格单元粘贴数百个哈希。
- 最终manifest文件哈希可从 `paper-table-inventory.json:input_files_sha256["manifest.json"]` 获取。
- cell账本规范哈希可从manifest与verification共同的 `summary_input_cell_ledger_sha256` 获取；逐cell文件哈希来自 `cell_file_ledger.csv`。
- 构建器没有发布完整solution/trace独立哈希总账，也没有archive审计总账。只能报告verification的完整性结果，不能声称存在单一solution/trace聚合哈希。

## 9. 推荐的完整机械回填流程

1. 要求正式manifest阶段为verified，verification passed，且 `paper-table-inventory.json` 最后发布。
2. 核对inventory声明的16项输出均存在且SHA256一致，并核对其记录的manifest输入哈希。
3. 读取registry并断言每个 `(hypothesis_id, metric_name)` 唯一；断言registry和adjudication ID集合均严格为四个允许值。
4. 整体插入四张主表与三张附表的生成Markdown，避免把SE误作SD或漏掉有效 `n`。
5. 用RID回填正文单值；按冻结顺序受限展开RID-FAMILY。
6. 仅在全部输入主值非空时执行范围、极值和合计等DERIVE。
7. adjudication只用于其明确统计层级；逐偏好方向和复合迁移结论另行审阅。
8. 从最终manifest按“一run一行”规则生成A.6，对未记录字段明确写“未记录/不适用”。
9. 对所有回填数字重新执行“源表→稿件表→正文”双阶段核验，并确认稿件中已不存在未解释占位符。

## 10. 状态

**状态：第二次独立复审通过，审计已闭环（CLEARED FOR MECHANICAL BACKFILL）。**

## 11. 第二次独立复审

- 复审稿件 SHA256：`9c9469a16ad275c997f8e3456442166a91d477a121c194441ff6959347b4024c`。
- 复审结论：**PASS**；未发现遗留阻断。
- 第一轮识别的五项问题均已消除：cell时间与方法级run背景已分离；A.6的manifest机械规则已补齐；表7已补报相对GA有效实例数；表7的FIELD均按 `model_id/test_scale` 唯一定位并明确双字段拼接；E1风险RID族已限定为builder实际方法域。
- 核心静态计数：148个唯一具体RID全部符合builder注册规则；RID-FAMILY展开为418个引用；75个FIELD块均通过路径、字段和行选择器核对；13个Markdown表块均无列数不一致。
- 边界复核：`FINAL-MANIFEST`与 `all_restarts_feasible` 均无残留；Windows环境采用release/version中性表述；`verification/verification_report.json` 路径正确；未发现显著性、无条件因果、全局最优或扩展至未测试规模的越界断言。

本次复审为只读检查；上述通过结论只适用于该SHA256稿件快照。正式结果回填后仍须执行“源表→稿件表→正文”的双阶段数字一致性核验。

# Reviewer 02：builder 精确修订计划

本文件只给出实施方案；本轮未修改 builder、测试、既有表格或论文稿。以下行号均指修订前的 `output/supplementary-experiments/scripts/build_paper_tables.py` 与 `tests/test_build_paper_tables.py`。

## 1. 新增由锁定协议派生的表5

1. 在 builder 的 `PAPER_TABLE_FILES`（当前第34—42行）加入 `"table5": "table5_scale_design"`。该常量目前不参与控制流，但应完整反映正文表集合。
2. 在 `_table4()` 结束后、`_generalization_core()` 前（当前第1884—1887行之间）新增 `_table5_scale_design(protocol) -> tuple[pd.DataFrame, str]`。数据源只能是 `_load_inputs()` 已读取并校验的 run 内 `protocol_config.json`：当前第508—524行已完成 verified-run、协议形状和 canonical hash 绑定，因此不得改为读取仓库中的可变配置，也不得从论文稿反向抄表。
3. 生成顺序固定为 `protocol["models"]` 中的2个训练行，再按 `_scales(protocol)` 生成4个测试行，共6行。训练行用 `model.train_scale` 解析同一协议中的规模向量；测试行直接读取 `protocol["test_scales"][scale_id]`。建议 CSV 精确字段如下：

   - `object_type`：`training` 或 `test`；
   - `object_type_label`：`训练规模` 或 `测试规模`；
   - `design_id`：`Train-S`、`Train-L`、`Test-1`—`Test-4`；
   - `source_test_scale`：训练行所引用的 `train_scale`，测试行为自身ID；
   - `producers_count`、`waste_types_count`、`facilities_count`、`vehicles_count`、`periods_count`：协议中的五个显式规模字段；
   - `scale_signature`：以上五项按协议 `scale_fields` 顺序拼成 `3/2/2/3/2` 等展示值；
   - `quantity`：训练行固定为该行所代表的1份模型，测试行取 `_expected_instances(protocol, scale_id)`；
   - `quantity_unit`：`model` 或 `instance`；
   - `quantity_display`：`1份模型` 或动态的 `{quantity}个实例`；
   - `description`：训练行为 `生成{checkpoint}`；测试端点为 `与Train-S/Train-L同规模`，非训练端点按协议顺序标为 `未见中间规模1/2`。

   `scale_signature` 应由 `protocol["scale_fields"]` 驱动，同时要求该列表与模块常量 `SCALE_FIELDS` 一致；不要再次硬编码五字段的次序。训练规模数量不要写成“1个训练实例”，因为该数量不在协议的模型条目中；表5只据协议可证明地写“1份模型”。
4. Markdown 标题固定为 `表5  两种训练规模与四种测试规模设置`，列为“类型 / 编号 / 产废节点/废物类型/处理处置节点/车辆/周期 / 实例或模型数量 / 说明”，正文值由上述 CSV 行同源生成。表注只说明五项规模的排列顺序、四个测试集共用锁定的非规模生成参数、测试实例数量来自协议；不写任何结果数值。
5. 在 `build_paper_tables()`（当前第2972行起）调用新函数；在 `frames`（第3011—3021行）加入 `table5_scale_design.csv`，在 `markdowns`（第3022—3030行）加入 `table5_scale_design.md`。表5是设计表，不对应E1/E2/G指标，故不要传入 `_registry()` 或 `_adjudication()`，两者行数及 result-id 集合应保持不变。
6. 更新 CLI 描述（当前第3081行）为 “Tables 3, 4, 5, 6, 7”。

## 2. 表3、表4标题恢复“50个”

- `_table3()` 当前第1684行改为动态标题 `表3  小规模{expected_n}个未见实例的求解质量与最优性检验`。
- `_table4()` 当前第1861行改为动态标题 `表4  大规模{expected_n}个未见实例的算法性能比较`。

正式协议下两者即恢复为“50个”；smoke协议下应诚实显示“1个”，不能为满足正式稿而硬编码50。

## 3. 压缩表3、表4、表7的主文表注

只改 `_markdown_table(..., notes=...)` 的主文 Markdown，不删除 CSV 中的并发审计字段，也不改 `_concurrency_scope()`（当前第1237—1268行）、附表全方法的并发说明或 registry 的时间注释。这样 batch、workers、max_cells、整批墙钟和 superseded 批次仍完整保留在 manifest `evaluation_runs`、附录运行表以及可审计字段中，但不再逐批塞入主表。

- `_table3()` 当前第1699—1708行压成四句：
  1. 独立实例、计划n、均值±样本SD，并在同句解释异常计数四元组；
  2. Gap只用MILP证明optimal的实例，同时说明ΔJ正值含义；
  3. 动态报告Train-S一次性训练耗时，并明确不计入单元推理时间；
  4. PPO/GA/MILP时间是并发条件下的求解段观测值，不能据此作隔离运行下纯算法速度的因果比较；完整批次信息见附录运行表和 `manifest.evaluation_runs`。
- `_table4()` 当前第1874—1882行同样压成四句：统计单位/SD及异常四元组；ΔJ口径；Train-L训练时间不计入推理；PPO/GA并发求解段时间的解释边界及完整run信息去向。
- `_table7()` 当前第2183—2190行同样压成四句：实例内先等权平均偏好后的实例层均值±SD及异常四元组；相对GA改善率口径；两模型一次性训练时间不计入推理；PPO并发求解段时间的解释边界及完整run信息去向。

必须保留 `_table3/_table4` DataFrame 中的 `*_runtime_concurrency_scope`（当前第1657—1659、1837—1838行）和 `_table7` 的 `runtime_concurrency_scope`（第2127行），也必须保留 registry 对各时间指标的完整 `_concurrency_scope()` 注释；本项只解决论文主表表注冗长问题。

## 4. inventory 与原子发布的预期变化

现状为16个被哈希声明的非inventory输出、9个CSV行数条目；新增表5后应变为：

- `output_files_sha256`：18项（原16项 + 表5 CSV/Markdown）；
- `output_row_counts`：10项，其中 `table5_scale_design.csv = 6`；
- `build_paper_tables()` 返回19条路径（18个成员文件 + 最后发布的inventory）；
- `results-registry.csv` 仍为521行，`adjudication-log.csv` 仍为4行；不因设计表而增加结果记录。

当前原子发布逻辑无需改写：所有 frame/Markdown 先写入 staging（第3031—3039行），哈希后生成inventory（第3041—3053行），删除旧inventory完成标记后替换成员文件（第3061—3070行），最后替换inventory（第3071行）。将表5加入 `frames/markdowns` 后会自动进入该流程。失败时可能已有部分成员被替换，但inventory必须继续保持不存在；消费者应只把最后出现的inventory视为完整发布标记。

## 5. 测试修改与新增清单

1. 修改 `test_inventory_is_published_last_as_complete_output_marker`（当前第529—563行）：把 `len(declared) == 16` 更新为18，并显式断言表5 CSV/Markdown均被声明、存在且哈希匹配；保留注入 `results-registry.csv` 发布失败后inventory不可见的断言。
2. 扩展 `test_smoke_requires_flag_and_build_inventory_has_matching_ids`（当前第458—475行）：断言两份表5文件在返回路径中；CSV正好6行且列集合/行顺序符合上节约定；smoke的四个测试行 `quantity=1`，从而证明数量来自锁定协议而非硬编码50；inventory中 `output_row_counts["table5_scale_design.csv"] == 6` 且声明输出数为18。
3. 新增正式协议的纯函数测试（不运行求解器）：直接对正式配置调用 `_table5_scale_design()`，断言训练/测试行顺序、四个测试规模签名分别为 `3/2/2/3/2`、`6/2/2/4/3`、`10/3/3/5/3`、`20/4/3/8/4`，测试数量均为50，训练端点与对应测试端点五字段完全相等。
4. 新增 Markdown 回归断言：smoke输出的表3/表4标题含“1个”而非“50个”；正式纯函数或正式重建产物的标题含“50个”；表3/4/7主表Markdown不再出现 `batch`、`workers=`、`max_cells=`、`整批墙钟=`，但包含 `manifest.evaluation_runs` 和“不能作隔离运行下纯算法速度的因果比较”。同时断言 `appendix_all_methods.csv`、registry时间行或其并发字段仍非空，防止压缩主表时误删审计信息。
5. 全部单元测试通过后，使用已验证正式run重新执行 builder，再逐项核对18个声明哈希、10个CSV行数、表5六行内容以及inventory最后发布语义。该步骤只重建派生表，不重跑9750个求解单元。

## 6. 主要风险与防护

- **协议来源漂移**：若直接读仓库配置，可能与正式run不一致；必须只用 `_load_inputs()` 返回且已哈希绑定的 `protocol`。
- **smoke标题/数量造假**：标题与测试数量必须使用 `expected_n`，正式才是50；测试禁止硬编码。
- **训练数量越界推断**：协议证明一条模型记录对应一份模型，但不直接声明训练实例个数；CSV/Markdown不要凭稿件补写“1个训练实例”。
- **规模字段顺序错位**：`scale_signature` 必须依据锁定 `scale_fields` 并校验等于 `SCALE_FIELDS`，不能依赖任意字典遍历顺序。
- **审计信息被误删**：压缩仅限三张主表Markdown表注；CSV并发字段、全方法附表、registry和manifest必须保留完整run信息。
- **产物集失配**：代码变更会改变 builder SHA 和全部受影响Markdown/哈希；正式重建后必须同步论文表5及表3/4/7，并更新稿件中的产物哈希，禁止手改生成表。
- **发布中断**：继续以inventory为唯一完成标记；新增表5不得绕过staging或在inventory之后发布。

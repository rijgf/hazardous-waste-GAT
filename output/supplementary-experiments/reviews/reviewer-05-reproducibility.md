# Reviewer 05：复现性、哈希与种子审查

- 审查日期：2026-09-05（Asia/Shanghai）
- 审查角色：Reviewer 5（复现性／哈希／种子）
- 审查结论：**REVISE**
- 评分：**88/100**
- 是否需要重跑 9,750 个正式求解单元：**否**
- 问题分级：阻断 0；重要 3；轻微 3
- 被审正式初稿 SHA-256：`3fee329fe38c80beeafad049cb89f8c6c2860906fef22506dec44795b3fee912`

结论分为两层。第一，正式数值产物通过审查：种子集合、两份模型、冻结推理、9,750 个唯一单元、哈希链和验证门均无结果级错误。第二，当前“replication”目录及配套说明还不足以让第三方独立重跑，运行时确定性开关的表述也不够精确。因此判为 **REVISE**，但所需工作是补齐／校正复现材料与稿件表述，而不是重新训练模型或重跑正式实验。

## 一、需要修订的问题

### 重要 I-1：紧凑 replication 是可审计结果归档，不是独立可执行的复现包

**证据。** `src/supplementary_experiment.py:2287-2291` 将导出目标明确写为“compact auditable outputs”；`src/supplementary_experiment.py:2332-2352` 只复制 manifest、协议、schedule、汇总表、verification、训练历史和两份 checkpoint。实际 `output/supplementary-experiments/replication/inventory.json:4-91` 也只登记 18 个 payload；目录没有 README、精确源码、`training_instances`、`test_instances`、`initial_solutions`、`cells`、`solutions` 或 `traces`。与此同时，包内原样复制的 manifest 仍引用这些未携带的文件。导出实现本身没有错误，且 18/18 payload 的路径、字节数和 SHA-256 均通过；问题在于材料用途边界未被完整说明。

**影响。** 第三方可以核对已发布的模型、schedule、表格和 verification 哈希，但不能仅凭该目录执行 `verify`，不能重建表格所依赖的 9,750 个 cell，也不能从头重跑正式协议。初始化时 Git 工作树还是 dirty（`outputs/supplementary_experiment_v1/manifest.json:56-68`），所以 manifest 中的旧 commit 也不能单独恢复哈希锁定源码。

**所需修订。** 二选一即可，不要求重跑正式实验：

1. 将该目录明确命名／说明为“紧凑可审计结果归档”，添加 README，逐项写明可核验内容、不可执行边界、正式源运行目录或持久化归档位置，以及从 init 到 export 的完整命令顺序；或
2. 补入哈希锁定的 13 个源码文件、依赖规范和完成独立重跑所需的输入／再生说明。若声称能够直接复核现有正式结果，还必须提供 cell/solution/trace 或其可访问的不可变归档。

### 重要 I-2：没有完整依赖锁，初始化快照也被稿件呈现成了全程运行状态

**证据。** 仓库扫描未发现 `requirements*.txt`、`environment*.yml`、`pyproject.toml`、`poetry.lock`、`Pipfile*` 或 conda lock 文件。`src/reproducibility.py:148-196` 记录 Python、平台、少量包版本及 PyTorch/CUDA状态，但未锁定完整传递依赖、GPU driver、HiGHS精确版本或线程环境。`src/supplementary_experiment.py:20-22` 用 `setdefault` 设置 `CUBLAS_WORKSPACE_CONFIG`，却未记录进程实际采用的值。初始化 manifest 的三个开关是 false（`outputs/supplementary_experiment_v1/manifest.json:110-112`）；训练入口随后把 deterministic algorithms 和 cuDNN deterministic 打开（`src/supplementary_experiment.py:559-562`），冻结 PPO 构造过程也会把 cuDNN deterministic 打开（`src/ppo_improver.py:363-368`）。因此，正式初稿 `output/supplementary-experiments/drafts/数值实验与结果分析_正式结果初稿.md:318-327` 中“正式运行环境／PyTorch确定性相关开关=false”的行，只是初始化时快照，并非训练与推理全程状态；正文第 37 行称“完整环境”也略强于实际记录范围。

**影响。** 当前材料足以追踪本机本次运行，却不足以支持跨机器、跨驱动或位级一致的复现主张。初稿没有直接声称跨硬件位级一致，这是优点；但表 A3 容易让读者误解实际运行开关。

**所需修订。** 增加锁定依赖文件和运行 README；记录实际 `CUBLAS_WORKSPACE_CONFIG`、GPU driver、solver/HiGHS 与线程设置。将表 A3 改称“初始化环境快照”，并另列训练阶段与冻结推理阶段实际设置，或在表注中明确其时点和不覆盖全程。稿件中避免“完整环境”“跨硬件确定性”之类超出证据的措辞。

### 重要 I-3：本次 cell seed 全部唯一，但正式 verify 没有把种子唯一性作为机器门

**证据。** `src/supplementary_protocol.py:126-153` 通过 3 位 namespace lane 加 29 位 SHA-256 截断值派生 uint32 种子；`src/supplementary_protocol.py:437-466` 校验八个 namespace tag 完备且互异；`src/supplementary_protocol.py:156-220` 按模型、规模、实例、偏好、restart 与初始解身份派生求解种子。可是 `src/supplementary_experiment.py:883-884` 只检查 compact `cell_id` 是否碰撞，`src/supplementary_experiment.py:2027-2284` 的正式 verify 也没有直接断言 `len(seeds)==len(set(seeds))`，所以不同逻辑单元若偶然得到同一个 29 位 lane 内值，仍可能通过现有门。

**本次独立核查结果。** Reviewer 从锁定协议和 schedule 重派生并比对了全部 10,154 个种子记录：训练实例 2、训练算法 2、测试实例 200、初始解 200、PPO 6,000、GA 3,000、启发式 500、MILP 250；**10,154/10,154 唯一，派生不一致 0**。9,750 个 evaluation cell 的 seed、`cell_id` 和 `logical_cell_id` 也各为 9,750 个唯一值。训练与测试种子交集为空；200 个测试实例内容哈希互异，并且与两份训练实例内容哈希均不重合。因此这是验证门缺口，不是当前结果污染。

**所需修订。** 为本次发布保留一份机器可读 seed-audit（含各 namespace 数量、唯一数、交集和派生一致性）；下一版 protocol/verify 将 cell seed 唯一性、训练—测试种子互斥及训练—测试实例内容哈希互斥纳入硬门。无需为此重跑当前求解单元。

### 轻微 M-1：产物能证明“两个已提交训练结果”，不能严格证明进程层面从未发生中断重试

`src/supplementary_experiment.py:569-575` 会复用已登记且哈希有效的 checkpoint，`src/supplementary_experiment.py:577-638` 可从 commit/prepared 记录恢复；但 `improver.train` 在 `src/supplementary_experiment.py:643-651` 执行，prepared 记录直到 `src/supplementary_experiment.py:680-683` 才落盘。若进程在两者之间退出，下一次调用可能重新训练，而当前账本不会留下“已启动但未提交”的尝试记录。现有正式目录确实只有两个 checkpoint、两个 record 和各一份 20 行训练历史，verification 也报告 model/checkpoint 均为 2（`verification_report.json:35-36`）。建议把初稿第 9、191 行的“一次性训练”理解并表述为“每个模型一份正式提交训练记录／训练耗时”，不要扩展成“系统可证明从未发生任何中断尝试”。

### 轻微 M-2：paper inventory 保存了机器专属绝对路径

`output/supplementary-experiments/tables/paper-table-inventory.json:91` 的 `source_run_dir` 是 `C:\Users\...` 绝对路径。该字段不参与数值或哈希正确性，但会泄露本机目录结构并降低跨机器可移植性。下一次发布宜改为仓库相对路径，或同时提供相对路径并把绝对路径标为非规范性本地信息。

### 轻微 M-3：复现配套说明仍保留中间进度占位

`output/supplementary-experiments/scripts/script-index.md:23` 称“具体种子写入 manifest”，实际上 evaluation cell seed 位于由 manifest 引用的 schedule；同文件第 27 行仍写“待正式结果生成后补充”。`output/supplementary-experiments/logs/process-log-scholar-analyze-2026-09-04.md:23` 仍把正式 evaluation 标为“进行中”。`output/supplementary-experiments/logs/formal-provenance-facts.md:183-192` 也保留最终批次待回填语气。这些内容不会改变正式产物，但在提交 GitHub 前应更新或明确标成历史过程记录，避免与 `manifest stage=verified` 冲突。

## 二、通过的独立检查

| 检查项 | 结果 | 精确证据 |
|---|---|---|
| 协议规模与运行数 | PASS | 配置含两模型、四规模×50、五偏好及各方法 restart（`configs/supplementary_experiment.json:41-123`）；协议硬锁 6,000/3,000/500/250/9,750（`src/supplementary_protocol.py:647-656`）。 |
| 训练／测试隔离 | PASS | 八个互斥 namespace（`configs/supplementary_experiment.json:4-13`）；测试集只在两个模型冻结后生成（`src/supplementary_experiment.py:710-724`）；独立重派生得到训练—测试 seed 交集 0、实例内容哈希交集 0。 |
| 两模型正式训练产物 | PASS | manifest 各登记一份训练实例、训练 seed、checkpoint、策略状态、history 与 frozen_at（`outputs/supplementary_experiment_v1/manifest.json:115-162`）；checkpoint 重算哈希及策略 state hash 均吻合。 |
| 冻结推理 | PASS | 冻结构造 `trainable=False`、optimizer 为 `None`、`requires_grad_(False)`、eval（`src/ppo_improver.py:344-395,419-489`）；`improve` 使用 `@torch.inference_mode` 并比较前后 state hash（`src/ppo_improver.py:825-843,964-968`）；执行器逐 cell 再核 checkpoint/state（`src/supplementary_experiment.py:1075-1145`）。独立解析 6,000 个 PPO cell：expected/before/after policy hash 与 checkpoint before/after 全部一致，错误 0。 |
| 运行计数与唯一键 | PASS | schedule 9,750，`cell_id` 和 `logical_cell_id` 均唯一；raw results 与 cell ledger 各 9,750 个唯一 cell。六个 evaluation run 的完成数为 PPO 60+120+5,820、GA 3,000、启发式 500、MILP 250，失败均为 0（`manifest.json:3182-3288`）。 |
| verify 关键门 | PASS | 期望／观察计数一致，模型 2、checkpoint 2、测试实例 200、缺失 0、失败 0、完整性错误 0、artifact 75/错误 0、solver replay 15/错误 0、source changes 0、passed=true（`outputs/supplementary_experiment_v1/verification/verification_report.json:21-52`）；manifest 为 `stage=verified`（`manifest.json:7,3334-3337`）。 |
| 源码与结果哈希锁 | PASS | manifest 锁定 13 个源文件及 source bundle（`manifest.json:40-55`）；schedule、cell input、checkpoint、policy、solution、trace、cell ledger、10 张表和 verification 均形成闭合哈希链。 |
| 论文表 inventory | PASS | `paper-table-inventory.json:28-42` 的 14 个输入、`:53-69` 的 16 个输出全部存在且 SHA-256 一致；`:71-80` 的 9 个 CSV 行数全部一致；protocol hash 与四个 result ID 闭合（`:82-88`）。 |
| 紧凑归档完整性 | PASS（限“归档”含义） | replication inventory SHA-256 为 `7e1d236784b1558fb682f6d98e8c33a72ff5f27f3033c3457f1247b17bc0cfdc`；18/18 payload 的存在性、字节数、哈希及与正式源文件的一致性均通过，无额外未列文件。 |

## 三、正式初稿陈述核对

- **一致：** 第 9 行明确两份模型、各一个训练实例和训练种子，并说明测试期不更新、不微调、不按测试结果重选 checkpoint；与代码和产物一致。
- **一致：** 第 11 行的 9,750 总单元及 6,000/3,000/500/250 分解、0 失败／0 缺失，与 schedule、manifest 和 verification 一致。
- **一致：** 第 67、146、150、168、170、378 行把推论限定为两份固定 checkpoint、单训练实例／种子及当前四规模，未把测试实例重复错误解释为训练不确定性，边界表述合格。
- **一致：** 第 176 行“训练与测试种子集合互斥”通过独立全量派生核查。
- **一致：** 第 342-378 行的关键哈希、200 个测试实例、75 项 artifact 复核、15 个 solver replay 与 `passed=true` 均与正式产物一致，且正确说明 75/15 不是全量重放。
- **需修订：** 第 37、318-327 行应按重要问题 I-2 区分“初始化快照”与训练／冻结推理阶段实际开关；初稿还应增加重要问题 I-1 所述的复现包用途边界和获取／执行说明。

## 四、最终意见

**REVISE，88/100。** 没有阻断性问题，没有发现需要重训、重算或改写正式结果表的证据。完成以下发布前动作即可申请 Reviewer 5 复核：明确紧凑归档不是单目录可执行复现包并补 README／命令链；补依赖和实际运行时设置说明；附机器可读的全量 seed 唯一性审计；修正表 A3 的快照时点；清理或标记过时的进度文档。

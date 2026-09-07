# 独立复审5：复现性、哈希与种子

## 结论与评分

**结论：PASS**  
**评分：97/100**  
**问题分级：阻断0；重要0；轻微0。**

本次以独立复审者身份只读核对审定稿、正式manifest与verification、`REPRODUCIBILITY.md`、运行后环境freeze、机器可读seed audit、紧凑归档inventory及paper-table inventory。审查对象为SHA-256=`c6b1adf87b488d49820f73fcc73ade402e643c80b91b3c4818d7ab2da702721c`的 `output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md`；下列稿件行号均指该版本。初审Reviewer 5的发布前整改条件均已落实，未发现需要再次修改稿件或复现材料的问题。

扣除3分反映当前证据仍不是最高等级的独立、位级复现包：依赖freeze采集于运行后，实际CUBLAS值、HiGHS精确版本与MILP线程数未记录，seed唯一性尚未进入core verify，紧凑归档也未携带全量逐cell工件。这些限制均已准确披露并由替代审计补证，故不构成本轮REVISE事项。

## 逐项复核

### 1. 正式提交训练记录边界

- 正式manifest只含2个模型条目与2个训练实例条目；Train-S、Train-L各有一个checkpoint和一份20行训练历史，文件哈希均与manifest一致。
- 审定稿第9行及表A1第184—196行写成“每个模型只有一份正式提交训练记录”，并明确现有账本不能证明操作系统进程层面从未发生提交前中断；没有把正式产物数量扩大为“从未重启训练”的不可审计断言。
- `REPRODUCIBILITY.md`第6节采用同一边界，并说明冻结推理的checkpoint、策略状态与训练历史证据。

**判定：通过。**

### 2. 初始化、训练与冻结推理设置

- manifest环境节点只记录初始化时点：deterministic algorithms=false、cuDNN benchmark=false、cuDNN deterministic=false。
- 训练入口源码设置 `torch.use_deterministic_algorithms(True, warn_only=True)`、cuDNN deterministic=true、benchmark=false；冻结PPO构造设置cuDNN deterministic=true、benchmark=false，并使用 `trainable=False`、无optimizer、`requires_grad_(False)`和inference mode。evaluation入口没有显式调用全局 `torch.use_deterministic_algorithms`。
- 审定稿第37行和表A6第320—333行将三阶段严格分开，没有把初始化快照误写成全程状态；`REPRODUCIBILITY.md`第2、6、7节及environment freeze采用一致口径。

**判定：通过。**

### 3. 未记录环境项没有被推断

- 审定稿表A6第330—333行把实际 `CUBLAS_WORKSPACE_CONFIG`、HiGHS精确版本和MILP solver线程明确写成“未记录”，并把driver 591.86与pip freeze标为2026-09-05运行后快照。
- 表A7第341—348行对六个evaluation run逐条保留：PPO、GA与启发式solver线程“不适用”，MILP线程“未记录”，所有run的device均为“批次未记录”；没有以workers、当前GPU或库默认值反推缺失字段。
- environment freeze共有37条非注释pip记录；与当前 `py -m pip freeze` 的37条逐项一致，但文件仍诚实声明其为运行后快照而非求解前依赖锁。

**判定：通过。**

### 4. Seed audit及core硬门边界

- `audits/seed-audit.json`状态为PASS：10,154/10,154条seed唯一，重复0；9,750个evaluation seed、cell_id和logical_cell_id各自全部唯一；seed派生偏差0、evaluation身份偏差0；训练—测试/评估seed交集0，训练—测试实例内容哈希交集0。
- JSON内协议、正式配置、schedule与manifest哈希均命中当前文件；Markdown摘要与JSON一致。
- 审定稿第386行明确这是一份独立补充审计，当前正式verify尚未把相关断言设为core硬门，硬门修改推迟至下一协议版本以保持已验证source bundle不变。没有把补充audit伪称为原始verification的一部分。

**判定：通过。**

### 5. 紧凑归档能力边界

- `replication/inventory.json`登记18项payload，18/18均存在，字节数与SHA-256全部一致；连同inventory共19个文件、27,799,907字节。完整正式run实有29,672个文件、1,099,073,679字节，二者能力范围明显不同。
- `REPRODUCIBILITY.md`第1、3、8节明确紧凑目录不含training/test instances、initial solutions、cells、solutions、traces或锁定源码，不能单目录执行既有verify或重建9,750个cell；同时给出从空目录运行的完整命令链和逐字节审计所需条件。
- 审定稿第170、388行同步称其为“紧凑可审计结果归档”，而非独立可执行复现包。

**判定：通过。**

### 6. 表A8关键哈希

表A8第352—369行的14个锚点全部命中当前证据：

| 锚点类型 | 独立核对结果 |
|---|---|
| protocol canonical hash | 与manifest、schedule及seed audit一致 |
| protocol配置、schedule、manifest、verification文件哈希 | 4/4命中当前正式run文件；紧凑副本亦一致 |
| source bundle | manifest锁定值一致；13/13当前源文件哈希与init记录相同，verification的source changes为0 |
| cell账本 | canonical row hash同时匹配manifest与verification；CSV文件哈希另列且命中，没有混淆两种语义 |
| compact inventory、paper inventory、builder、REPRODUCIBILITY、seed audit JSON、environment freeze | 6/6命中当前文件 |

paper-table inventory另行核对为formal模式、repository-relative来源；14项输入与18项非自身输出全部存在且哈希不匹配数为0，10个CSV行数登记完整。

**判定：通过。**

### 7. 表A9 Markdown表哈希

表A9第371—382行登记的表3、表4、表5、表6、表7及三张builder附表共8个SHA-256，逐一重算后为8/8命中；这些值也与paper-table inventory的对应输出字段一致。

**判定：通过。**

### 8. 75次与15次验证对象

- verification report为 `passed=true`，计划与完成计数均为9,750，missing、failed和integrity error均为0；artifact recheck为75/错误0，solver replay为15/错误0。
- 验证源码显示，75次artifact检查读取已提交方案并重新计算可行性、plan hash及质量字段；15次才重新调用相应求解方法，并检查可行性及按方法定义的目标/计划一致性。
- 审定稿第384行和 `REPRODUCIBILITY.md`第8节准确区分75次artifact metric recomputation与15次solver replay，并明确二者都不是对9,750个单元的全量重算。

**判定：通过。**

## 反向错误扫描

- 未发现“完整依赖锁”“完整运行环境”或“跨硬件位级一致”之类超出证据的表述。
- 未发现对CUBLAS实际值、HiGHS精确版本、solver线程或run级device的推断值。
- 未发现把10,154条seed audit写成core verify既有硬门的表述。
- 未发现把紧凑归档称为独立可执行包、全量正式run或可单独复核全部cell的表述。
- 未发现A8 canonical-row hash与CSV file hash混用；A8、A9也未发现陈旧哈希。
- 未发现把75次artifact复算误写成solver replay或全量重放的表述。

## 最终意见

初审Reviewer 5的3项重要问题与3项轻微问题均已得到与当前证据相称的处理。现版对“可追溯”“近似重建”“从头重跑”“逐字节审计”四种能力边界区分清楚，哈希与种子声明可由当前文件独立核对。建议通过复现性、哈希与种子复审。

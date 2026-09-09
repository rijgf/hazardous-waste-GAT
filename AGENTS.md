# 本仓库的实验约定

## 冻结 V9 全部 PPO 的 12000 预算重跑（v11，2026-09-10，实验已完成）

- 用户明确指定 V9、每偏好12000候选、PT不重训、重跑全部PPO实验，完成后提交并推送GitHub main。本段优先于历史探索的仅单点及不自动推送约定。只更新正式Markdown，不更新Word；不引入V10探索设置。
- 登记仍为 `configs/ppo_models_frontier_v9.json`，两份PT及其SHA256与下方V9完全一致。保留原实例、绑定b、偏好与种子、width8、temperature2、horizon1024和去重；唯一搜索参数变化为每偏好1920→12000，完整21偏好前沿上限252000。真实耗尽时提前停止，不能填满计数或转移余量。
- 新入口 `run_ppo_budget_v11.py`，目录 `output/pareto-ppo-budget-v11/`，协议SHA256 `db90a62fa5dab3d2180e1f640239fba195fe6d9470382f6c91ecdcee96e68ce5`。48次新完整前沿（跨规模24、非基准敏感性24），15次小规模求解；大规模比较与敏感性基准复用新Train-L/Test-4三次结果。每次完整前沿独立，不取三次最优来替代覆盖均值。
- NSGA-II保持原40320/120960，复用15档案及真实旧耗时；MILP复用V9五组3600s上限已完成记录，不重新求解。PPO上限为B1的6.25倍、B2的约2.0833倍，不能称为等预算或公平速度比较。
- 回退包 `backups/ppo-v9-before-budget-v11/`。新动作日志为gzip无损JSONL；每个前沿保存真实PID、UTC起止、执行ID和完成哈希。预检前1920动作与V9逐字节解压一致。运行冻结源码不得改动；新报告与独立审核另存。阶段预检不等于交付PASS，最终须完成全日志CRC/计数、63档案及15小解完整MILP重放、表图和正式稿哈希核验。
- 48次完整前沿和15次小规模求解全部完成，失败0；不要重复启动。前沿实际9589192/12096000次候选，259个偏好真实提前耗尽；小规模实际17040/180000，15次均提前耗尽。新表5两档NSGA预算、各3配对均N→P=0、P→N=1；仅是本固定案例与不同明确预算下的档案覆盖结论，不表示真实完整前沿或无偏泛化。大模型0.5/0.5三次平均J=0.1952599818，三个该偏好最终点风险仍高于历史NSGA最强标量点，不把单点标量胜出称为单点Pareto支配。
- 本轮完整数值审核已回放63档案、1023日志和13实例/情景的6390份去重保存及最终方案，原MILP约束违规为0；全部171项回归测试通过。正式文稿已更新，保留三种敏感性图和文字LaTeX公式。最终含文稿审核入口为 `py -B audit_ppo_budget_v11.py --require-report`，证据在 `output/pareto-ppo-budget-v11/audit/`；整体交付PASS必须对应当前 `numerics-with-report.json` 与独立终验的正式文稿、表图和来源哈希。
- 含正式稿数值终验已PASS：`numerics-with-report.json` SHA256 `3fc7bdeef3a49a91d551081e86dcf849c6000a1d2f8269339382920eb3b513db`，正式Markdown SHA256 `198666d3bdb2b55b1ce4e69babca519533f8cbf8f391509ddf68bd01776a9324`。从实际Git暂存区导出的干净副本亦通过171项测试和协议源/PT/实例哈希校验；不以本地未暂存文件充当复现依赖。


## 优化 PPO 完整实验重跑（v9，2026-09-09，已完成）

- 用户本轮授权按 v8 优化逻辑重训大小模型、重跑原文稿全部实验，额外智能体独立核验交付。范围恢复完整21偏好前沿，不再仅限0.5单点；不更新Word，未要求提交或推送。
- 两份新模型已完成，登记 `configs/ppo_models_frontier_v9.json`。小模型 `outputs/ppo_frontier_v9/models/small_model.pt`，SHA256 `2d22202e39eeb681b38f1dd34dcf84f55c125828877cd66c1241ec2a7fdfcf00`；大模型 `outputs/ppo_frontier_v9/models/large_model.pt`，SHA256 `0269a9a52a6e25ef1ed9d89ba8861ba35aa09e5d9b37643410baea1ee64fce54`。不要重复训练或覆盖。两者各24训练实例、36864动作、1152 Adam、96维、1层Transformer、原Linear价值头，预排episode覆盖21偏好，seed20260909。完整24×21偏好动作计数在 `outputs/ppo_frontier_v9/training/`。
- 本地Chen论文第15页和第16页表7/8的1000/3000为iterations，不能证实等价于本仓库candidate_attempts。用户随后明确“预算还是不变，总结好结果吧（可以复用nsga2）”：本轮保持原40320/120960次候选预算，复用v4的12次NSGA演化、15份预算档案及真实旧耗时；不重跑NSGA，也不改成1000秒或1000/3000候选。复用记录保留原字节、原协议哈希，并由v9协议登记来源文件哈希。旧NSGA为16并发、每进程1线程；新PPO初始3并发，后辅助调度的重叠上界15（不是实测全局峰值），每进程1线程，不据此宣称严格速度倍数。
- 新入口 `run_ppo_frontier_v9.py`：本轮已执行 `prepare --nsga-budgets 40320 120960 --reuse-nsga`，协议SHA256为 `5452d423c3293be00e5e5ce578b57e2e4cb0bb2c248b5d3ba32e5b555bf59ce5`，15份NSGA档案字节复用完成。48次正式PPO前沿及15次small求解全部完成，不重复启动。PPO按原文稿保留每偏好1920上限、21偏好40320上限，宽度8、温度2、horizon1024、同状态去重。正概率动作支持耗尽时真实提前停止，保留半批及实际次数，不补虚构计数：前沿实际1845401/1935360次尝试、145偏好提前耗尽；small实际17040/28800次，15次提前耗尽。
- 五组新3600秒上限MILP已全部完成并normalize，输出 `output/pareto-ppo-v9/milp/`，不要重跑。五组solver日志均status0/success/gap0；p3的raw proven_optimal=false来自原整数容差与目标评价交接不一致，不是限时。规范化后完整矩阵及目标一致性通过；报告层 `milp_optimality_verdict` 显式核验solver证明与规范化解，单独记录derived最优状态并保留全部原flag和hash，最终独立审核再次检查。
- 报告已完成暂存、独立检查及 `--publish`。`py -B audit_ppo_frontier_v9.py --require-report` 正式终验PASS：63前沿档案、1023偏好动作日志、27独立覆盖配对、5组表4择优行及5组MILP交接核验；13案例3768份按实例去重保存方案原MILP矩阵0违规，最大行残差1.4211e-14。终验 `output/pareto-ppo-v9/audit/numerics-with-report.json` SHA256 `4f5aa75f63635b1adb27f502c290b5547839839fb7e4a3d133add74b0be8b806`。初次 `numerics.json` 为非正式中间检查，不引用为交付PASS。162项测试通过，三图视觉记录为 `audit/visual-qa-v9.md`。动作核验含记录计数、有限logprob及同输入不重复，不声称独立重执行全部神经概率。
- 正式文稿已由 `report_ppo_frontier_v9.py --publish` 更新，SHA256 `60a436e7d0264149fa29e80f13392078022f221113982c3f3125264360f36861`。原文稿备份为 `backups/ppo-v8-before-full-frontier-v9/数值实验与结果分析_论文稿_v4.md`，SHA256 `c7e183186517bc800549c40a2fb65487146a37c6a6c9b3aee6443b3b1e70d904`。新协议说明为 `docs/ppo_frontier_v9_protocol.md`。旧源、模型及NSGA档案保留，Word未更新。
- 主3worker队列计算capacity情景时，另外以同一冻结CLI、精确task ID并发启动最后12个此前未启动的coload情景任务，现已全部完成并由主队列正确REUSED。记录在 `output/pareto-ppo-v9/orchestration/coload-aux-20260909T142107Z/manifest.json`。11份辅助入口execution存在，`PPO-coload-130-large-r2`的一份原始execution缺失、原因未确定，不能补造；命令/PID/日志/completed及21动作日志完整并已交叉核验，正文及独立报告保留此warning。
- 本轮主要结果：小规模5组MILP均在声明容差内最优交接，PPO2组达到相同J；Test-3的Train-L相对NSGA为N→P=0/P→N=1；Test-4为B1 N→P=0.7781±0.1307/P→N=0，B2 N→P=1/P→N=0，完整前沿仍落后NSGA。不得套用v8单偏好12000候选胜出结论，本轮多偏好模型每偏好只有1920上限；不据不利结果追加预算或改实例。
- 独立智能体的最终完整性结果为 `output/pareto-ppo-v9/audit/independent-completeness-final-v9.json` 及同名Markdown：PASS_WITH_1_DISCLOSED_WARNING，0项关键缺陷；唯一warning为上文一份辅助执行元数据缺失。完整性核验覆盖正式文稿、表图、来源、预算和最终数值审核哈希。以后引用PASS前仍应核对当前文件哈希；不能把这轮单实例案例报告为无偏多实例泛化。

## PPO 固定预算深化探索（2026-09-09，v8 最新探索登记）

- 用户允许优化输入、网络宽度及算子，但不增加每次候选预算、不叠加 Transformer；继续只做固定 Test-4、0.5/0.5。PPO 选择算子及全部动作对象，不能改为启发式对象选择。
- 当前探索保留版本为 `configs/ppo_objects_v8_exploratory.json`：检查点 `output/ppo-objects-v8/train/conditional96/large_model.pt`，SHA256 `3a2dcf8e1476901db5fb224e2c8d37f47e031f104ca63e14797a38f538fb9594`。入口 `run_ppo_objects_v8c.py`，96 维、1 层 Transformer、三个独立参数的条件化 512 类对象头、处理能力掩码、`--memoize`、温度 2、宽度 8、horizon 1024、每次 12000 候选。
- 状态记忆只在计划及截断后的进度输入均不变时保留已试完整动作；从 PPO 剩余联合概率中直接抽样，不额外重抽或评估。任一输入改变就清空。没有增加算子、目标导向插入或 NSGA 热启动。
- 三个原种子平均 J=0.1877679943，较 v7b 低 4.65%，较最强已有 NSGA 标量点低 17.89%。只是固定偏好的标量比较：PPO 平均成本更低、风险仍更高，不能声称 Pareto 支配或无偏泛化。对象头均匀、全部均匀消融均保留。
- 用户追加的价值头非线性对照也已完成：原共享上下文已含 GELU，最后 Linear 本身就是全连接层。只增加价值头隐藏层后，普通搜索均值从 0.2059328594 到 0.2049907459，状态记忆版从 0.1877679943 退至 0.1951925485，因此不替换当前保留模型。对照模型在 `output/ppo-objects-v8/train/critic_mlp96/`，旧 PT 均不覆盖。
- 本轮六份新模型均从原 24 个大规模 TRAIN 实例从头训练，每份 36864 次训练动作、1152 次 Adam 更新、1 层 Transformer。只保存预定最终检查点；架构/版本按固定测试反馈探索，必须明确这一局限。
- 结果与失败记录见 `output/ppo-objects-v8/results.md`，协议 `docs/ppo_objects_v8_protocol.md`。69 次完成试验、1 次未完成启动及同种子重试全部留档。完整动作概率/状态路径/原 MILP 核验入口 `py -B audit_ppo_objects_v8_final.py --workers 5`；生成报告后再运行 `py -B audit_ppo_objects_v8_report.py`。PASS 必须对应当前文件哈希，不能引用过期审核。
- v7b 回退包在 `backups/ppo-v7b-before-fixed-budget-optimization-20260909/`。本探索登记不自动替代下方正式 v4 冻结模型，不改正式论文/Word/NSGA 前沿，也不自动提交或推送 GitHub。

## PPO 全对象决策探索（2026-09-09，最新用户要求）

- 用户要求修复路线归属、候选边及对象索引，由 PPO 决定算子和全部动作对象，保留三个独立 512 分类头。下方 v6 的启发式/随机对象选择不能作为此要求的成功结果。
- 当前协议：`docs/ppo_objects_v7_protocol.md`；独立代码：`src/ppo_objects_v7.py`、`run_ppo_objects_v7.py`。改造前快照为 `backups/ppo-v6-before-object-encoder-20260909/code.zip`，旧 v4/v5/v6 文件保留。
- 只比较固定 Test-4、偏好 0.5/0.5，NSGA-II 最强已有标量点 J=0.2286833693057271。禁止用 cheapest insertion、修复重分配或 NSGA 方案替网络选动作对象。
- 新训练只读取 v4 的 24 个大规模训练实例和固定 b，检查点及来源在 `output/ppo-objects-v7/train/`，探索结果在 `output/ppo-objects-v7/evaluate/`。新模型未登记为正式默认，不覆盖冻结 pt。
- 保留失败试验，执行完整约束重放和无学习对照；固定测试上的反复调试不能表述为无偏泛化验证。当前授权不包含自动推送 GitHub。
- v7b 全对象探索已完成，登记为 `configs/ppo_objects_v7_exploratory.json`，检查点 `output/ppo-objects-v7/train/v7b_capacity_context/large_model.pt`，SHA256 `7d0cf70b185ec544b5bf52677566ff4ce35f208261ab39ed6f2f9187f44acc40`。温度2、12000候选、三个固定种子平均 J=0.1969241911，均超过旧NSGA最强点；对象头消融平均收益仅1.94%，不能夸大为全部收益来自学习。
- 报告 `output/ppo-objects-v7/results.md` 保留26次探索；完整来源/动作概率/原MILP核验入口 `py -B audit_ppo_objects_v7.py --workers 4`，120项测试通过。该模型是0.5/0.5专用探索版本，其他偏好未验证，不自动替代下面v4两个正式冻结模型。

## 单点自主算法探索（2026-09-09，v6 历史结果）

- 用户要求保存当前版本后自主修改/简化算法、探索和修复，直至大规模风险偏好(0.5,0.5)结果优于已有NSGA-II；不再默认每轮计算完整前沿，尽量不增加算子复杂度。
- 回退点为`backups/ppo-v4-before-exploration-20260909/`，源代码压缩包与三份模型副本均有哈希。保留历史失败实验，禁止修改NSGA-II档案、固定测试实例或绑定b来制造胜出。
- 本轮固定Test-4原实例，标量目标为`0.5*C/b_C+0.5*R/b_R`。更严格的既有比较目标采用两档预算、三次重复的全部NSGA-II前沿中最小值0.2286833693057271，实际引用须从档案重算。
- 每轮保存方法版本、参数、种子、候选评估数、时间、C/R/J、结果方案及合法性检查；按测试结果调试属于单实例探索，不能宣称无偏泛化或算法普遍优越。不得从NSGA-II解热启动后冒充独立PPO结果。
- 新算法和新模型另存；若旧PT只作为混合搜索的一部分，必须准确标注并检查消融，不能把其余启发式收益归因于PPO学习。当前授权允许实现与必要重训，但不是自动推送GitHub的授权。
- 本轮已实现独立六算子分支`src/compact_search.py`，入口`run_ppo_scalar_exploration.py`；七次8000候选单点搜索完成。PPO算子引导版三个原定种子J为0.1859716503、0.1999328134、0.1973335555，均小于最强既有NSGA-II点0.2286833693；无PPO消融同样三次胜出，均值接近，不能宣称学习引导已显著优于普通简化搜索。本轮未重训、替换PT或恢复v5优势合成修改。
- 本轮终验与报告为`output/ppo-scalar-exploration-v6/audit.json`和`results.md`；`py -B audit_ppo_scalar_exploration.py`核验七份新结果和六个NSGA-II基准点的完整MILP约束、目标与哈希，PASS须对应实际交付文件。当前110项测试通过。
- 共同校验器`src/solution_utils.py`补齐BG/BD服务前总容量检查，未放宽数学约束。旧v4完整源码锁会识别这项版本变更；历史复现应使用回退包，不重写原协议哈希。旧8算子和PPO推理源码原件仍保留，新六算子分支不冒充旧训练网络的原生对象输出。

## 优势合成修复已按用户要求回退（2026-09-09，优先于下方v5记录）

- 当前`src/ppo_improver.py`已恢复到v4原始字节版本（SHA256 `025b0c9ecb3b686ea89fcf42929596d383c7791335dc77d7c53f01fb70d9571b`）。默认仍使用原v4冻结模型，不启用v5模型。
- v5失败对照的模型、结果、协议和历史审核保留，不删除或改写。v5修复源码及专属测试归档在`output/ppo-scalar-advantage-v5/source-v5/`；专属测试已移出当前测试发现目录。
- 下方v5段落描述的是回退前的历史运行状态。当前代码不再满足v5源码锁；直接运行其训练/审核入口应拒绝版本不匹配，不能修改协议哈希绕过。复现v5须在独立副本中使用对应归档源码和测试。
- 本次后续工作仅诊断其他问题，不自动应用新修复、重训或修改NSGA-II历史结果。临时诊断程序在内存执行，不保存。
- 回退后诊断记录见`docs/ppo_v4_followup_diagnosis_20260909.md`：发现60步局部搜索的访问数可达范围限制、无变化动作误走−0.2失败惩罚分支、未选路段参数不可见反例。固定1920候选的60×32/240×8单偏好单种子诊断中，大规模J由0.494896降至0.419853；不是新的正式前沿比较，不据此替换既有表格或训练参数。诊断方案通过完整MILP复核，106项当前测试通过。

## 大规模优势合成修复对照（2026-09-09，仅限本次Test-4前沿比较）

- 用户授权将PPO训练中的优势处理改为“按每个样本的偏好合成标量优势，再整体标准化”，并仅重新训练大规模模型、重做单个大规模前沿对比。其他训练参数、网络、数据、种子及推理预算不变。
- 新大规模模型已训练完成并冻结：`outputs/ppo_scalar_advantage_v5/models/large_model.pt`，SHA256为`574d50293757caa8d704466fa0c38f953feb295f02485283e3969f409a86914a`；登记为`configs/ppo_large_scalar_advantage_v5.json`，别名`New-L-v5-scalar-advantage`。
- 该登记仅用于v5修复对照，不自动替换其他历史实验的v4模型对。小规模模型未重训；旧两份PT和NSGA-II结果均保留。
- 本次测试使用原v4的`Test-4/000`及绑定b；3次完整PPO前沿，各21组偏好×60步×32候选。NSGA-II的B1/B2档案直接复用，不重新生成或求解。
- 运行入口为`run_scalar_advantage_v5.py`，独立对照结果目录为`output/ppo-scalar-advantage-v5/`。本轮训练、3次前沿比较及`py -B audit_scalar_advantage_v5.py`终验均已完成：`audit.json`为PASS，767份按方法去重的前沿方案通过原MILP完整约束、变量界、整数性及成本/风险重算；3325份历史输入文件哈希保持不变。审核须与当前实际文件哈希对应。
- 本轮两档NSGA-II预算、3次重复均为NSGA-II覆盖新PPO 100%、反向0%；不能声称单独修复优势合成解决了性能差距。进一步的路线状态编码信息缺失和不等长微批次权重问题记录在`output/ppo-scalar-advantage-v5/further_diagnosis.md`，均未混入本轮修复或重训，不能把反例当作对实际性能损失的定量归因。
- 当前PPO源码已更新，v4原源码按原字节归档在`output/ppo-scalar-advantage-v5/source-before/src/ppo_improver.py`。不要修改旧协议的源码哈希来掩盖版本变化；旧源码锁的完整复现须使用对应历史源码版本。实验边界和GPU确定性限制见`docs/ppo_scalar_advantage_v5.md`。

## 当前参数修订后的冻结PPO模型（2026-09-08，优先于下方历史约定）

用户明确追加要求：处理能力与共载参数修订后，两个规模的PPO都必须重新训练。新训练已完成并通过独立训练来源核验；后续修订参数实验使用以下两份冻结模型，不覆盖旧模型：

- 小规模：`outputs/parameter_revision_v4/models/small_model.pt`（`New-S-v4`）。
- 大规模：`outputs/parameter_revision_v4/models/large_model.pt`（`New-L-v4`）。
- 当前权威登记为`configs/ppo_models_parameter_revision_v4.json`，实验前校验其中的模型与数据清单哈希。旧`configs/frozen_ppo_models.json`保留用于v2/v3来源核验，不作为修订参数实验的默认登记。
- 新训练集为`datasets/parameter_revision_v4/train/`，仍为两个规模各24个实例；网络、训练参数及种子保持旧规格，模型从头训练后冻结，不在测试上微调或按测试成绩选择检查点。
- 后续实验仍按所需规模各1个实例；修订参数实验默认读取`datasets/parameter_revision_v4/test/<规模>/000.json`。表4另使用`output/pareto-parameter-revision-v4/instances/small-fixed.json`，不把它与Test-1混为同一个实例。
- 新参数与公共参考方案规则见`docs/parameter_revision_v4_protocol.md`。每个实例/敏感性版本均读取自身预先绑定的b，输入、奖励和择优一致，搜索期间不重算b。不能将旧模型、旧参数实例及新结果静默混用。
- 本轮正式实验与文稿已完成：60个前沿任务产生63份档案，另有15次小规模PPO和5组3600秒上限MILP；其中3组MILP已证最优、2组仅限时可行。含文稿的独立终验见`output/pareto-parameter-revision-v4/audit/independent_numerics.json`（PASS），须与实际文件哈希对应。完整终验命令为`py -B audit_parameter_revision.py --require-report`，不能把仅预检或不含文稿的审核冒充最终交付核验。

## 历史冻结PPO模型（v2/v3来源保留，不再作为修订参数实验默认）

参数修订前，用户曾确认使用`reference_generalization_v2`训练得到的两份pt。以下规则保留用于历史实验复现；修订参数实验遵循上方v4约定。

- 小规模：`outputs/reference_generalization_v2/models/small_model.pt`（本轮结果中的`New-S`）。
- 大规模：`outputs/reference_generalization_v2/models/large_model.pt`（本轮结果中的`New-L`）。
- 权威路径、SHA256及训练来源登记在`configs/frozen_ppo_models.json`。开始涉及PPO的实验前先读取该登记，并校验将要使用的模型文件哈希。缺失或不匹配时停止并向用户说明，不自动重训、替换或回退到旧pt。
- 默认只做冻结推理：使用`PPOImprover.from_frozen_checkpoint(..., objective_refs=(b_C, b_R))`，不调用训练或优化器更新，不覆盖这两份检查点。只有用户明确要求重新训练或更换默认模型时才改变此约定；新模型另存，保留本次原件。
- 两份模型使用`objective_normalization=instance_reference`。必须传入目标实例自身预先固定的b_C、b_R；输入、奖励及求解择优保持相同归一化。不能使用旧的1000/100编码，也不能随风险偏好、重启、搜索起点或搜索过程重新计算b。
- 配套训练/测试数据说明见`datasets/reference_generalization_v2/README.md`；本轮结果见`outputs/reference_generalization_v2/results.md`。复现该轮结果时使用其记录的实例、偏好、种子及CPU评估条件；变更设备会改变随机数流，须在新实验中说明。
- 历史`Old-S`/`Old-L`仅在明确需要历史对照的实验中使用，不能冒充或替代默认模型。实验记录应标明实际模型路径与SHA256。

## 后续实验默认先用单实例（2026-09-08）

用户新增要求：做实验时先只用一个实例，结果不理想再考虑更换，不默认批量跑多实例。

- 对每个本次实验确实需要的规模，先固定使用1个实例。若只研究一个规模，总共只用1个实例；只有明确涉及多个规模时，才为各个所需规模各用1个。不因数据集中有四个规模就自动跑四个规模。
- 用户未指定实例时，按编号顺序先选`datasets/reference_generalization_v2/test/<规模>/000.json`，不根据历史成绩挑选起始实例。同一比较中的PPO、GA、MIP等方法使用同一个实例及其固定b_C、b_R；不同风险偏好和重复求解也不更换实例。
- “单实例”是减少独立实例数，不意味着把重复求解次数改为1；偏好、重启次数和求解预算按具体实验约定执行。不改变已冻结的两份pt或训练参数。
- 结果不理想时可以再更换实例，但须保留已运行实例、所有结果及更换原因；更换后的全部方法统一使用新实例自身保存的b值。不能通过更换实例删除或隐瞒不利结果。
- 单实例以及查看结果后换例的实验作为探索性/案例实验报告，不能将挑选后的结果表述为无偏的多实例泛化结论。正式多实例验证须另行明确安排。
- 保留现有40个测试实例作为备选库，不删除、不重写已锁定的数据集或历史3000次实验结果。`run_reference_generalization.py evaluate/all`仍是历史全量复现入口，不要将其直接作为未来单实例实验的默认入口。

这是仓库内的持久约定；后续用户的明确新指示优先。本约定不表示已设置操作系统级文件只读，也不表示模型已推送到远程仓库。

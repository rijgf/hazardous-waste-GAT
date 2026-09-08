# 本仓库的实验约定

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

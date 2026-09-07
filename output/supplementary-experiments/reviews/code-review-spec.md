# Code Review — Spec 轴

## 审查范围与结论

- 固定点：`0a71b0bb1b2445ce90989d453cfba357d0d12458`。
- 审查时 `HEAD` 与固定点相同，`git log <fixed-point>..HEAD` 无新增提交；因此本次按委托要求审查固定点之后的**当前工作树**，包括已跟踪修改、未跟踪的补充实验实现/测试/论文表产物，以及本机 ignored 的完整正式运行 `outputs/supplementary_experiment_v1/`。
- 规格源：`供应链管理写作/补充实验.md`。
- 本报告只做 Spec 轴；未执行 Standards 轴，也未修改实现、正式运行、结果表或稿件。
- 结论：**PASS。未发现 P0、P1、P2 或 P3 规格偏差；无阻断项。**

| 严重度 | 数量 | 结论 |
|---|---:|---|
| P0 | 0 | 无 |
| P1 | 0 | 无 |
| P2 | 0 | 无 |
| P3 | 0 | 无 |

## 逐项规格闭合

| 规格项 | 判定 | 实现与产物证据 |
|---|---|---|
| 仅开展多实例统计与泛化，不开展敏感性/消融（`供应链管理写作/补充实验.md:3,119,154`） | PASS | 锁定配置只有 E1/E2/G 所需方法与规模（`configs/supplementary_experiment.json:41-123`），协议校验器固定四规模、两模型和方法范围（`src/supplementary_protocol.py:499-545`）。审定稿明确声明未开展参数敏感性或网络/特征消融（`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md:154,168`）。未发现额外敏感性/消融运行产物。 |
| 两份模型、各一份正式训练结果、冻结测试，不逐实例训练（`供应链管理写作/补充实验.md:4-6,16,40-49`） | PASS | 配置严格限定 Train-S/small_model.pt 与 Train-L/large_model.pt（`configs/supplementary_experiment.json:41-51`）；训练入口对已提交且哈希有效的 checkpoint 直接复用，并以 prepared/commit 记录原子提交（`src/supplementary_experiment.py:545-707`）；只有两份模型均冻结后才允许锁测试集（`src/supplementary_experiment.py:710-724`）。冻结加载使用 `trainable=False`、无 optimizer、`requires_grad_(False)`、`eval()` 和 `torch.inference_mode()`（`src/ppo_improver.py:344-395,419-489,503-515,825-843,964-967`），每个 PPO cell 又核对推理前后 policy/checkpoint 哈希（`src/supplementary_experiment.py:1075-1145`）。正式报告确认模型 2、checkpoint 2（`outputs/supplementary_experiment_v1/verification/verification_report.json:35-37`）。稿件对“各一次”的证据边界准确限定为各一份正式提交训练记录（`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md:9`）。 |
| 冻结后生成四组各 50 个独立未见实例；训练/测试互斥；两模型使用同一测试输入（`供应链管理写作/补充实验.md:5,14-18,43-49,117-134,196-197`） | PASS | Test-1…Test-4 均锁为 50 个实例且规模分别为 3/2/2/3/2、6/2/2/4/3、10/3/3/5/3、20/4/3/8/4（`configs/supplementary_experiment.json:53-94`；校验门见 `src/supplementary_protocol.py:499-517`）。测试集仅在两 checkpoint 冻结后生成，并保存实例、初始解、归一化尺度及其哈希（`src/supplementary_experiment.py:710-800`）；schedule 对同一 scale/index 复用这些锁定输入（`src/supplementary_experiment.py:824-880`）。独立重算得到 200 个 scale/index 组、共享输入哈希差异 0。seed audit 显示训练/测试 seed 交集 0、训练/测试实例内容哈希交集 0（`output/supplementary-experiments/audits/seed-audit.json:103-111`）。 |
| 2×4 完全交叉，E1/E2 端点复用且不重复计数（`供应链管理写作/补充实验.md:26-28,117-134`） | PASS | PPO 范围为 2 模型×4 规模，GA 为一套 4 规模基线，启发式仅 Test-1/Test-4，MILP 仅 Test-1（`configs/supplementary_experiment.json:103-123`）；schedule view 映射复用 E1/E2 端点并将 G 绑定到同一 cell（`src/supplementary_experiment.py:855-880`）。正式 schedule 为 9,750 个唯一 cell/9,750 个唯一 cell_id，其中 PPO 6,000、GA 3,000、启发式 500、MILP 250；manifest 登记 9,750（`outputs/supplementary_experiment_v1/manifest.json:3175-3178`），验证报告观察值完全一致（`outputs/supplementary_experiment_v1/verification/verification_report.json:21-34`）。 |
| 重启数、失败分母与实例级统计：PPO/GA 3 次，启发式/MILP 1 次；先重启均值，再以 n=50 的实例为样本（`供应链管理写作/补充实验.md:18,55,65,69-83`） | PASS | 配置锁定 3/3/1/1（`configs/supplementary_experiment.json:102-123`），协议校验总网格为 6000/3000/500/250（`src/supplementary_protocol.py:527-545,647-656`）。实例汇总要求观察到全部应有重启且全部完成/可行才给主质量值，同时保留技术失败、不可行、缺失分母（`src/supplementary_experiment.py:1490-1551`）；偏好层再以实例为单位计算均值和 `ddof=1` 样本标准差（`src/supplementary_experiment.py:1554-1610`）。审定稿明确 n=50 而非 n=150，并披露完整分母（`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md:33,45,75,116,144-150`）。 |
| MILP 180 s；Gap 只用同实例、同偏好且证明 optimal 的基准，并报告有效 n（`供应链管理写作/补充实验.md:55,80,87-97`） | PASS | 时限锁为 180 s（`configs/supplementary_experiment.json:181`；硬校验见 `src/supplementary_protocol.py:604-605`），求解入口传入该时限（`src/supplementary_experiment.py:1212-1225`）。Gap 的有效门显式要求 `milp_status.casefold() == "optimal"`，再逐实例配对计算（`src/supplementary_experiment.py:1650-1697`），汇总保留 `optimal_reference_n`/`valid_n`（`src/supplementary_experiment.py:1878-1900`）。表 3 分偏好报告 50、49、26、2、2 个有效基准并明确排除非 optimal incumbent（`output/supplementary-experiments/tables/table3_small_scale.md:5-11`）；稿件未计算跨偏好合并 Gap（`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md:27-33,63,170-172`）。 |
| 哈希锁、回放与全部表可由保存结果复算（`供应链管理写作/补充实验.md:47,201,204`） | PASS | 初始化锁 13 个源码文件，后续阶段拒绝源码哈希变化（`src/supplementary_experiment.py:307-337,463-478`）；schedule 将协议、实例、初始解、归一化尺度、源码和 checkpoint/policy 哈希绑定到 cell（`src/supplementary_experiment.py:805-899`）；汇总完全从已提交 cell 重建并重算指标/plan hash（`src/supplementary_experiment.py:1780-1854`）。验证门检查全部 9,750 个身份/文件链，并分层复核 75 个 artifact、重放 15 个 solver cell（`src/supplementary_experiment.py:2027-2284`）。正式报告为 200 个测试实例、missing/failed/integrity error 均 0、75/75 artifact 和 15/15 solver replay 错误均 0、源码变化 0、`passed=true`（`outputs/supplementary_experiment_v1/verification/verification_report.json:35-52`），manifest 为 `stage=verified`（`outputs/supplementary_experiment_v1/manifest.json:7,3334-3337`）。seed audit 另确认 10,154/10,154 seed 唯一、9,750/9,750 evaluation seed 唯一且派生偏差 0（`output/supplementary-experiments/audits/seed-audit.json:72-111`）。 |
| 表 3–7 与论文写作交付（`供应链管理写作/补充实验.md:85-176,184-204`） | PASS | builder 明确生成表 3、4、5、6、7（`output/supplementary-experiments/scripts/build_paper_tables.py:3086-3143`）。五张 Markdown 表均存在且结构符合规格（`output/supplementary-experiments/tables/table3_small_scale.md:1-11`、`output/supplementary-experiments/tables/table4_large_scale.md:1-11`、`output/supplementary-experiments/tables/table5_scale_design.md:1-12`、`output/supplementary-experiments/tables/table6_generalization_matrix.md:1-8`、`output/supplementary-experiments/tables/table7_generalization_operational.md:1-14`）。paper inventory 登记五表 CSV/Markdown（`output/supplementary-experiments/tables/paper-table-inventory.json:53-83`）；本次独立复核其 14 个输入哈希、18 个非自身输出哈希和 10 个 CSV 行数，差异均为 0，inventory 自身 SHA-256 为 `76edf5c7f05035676b97c6d1dc12f09195b2d2000f87aa5af3788b9abf16e0f3`。审定稿 SHA-256 为 `c6b1adf87b488d49820f73fcc73ade402e643c80b91b3c4818d7ab2da702721c`；其中表 3–7 的表体与表注逐行等于五份 builder Markdown，正文覆盖实验目的、口径、关键数字、风险分项、管理含义和外推边界（`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md:3-172`），无待填占位符。 |

## 验收门控动作（非 finding）

审查时正式目标路径 `供应链管理写作/数值实验与结果分析_论文稿.md` 仍是验收前旧稿：例如第 16 行写“两类规模目前各生成1个实例”，第 74–76 行的“表3”仍是旧的最优重启风险构成，第 101 行仍把多实例与泛化列为未来工作。审查委托者已确认，这一状态是 scholar-analyze 流程要求的**用户验收门**，不是本轮实现缺陷，故不计入 P0–P3。

验收通过后的预期动作是将已审定且通过上述核对的 `output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md` 写入用户指定目标路径，再在提交/推送前复核目标文件哈希与占位符。若最终提交仍保留旧目标稿，届时才构成写作交付不完整。

## 独立验证记录

- `py -B -m unittest tests.test_supplementary_protocol tests.test_supplementary_runner tests.test_frozen_ppo tests.test_build_paper_tables tests.test_ga_decoder -v`：**46 tests，OK**。
- 当前正式 manifest：`stage=verified`、`smoke=false`、2 个模型、200 个测试实例、9,750 个唯一计划 cell。
- 当前 paper inventory：14/14 输入哈希、18/18 输出哈希、10/10 CSV 行数均一致；registry 与 adjudication 的 ID 集合均严格为 `E1`、`E2`、`G-Train-S`、`G-Train-L`，adjudication 每个 ID 恰好一行。
- 表 3–7 及三张 builder 附表在审定稿中的表体/表注均与已哈希生成稿逐行一致。

**最终判定：Spec PASS；P0=0，P1=0，P2=0，P3=0；无阻断项。**

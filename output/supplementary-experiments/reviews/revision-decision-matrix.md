# 五份审稿意见的修订决策矩阵

## 1. 用途、范围与判定原则

本矩阵综合 `reviewer-01-statistics.md`、`reviewer-02-spec-tables.md`、`reviewer-03-validity.md`、`reviewer-04-prose-alignment.md` 和 `reviewer-05-reproducibility.md`，审查对象为 SHA-256=`3fee329fe38c80beeafad049cb89f8c6c2860906fef22506dec44795b3fee912` 的正式结果初稿。本文件只记录修订决定、理由、责任文件和验收口径；编制本矩阵时不修改该初稿，也不修改 `供应链管理写作/数值实验与结果分析_论文稿.md`。

状态含义如下：

- **ACCEPTED**：本轮完整采纳；如依赖产物已经生成，则正文同步仍须等待最终 reviser 阶段。
- **PARTIAL**：采纳能够由现有证据支持的部分，并明确拒绝或推迟超出证据的部分。
- **DEFERRED**：建议成立，但实施会改变已经验证的 formal source bundle 或协议语义，移至下一协议版本。
- **REJECTED**：与本轮预设范围冲突、需要伪造未记录信息，或会造成超出证据的复现/因果/部署主张，故不实施。

统一证据规则：正文数字只可来自正式 `tables/*`、`manifest.json`、`verification_report.json` 及其哈希绑定的下游产物；不从审稿意见反向造数。所有正式结论保持描述性，不新增显著性检验，不把并发计时解释为隔离条件下的算法速度因果效应，不把未证明 `optimal` 的 incumbent 称为全局最优。

## 2. 数值、分母与解释边界

| ID | 来源 | 审稿意见或风险 | 决定 | 理由与实施口径 | 将修改的文件 | 验收标准 |
|---|---|---|---|---|---|---|
| N01 | R1-I1；R3-I1；R4-I1 | 实验G缺少总分母及逐实例—偏好方向异质性，组均值为正易被误读为逐单元支配 | **ACCEPTED** | 报告400/400个实例宏平均，400个实例层的等权偏好平均改善率均为正；另报告2000/2000个逐实例—偏好有效配对，在容差$10^{-9}$下PPO目标较低、持平、较高分别为1871、110、19个。明确这些是描述性组均值/宏平均优势，不是逐实例—偏好支配。 | 最终 reviser 阶段：`output/supplementary-experiments/drafts/数值实验与结果分析_正式结果初稿.md`；最终整合后才更新 `供应链管理写作/数值实验与结果分析_论文稿.md` | 数字与正式配对表一致；全文不得出现“所有实例—偏好均优于GA”等越界表述。 |
| N02 | R1-M1；R4-I2 | 表3把所有非 `optimal` incumbent统称为“限时解” | **ACCEPTED** | 改为“任何未证明 `optimal` 的 incumbent（包括限时或其他中止）均不进入Gap”；不从 `solver_status_1` 推断唯一停止原因。 | 正式表3Markdown已由builder更新；最终 reviser 同步上述两份稿件 | 表注及正文不再把121个非 `optimal` 状态一概称为限时。 |
| N03 | R1-M2 | canonical cell账本行哈希与账本CSV文件哈希易混淆 | **ACCEPTED** | 将 `bff5643b…` 明确标成“汇总输入cell账本（canonical row hash）”；如列CSV文件哈希，则另列 `67b0ade1…` 并标明对象，不把二者混作同一SHA。 | 最终 reviser 阶段的初稿与正式目标稿；必要时同步哈希表注 | 每个哈希都有唯一对象名；与manifest、verification及paper inventory的字段语义一致。 |
| N04 | R1-M3 | 75项artifact复核与15项solver replay的对象需更精确 | **ACCEPTED** | 写明9750个cell完成身份/文件完整性核验；75项为分层抽样方案的可行性、plan hash与质量指标复算；15项才是solver重放。不得把75项称为trace算法语义重放。 | 最终 reviser 阶段的初稿与正式目标稿 | 9750、75、15三层核验对象清楚且与verification字段一致。 |
| N05 | R3-I2；R4-I3 | 不同规模的归一化$J$使用各自实例参考，不能横向比较 | **ACCEPTED** | 在表6注和结果段明示：跨规模$\bar J$绝对水平不可用于排序任务难度、绝对质量或推断随规模的趋势；跨规模判断只依赖同规模配对、可行性与完整性。 | 最终 reviser 阶段的初稿与正式目标稿 | 不出现由Test-1—Test-4的$J$水平推导规模效应的句子。 |
| N06 | R3-I3 | 局限未明确披露本轮无敏感性/消融、兼容概率固定及GA预算随规模变化 | **ACCEPTED** | 增加范围说明：未扰动容量、风险系数、兼容比例等参数，未做结构/特征消融；兼容概率固定为1.0，不覆盖稀疏兼容；Test-1/2的GA预算为30×60，Test-3/4为20×40，故跨规模相对GA差异混有基线预算变化，不能解释为规模净效应。 | 最终 reviser 阶段的初稿与正式目标稿 | 局限段同时包含三项边界；不得出现参数稳健性、组件贡献或纯规模效应结论。 |
| N07 | R3-I3；本轮实验范围 | 为增强稳健性而补跑协议明确排除的敏感性分析或消融实验 | **REJECTED** | 设计文件已明确本轮不做此类实验；补跑会扩展协议范围并产生另一套证据包。当前问题通过边界披露解决，不改变正式9750个求解单元。 | 无；仅在上述两份稿件的局限中记录“未开展” | 不新增敏感性/消融数据、图表或结果性表述。 |
| N08 | R3-I4 | 将合成算例结果直接写成现实部署用途 | **ACCEPTED** | 将“可作为”收缩为“在合成算例、既定参数区间和严格可行复核下显示潜力，尚未经真实企业数据验证”；实际部署前要求多训练实例、多训练种子复训及真实分布、兼容关系和约束偏移检验。 | 最终 reviser 阶段的初稿与正式目标稿 | 管理含义是潜力与条件句，不形成现实有效性或部署认证主张。 |
| N09 | R3-M1；R4-M2 | “可用/保持可用的迁移”没有预注册阈值 | **ACCEPTED** | 用可审计条件替代“可用”：完整、严格可行、期末清零，且八个模型—规模组合的实例宏平均相对GA改善率为正；仍明确这是描述性结果。 | 最终 reviser 阶段的初稿与正式目标稿 | 不出现无定义的“业务可用”判断；条件均能由正式表追溯。 |
| N10 | R3-M2 | 处理处置端库存风险为0易被误读为结构性质 | **ACCEPTED** | 就地说明该0值仅为当前生成器、容量配置和所得方案的观测结果，不是理论模型或现实系统的一般性质。 | 最终 reviser 阶段的初稿与正式目标稿 | 风险机制解释紧邻边界句；不作结构性或现实普遍化。 |
| N11 | R3-M3；R5-I2 | 单seed与确定性开关的关系需更审慎 | **ACCEPTED** | seed登记只支持溯源；单训练实例/单训练seed不能估计训练随机变异，初始化快照中的false也不能代表训练/推理全程，更不保证跨硬件位级复现。 | 最终 reviser 阶段的初稿与正式目标稿；`output/supplementary-experiments/REPRODUCIBILITY.md`与环境freeze | 限制同时覆盖样本层、训练随机性和潜在非确定性。 |
| N12 | R4-I2；R1通过项 | “129个MILP精确配对的平均Gap”像是存在合并Gap | **ACCEPTED** | Gap只在五个偏好各自的MILP-`optimal`配对有效集合内报告；有效$n$为50、49、26、2、2，总计129仅是覆盖计数，不计算或暗示跨偏好合并Gap。可写五层范围0.94%–4.06%，并同时保留$n$为2的边界。 | 最终 reviser 阶段的初稿与正式目标稿 | 不出现“129个单元的总体/综合平均Gap”；所有Gap语句带对应有效集合。 |
| N13 | R1、R3、R4通过项；预设要求 | 描述性分析、不作显著性推断 | **ACCEPTED（保留）** | 不增设$p$值、置信区间或“显著优于”；“均值更低/改善率为正”只描述已观察到的样本。 | 最终 reviser 阶段的初稿与正式目标稿 | 全文扫描不含未经设计支持的显著性与因果措辞。 |
| N14 | R1、R4通过项；预设要求 | $L$的含义需与MILP Gap和严格泛化损失分开 | **ACCEPTED（保留）** | $L$只称同一测试规模上两份固定checkpoint的相对差距；不称统计泛化损失、不与MILP Gap等同，也不据单次训练归因于训练规模。 | 最终 reviser 阶段的初稿与正式目标稿 | 每处$L$均限定“两份固定模型、同一测试规模、相对差距”。 |
| N15 | R1、R2、R5通过项；预设要求 | 完整$n$、失败/缺失及并发时间边界必须保留 | **ACCEPTED（保留）** | 报告9750/9750完成、失败0、缺失0及各表有效$n$；时间只称同机并发条件下的求解段观测值，训练时间不计入推理，不能作隔离运行下纯算法速度的因果比较。 | 表3/4/7及最终 reviser 阶段两份稿件 | 不因压缩表注而删掉完整性或计时边界；详细run信息移至附录。 |

## 3. 表格、编号与正文节奏

| ID | 来源 | 审稿意见或风险 | 决定 | 理由与实施口径 | 将修改的文件 | 验收标准 |
|---|---|---|---|---|---|---|
| T01 | R2-I1 | 表5为手工表，未进入builder和哈希清单 | **ACCEPTED（builder已实施）** | 表5只从已验证run内锁定协议生成，纳入原子发布和inventory；训练行写“1份模型”，不凭协议外信息写训练实例数。 | 已修改：`output/supplementary-experiments/scripts/build_paper_tables.py`、`tests/test_build_paper_tables.py`；已生成：`tables/table5_scale_design.csv`、`tables/table5_scale_design.md`、`tables/paper-table-inventory.json`；最终 reviser 同步稿件 | 表5共6行；inventory声明18个非自身输出、10个CSV行数项；相关14项测试全部通过；稿件表5逐字取自生成Markdown。 |
| T02 | R2-M2 | 表3、表4标题未写动态实例数 | **ACCEPTED（builder已实施）** | 标题由协议 `expected_n` 动态生成；正式输出为50个，smoke输出为1个，禁止硬编码。 | 已修改builder与测试；已生成 `tables/table3_small_scale.md`、`tables/table4_large_scale.md`；最终 reviser 同步稿件 | 正式标题含“50个未见实例”，smoke测试含“1个”。 |
| T03 | R2-I2；R4-I4 | 表3/4/7主表注过长并重复run级审计信息 | **ACCEPTED（builder已实施）** | 主表注仅保留统计单位/SD、比较指标口径、训练—推理边界和并发解释边界；batch、workers、max_cells及整批墙钟保留在CSV、registry、manifest和附录运行表。 | 已修改builder与测试；已生成 `tables/table3_small_scale.md`、`tables/table4_large_scale.md`、`tables/table7_generalization_operational.md`；最终 reviser 同步稿件 | 三个主表Markdown不含逐批长注，但仍指向 `manifest.evaluation_runs`；审计字段未删除。 |
| T04 | R2-M1；R4-I4 | 附表编号不连续、交叉引用含糊 | **ACCEPTED** | 连续编号为A1训练记录、A2全方法、A3风险构成、A4 MILP状态、A5失败审计、A6初始化环境快照、A7 evaluation_runs、A8关键哈希、A9 Markdown哈希，并同步正文引用。 | 最终 reviser 阶段的初稿与正式目标稿 | 每张附表唯一编号；所有“见附表”改为可定位的表号；不存在重号/跳号。 |
| T05 | R2-M3 | 训练实例措辞像是从测试集抽取 | **ACCEPTED** | 改为“与Test-1/Test-4同规模且与测试集互斥的训练实例”；同时保留每模型一份正式提交训练记录和冻结checkpoint。 | 最终 reviser 阶段的初稿与正式目标稿 | 不再出现“在一个Test-1/Test-4训练实例上训练”的歧义表述。 |
| T06 | R2-M4 | E1/E2正文缺少启发式基线收束 | **ACCEPTED** | 说明启发式在自身归一化口径下$J=1$，PPO与GA五个偏好组的平均$J$均低于该参考；不得把启发式初始化时长与PPO/GA核心求解段直接排序。 | 最终 reviser 阶段的初稿与正式目标稿 | E1、E2各有一句简洁收束，数字来自全方法附表且无跨阶段时间比较。 |
| T07 | R4-M1 | E2称比较GA和启发式，但表4只列GA | **ACCEPTED** | 明示表4聚焦PPO—GA，启发式完整结果见表A2；不为此扩宽主表。 | 最终 reviser 阶段的初稿与正式目标稿 | 读者可由正文准确定位启发式结果，主表保持紧凑。 |

## 4. 复现材料、环境与种子

| ID | 来源 | 审稿意见或风险 | 决定 | 理由与实施口径 | 将修改的文件 | 验收标准 |
|---|---|---|---|---|---|---|
| R01 | R5-I1 | 当前 `replication` 目录的用途边界不清 | **ACCEPTED** | 增加总复现说明并明确该目录是“紧凑可审计结果归档”：列出可核验与不可仅凭该目录执行的事项、正式源run/持久化位置、从init到export的命令链及哈希核对入口。 | `output/supplementary-experiments/REPRODUCIBILITY.md`、`output/supplementary-experiments/reviews/reproducibility-remediation.md`；最终 reviser 在复现说明中引用 | 说明不得声称归档内含实际未携带的training/test instances、cells、solutions或traces，也不得声称可单目录完整重跑。 |
| R02 | R5-I1 | 将紧凑归档在本轮扩展成独立可执行、携带全部原始cell/solution/trace的复现包 | **PARTIAL** | 采纳“解释用途与命令链”的方案；不在本轮复制巨量原始工件或把紧凑归档改造成完整可执行包。若未来发布不可变全量归档，应另立版本并重新做inventory/哈希核验。 | 本轮只新增 `output/supplementary-experiments/REPRODUCIBILITY.md` 和remediation报告并修订稿件说明；不改 `replication/inventory.json`，不复制正式原始run的大型目录 | 归档能力声明与当前inventory实际登记的payload严格一致。 |
| R03 | R5-I2；R3-M3 | 初始化环境快照被呈现成全程设置，且环境记录不完备 | **ACCEPTED** | 附表A6改称“初始化环境快照”；分开说明初始化、训练和冻结PPO阶段可由源码/记录证明的设置。device、GPU driver、实际CUBLAS值、HiGHS精确版本及solver线程等未记录项一律写“未记录”，不反推、不伪造。 | `output/supplementary-experiments/REPRODUCIBILITY.md`、`output/supplementary-experiments/environment-freeze-2026-09-05.txt`、`output/supplementary-experiments/reviews/reproducibility-remediation.md`；最终 reviser 阶段的初稿与正式目标稿 | 时间点清楚；已记录值可追溯；所有历史未知项显式为“未记录”；不出现“完整环境”或跨硬件位级确定性主张。 |
| R04 | R5-I2 | 缺完整依赖锁，希望补锁文件 | **PARTIAL** | 接受补充依赖/环境快照和安装说明；发布时环境freeze必须明确不是正式运行开始前的精确传递依赖锁，不能伪称历史运行状态。严格可重建的预运行lock与实际runtime采集纳入下一协议版本。 | `output/supplementary-experiments/environment-freeze-2026-09-05.txt`、`output/supplementary-experiments/REPRODUCIBILITY.md`；下一协议版本的环境锁定文件 | 当前文档区分“已记录”“从源码可知”“2026-09-05发布时采集”“未记录”；不制造历史版本号。 |
| R05 | R5-I3 | 当前种子唯一性虽通过独立核查，但缺机器可读发布审计 | **ACCEPTED** | 发布seed audit：训练实例2、训练算法2、测试实例200、初始解200、PPO 6000、GA 3000、启发式500、MILP 250，共10154/10154唯一；9750个evaluation seed、cell_id、logical_cell_id均唯一；派生不一致0，训练—测试seed与实例内容哈希交集均为0。 | `output/supplementary-experiments/audits/seed-audit.json`、`output/supplementary-experiments/audits/seed-audit.md`、`output/supplementary-experiments/reviews/reproducibility-remediation.md`；总复现说明与最终稿引用 | 审计可由锁定protocol/schedule/manifest重算；总数、分项、交集和 mismatch 字段齐全；remediation报告登记文件SHA。 |
| R06 | R5-I3 | 将seed唯一性和训练—测试hash互斥加入core硬门 | **DEFERRED** | 建议合理，但修改 `src/supplementary_protocol.py` 或 `src/supplementary_experiment.py` 会改变已验证的formal source bundle。当前发布以seed audit补充证据；硬门在下一协议版本实施并重新锁定source bundle。 | 本轮不改core；下一版才修改上述两个 `src/` 文件及对应测试、协议版本和verification | 当前source bundle哈希保持不变；下一版本必须有碰撞/交集负向测试和正式verify门。 |
| R07 | R5-M1 | “一次性训练”可能被读成从未发生未提交中断尝试 | **ACCEPTED** | 只写“每个模型一份正式提交训练记录/一次正式提交训练耗时”。现有账本能证明两份已提交结果，不能证明进程层面从未发生未落盘的中断尝试。 | 最终 reviser 阶段的初稿与正式目标稿；`output/supplementary-experiments/REPRODUCIBILITY.md` | 不出现“系统证明从未重试/从未中断”等不可审计断言。 |
| R08 | R5-M2 | paper inventory泄露机器专属绝对路径 | **ACCEPTED（builder已实施）** | 仓库内正式run使用 `outputs/supplementary_experiment_v1`，并标记 `source_run_dir_path_kind=repository-relative`；仓库外才安全回退为 `absolute-external`。 | 已修改builder、测试与 `tables/paper-table-inventory.json` | 正式inventory不含用户绝对路径；来源路径类型显式；输出哈希核验仍为0不匹配。 |
| R09 | R5-M3 | 脚本索引与过程日志含过时“待补/进行中”表述 | **ACCEPTED** | 旧文件保留为历史过程记录，不追溯改写；由新的权威复现说明和remediation报告逐项标成历史/非规范性，并纠正evaluation seed位于schedule且由manifest引用的关系。 | 新增 `output/supplementary-experiments/REPRODUCIBILITY.md`、`output/supplementary-experiments/reviews/reproducibility-remediation.md`；不直接修改旧 `script-index.md`、process log或`formal-provenance-facts.md` | 新说明不得沿用“进行中/待生成”为当前状态，并明确最终状态以 `manifest.stage=verified` 和verification为准。 |
| R10 | R5-I2；预设A.6要求 | 为填满环境表而推断device或solver线程数 | **REJECTED** | 附录A.6中的表A7必须逐run忠实转录 `manifest.evaluation_runs`；PPO/GA/启发式solver线程为“不适用”，MILP未记录则写“未记录”，device缺失亦写“未记录”。不能由机器当前状态、默认值或并发worker数反推。 | 不新增推断值；最终 reviser 仅同步正式记录与明确缺失 | 六个evaluation run逐条齐全，字段与manifest一致；无伪造device/线程/solver版本。 |

## 5. 实施顺序与合并门

1. **派生表层。** 先完成并验证builder生成的表3、表4、表5、表7及inventory；该阶段已经由 `builder-revision-implementation.md` 记录为完成，正式锁定输入未改写。
2. **复现材料层。** 完成总复现说明、发布时环境freeze、JSON/Markdown seed audit及remediation报告；旧过程日志保留历史原貌并在新说明中标为非规范性。每个新增文件先做结构、计数和SHA复核。
3. **最终reviser层。** 只在上述正式产物稳定后，按本矩阵将生成表嵌入正式初稿，完成数值/边界修文、附表连续编号和交叉引用。未经该阶段复核，不把中间稿覆盖到正式目标稿。
4. **下一协议版本。** 实施seed硬门、训练—测试内容hash硬门、预运行依赖锁与更完整runtime环境采集；届时必须更新协议版本、source bundle及对应验证证据，不追溯性改动当前formal run。

最终合并须同时通过以下门：所有正文数字可回指正式生成表或manifest/verification；Gap只在五个偏好各自的129个 `optimal` 配对中的对应有效集合内使用；$L$只称两份固定模型相对差距；主张均为描述性；9750/9750完成、失败0、缺失0与表内完整$n$保留；并发时间和单训练实例/单训练seed限制不被弱化；不新增本轮排除的敏感性/消融结果；不修改当前core源码及已验证formal source bundle。

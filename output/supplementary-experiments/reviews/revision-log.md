# 正式结果审定稿修订日志

- 修订日期：2026-09-05（Asia/Shanghai）
- 修订模式：REVISE（Results）
- 修订依据：五份Reviewer报告、`revision-decision-matrix.md`、正式builder表、`formal-results-analysis.md`及复现整改材料
- 输入初稿：`output/supplementary-experiments/drafts/数值实验与结果分析_正式结果初稿.md`
- 输入初稿SHA-256：`3fee329fe38c80beeafad049cb89f8c6c2860906fef22506dec44795b3fee912`
- 输出审定稿：`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md`
- 输出审定稿SHA-256：`c6b1adf87b488d49820f73fcc73ade402e643c80b91b3c4818d7ab2da702721c`
- 正式目标稿：本轮未修改 `供应链管理写作/数值实验与结果分析_论文稿.md`

## 1. 总体修订策略

审定稿沿用本地众包论文数值实验章节的叙述节奏，即先给出全节路线图，再按实验分别交代目的、规模、基线与重复，继而在表前限定统计口径、在表后提炼少量关键数字，最后给出分项解释与边界。只借鉴章节组织与信息密度，不复制该论文的句子、数字或结论。稿件未新增外部文献引证；全部事实性数字均来自本项目正式表、manifest、verification、独立分析或哈希绑定的补充审计。

正式初稿被保留为未修改的审稿基线，所有修订写入独立审定稿。主表3—7及三张builder附表均直接嵌入当前Markdown产物；除把一级标题改成粗体并为三张附表补连续编号外，表体与表注逐字保持不变。

## 2. 审稿意见逐条映射

### Reviewer 1：计算与统计聚合

| 意见 | 决定 | 审定稿修改位置 | 落实内容 |
|---|---|---|---|
| R1-I1：G缺总分母和方向异质性 | ACCEPTED | 第148—150行，实验G“结果分析与范围限制” | 增加400/400个实例宏平均、2000/2000个逐实例—偏好单元；明确400个实例层改善率均为正，并按容差$10^{-9}$披露1871正、110平、19负，排除逐单元支配解释。 |
| R1-M1：非optimal状态被统称限时 | ACCEPTED | 表3（第47行起）及表A4（第288行起） | 使用builder现行表注：“任何未证明optimal的incumbent（包括限时或其他中止）均不进入Gap”；不从`solver_status_1`推断唯一停止原因。 |
| R1-M2：canonical row hash与CSV文件哈希易混 | ACCEPTED | 表A8（第352行起） | 将`bff5643b…`标为cell账本canonical row hash，另列`cell_file_ledger.csv`文件哈希`67b0ade1…`。 |
| R1-M3：75与15的验证对象需精确 | ACCEPTED | 第384行，表A9后验证说明 | 写明9750个单元的身份/文件完整性核验，75次为分层artifact metric recomputation，15次为solver replay；未称全量重放。 |

### Reviewer 2：规格、表格与产物

| 意见 | 决定 | 审定稿修改位置 | 落实内容 |
|---|---|---|---|
| R2-I1：表5未由builder生成 | ACCEPTED | 表5（第103行起）、表A9（第371行起） | 嵌入由锁定协议生成的6行表5；训练行仅写“1份模型”，并登记当前Markdown哈希。 |
| R2-I2：主表注过长 | ACCEPTED | 表3、表4、表7（第47、77、131行起） | 嵌入builder压缩后的表注；完整batch、workers与墙钟转由表A7及manifest承载。 |
| R2-M1：附表号不连续 | ACCEPTED | 表A1—A9（第184—371行） | 训练记录、全方法、风险、MILP状态、失败、环境、runs、关键哈希、Markdown哈希连续编号；正文改用具体表号。 |
| R2-M2：表3/4标题缺“50个” | ACCEPTED | 表3、表4（第47、77行） | 使用动态builder标题，正式输出均明确“50个未见实例”。 |
| R2-M3：训练实例措辞像来自测试集 | ACCEPTED | 第9行及表A1（第184行起） | 明确训练实例仅与Test-1/Test-4同规模，并与200个测试实例在seed及内容哈希上互斥。 |
| R2-M4：启发式基线缺正文收束 | ACCEPTED | 第61、91行；完整结果见表A2（第204行起） | E1/E2均说明启发式因归一化定义而有$J=1$，PPO与GA五组均值低于该参考；未把初始化时间与核心求解段排序。 |

### Reviewer 3：稳健性与有效性威胁

| 意见 | 决定 | 审定稿修改位置 | 落实内容 |
|---|---|---|---|
| R3-I1：19个反向单元未披露 | ACCEPTED | 第148—150行 | 完整披露1871/110/19及容差，并把结论限定为平均表现。 |
| R3-I2：跨规模$J$不可横向比较 | ACCEPTED | 表6后第127行；第168行局限 | 明确各规模采用自身实例参考，$\bar J$绝对水平不能排序难度或推断规模趋势。 |
| R3-I3：无敏感性/消融、兼容率与GA预算边界 | ACCEPTED（修文）；REJECTED（补跑） | 第154、168行；附录A.1—A.2 | 明示本轮未做敏感性或结构/特征消融，兼容概率固定为1.0；披露GA的30×60与20×40预算，跨规模差异不能识别纯规模效应。未补跑协议排除实验。 |
| R3-I4：管理含义超出合成证据 | ACCEPTED | 第162—164行 | 将现实工具断言收缩为合成算例中“显示潜力、尚未经真实企业数据验证”，并加入多训练实例、多seed与真实分布/约束偏移验证要求。 |
| R3-M1：“可用迁移”无操作定义 | ACCEPTED | 第148—154、172行 | 改用完整宏实例、严格可行、期末清零、异常为0和实例宏平均改善率为正等可审计条件。 |
| R3-M2：设施库存风险为0被一般化 | ACCEPTED | 第65、160行 | 明确处理处置端库存风险为0只是当前生成参数、容量配置和所得方案下的观测值。 |
| R3-M3：单seed与非确定性设置 | ACCEPTED | 第37、168行及表A6（第320行起） | 区分初始化、训练和冻结推理设置；说明seed支持溯源但不估计训练随机变异，也不保证跨硬件位级一致。 |

### Reviewer 4：正文—数字一致性与行文

| 意见 | 决定 | 审定稿修改位置 | 落实内容 |
|---|---|---|---|
| R4-I1：组均值为正易被理解为逐实例支配 | ACCEPTED | 第61、91、148—150行 | E1/E2均明确“组均值”；G同时给实例宏平均一致性与19个反向偏好单元。 |
| R4-I2：结论像在报告129个单元的合并Gap | ACCEPTED | 第63、172行 | 逐偏好保留有效$n$；结论写PPO Gap均值0.94%—4.06%、$n=2$—50，并明确不计算跨偏好合并Gap。 |
| R4-I3：表6后缺跨规模归一化边界 | ACCEPTED | 第127行 | 在表6后立即说明跨规模$J$不可作水平比较。 |
| R4-I4：主表注冗长且附表编号混乱 | ACCEPTED | 表3/4/7及表A1—A9 | 使用短表注并连续编号；run级信息集中于表A7。 |
| R4-M1：E2称比较启发式但表4未列 | ACCEPTED | 第71、91行 | 明确表4聚焦PPO—GA，启发式完整结果见表A2。 |
| R4-M2：“保持可用”证据过强 | ACCEPTED | 第152、162—164行 | 删除无阈值“可用”表述，改为可审计描述性条件及合成场景潜力。 |

### Reviewer 5：复现性、哈希与种子

| 意见 | 决定 | 审定稿修改位置 | 落实内容 |
|---|---|---|---|
| R5-I1：compact replication不是独立可执行包 | ACCEPTED（采用说明方案） | 第170、388行 | 明确compact archive为紧凑可审计结果归档，不含完整instances/cells/solutions/traces或锁定源码，不能单目录执行既有verify；引用REPRODUCIBILITY说明。 |
| R5-I2：环境锁与阶段设置混淆 | PARTIAL | 第37行、表A6及表A8 | 区分init、训练、冻结推理；补cuDNN benchmark=false和evaluation入口未显式调用全局确定性API；CUBLAS实际值、HiGHS精确版本、solver线程保持“未记录”。运行后freeze不伪称运行前依赖锁。 |
| R5-I3：seed唯一性未进正式硬门 | ACCEPTED（本次audit）；DEFERRED（core硬门） | 第386行 | 引用机器可读seed audit，报告10154/10154唯一、evaluation身份唯一、交集与派生偏差为0；不修改已锁定core，硬门留待下一协议版本。 |
| R5-M1：“一次训练”不能证明从未中断 | ACCEPTED | 第9行、表A1 | 改成“每模型一份正式提交训练记录”，并明确不声称不存在提交前中断。 |
| R5-M2：paper inventory绝对路径 | ACCEPTED（builder已完成） | 表A8、第388行所引REPRODUCIBILITY | 使用当前repository-relative来源和当前paper inventory哈希，不保留机器专属用户路径。 |
| R5-M3：旧配套说明含中间状态 | ACCEPTED | 第388行所引REPRODUCIBILITY | 以新复现说明、正式manifest与verification为规范性状态证据；旧日志保留为历史记录，不把“进行中”写成当前状态。 |

## 3. 未在本轮实施的建议

| 项目 | 处置 | 理由 |
|---|---|---|
| 新增敏感性分析或消融实验 | REJECTED | 本轮协议明确排除，补跑会形成新的实验版本；改用局限披露。 |
| 修改`src/supplementary_protocol.py`或`src/supplementary_experiment.py`加入seed硬门 | DEFERRED | 会改变已经验证的formal source bundle；本次由独立seed audit补证，下一协议版本再实施。 |
| 将紧凑归档扩成携带所有逐cell工件的独立复现包 | PARTIAL | 本轮采纳诚实用途说明与完整命令链，不复制约1.1 GB正式运行目录。 |
| 追溯性编造运行时CUBLAS、HiGHS或solver线程 | REJECTED | 正式记录没有这些字段；未知值保持“未记录”。 |
| 把运行后pip freeze称为正式运行前锁文件 | REJECTED | freeze仅支持发布时环境溯源与近似重建。 |

## 4. 解释性主张审计

| 主张 | 直接证据 | 跨组成立 | 组内/单元成立 | 证据属性 | 结论 |
|---|---|---:|---:|---|---|
| E1五个偏好组的PPO平均$J$低于GA | 表3五个$\Delta J$组均值 | 是 | 非逐单元保证 | 测量 | KEEP，并在G披露19个反向单元 |
| E2五个偏好组的PPO平均$J$低于GA | 表4五个$\Delta J$组均值 | 是 | 非逐单元保证 | 测量 | KEEP，明确组均值 |
| E1的PPO Gap较小 | 五个偏好各自optimal子样本，0.94%—4.06% | 分层成立 | 有效$n=2$—50 | 测量 | REVISE，不计算129个合并Gap |
| G在实例宏平均层面方向一致 | 400/400个实例宏平均改善率为正 | 是 | 实例层成立 | 测量 | KEEP，并与2000个偏好单元分开 |
| PPO逐实例—偏好全部优于GA | 1871正、110平、19负 | 否 | 否 | 测量 | REJECT，不写逐单元支配 |
| Train-S代表小规模训练优于大规模训练 | 两份固定模型的$L=0.81%$—1.39% | 仅两检查点 | 未覆盖训练重复 | 测量 | REVISE，仅称两固定模型相对差距 |
| 风险下降对应若干风险分量下降 | 表A3的运输、共载、产废端库存分量 | E1/E2均有对应 | 无轨迹机制证据 | 测量 | KEEP为构成描述，不作决策渠道因果推断 |
| PPO已适合真实部署 | 仅合成数据、单训练实例/seed | 否 | 否 | 超出测量 | REJECT，改为合成算例中显示潜力 |

## 5. 表格与复现材料核验

- 八张生成Markdown表全部嵌入：表3、表4、表5、表6、表7及表A2、表A3、表A4。
- 自动逐字比对结果：八张表去除源文件一级标题后的完整表体与表注，在审定稿中均恰好出现1次；标题均恰好出现1次。
- 附表编号：A1—A9各出现1次，无重号和跳号。
- 表A7逐run保留manifest.evaluation_runs原顺序的6条记录；未为run伪造device或solver线程。
- 表A8分别登记canonical row hash与`cell_file_ledger.csv`文件哈希，并加入当前builder、REPRODUCIBILITY、seed audit与运行后环境freeze哈希。
- 引用的复现材料：`output/supplementary-experiments/REPRODUCIBILITY.md`、`environment-freeze-2026-09-05.txt`、`audits/seed-audit.json`及`audits/seed-audit.md`。

## 6. 写作与完整性检查

- 本节报告原始计算实验，不含外部文献引证；引用数为0，`[CITATION NEEDED]`为0。Source Integrity协议对Results的文献三代理面板不触发。
- 数字来源审计：主表与三张builder附表逐字嵌入；其余关键数字逐项核对formal-results-analysis、manifest、verification、protocol与复现整改材料。
- 推断纪律：不作显著性检验，不声称因果速度、不声称逐实例支配、不把非optimal incumbent写成$J^*$，不外推真实部署。
- 占位扫描：`TODO`、`TBD`、`待回填`、`RID`、`FIELD`、`DERIVE`及补丁前缀均为0。
- 初稿保持：输入初稿SHA-256仍为`3fee329fe38c80beeafad049cb89f8c6c2860906fef22506dec44795b3fee912`；正式目标稿本轮未修改。

## 7. 复审结果

| Reviewer | 复审报告 | 分数 | 结论 |
|---|---|---:|---|
| Reviewer 1（计算与统计聚合） | `post-reviewer-01-statistics.md` | 100/100 | PASS |
| Reviewer 2（实验规格与表格产物） | `post-reviewer-02-spec-tables.md` | 100/100 | PASS |
| Reviewer 3（稳健性与有效性威胁） | `post-reviewer-03-validity.md` | 98/100 | PASS |
| Reviewer 4（正文—数字一致性与学术行文） | `post-reviewer-04-prose-alignment.md` | 100/100 | PASS |
| Reviewer 5（复现性、哈希与种子） | `post-reviewer-05-reproducibility.md` | 97/100 | PASS |
| **五轴等权均值** | — | **99.0/100** | **PASS** |

五份独立复审均为PASS，复审问题合计为阻断0、重要0、轻微0，综合结论为**PASS**。复审登记前后均未修改审定稿；`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md`的SHA-256保持为`c6b1adf87b488d49820f73fcc73ade402e643c80b91b3c4818d7ab2da702721c`。

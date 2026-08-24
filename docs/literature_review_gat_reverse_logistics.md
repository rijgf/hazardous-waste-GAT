# 多周期多目标逆向物流与 GAT 神经求解器文献调研

> 调研日期：2026-08-23；节点—边设计补充调研：2026-08-24
> 调研目的：为“一个模型、不同规模、连续成本—风险偏好”的危险废物多周期逆向物流求解器提供方法依据。
> 文档性质：设计前调研，不等同于代码已经实现，也不把邻近问题上的实验结论直接当成本问题结论。

## 1. 结论摘要

本轮调研得到七项直接结论。

1. 与本项目最接近的危险废物研究通常把选址、路径、处理和库存联合建模为多目标 MILP/MINLP；小规模使用精确求解或 ε-约束生成 Pareto 解，大规模主要使用 NSGA-II、仿真优化、分解算法和滚动时域启发式。
2. 多周期废物收运的难点不是单期路径本身，而是“当前路径决定后续库存与剩余能力”。因此，任何只看局部窗口的学习方法都需要显式的未来义务、剩余处理能力和终端可行性机制。
3. GAT 的尺寸无关参数共享为跨规模提供了结构基础，但不自动产生规模外推。训练规模混合、无量纲特征、动态动作集合、完全图的二次复杂度控制和严格的 OOD 评估同样重要。
4. 原始节点型 GAT 不能充分利用多维边属性。本项目首版以 Lei 等的 residual edge-GAT 为基准，让距离、事故概率、成本和风险进入注意力；GATv2 只作为后续受控消融，不在首版同时增加关系门控和复杂异构结构。
5. “同一个模型覆盖全部成本—风险偏好”已有多目标强化学习和多目标组合优化研究支持。首版应采用共享 GAT + 偏好条件化解码器 + 向量 Critic；MoE 是可选增容机制，不是实现偏好泛化的必要条件。
6. 最后时期库存清零必须是硬约束。奖励惩罚、价值函数或末期修复只能辅助训练，不能替代逐窗口可承接性检查、确定性补全和最终严格校验。
7. 路径规划、取送货、废物收运和多周期设施选址中的图注意力模型，大多只把物理地点或待服务任务作为节点。本项目进一步利用“需求不可拆分、车辆同质、已有完整计划再改进”的结构：只把产生者—废物取货任务和处置设施作为 GAT 节点；不设置 `period`、`vehicle_period`、`route_edge`、`future_waste` 和 `future_facility_capacity` 等虚拟节点；也不向神经网络输入车辆编号。当前选中边形成的每个闭合回路直接表示一条无编号路线，车辆容量和可用数量只作为全局资源与硬约束。

据此，本项目建议的首个可验证版本为：

```text
多规模实例联合训练
→ $(i,s)$ 取货任务/处置设施构成的 Lei 式完全有向图
→ 当前解边标记叠加在完整候选图上
→ 当前/下一期库存、未来义务、无编号路线状态和成本—风险偏好作为动态条件
→ Lei 式 residual edge-GAT 共享编码器
→ 成本—风险偏好条件化共享解码器（默认无 MoE）
→ PPO 选择带硬掩码的局部改进动作
→ 剩余时域可行性预检 / 补全
→ 终期库存严格清零
```

## 2. 问题定位与调研边界

本项目同时包含以下要素：

- 逆向物流：废物从产生者流向处理设施；
- 多周期：产生、收集、处理和库存跨时期耦合；
- 多目标：经济成本和风险存在偏好权衡；
- 多品类与兼容性：废物类型、共载和处理技术受限；
- 路径—库存—处理联合决策；
- 希望使用同一个神经网络检查点处理不同实体数、时期数和偏好。

现有文献通常只覆盖其中一部分。因此，本调研把证据分为三层：

| 证据层 | 主要问题 | 可迁移内容 | 不可直接外推内容 |
|---|---|---|---|
| 危险废物与逆向物流运筹优化 | 成本、风险、选址、路径、库存、多周期 | 目标、约束、终期库存、传统基线 | 神经网络结构和跨规模学习能力 |
| 废物收运与库存路径 | 滚动时域、随机产生、短期路径与长期库存 | 窗口推进、未来价值、末端效应 | 危险废物技术/兼容约束 |
| 神经组合优化与多目标 RL | 图编码、指针动作、偏好条件化、尺寸泛化 | GAT/PPO、统一模型、训练评估协议 | 对本问题可行率和最优差距的直接保证 |

本轮为面向设计的定向调研，不是覆盖所有数据库的系统综述。下文若称“未发现”，仅表示在本轮检索到的直接相关研究中未发现，不能等价为严格的首创性结论。

## 3. 类似问题采用了哪些方法

### 3.1 危险废物：成本—风险联合优化

[Alumur 与 Kara（2007）](https://doi.org/10.1016/j.cor.2005.06.012)建立危险废物多目标选址—路径模型，联合决定处理/处置设施与运输路径，目标包括总成本和运输风险。该类模型说明风险不是普通路径长度的替代量，而需要作为独立目标或约束处理。

[Sheu（2007）](https://doi.org/10.1016/j.cor.2005.06.009)研究多源危险废物的区域协同逆向物流，用多目标分析同时处理逆向物流运营成本和风险。其启示是产生者、设施、处理与运输关系应被统一表达，而不是把路径网络从处理网络中割裂出来。

[Yu 与 Solvang（2016）](https://doi.org/10.3390/ijerph13060548)提出危险废物选址—路径多目标 MIP，并用增强 ε-约束法生成成本—风险 Pareto 曲线。相较仅给定一组加权和，ε-约束更适合作为小规模 Pareto 质量基准，因为非凸离散 Pareto 前沿可能无法被有限个加权和完整覆盖。

### 3.2 多周期、库存和不确定性

[Rabbani、Heidari 与 Yazdanparast（2019）](https://doi.org/10.1016/j.ejor.2018.07.024)把工业危险废物的选址、路径和库存放入随机多周期多目标模型，并以 NSGA-II 与 Monte Carlo 仿真组成 simheuristic。其方法说明大规模与不确定场景常需要“优化器 + 仿真/评价器”的混合框架。

[Ma 与 Li（2021）](https://doi.org/10.3934/jimo.2019117)处理时变废物产生量、允许延迟收集的多周期收运，目标同时包含总成本、运输风险和跨时期风险稳定性；其多目标模型经加权和与线性化转成混合 0-1 线性规划。对本项目的启示是：跨期库存风险不仅体现为总量，还可能体现为风险在时期之间的集中程度。

[Reddy 等（2022）](https://doi.org/10.1016/j.ejor.2022.03.014)研究多周期绿色逆向物流网络设计，联合设施、车辆、流量、库存与排放，并用改进 Benders 分解处理规模问题。其模型明确设置期末库存为零，说明“规划期末状态”在多周期逆向物流中可以且经常需要作为显式约束，而不是软偏好。

[Azizi 与 Hu（2021）](https://doi.org/10.3390/su132413596)采用多阶段随机规划处理多层级、多周期逆向物流和批量决策。该工作提示：若未来扩展到随机产生量，应区分当前可执行决策与未来情景下的补救决策；把未来随机量当成已知确定输入会高估策略能力。

### 3.3 滚动时域与启发式

[Andersen 与 Wøhlk（2016）](https://doi.org/10.1016/j.ejor.2015.08.035)针对多周期可回收物收集，在真实数据上采用滚动时域与变邻域搜索，每个时期重新优化短期计划。它支持“局部窗口 + 重优化”作为大规模多期废物收运的可行框架。

[Spinelli 等（2025）](https://doi.org/10.1016/j.ejor.2024.11.041)针对随机废物累积的库存路径问题，使用多阶段随机规划和滚动时域启发式，并在真实小/大规模实例上测试。其结果支持窗口化求解，但不意味着两期窗口天然保留长期最优性。

[Ben Ahmed 等（2024）](https://doi.org/10.1016/j.tre.2024.103447)专门讨论库存路径滚动时域中的“end-of-times”偏差，并比较终端库存下界、安全库存、折扣以及末端时期松弛等缓解办法。对本项目而言，双期输入若没有远期摘要和终端机制，会系统性地把库存推向网络看不到的未来。

### 3.4 方法谱系总结

| 方法族 | 典型做法 | 优点 | 对本项目的主要局限 | 建议角色 |
|---|---|---|---|---|
| MILP/MINLP | 加权和、ε-约束、分支定界 | 可验证可行性；小规模可得最优解 | 随规模快速变慢 | 小规模真值、修复子问题、可行性 oracle |
| 分解算法 | Benders、分阶段模型 | 利用网络设计与流量子结构 | 路径细节与整数耦合仍困难 | 中规模强基线或末期补全器 |
| 多目标进化 | NSGA-II、MOPSO、自适应 GA | 一次搜索多个 Pareto 解；实现灵活 | 函数评价多；缺少跨实例知识复用 | 大规模 Pareto 基线 |
| Simheuristic | NSGA-II/GA + Monte Carlo | 可处理随机产生量与风险 | 计算成本高 | 不确定扩展基线 |
| VNS/ALNS/滚动时域 | 短窗口反复重优化 | 实用、可解释、易维护可行性 | 有末端效应；手工算子较多 | GAT-PPO 环境骨架与强启发式基线 |
| 神经改进启发式 | 学习选择算子或候选边 | 推理快；可跨实例复用 | 泛化与硬约束无自动保证 | 本项目拟研究方法 |

## 4. GAT 用于本问题时要注意什么

### 4.1 原始 GAT 只是起点

[Veličković 等（2018）](https://openreview.net/forum?id=rJXMpikCZ)提出 GAT，以掩码自注意力聚合邻居，并展示了对未见图的归纳使用能力。但“可处理未见图”不等于“能对更大、更紧或不同分布的优化实例保持解质量”。

[Brody、Alon 与 Yahav（2022）](https://openreview.net/forum?id=F72ximsx7C1)指出原始 GAT 的邻居排序具有静态注意力限制，并提出更具表达力的 GATv2。这说明 GATv2 值得作为结构消融，但为了先建立清晰、可复现的 Lei 基准，首版采用 residual edge-GAT；只有消融显示查询相关注意力带来稳定增益时再替换注意力形式。

### 4.2 边特征必须进入消息和注意力

原始 GAT 主要依据节点表示计算注意力；[Gong 与 Cheng（2019）](https://openaccess.thecvf.com/content_CVPR_2019/html/Gong_Exploiting_Edge_Features_for_Graph_Neural_Networks_CVPR_2019_paper.html)指出这种做法不能充分利用多维边属性。本问题的核心信号大量存在于边上：

- 距离与运输成本；
- 事故概率和单位载荷风险；
- 当前路径相邻关系与路线位置标记；
- 取货任务—设施候选对的处理技术与承接能力；
- 运输弧的道路可行性、行驶时间和风险。

设施剩余能力、产废点库存和同一实体的跨期变化更适合作为节点的动态特征或历史编码，不必为每个时期复制节点。

因此首版采用单一 `travel` 关系，并让边嵌入直接参与注意力：

$$
e_{ij}^{(\ell)}
=\sigma\!\left(
(g^{(\ell)})^{\top}W^{(\ell)}
[h_i^{(\ell-1)}\Vert h_j^{(\ell-1)}\Vert \hat z_{ij}]
\right),
\qquad
\alpha_{ij}^{(\ell)}
=\operatorname{softmax}_{j}(e_{ij}^{(\ell)}),
$$

而不是忽略边属性。技术、共载和处理能力等组合约束可由任务节点的废物标识、候选动作特征和掩码处理，不要求增加关系节点或关系专属 GAT 参数。

### 4.3 完全图过大时，稀疏消融要注意候选覆盖

Lei 式完全图是首版基准。只有规模实验显示其二次边数不可承受时，才考虑稀疏图消融；只保留固定 $k$ 近邻可能删除风险更低但距离较远的关键弧。此时候选边宜取以下集合的并集：

1. 当前方案中的全部真实路径边；
2. 几何距离最近的 $k_d$ 条边；
3. 风险指标最优的 $k_r$ 条边；
4. 兼容且有能力的设施边；
5. 启发式/MILP 暖启动曾使用的边；
6. 少量随机或多样性边，用于避免候选图永久封闭。

训练与测试必须使用同一候选规则，并单独报告“最优/基线边被候选图覆盖的比例”。否则模型表现差可能来自图构造，而不是策略学习。

### 4.4 消息传递深度不能无限增加

[Di Giovanni 等（2023）](https://proceedings.mlr.press/v202/di-giovanni23a.html)分析了消息传递网络中的 over-squashing：远距离、指数增长的信息被压入固定维表示。对本问题，不应单纯靠堆叠很多 GAT 层传播“产生者 → 车辆 → 设施 → 时期 → 远期义务”。建议：

- 使用与 Lei 基准一致的 4 层 residual edge-GAT 作为起点；
- 使用图级池化、历史编码和未来义务汇总特征，避免依赖大量时期节点传播时间信息；
- 只有消融实验确认必要时，才增加少量摘要节点，而不是按实体和时期完整展开；
- 检测注意力熵、节点表示相似度和梯度随图规模的变化。

### 4.5 硬约束优先于惩罚学习

可行动作应在 softmax 前屏蔽。容量、兼容性、处理技术、冻结时期和明显无法完成终期清零的动作不进入策略分布。对仍需组合检查的动作，执行确定性局部修复和剩余时域可行性预检；失败动作不改变环境状态。

终端库存与一般“偏好目标”不同：无论成本权重还是风险权重是多少，可行性都不能被偏好抵消。因此不应把库存未清零仅当作第三个可权衡目标。

## 5. 如何体现同一模型的泛化性

### 5.1 架构可接收变长图，不等于已经泛化

[Joshi 等（2022）](https://doi.org/10.1007/s10601-022-09327-y)指出，在小规模上训练的神经组合优化模型通常不能自然外推到实用的大规模实例。架构的尺寸无关只是必要条件。

[Zhou 等（2023）](https://proceedings.mlr.press/v202/zhou23o.html)把同时跨规模和跨分布称为 omni-generalization，并使用元学习增强测试任务适应。[Bi 等（2022）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ca70528fb11dc8086c6a623da9f3fee6-Abstract-Conference.html)通过多分布教师蒸馏改善跨分布表现。[Drakulic 等（2023）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract.html)则说明状态约简与组合对称性能够改善尺寸外推。

这些工作共同说明，泛化应同时由表示、训练分布、目标尺度和评估协议构成。

### 5.2 尺寸无关表示要求

同一检查点应满足：

- 不使用生产者、车辆、设施的绝对编号嵌入；
- 不使用固定 `object_count` 输出槽位；
- 动作通过当前图节点/边指针产生；
- 参数在同类型节点和同关系边之间共享；
- 图级聚合使用类型内 mean/max/attention pooling，并显式提供 $\log |I|,\log |J|,\log |K|,\log H$；
- 连续量使用容量比、剩余能力比、相对距离、单位载荷风险等无量纲特征；
- 图批处理采用不相交图，损失按图而不是按节点数等权，避免大图主导梯度。

### 5.3 训练分布必须主动覆盖变化

每个训练 episode 联合随机化：

| 维度 | 训练变化 |
|---|---|
| 规模 | $|I|,|S|,|J|,|K|,H$ 的混合取值 |
| 空间 | 均匀、聚类、中心—边缘、现实坐标扰动 |
| 产生 | 总量、时序峰值、类型不平衡、初始库存 |
| 能力 | 车辆数/容量、设施存储与处理紧度 |
| 拓扑 | 技术矩阵、兼容矩阵、运输动作掩码稀疏度（GAT 邻接仍为完全图） |
| 目标 | 成本、事故风险、后果、共载和库存风险尺度 |
| 初始解 | 贪心、GA、扰动可行解及不同质量层级 |
| 偏好 | 两个端点、端点邻域和连续中间偏好 |

使用“混合批次 + 难度课程 + 旧规模回放”，不采用从小规模训练完成后永久切换到大规模的单向课程。

### 5.4 泛化声明的最低实验标准

必须只使用一个冻结检查点，分别测试：

- IID：未见随机种子；
- 规模内插：训练规模之间的组合；
- 规模外推：实体数或时期数超过训练上界；
- 参数偏移：产生量、能力、成本和风险分布变化；
- 空间偏移：均匀训练、聚类或现实地理测试；
- 结构偏移：未见技术/兼容矩阵与约束紧度；
- 偏好外推：训练未直接采样的稠密偏好网格。

主要指标应包括：严格可行率、终期库存清零率、成本/风险、加权目标、Hypervolume、IGD、相对 MILP/GA 的 gap、推理时间、显存以及随规模增长的性能退化斜率。

如果对每个新实例进行梯度更新，应把结果标为“测试时适应”，不能计入零样本同模型泛化的主结果。

## 6. 一个模型如何覆盖不同成本—风险偏好

[Abdolmaleki 等（2020）](https://proceedings.mlr.press/v119/abdolmaleki20a.html)强调多目标在不同单位和尺度下需要尺度不变的偏好处理，并分别保留目标动作分布。[Lin、Yang 与 Zhang（2022）](https://arxiv.org/abs/2203.15386)用单个偏好条件模型近似多目标组合优化的 Pareto 集。[Fan 等（2025）](https://doi.org/10.1109/TNNLS.2024.3371706)进一步在多目标 VRP 中同时利用实例、偏好和规模条件，并报告一个模型跨多种规模的效果。

据此，首版建议：

1. GAT 主干学习共享结构，不为每个偏好复制模型；
2. 偏好 $\alpha=(\alpha_c,\alpha_r)$ 经过 Fourier/MLP 编码；
3. 在每层 Actor MLP 或指针查询中使用 FiLM/门控调制；
4. Critic 输出成本、风险两个价值分量，训练后才按偏好组合；
5. 轨迹保存二维回报，避免在数据层过早丢失目标分量；
6. 训练时端点、中间和端点邻域分层采样；
7. 对同一实例、不同偏好的策略行为加入适度差异性诊断，防止忽略偏好。

### 6.1 MoE 是否必要

不必要。[MVMoE（Zhou 等，2024）](https://proceedings.mlr.press/v235/zhou24c.html)表明 MoE 能提升多 VRP 变体统一求解和迁移能力；[POCCO（Fan 等，2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html)也表明偏好驱动的条件计算可能改善多目标组合优化。但这些证据不能推出本问题必须使用 MoE。

推荐顺序为：

1. **主模型**：共享 GAT + 共享偏好条件化 MLP/指针头；
2. **容量不足诊断**：检查端点与中间偏好性能、策略是否忽略偏好、梯度冲突；
3. **可选增强**：只在 GAT 后的解码器增加 2～4 个小专家，使用连续软路由；
4. **消融结论**：只有 MoE 在相同训练预算、参数量和推理预算下稳定提升 Hypervolume/OOD 表现，才进入正式模型。

把 MoE 限制在解码端可以保留同一个共享图表示，减少专家按规模或分布分裂、造成泛化碎片化的风险。

## 7. 最后时期库存如何处理

### 7.1 为什么不能只在最后一步惩罚

如果前期连续把收集和处理延后，到最后一期可能同时出现：

- 产生者库存超过最后一期车辆总运力；
- 某类废物只能由少数设施处理，剩余处理能力不足；
- 设施先收到废物但没有足够处理能力清零；
- 路径或共载兼容性使理论总容量无法实际利用。

此时再施加大惩罚无法恢复可行性，只会产生大量失败轨迹。滚动时域文献中的末端效应也说明，局部目标会系统性忽视窗口之后的代价。

### 7.2 四层终端保障

#### 第一层：累计必要条件和最迟服务期

对每个废物类型 $s$、窗口右边界 $b$，计算剩余义务：

$$
D^{\mathrm{rem}}_{s,b}
=\sum_i I^P_{i,s,b}
+\sum_j I^F_{j,s,b}
+\sum_i\sum_{t>b}g_{i,s,t}.
$$

令 $\bar P_{j,s,t}$ 为设施对废物 $s$ 在时期 $t$ 的最大可用处理量，
$P^{\mathrm{comm}}_{j,s,t}$ 为已冻结方案占用的处理量，并定义
$\bar P^{\mathrm{free}}_{j,s,t}=[\bar P_{j,s,t}-P^{\mathrm{comm}}_{j,s,t}]_+$。
至少要求：

$$
D^{\mathrm{rem}}_{s,b}
\le
\sum_{j:\tau_{j,s}=1}\sum_{t>b}\bar P^{\mathrm{free}}_{j,s,t}.
$$

该式使用已知的剩余处理能力，而不是尚未决定的未来处理决策量。若多类废物共享设施能力，右端仍只是偏宽松的必要条件，还要在 oracle 中检查跨类型竞争。同时以剩余车辆运力给出聚合必要条件，并根据产生者存储容量和未来产生量反推每个服务事件的最迟收集期。会破坏这些必要条件的延后动作直接屏蔽。

#### 第二层：剩余时域可行性 oracle

必要条件不考虑路线组合，仍可能误判。因此每次提交窗口前，对未冻结时期运行一个快速补全器：

```text
固定已提交时期
→ 保留当前未来方案作为热启动
→ 贪心/流模型补全剩余收集与处理
→ 检查全部库存守恒、容量、技术和终期清零
```

若快速补全失败，可在最后 $h_{\mathrm{repair}}=2$ 或 $3$ 期，或在受影响节点上调用保留的 MILP 环境求一个小型可行性子问题。这里不用 $K$ 表示末期窗口，因为本项目已用 $K$ 表示车辆集合。动作只有在存在可行补全时才允许提交。这比只在 episode 结束时判失败更接近滚动优化中的递归可行性思想。

#### 第三层：终端势函数和辅助预测头

将剩余义务、处理松弛、车辆松弛、最迟服务违约量组成势函数，使用前后势差进行奖励塑形。同时训练一个 `completion_feasibility_head` 预测剩余方案能否完成，用于候选排序。二者只提高学习效率，不改变硬约束判定。

#### 第四层：最终严格校验与回退

最终窗口要求：

$$
I^P_{i,s,H}=0,
\qquad
I^F_{j,s,H}=0,
$$

并调用与 MILP/GA 共用的完整约束检查器。最终修复应优先把任务拉回最后 2～3 期共同处理，而不是把所有剩余任务塞入最后一期。若修复失败，丢弃该候选并返回历史最佳严格可行解。

### 7.3 终期库存与成本—风险偏好的关系

终期清零属于可行域，不参与 $(\alpha_c,\alpha_r)$ 权衡：

```text
先满足终期清零及其他硬约束
再在可行解集合内按成本—风险偏好比较
```

如果业务允许期末残余库存，应通过显式配置切换为另一种问题定义，例如设置允许上限、残值或延期罚金；不能在同一实验中把“必须清零”和“允许付罚金残留”混为一谈。

## 8. 推荐研究路线

### 阶段 A：非 MoE 单模型基线

- Lei 式单关系 residual edge-GAT；
- 产生者—废物取货任务和处置设施构成的 Lei 式完全有向图；
- 当前/下一期库存和未来义务进入节点动态特征或全局上下文；
- 同质车辆不使用编号嵌入；从设施已选出边开始沿任务唯一后继追踪得到无编号闭合路线，容量与路线数上限由硬约束处理；
- 偏好条件化共享 Actor；
- 成本/风险向量 Critic；
- 动态指针动作与硬掩码；
- 可行性 oracle 和终端回退。

### 阶段 B：证明泛化而非只证明可运行

- 多规模、多分布、全偏好联合训练；
- 一个冻结检查点完成所有主测试；
- 对小规模以 MILP/ε-约束为基准；
- 对大规模以 GA、VNS/滚动启发式为基准；
- 报告 Pareto 与终期可行性指标。

### 阶段 C：有证据后再增加复杂度

- 解码器 MoE；
- 元学习或多分布蒸馏；
- 测试时轻量适应；
- 随机产生量和情景树。

## 9. 需要避免的结论

- “GAT 可以输入任意大小图，所以必然泛化到任意规模。”
- “一个连续偏好输入即可保证完整覆盖 Pareto 前沿。”
- “两个成本/风险专家按权重线性混合，就一定得到正确的中间偏好策略。”
- “终期库存给足够大惩罚，模型最终自然会学会清零。”
- “在同一随机生成分布上换一个节点数，就已经证明真实场景泛化。”
- “测试时对每个实例微调后表现好，等同于冻结的同一模型零样本泛化好。”

## 10. 补充调研：类似问题中的节点与边设计

### 10.1 调研结论与方法边界

本次补充调研重点回答“是否必须把车辆、时期、路线边和未来状态全部节点化”。检索到的研究给出的答案基本是否定的。常见设计规律是：

1. **节点表示物理地点或服务任务**，例如仓库、客户、取货任务、送货任务、充电站、废物来源点和处理设施；
2. **边表示移动或业务关系**，例如距离、时间、能耗、成本、风险、处理兼容性和时间窗可达性；
3. **车辆作为解码状态或智能体**，车辆当前位置、剩余载重、电量和已行驶时间通常不转换成图节点；
4. **时期作为动态状态**，通过当前库存、剩余时期比例、历史序列和未来需求汇总表达，很少为每个物理实体复制时期节点；
5. **业务约束通过动作掩码实现**，例如容量、先取后送、废物兼容、处理能力和终期可行性；
6. **多目标偏好作为全局条件**，成本—风险权重输入解码器或超网络，不设置“成本节点”和“风险节点”。

需要严格区分术语：Lei 等和 Zhang 等属于明确以 GAT/edge-GAT 为核心的路线模型；Edge-DIRECT、边增强注意力以及异构注意力模型虽然同样使用图和注意力，但不都属于原始定义下的 GAT；Kool、MAPDP、Pareto Set Learning 等属于稠密 Transformer 式图注意力。后几类仍能为节点、边和状态分工提供直接设计依据，但论文中不应统一称为 GAT。

### 10.2 代表性文献的具体定义

| 文献 | 方法性质 | 节点定义 | 边和联通关系 | 车辆、时期或偏好的处理 |
|---|---|---|---|---|
| [Lei et al.（2022）](https://doi.org/10.1016/j.neucom.2022.08.005) | 残差 edge-GAT | TSP 为城市；CVRP 为一个仓库和若干客户，客户特征含坐标与归一化需求 | 构造完全有向图，边属性为节点间距离；节点对及边属性共同参与注意力 | 当前位置、剩余容量进入解码上下文；车辆不是节点 |
| [Zhang et al.（2023）](https://doi.org/10.1016/j.physa.2023.128451) | GAT-RL，多仓库 VRP | 多个仓库和客户；客户特征含坐标、需求、时间窗及早到/迟到惩罚 | 可获取资料确认了节点输入和灵活匹配解码，但没有确认正文中的精确邻接规则，因此不推断其为完全图或稀疏图 | 车辆通过匹配/解码策略处理，不建立车辆节点 |
| [Edge-DIRECT（2024）](https://arxiv.org/abs/2407.01615) | 显式图约束与 edge-enhanced attention | 仓库、客户、充电站 | 若存在从节点 $i$ 的时间窗出发并能在节点 $j$ 时间窗内到达的方案，则建立有向边 $i\to j$；边属性含行驶时间和能耗 | 车型、载重、电量和当前位置进入双解码器状态 |
| [Liu et al.（2023）](https://doi.org/10.2514/1.I011171) | 边增强注意力，无人机取送与充电 | 仓库、充电站、取货任务、送货任务 | 仓库可到取货点/充电站；取货点只到对应送货点或充电站；送货点可到后续取货点、充电站或仓库；充电站与其他类型双向连接；边属性含距离和能耗 | 无人机状态进入解码上下文，不建立无人机节点 |
| [Li et al.（2022）](https://doi.org/10.1109/TITS.2021.3056120) | 异构稠密注意力，非严格 GAT | 一个仓库、若干取货任务及其配对送货任务 | 使用取货—送货、取货—取货、送货—送货等语义注意力通道；配对和先取后送由嵌入结构及动作掩码表达 | 不设置配对关系节点或车辆节点 |
| [MAPDP（2022）](https://doi.org/10.1609/aaai.v36i9.21236) | 多智能体稠密注意力，非严格 GAT | 仓库、取货点、送货点 | 物理任务节点之间采用稠密注意力，先取后送和容量由掩码约束 | 多辆车是共享编码器的多个解码智能体，不是图节点 |
| [Multi-Type Attention（2024）](https://doi.org/10.1109/TITS.2024.3413077) | 多视图稠密注意力，非严格 GAT | 只有仓库和客户 | 同一批节点形成仓库视图、客户视图和全节点视图；没有为每种语义关系新增节点 | 剩余容量、累计距离和当前路径保存在路线状态中 |
| [LAtt-PR（2026）](https://doi.org/10.3390/math14040612) | WEEE 多周期设施选址，包含 GAT | 废弃电器来源点和候选回收设施 | 空间模块处理设施及来源之间的空间/服务关系 | 历史时期由特征和 LSTM 表达，不设置逐期节点 |
| [多任务深度强化学习库存—路径模型（2025）](https://doi.org/10.1007/s44176-025-00053-2) | GIN，非 GAT，但问题结构接近 | 供应点/仓库和零售点，节点特征含位置、库存、补货量和需求 | 运输网络边用于图聚合 | 多时期通过 MDP 状态和历史表达，不设置时期节点 |
| [Learning Heuristics for Arc Routing Problems（2023）](https://doi.org/10.1016/j.iswa.2023.200300) | 废物收运邻近问题，GNN | 把每条需要服务的街道边转换成线图中的任务节点 | 原路网中的两条服务边共享端点时，对应任务节点相连 | 当前部分路线作为节点标记，不设置路线边节点 |
| [Direction-aware Deep Policy Learning for CARP（2026）](https://doi.org/10.1016/j.engappai.2026.114695) | 方向感知图注意力，弧路径问题 | 每条无向必服务道路拆成两个有向弧节点；车场作为零需求自环弧节点 | 弧节点形成完全有向图，边权为前一弧终点到后一弧起点的最短路成本 | 车辆剩余容量进入解码上下文和可行性掩码 |
| [Pareto Set Learning（2022）](https://arxiv.org/abs/2203.15386) | 多目标稠密注意力，非严格 GAT | 多目标 CVRP 仍只使用仓库和客户节点 | 使用全连接注意力编码运输节点 | 偏好向量输入条件模型/超网络；不设置目标节点 |

从与本项目最接近的“多周期废物 + GAT”证据看，LAtt-PR 把空间关系交给 GAT、把时间关系交给 LSTM；从明确的 edge-GAT 路径模型看，Lei 等只使用仓库/客户节点和距离边；从成本—风险连续偏好看，Pareto Set Learning 把偏好放入条件模型。这三类证据共同支持“任务—设施图 + 动态上下文”的简化路线。

### 10.3 对本项目图结构的修订建议

#### 先说明构图目的

本项目构图不是为了复刻一张道路地图，而是为了让同一个 GAT 同时看到：问题中全部可比较的运输连接、当前完整计划已经选择的路线、当前与下一期库存，以及不能违反的业务约束。如果只输入当前解中已经选中的边，网络将看不到尚未使用但可能改善方案的候选边；如果把每个时期和每辆同质车辆都复制成节点，图规模和编号对称又会妨碍跨规模泛化。

因此，调研结果对应的项目方案是：以 Lei 式完整有向图表达候选空间，在同一张图上叠加当前解标记和跨期动态特征，再由独立硬掩码限制实际动作。

#### 整体输入是什么

先定义本节使用的基本符号：

- $I$：产废地点集合，$i$ 表示其中一个产废地点；
- $S$：废物类型集合，$s$ 表示其中一种废物；
- $J$：处置设施集合，$j$ 表示其中一个设施；
- $H$：规划期包含的总时期数；
- $t$：当前滚动窗口的起始时期；
- $u_{i,s}$：产废地点 $i$ 中废物 $s$ 对应的取货任务节点；
- $d_j$：处置设施 $j$ 对应的设施节点。

全部节点组成集合 $V$：

$$
V=\{u_{i,s}:i\in I,s\in S\}\cup\{d_j:j\in J\}.
$$

这里没有时期节点、车辆节点、路线边节点或未来库存节点。当前模型没有独立车场，因此处置设施同时作为闭合路线的起点和终点；如果业务模型以后加入独立车场，再增加第三类物理路径节点。

一次输入由以下信息组成：

- 节点集合 $V$；
- 完整有向编码边集合 $E_{\mathrm{GAT}}$；
- 当前期、下一期和更远时期摘要组成的节点特征 $X_t$；
- 道路、成本、风险和兼容性组成的业务边特征 $Z_t$；
- 静态边掩码和动态动作掩码的总称 $M_t$；
- 当前期与下一期的选中边标记 $Y_t$；
- 时期、规模、同质车队资源和成本—风险偏好组成的全局条件 $C_t$。

定义完这些组成部分后，完整输入可写为：

$$
G_t=\left(V,E_{\mathrm{GAT}},X_t,Z_t,M_t,Y_t,C_t\right).
$$

$G_t$ 的准确含义是“带有两期库存状态、硬约束和当前解标记的完整候选有向图”，不是纯道路图，也不是只含已选边的解图。实现时可以把 $Y_t$ 拼入 $Z_t$，但概念上仍应区分“边的业务属性”和“当前是否选中”。

#### 节点怎样承载时期与库存

取货任务节点 $u_{i,s}$ 是跨期持续的库存账户和可选服务对象，不是一次性订单。先定义：

- $I^P_{i,s,t-1}$：时期 $t$ 开始前，产废地点 $i$ 尚未运走的废物 $s$ 库存；
- $g_{i,s,t}$：时期 $t$ 新产生的废物量；
- $BG_{i,s,t}$：时期 $t$ 可收运总量，等于期初库存与本期产生量之和。

三者满足：

$$
BG_{i,s,t}=I^P_{i,s,t-1}+g_{i,s,t}.
$$

需求不可拆分意味着：若时期 $t$ 访问节点 $u_{i,s}$，就收走当期全部 $BG_{i,s,t}$；若不访问，则全部结转。每个任务节点同时保存当前期库存与产生量、下一期产生量与当前计划下的预计库存，以及更远时期累计产生量、最迟服务松弛和库存压力。同一产废地点的不同废物任务共享坐标和产废端容量，但分别保存废物属性、库存、风险和兼容信息。

设施节点 $d_j$ 保存位置、是否开放、总存储能力和聚合处理能力；针对具体废物 $s$ 的处理技术、处理成本、设施库存风险和分类型处理能力放在任务—设施条件边特征中。这样废物类型数变化只改变任务节点数量，不改变单节点输入维度。

时期不改变图的节点和邻接。$H\ge2$ 时，一次输入同时表示窗口 $\{t,t+1\}$；更远时期只输入累计产生量、累计处理能力、剩余时期和最迟服务紧迫度。窗口向前滚动时只更新动态特征与当前解标记，不复制节点。

#### 有向边与联通关系怎样定义

用 $u$ 和 $v$ 表示节点集合 $V$ 中任意两个节点，$(u,v)$ 表示从 $u$ 到 $v$ 的有向连接。按照 Lei 的完全图原则，编码边集合为：

$$
E_{\mathrm{GAT}}=V\times V.
$$

$V\times V$ 表示所有有序节点对，因此 A→B 与 B→A 是两条不同边。自环也保留用于 GAT 消息传播。完整编码边只表示节点之间可以交换信息，不表示所有边都能被运输路线选择。

实际运输联通规则如下：

| 起点 | 终点 | 运输动作何时允许 |
|---|---|---|
| 处置设施 $d_j$ | 任务 $u_{i,s}$ | 道路可达，并且该设施能够处理整条路线最终携带的废物 |
| 任务 $u_{i,s}$ | 其他地点任务 $u_{i',s'}$ | $i\ne i'$、道路可达，并且整条路线满足容量与共载兼容 |
| 任务 $u_{i,s}$ | 同址任务 $u_{i,s'}$ | 当前 MIP 禁止同址任务直连，作为运输动作永久屏蔽；编码边仍保留 |
| 任务 $u_{i,s}$ | 处置设施 $d_j$ | 道路可达，设施能处理路线全部载荷，存储和处理能力可承接 |
| 处置设施 $d_j$ | 处置设施 $d_{j'}$ | 当前模型不允许设施间转运，永久屏蔽 |
| 任意节点 $u$ | 自身 $u$ | 只作 GAT 自环，永久禁止作为运输动作 |

若两个地点没有直达道路但存在可行道路路径，则有向物流边使用相应最短可行路径的距离、时间和风险；只有完全不可达时才把道路可达标记设为 0 并永久屏蔽运输动作。

每条边的基础字段包括有向距离、时间、事故概率、归一化运输成本、道路基础风险、道路可达性和节点类型对。任务—任务边追加废物兼容与共载风险；任务—设施边追加处理技术、处理成本、设施分废物库存风险和当前/未来能力松弛。不适用字段置零并增加适用性标记，避免把“不适用”和“数值恰好为零”混为一谈。

#### 完整候选图如何叠加当前解

定义 $y_{uv,t}$ 为当前解的选中标记：时期 $t$ 使用边 $u\rightarrow v$ 时取 1，否则取 0。当前两期窗口同时保存 $y_{uv,t}$ 和 $y_{uv,t+1}$。因此网络能够区分：

- 已被当前路线使用的边；
- 当前没有使用、但可以作为改进候选的边；
- 因道路或业务规则而永久不能使用的边。

只保留所有 $y_{uv,t}=1$ 的边会得到时期 $t$ 的当前解子图。严格可行时，每个已访问任务恰有一个前驱和一个后继。从设施的每条已选出边开始，沿任务的唯一后继连续追踪，最终必须回到同一设施，由此得到若干条“设施→任务→……→同一设施”的闭合回路。不同路线可以共享设施节点，但运输边互不重复；这些闭合回路表达访问顺序、路线分组、处置设施选择和路线数量。统一检查器还必须排除任何与设施脱离的任务子回路。

#### 为什么不再输入车辆编号

先定义 $Q$ 为每辆同质车辆的载重上限，定义 $K_t^{\max}$ 为时期 $t$ 最多可以使用的车辆数。由于车辆同质，每个闭合回路只需要占用一辆车，给它标成“车辆1”还是“车辆3”不会改变成本、风险或可行性。因此神经网络不输入车辆下标，也不学习车辆 ID。

当前选中边形成的每条闭合回路作为一条无编号路线。路线总载荷由其任务节点当期可收运量求和得到；路线容量余量、所含废物属性和节点表示池化可作为候选动作特征。新建路线使用统一的空路线占位，并检查新建后的路线数是否超过 $K_t^{\max}$。现有 MIP 仍可保留车辆下标用于内部约束和输出，求解后把无编号路线任意分配给同质车辆即可。

#### 静态掩码、动态掩码与期末约束

静态边掩码处理不会随当前解改变的禁止关系，例如完全不可达、设施间转运、同址任务直连和自环运输。动态动作掩码处理依赖整条路线和跨期状态的约束，例如路线容量、全部已装废物的兼容性、目标设施对整条路线的处理能力、库存结转和剩余时域可承接性。

期末库存清零不是两个节点之间的连接关系，因此不新增“末期节点”或“末期边”。节点与全局条件提供剩余义务、能力松弛和最迟服务紧迫度；动作掩码、可行性预检和最终统一检查器保证期末清空。奖励塑形只帮助学习，不能把不可行方案变成可行方案。

综上，首版图输入收敛为：**2 类路径节点、1 种共享 `travel` 边、1 张完整有向候选图、2 期详细状态、当前解边标记、无编号路线状态以及独立硬掩码**。稀疏 kNN 图仅作为完整图显存不可接受时的效率消融，不是默认结构。

### 10.4 正文获取状态与待进一步核验文献

本轮没有找到同时完整覆盖“危险废物逆向物流 + 多周期库存路径 + 多目标偏好 + 端到端 GAT”的单篇论文；这只是本轮检索结果，不能作为首创性证明。以下文献题目和 DOI 值得进一步获取全文：

1. *Graph attention reinforcement learning with flexible matching policies for multi-depot vehicle routing problems*，DOI：[10.1016/j.physa.2023.128451](https://doi.org/10.1016/j.physa.2023.128451)。已确认节点及特征，未从可获取正文确认精确邻接规则。
2. *An improved graph attention network combined with reinforcement learning for capacitated vehicle routing problem*，DOI：[10.1007/s10489-025-06932-1](https://doi.org/10.1007/s10489-025-06932-1)。本轮未取得可核验的完整方法正文。
3. *Heterogeneous Attention-Based Graph Convolutional Network for Solving Asymmetric Pickup and Delivery Problem*，DOI：[10.1109/TASE.2025.3548141](https://doi.org/10.1109/TASE.2025.3548141)。本轮未取得可核验的完整方法正文。
4. *Eco-routing for electric logistic vehicles using deep reinforcement learning with graph attention networks*，DOI：[10.1016/j.tre.2026.104908](https://doi.org/10.1016/j.tre.2026.104908)。可获取接受稿页面，但本轮尚未逐项核验其节点和邻接定义。

## 11. 核心参考文献

### 危险废物、逆向物流与多周期库存

1. Alumur, S.; Kara, B. Y. [A new model for the hazardous waste location-routing problem](https://doi.org/10.1016/j.cor.2005.06.012). *Computers & Operations Research*, 2007.
2. Sheu, J.-B. [A coordinated reverse logistics system for regional management of multi-source hazardous wastes](https://doi.org/10.1016/j.cor.2005.06.009). *Computers & Operations Research*, 2007.
3. Yu, H.; Solvang, W. D. [An Improved Multi-Objective Programming with Augmented ε-Constraint Method for Hazardous Waste Location-Routing Problems](https://doi.org/10.3390/ijerph13060548). *IJERPH*, 2016.
4. Rabbani, M.; Heidari, R.; Yazdanparast, R. [A stochastic multi-period industrial hazardous waste location-routing problem: Integrating NSGA-II and Monte Carlo simulation](https://doi.org/10.1016/j.ejor.2018.07.024). *EJOR*, 2019.
5. Ma, H.; Li, X. [Multi-period hazardous waste collection planning with consideration of risk stability](https://doi.org/10.3934/jimo.2019117). *Journal of Industrial & Management Optimization*, 2021.
6. Reddy, K. N.; Kumar, A.; Choudhary, A.; Cheng, T. C. E. [Multi-period green reverse logistics network design: An improved Benders-decomposition-based heuristic approach](https://doi.org/10.1016/j.ejor.2022.03.014). *EJOR*, 2022.
7. Azizi, V.; Hu, G. [A Multi-Stage Stochastic Programming Model for the Multi-Echelon Multi-Period Reverse Logistics Problem](https://doi.org/10.3390/su132413596). *Sustainability*, 2021.
8. Andersen, M. E.; Wøhlk, S. [A variable neighborhood search for the multi-period collection of recyclable materials](https://doi.org/10.1016/j.ejor.2015.08.035). *EJOR*, 2016.
9. Spinelli, A. et al. [A rolling horizon heuristic approach for a multi-stage stochastic waste collection problem](https://doi.org/10.1016/j.ejor.2024.11.041). *EJOR*, 2025.
10. Ben Ahmed et al. [Optimizing the long-term costs of an Inventory Routing Problem using linear relaxation](https://doi.org/10.1016/j.tre.2024.103447). *Transportation Research Part E*, 2024.

### GAT、神经组合优化与泛化

11. Veličković, P. et al. [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ). ICLR, 2018.
12. Brody, S.; Alon, U.; Yahav, E. [How Attentive are Graph Attention Networks?](https://openreview.net/forum?id=F72ximsx7C1). ICLR, 2022.
13. Gong, L.; Cheng, Q. [Exploiting Edge Features for Graph Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Gong_Exploiting_Edge_Features_for_Graph_Neural_Networks_CVPR_2019_paper.html). CVPR, 2019.
14. Kool, W.; van Hoof, H.; Welling, M. [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm). ICLR, 2019.
15. Joshi, C. K. et al. [Learning the Travelling Salesperson Problem Requires Rethinking Generalization](https://doi.org/10.1007/s10601-022-09327-y). *Constraints*, 2022.
16. Zhou, J. et al. [Towards Omni-generalizable Neural Methods for Vehicle Routing Problems](https://proceedings.mlr.press/v202/zhou23o.html). ICML, 2023.
17. Bi, J. et al. [Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ca70528fb11dc8086c6a623da9f3fee6-Abstract-Conference.html). NeurIPS, 2022.
18. Drakulic, D. et al. [BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract.html). NeurIPS, 2023.

### 多目标偏好条件化与可选 MoE

19. Abdolmaleki, A. et al. [A Distributional View on Multi-Objective Policy Optimization](https://proceedings.mlr.press/v119/abdolmaleki20a.html). ICML, 2020.
20. Lin, X.; Yang, Z.; Zhang, Q. [Pareto Set Learning for Neural Multi-Objective Combinatorial Optimization](https://arxiv.org/abs/2203.15386). ICLR, 2022.
21. Chen, J. et al. [Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html). NeurIPS, 2023.
22. Fan, M. et al. [Conditional Neural Heuristic for Multi-objective Vehicle Routing Problems](https://doi.org/10.1109/TNNLS.2024.3371706). *IEEE TNNLS*, 2025.
23. Zhou, J. et al. [MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts](https://proceedings.mlr.press/v235/zhou24c.html). ICML, 2024.
24. Fan, M. et al. [Preference-Driven Multi-Objective Combinatorial Optimization with Conditional Computation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html). NeurIPS, 2025.

### 节点—边构造补充文献

25. Lei, K. et al. [Solve routing problems with a residual edge-graph attention neural network](https://doi.org/10.1016/j.neucom.2022.08.005). *Neurocomputing*, 2022.
26. Zhang, K.; Lin, X.; Li, M. [Graph attention reinforcement learning with flexible matching policies for multi-depot vehicle routing problems](https://doi.org/10.1016/j.physa.2023.128451). *Physica A*, 2023.
27. Mozhdehi, A. et al. [Edge-DIRECT: A Deep Reinforcement Learning-based Method for Solving Heterogeneous Electric Vehicle Routing Problem with Time Window Constraints](https://arxiv.org/abs/2407.01615). Canadian AI, 2024.
28. Liu, R.; Shin, H. S.; Tsourdos, A. [Edge-Enhanced Attentions for Drone Delivery in Presence of Winds and Recharging Stations](https://doi.org/10.2514/1.I011171). *Journal of Aerospace Information Systems*, 2023.
29. Li, J. et al. [Heterogeneous Attentions for Solving Pickup and Delivery Problem via Deep Reinforcement Learning](https://doi.org/10.1109/TITS.2021.3056120). *IEEE Transactions on Intelligent Transportation Systems*, 2022.
30. Zong, Z. et al. [MAPDP: Cooperative Multi-Agent Reinforcement Learning to Solve Pickup and Delivery Problems](https://doi.org/10.1609/aaai.v36i9.21236). AAAI, 2022.
31. Li, J. et al. [Multi-Type Attention for Solving Multi-Depot Vehicle Routing Problems](https://doi.org/10.1109/TITS.2024.3413077). *IEEE Transactions on Intelligent Transportation Systems*, 2024.
32. Qu et al. [LAtt-PR: Hybrid Reinforced Adaptive Optimization for Conquering Spatiotemporal Uncertainties in Dynamic Multi-Period WEEE Facility Location](https://doi.org/10.3390/math14040612). *Mathematics*, 2026.
33. [Enhanced multi-task deep reinforcement learning for the integrated inventory-routing problem under VMI mode](https://doi.org/10.1007/s44176-025-00053-2). *Management System Engineering*, 2025.
34. [Learning heuristics for arc routing problems](https://doi.org/10.1016/j.iswa.2023.200300). *Intelligent Systems with Applications*, 2023.
35. [Direction-aware deep policy learning for efficient capacitated arc routing](https://doi.org/10.1016/j.engappai.2026.114695). *Engineering Applications of Artificial Intelligence*, 2026.

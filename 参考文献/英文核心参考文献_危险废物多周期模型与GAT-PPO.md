> 2026-09-08同步说明：下文20篇为历史检索记录，不代表当前本地文献数量或综述引用范围。当前范围为参考文献文件夹全部独立论文15篇，加用户指定的和剑波等、李政辉等2篇，共17篇。

# 危险废物多周期库存—路径模型与 Residual Edge-GAT–PPO 方法：英文核心参考文献

> [!WARNING]
> **历史版本，请勿用于当前稿件。** 当前实际方法已确认为 PPO + Transformer，而不是 GAT。当前以本地PDF为准的17篇独立文献清单见：[英文核心参考文献_危险废物多周期模型与PPO-Transformer.md](./英文核心参考文献_危险废物多周期模型与PPO-Transformer.md)。本文件仅保留作替换审计与版本追溯。

> 检索与核验日期：2026-08-25  
> 文献数量：20 篇  
> 引用格式：APA 7  
> 核验口径：Crossref/DOI 元数据 + 出版社或会议官方页面；分区提示统一采用可核验的 2024 JCR 口径

## 1. 检索对象与结论

本清单针对当前工作目录中的数学模型与求解设计组织文献。数学模型的核心要素包括：危险废物逆向收运、确定性多周期、产废端与处理设施端库存守恒、车辆闭合路径、多废物兼容混装、处理技术与处理能力、运输/库存/混装风险、成本—风险权衡以及规划期末库存清零。求解框架的核心要素包括：完全有向业务图、显式边属性、Residual Edge-GAT、偏好条件 PPO、硬可行性掩码，以及从一个完整可行解出发的局部改进策略。

### 1.1 核心判断

在本次核验的文献范围内，没有发现一篇论文同时覆盖“危险废物逆向物流 + 多周期双端库存 + 车辆路径 + 兼容混装及成对附加风险 + 偏好条件 Residual Edge-GAT–PPO 局部改进”。现有研究分别覆盖了这些模块，因此本文较稳妥的创新定位是**组合式集成与面向特定业务约束的方法扩展**，而不是未经系统综述便声称“首次提出”。

20 篇文献中，有 12 篇发表于按 2024 JCR 可核验为 Q1 的期刊；其余主要是 PPO、GAT、神经组合优化领域的经典预印本或 ICLR、NeurIPS、IROS、ICANN 同行评审会议论文。会议不适用 JCR/中科院期刊分区。这里的“Q1”也不等同于中科院一区、UTD24 或 FT50，正式投稿前应按学校数据库与目标年份重新核对。

### 1.2 最值得优先阅读的文献

1. **总体数学模型最接近：** Rabbani et al. (2019)。
2. **混装载荷风险最接近：** Paredes-Belmar et al. (2017)。
3. **废物兼容矩阵最接近：** Rabbani et al. (2018)。
4. **多周期库存—路径最接近：** Ma and Li (2021)；Elbek and Wøhlk (2016)。
5. **Residual Edge-GAT + PPO 路由最接近：** Lei et al. (2022)。
6. **PPO 局部改进完整路线最接近：** Guan et al. (2025)，但其编码器是注意力 Transformer，不是 GAT。
7. **偏好条件统一模型最接近：** Lin et al. (2022)，但其多目标强化学习并非 PPO。
8. **管理/工业规划中的直接 GAT + PPO：** Liu et al. (2024)；Liu et al. (2025)。

## 2. 数学模型文献对照

符号说明：✓ = 明确覆盖；△ = 部分覆盖或语义相近；— = 未覆盖或无法从原始来源确认。

| # | 文献 | 危废/逆向 | 多周期 | 车辆路径 | 库存联动 | 混装/兼容 | 成本—风险 | 在本文中的主要作用 |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | Alumur & Kara (2007) | ✓ | — | △ | — | △（废物—处理技术） | ✓ | 危险废物位置—路径与技术兼容经典基础 |
| 2 | Sheu (2007) | ✓ | △ | △ | △（储存/处理） | — | ✓ | 危废协同逆向物流与系统风险边界 |
| 3 | Elbek & Wøhlk (2016) | △（可回收物） | ✓ | ✓ | ✓ | △（分舱多品类） | — | 多周期累积库存—收集路径 |
| 4 | Paredes-Belmar et al. (2017) | ✓ | — | ✓ | — | ✓（多产品同车） | ✓ | 载荷组合改变路段风险 |
| 5 | Rabbani et al. (2018) | ✓ | — | ✓ | — | ✓（不相容废物） | ✓ | 兼容矩阵与异质废物共运限制 |
| 6 | Rabbani et al. (2019) | ✓ | ✓ | ✓ | ✓ | — | ✓ | 多周期危废位置—路径—库存总体对照 |
| 7 | Ma & Li (2021) | ✓ | ✓ | ✓ | △（延期/结转） | — | ✓ | 多周期延迟收集、风险稳定性与加权偏好 |
| 8 | Wu et al. (2022) | △（危险品正向分销） | △ | △ | ✓ | ✓（不相容及叠加风险） | ✓ | 库存风险与多类别危险品交互风险 |

### 2.1 Alumur and Kara (2007) — 危险废物位置—路径经典基础

**匹配度：高；2024 JCR Q1 期刊。** 论文联合决定处理中心及其技术、处置中心和不同废物流的运输路径，同时最小化成本与运输风险。它直接支撑“废物—处理技术兼容”“设施能力/技术与路径联动”和成本—风险双目标，但不含多周期双端库存、同车混装风险或逐车闭合巡回。

**在本文中的建议用途：** 用于数学模型综述开端，说明危险废物运输不能脱离处理技术、设施能力和风险目标单独优化。

**APA 7：** Alumur, S., & Kara, B. Y. (2007). A new model for the hazardous waste location-routing problem. *Computers & Operations Research, 34*(5), 1406–1423. https://doi.org/10.1016/j.cor.2005.06.012

- DOI：[10.1016/j.cor.2005.06.012](https://doi.org/10.1016/j.cor.2005.06.012)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0305054805001887)

### 2.2 Sheu (2007) — 危险废物协同逆向物流

**匹配度：高；2024 JCR Q1 期刊。** 论文面向多个产废源的区域危险废物系统，协调收集、储存、处理、运输与最终处置，并同时考虑运营成本和相关风险。其贡献更接近跨组织网络流协调而非逐车 VRP，因此适合定义逆向物流系统边界及库存/处理风险来源，不宜用来支撑本文的全部路径和混装约束。

**在本文中的建议用途：** 支撑“风险不仅发生在运输弧上，还发生在未收集、储存和处理环节”的目标函数分解。

**APA 7：** Sheu, J.-B. (2007). A coordinated reverse logistics system for regional management of multi-source hazardous wastes. *Computers & Operations Research, 34*(5), 1442–1462. https://doi.org/10.1016/j.cor.2005.06.009

- DOI：[10.1016/j.cor.2005.06.009](https://doi.org/10.1016/j.cor.2005.06.009)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0305054805001905)

### 2.3 Elbek and Wøhlk (2016) — 多周期多品类库存—收集路径

**匹配度：高（库存路径模块）；2024 JCR Q1 期刊。** 论文研究玻璃和纸张在收集点持续累积时的多周期收集计划，以带两个容器的车辆将两类物料送往不同处理设施，并采用滚动时域和变邻域搜索。它对“跨期累积库存—本期是否收集—路线”联动非常直接，但物料并非危废，两类物料物理分舱，也没有事故风险与设施端处理库存。

**在本文中的建议用途：** 支撑多周期库存路径、滚动时域、末端效应和多品类收集设计。

**APA 7：** Elbek, M., & Wøhlk, S. (2016). A variable neighborhood search for the multi-period collection of recyclable materials. *European Journal of Operational Research, 249*(2), 540–550. https://doi.org/10.1016/j.ejor.2015.08.035

- DOI：[10.1016/j.ejor.2015.08.035](https://doi.org/10.1016/j.ejor.2015.08.035)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0377221715007900)

### 2.4 Paredes-Belmar et al. (2017) — 多产品同车装载与风险

**匹配度：很高（混装模块）；2024 JCR Q1 期刊。** 论文允许同质容量车队在一次收集中共同装载多类工业危险物质，并最小化运输成本与暴露人口风险；车辆所载物质组合会改变路段风险。其风险表达主要由车上最高风险等级驱动，而本文拟采用兼容矩阵和数量相关的成对共载附加风险，因此两者既有直接联系，也存在明确扩展空间。

**在本文中的建议用途：** 作为“混装会改变运输风险、不能只把多品类流量相加”的最直接管理类依据。

**APA 7：** Paredes-Belmar, G., Bronfman, A., Marianov, V., & Latorre-Núñez, G. (2017). Hazardous materials collection with multiple-product loading. *Journal of Cleaner Production, 141*, 909–919. https://doi.org/10.1016/j.jclepro.2016.09.163

- DOI：[10.1016/j.jclepro.2016.09.163](https://doi.org/10.1016/j.jclepro.2016.09.163)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0959652616315049)

### 2.5 Rabbani et al. (2018) — 不相容废物类型约束

**匹配度：很高（兼容模块）；2024 JCR Q1 期刊。** 论文在多目标工业危险废物位置—路径问题中显式禁止不相容废物共同运输，并同时最小化成本、运输暴露风险和设施场址风险。其兼容性主要是硬禁止关系；本文还要处理“可以共载但风险上升”的连续附加风险，因而可将其作为兼容硬约束的直接基础。

**在本文中的建议用途：** 支撑废物—废物兼容矩阵、异质车队/容量约束及 NSGA-II、MOPSO 等传统多目标基线。

**APA 7：** Rabbani, M., Heidari, R., Farrokhi-Asl, H., & Rahimi, N. (2018). Using metaheuristic algorithms to solve a multi-objective industrial hazardous waste location-routing problem considering incompatible waste types. *Journal of Cleaner Production, 170*, 227–241. https://doi.org/10.1016/j.jclepro.2017.09.029

- DOI：[10.1016/j.jclepro.2017.09.029](https://doi.org/10.1016/j.jclepro.2017.09.029)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0959652617320176)

### 2.6 Rabbani et al. (2019) — 随机多周期危废位置—路径—库存

**匹配度：最高；2024 JCR Q1 期刊。** 论文把工业危险废物的设施选址、车辆路径、库存控制、多周期决策以及成本—风险多目标结合，并用 NSGA-II 与 Monte Carlo 仿真求解不确定模型。它是当前整体问题结构最接近的对照；主要差异是其包含战略选址与随机性，而当前模型以给定设施条件下的确定性收运—处理为主，且原始来源不能证明其含有本文式成对共载附加风险和双端期末清零。

**在本文中的建议用途：** 作为文献综述与模型比较表中的首要基准文献。

**APA 7：** Rabbani, M., Heidari, R., & Yazdanparast, R. (2019). A stochastic multi-period industrial hazardous waste location-routing problem: Integrating NSGA-II and Monte Carlo simulation. *European Journal of Operational Research, 272*(3), 945–961. https://doi.org/10.1016/j.ejor.2018.07.024

- DOI：[10.1016/j.ejor.2018.07.024](https://doi.org/10.1016/j.ejor.2018.07.024)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0377221718306295)

### 2.7 Ma and Li (2021) — 多周期延期收集与风险稳定性

**匹配度：很高；非 Q1（2024 JCR：OR & Management Science Q3；Engineering, Multidisciplinary Q2）。** 论文针对随时期变化的危险废物产生量决定各期收集量与车辆路线，允许不完全收集和延期收集，并优化成本、运输风险和跨期风险稳定性。它最适合支撑库存结转语义、跨期风险分布和加权偏好，但不含多废物兼容混装或设施处理库存。

**在本文中的建议用途：** 即使强调一区文献也建议保留，因为其主题匹配度高于许多泛化的逆向物流网络设计论文。

**APA 7：** Ma, H., & Li, X. (2021). Multi-period hazardous waste collection planning with consideration of risk stability. *Journal of Industrial & Management Optimization, 17*(1), 393–408. https://doi.org/10.3934/jimo.2019117

- DOI：[10.3934/jimo.2019117](https://doi.org/10.3934/jimo.2019117)
- 期刊原始网页：[AIMS Press](https://www.aimsciences.org/article/doi/10.3934/jimo.2019117)

### 2.8 Wu et al. (2022) — 库存风险与多类别危险品叠加风险

**匹配度：高（风险模块）；2024 JCR Q1 期刊。** 论文在多类别危险品分销网络中联合优化设施、库存与运输，并显式考虑运输/储存不相容、库存风险及事故后的叠加风险。其场景是正向危险品分销，且“叠加风险”不等同于本文同车同弧上的成对共载风险，但它能高质量支撑库存危险性与类别交互风险需要单独建模。

**在本文中的建议用途：** 用于解释风险目标中的库存风险、运输风险和混合危险品交互风险为何不可相互替代。

**APA 7：** Wu, W., Ma, J., Liu, R., & Jin, W. (2022). Multi-class hazmat distribution network design with inventory and superimposed risks. *Transportation Research Part E: Logistics and Transportation Review, 161*, Article 102693. https://doi.org/10.1016/j.tre.2022.102693

- DOI：[10.1016/j.tre.2022.102693](https://doi.org/10.1016/j.tre.2022.102693)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1366554522000850)

## 3. 方法文献对照

| # | 文献 | 图/注意力编码 | PPO | 规划/组合优化 | 完整解局部改进 | 偏好条件 | 精确归类 |
|---:|---|---|:---:|:---:|:---:|:---:|---|
| 9 | Schulman et al. (2017) | — | ✓ | — | — | — | PPO 原始技术报告 |
| 10 | Veličković et al. (2018) | 标准 GAT | — | — | — | — | GAT 原始论文 |
| 11 | Wang et al. (2021) | EGAT，显式边特征 | — | — | — | — | Edge-GAT 基础，不含 RL |
| 12 | Kool et al. (2019) | 多头注意力，非严格 GAT | —（REINFORCE） | ✓（路由） | — | — | 神经路由经典构造式策略 |
| 13 | Lin et al. (2022) | 注意力神经组合优化 | —（多目标 RL） | ✓ | — | ✓ | 单模型近似整个 Pareto 集 |
| 14 | Zhang et al. (2020) | GIN/GNN，非 GAT | ✓ | ✓（作业车间） | — | — | 图表示 + PPO 的工业调度经典实例 |
| 15 | Chen et al. (2021) | GAT 通信 | ✓/MAPPO式 | ✓（多机器人轨迹） | — | — | 较早的直接 MA-G-PPO 规划实例 |
| 16 | Lei et al. (2022) | Residual Edge-GAT | ✓，也比较 REINFORCE | ✓（路由） | — | — | 当前编码器—PPO 组合最接近 |
| 17 | Fellek et al. (2024) | Gated Deep GAT | ✓ | ✓（TSP） | — | — | 深层边增强 GAT + PPO |
| 18 | Liu et al. (2024) | 标准 GAT | ✓ | ✓（动态 JSSP） | — | — | 直接 GAT + PPO 工业规划 |
| 19 | Liu et al. (2025) | GAT | ✓ | ✓（动态混合流水车间） | — | — | 明确命名 PPO-GAT 的工业调度 |
| 20 | Guan et al. (2025) | 协同注意力 Transformer，非 GAT | ✓ | ✓（TSP/CVRP） | ✓ | — | 当前“完整解局部改进 + PPO”最接近 |

### 3.1 经典与基础技术文献

#### 3.1.1 Schulman et al. (2017) — PPO 原始论文

**性质：经典技术报告；期刊分区不适用。** 论文提出 PPO 算法族，以裁剪概率比的代理目标限制单次策略更新，并允许对同一批轨迹数据进行多轮小批量训练。它是当前 actor–critic、clipped surrogate objective、GAE 和多轮更新的直接算法来源，但不涉及图编码、组合优化、动作掩码或多目标偏好。

**APA 7：** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1707.06347

- DOI：[10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)（arXiv/DataCite 仓储 DOI，不是期刊 DOI）
- 原始网页：[arXiv](https://arxiv.org/abs/1707.06347)；[OpenAI 官方说明](https://openai.com/index/openai-baselines-ppo/)

#### 3.1.2 Veličković et al. (2018) — GAT 原始论文

**性质：ICLR 2018；期刊分区不适用。** 论文首次系统提出 Graph Attention Network，通过 masked self-attention 为邻域节点学习不同聚合权重，并以多头注意力提升稳定性。它支撑物流网络的非欧式节点表示；原始 GAT 并未系统处理运输距离、成本、风险和兼容性等显式边属性，因此不能单独等同于当前 Edge-GAT。

**APA 7：** Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*. https://openreview.net/forum?id=rJXMpikCZ

- DOI：[10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)（arXiv/DataCite 仓储 DOI；会议本身无 DOI）
- 官方网页：[ICLR](https://iclr.cc/virtual/2018/poster/299)；[OpenReview](https://openreview.net/forum?id=rJXMpikCZ)

#### 3.1.3 Wang et al. (2021) — Edge-featured GAT

**性质：ICANN 2021；期刊分区不适用。** EGAT 把边特征直接纳入注意力系数和图表示，并设置相互依赖的节点注意力块与边注意力块。该结构可直接支撑把距离、成本、事故概率、时间、处理兼容和共载关系写入边编码；但原论文任务是图/节点分类，不含强化学习、PPO 或路径规划。

**APA 7：** Wang, Z., Chen, J., & Chen, H. (2021). EGAT: Edge-featured graph attention network. In I. Farkaš, P. Masulli, S. Otte, & S. Wermter (Eds.), *Artificial neural networks and machine learning—ICANN 2021* (Lecture Notes in Computer Science, Vol. 12891, pp. 253–264). Springer. https://doi.org/10.1007/978-3-030-86362-3_21

- DOI：[10.1007/978-3-030-86362-3_21](https://doi.org/10.1007/978-3-030-86362-3_21)
- 出版社网页：[Springer](https://link.springer.com/chapter/10.1007/978-3-030-86362-3_21)

#### 3.1.4 Kool et al. (2019) — 注意力神经路由经典论文

**性质：ICLR 2019；期刊分区不适用。** 论文用多头注意力编码—解码器和 REINFORCE 学习 TSP、CVRP、SDVRP、OP 与 PCTSP 等路由问题，并在解码阶段屏蔽不可行动作。它是“注意力表示 + 动态可行性掩码 + 神经路由”的关键桥梁，但采用构造式策略和 REINFORCE，而不是 PPO、Edge-GAT 或完整解局部改进。

**APA 7：** Kool, W., van Hoof, H., & Welling, M. (2019). Attention, learn to solve routing problems! *International Conference on Learning Representations*. https://openreview.net/forum?id=ByxBFsRqYm

- DOI：[10.48550/arXiv.1803.08475](https://doi.org/10.48550/arXiv.1803.08475)（arXiv/DataCite 仓储 DOI；会议本身无 DOI）
- 官方网页：[ICLR](https://iclr.cc/virtual/2019/poster/1049)；[OpenReview](https://openreview.net/forum?id=ByxBFsRqYm)

#### 3.1.5 Lin et al. (2022) — 偏好条件多目标神经组合优化

**性质：ICLR 2022；期刊分区不适用。** 论文用一个显式偏好条件模型为任意权衡偏好直接生成近似 Pareto 解，并在多目标 TSP、VRP 和背包问题上验证。它是当前“同一冻结模型覆盖不同成本—风险权重”的最直接基础，但训练算法并非 PPO，也不包含危险废物业务约束或本文式局部改进。

**APA 7：** Lin, X., Yang, Z., & Zhang, Q. (2022). Pareto set learning for neural multi-objective combinatorial optimization. *International Conference on Learning Representations*. https://openreview.net/forum?id=QuObT9BTWo

- DOI：[10.48550/arXiv.2203.15386](https://doi.org/10.48550/arXiv.2203.15386)（arXiv/DataCite 仓储 DOI；会议本身无 DOI）
- 官方网页：[ICLR](https://iclr.cc/virtual/2022/poster/7076)；[OpenReview](https://openreview.net/forum?id=QuObT9BTWo)

### 3.2 PPO + 图注意/图网络求解规划问题

#### 3.2.1 Zhang et al. (2020) — 图表示 + PPO 的作业车间调度

**匹配度：高（规划范式）；NeurIPS 2020，期刊分区不适用。** 论文把作业车间状态表示为析取图，以图神经网络学习尺寸无关的状态嵌入，再用 PPO 学习优先派工规则，并考察向更大实例泛化。该文使用的是 GIN/GNN 而非 GAT，因此应称“图网络 + PPO”，不能写成标准 GAT-PPO。

**在本文中的建议用途：** 支撑“图状态表示—可行动作选择—跨规模规划”的工业运筹范式。

**APA 7：** Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020). Learning to dispatch for job shop scheduling via deep reinforcement learning. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 1621–1632). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html

- DOI：[10.48550/arXiv.2010.12367](https://doi.org/10.48550/arXiv.2010.12367)（arXiv/DataCite 仓储 DOI；NeurIPS 论文无出版 DOI）
- 官方网页：[NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html)

#### 3.2.2 Chen et al. (2021) — 较早的直接 GAT + PPO 规划实例

**匹配度：中高；IROS 2021，期刊分区不适用。** MA-G-PPO 用 GAT 在机器人智能体之间聚合局部观测，并由共享参数 PPO 策略规划持续监测轨迹；论文还研究有无 GAT 通信以及训练、部署智能体数量不一致时的泛化。该文说明 GAT 与 PPO 可以端到端协同用于轨迹规划，但图节点是机器人、GAT 主要承担通信，不是物流业务图或库存路径。

**说明：** 它是经核验的较早直接实例之一，但本清单不足以证明它是所有领域中“第一篇 PPO+GAT”。

**APA 7：** Chen, J., Baskaran, A., Zhang, Z., & Tokekar, P. (2021). Multi-agent reinforcement learning for visibility-based persistent monitoring. In *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 2563–2570). IEEE. https://doi.org/10.1109/IROS51168.2021.9635898

- DOI：[10.1109/IROS51168.2021.9635898](https://doi.org/10.1109/IROS51168.2021.9635898)
- 官方网页：[IEEE Xplore](https://ieeexplore.ieee.org/document/9635898)；[作者公开稿](https://raaslab.org/pubs/chen2021multi.pdf)

#### 3.2.3 Lei et al. (2022) — Residual Edge-GAT + PPO 路由

**匹配度：最高（编码器—训练器组合）；2024 JCR Q1 期刊。** 论文提出 residual edge-graph attention network，把边特征、残差连接和动态 mask 用于 TSP、CVRP 与 MDCVRP，并采用 PPO 或改进基线的 REINFORCE 训练。它与当前 Residual Edge-GAT–PPO 结构最接近，但从零构造路线，不含多周期、库存、混装、危险废物、偏好条件化，也不是对已有完整可行解进行局部改进。

**在本文中的建议用途：** 作为 Edge-GAT 编码器、残差堆叠、路由动作 mask 和 PPO 训练的首要技术依据；表述时应说明论文同时研究 PPO 与 REINFORCE。

**APA 7：** Lei, K., Guo, P., Wang, Y., Wu, X., & Zhao, W. (2022). Solve routing problems with a residual edge-graph attention neural network. *Neurocomputing, 508*, 79–98. https://doi.org/10.1016/j.neucom.2022.08.005

- DOI：[10.1016/j.neucom.2022.08.005](https://doi.org/10.1016/j.neucom.2022.08.005)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S092523122200978X)

#### 3.2.4 Fellek et al. (2024) — Gated Deep GAT + PPO 求解 TSP

**匹配度：高（纯技术扩展）；2024 JCR Q1 期刊。** G-DGANet 通过门控深层图注意、节点/边消息传递与图池化增强深层表示，并明确使用 PPO 训练 TSP 访问序列策略。它可支撑边嵌入、跨层信息流和深层 GAT 稳定性，但仅求解 TSP，仍是构造式策略，不含容量、时期、设施、库存、混装或多目标偏好。

**在本文中的建议用途：** 放在“除经典 GAT 外的高水平纯技术扩展”小节。

**APA 7：** Fellek, G., Farid, A., Fujimura, S., Yoshie, O., & Gebreyesus, G. (2024). G-DGANet: Gated deep graph attention network with reinforcement learning for solving traveling salesman problem. *Neurocomputing, 579*, Article 127392. https://doi.org/10.1016/j.neucom.2024.127392

- DOI：[10.1016/j.neucom.2024.127392](https://doi.org/10.1016/j.neucom.2024.127392)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231224001632)

#### 3.2.5 Liu et al. (2024) — 标准 GAT + PPO 的动态作业车间调度

**匹配度：很高（管理/工业规划）；2024 JCR Q1 期刊。** 论文以析取图描述包含机器故障和随机新作业的动态 JSSP，使用 GAT 生成节点嵌入，并由 PPO 训练 actor–critic 策略。它是“标准 GAT + PPO”求解动态工业规划问题的直接高质量实例，但不涉及车辆路径、库存、混装或逆向物流。

**在本文中的建议用途：** 支撑动态图状态更新、可行动作和在线重规划；可以称为直接 GAT-PPO，不需要降格为一般注意力模型。

**APA 7：** Liu, C.-L., Tseng, C.-J., & Weng, P.-H. (2024). Dynamic job-shop scheduling via graph attention networks and deep reinforcement learning. *IEEE Transactions on Industrial Informatics, 20*(6), 8662–8672. https://doi.org/10.1109/TII.2024.3371489

- DOI：[10.1109/TII.2024.3371489](https://doi.org/10.1109/TII.2024.3371489)
- 出版社网页：[IEEE Xplore](https://ieeexplore.ieee.org/document/10477770)

#### 3.2.6 Liu et al. (2025) — 明确命名 PPO-GAT 的动态混合流水车间

**匹配度：很高（管理/工业规划）；2024 JCR Q1 期刊。** 论文针对动态混合流水车间调度，以 GAT 编码动态图结构，以多信号差分奖励缓解奖励稀疏，并明确采用 PPO-GAT 进行实时决策。该文与当前方法在动态图、增量奖励和跨规模决策方面接近，但目标是总加权延误，不含车辆、库存、混装、危废或成本—风险偏好条件。

**在本文中的建议用途：** 与 Liu et al. (2024) 一起证明 PPO+GAT 已进入工业调度/规划，而本文将该范式迁移并扩展到危废多周期库存路径。

**APA 7：** Liu, Y., Fan, J., & Shen, W. (2025). A deep reinforcement learning approach with graph attention network and multi-signal differential reward for dynamic hybrid flow shop scheduling problem. *Journal of Manufacturing Systems, 80*, 643–661. https://doi.org/10.1016/j.jmsy.2025.03.028

- DOI：[10.1016/j.jmsy.2025.03.028](https://doi.org/10.1016/j.jmsy.2025.03.028)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0278612525000883)

#### 3.2.7 Guan et al. (2025) — 注意力 + PPO 的完整路线局部改进

**匹配度：最高（改进型策略）；2024 JCR Q1 期刊。** 论文提出 synergetic attention-driven transformer，从一个完整路线出发，通过节点对操作迭代改进 TSP/CVRP 解，并以 PPO 训练改进策略。它与当前“完整可行计划 + 局部修改动作 + PPO”范式最接近；但其编码器是多头协同注意 Transformer 而不是 GAT/Edge-GAT，也没有多周期库存、混装和偏好条件化。

**在本文中的建议用途：** 用于论证为何采用 improvement-type DRL 而非只从零构造路线，并作为 2-opt/节点对操作、多解码器和搜索多样性的比较对象。

**APA 7：** Guan, Q., Cao, H., Jia, L., Yan, D., & Chen, B. (2025). Synergetic attention-driven transformer: A deep reinforcement learning approach for vehicle routing problems. *Expert Systems with Applications, 274*, Article 126961. https://doi.org/10.1016/j.eswa.2025.126961

- DOI：[10.1016/j.eswa.2025.126961](https://doi.org/10.1016/j.eswa.2025.126961)
- 出版社网页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417425005834)

## 4. 可直接用于文章的综述组织方式

### 4.1 数学模型综述链条

可按“系统边界—多周期库存—混装兼容—研究缺口”组织，而不是逐篇罗列：

> Hazardous-waste logistics studies have long integrated treatment compatibility, facility decisions, routing cost, and transportation risk (Alumur & Kara, 2007), while coordinated reverse-logistics models have further incorporated collection, storage, treatment, and disposal risks across multiple waste sources (Sheu, 2007). Multi-period collection studies subsequently linked time-varying waste generation, delayed pickup, inventory carryover, and routing decisions (Rabbani et al., 2019; Ma & Li, 2021), echoing the broader inventory-routing structure observed in recyclable-material collection (Elbek & Wøhlk, 2016). Other studies have separately modeled multi-product loading, incompatible waste types, and the superimposed risks of multiple hazardous-material classes (Paredes-Belmar et al., 2017; Rabbani et al., 2018; Wu et al., 2022). Within the reviewed literature, however, these elements have not been jointly represented in a multi-period vehicle-routing model with inventories at both generators and treatment facilities, pairwise co-loading risk, and mandatory terminal inventory clearance.

### 4.2 方法综述链条

> PPO controls policy updates through a clipped surrogate objective (Schulman et al., 2017), whereas GAT learns adaptive neighborhood aggregation through masked self-attention (Veličković et al., 2018). Edge-featured extensions further allow explicit arc attributes to participate in attention computation (Wang et al., 2021). In neural combinatorial optimization, attention-based policies have demonstrated the value of dynamic feasibility masks for routing (Kool et al., 2019), and preference-conditioned models have shown that a single policy can approximate solutions under different multiobjective trade-offs (Lin et al., 2022). Recent studies have combined graph representations with PPO in job-shop scheduling and multi-robot planning (Zhang et al., 2020; Chen et al., 2021), while direct GAT–PPO variants have been developed for routing and dynamic industrial scheduling (Lei et al., 2022; Fellek et al., 2024; Liu et al., 2024, 2025). Improvement-type PPO policies can also iteratively refine a complete routing solution (Guan et al., 2025). These studies motivate a preference-conditioned Residual Edge-GAT–PPO improver, but none of the reviewed methods directly enforces the multi-period inventory, treatment-technology, co-loading compatibility, and terminal-clearance constraints of hazardous-waste reverse logistics.

## 5. 推荐的文内引用优先级

### A 级：建议必引

- 数学模型：Alumur and Kara (2007)、Paredes-Belmar et al. (2017)、Rabbani et al. (2018, 2019)、Ma and Li (2021)。
- 方法模型：Schulman et al. (2017)、Veličković et al. (2018)、Wang et al. (2021)、Lin et al. (2022)、Lei et al. (2022)、Guan et al. (2025)。

### B 级：按段落选引

- 逆向物流与库存风险：Sheu (2007)、Elbek and Wøhlk (2016)、Wu et al. (2022)。
- 图网络 + PPO 的规划应用：Zhang et al. (2020)、Chen et al. (2021)、Liu et al. (2024, 2025)。
- 深层 GAT 技术扩展：Fellek et al. (2024)。

## 6. 引用与创新表述边界

1. **不要把“注意力模型”一律写成 GAT。** Kool et al. (2019) 和 Guan et al. (2025) 是注意力/Transformer 路由模型；Zhang et al. (2020) 是 GIN/GNN；只有相应论文明确使用 GAT 时才写 GAT-PPO。
2. **不要把“使用 PPO”写成“首次提出 PPO”。** PPO 的算法源头是 Schulman et al. (2017)。Lei et al. (2022) 同时研究 PPO 与 REINFORCE，不能描述为只采用 PPO。
3. **不要把仓储 DOI 写成正式会议 DOI。** PPO、GAT、Kool、Lin 和 Zhang 的 `10.48550/arXiv...` 是 arXiv/DataCite DOI；相应会议论文没有独立出版 DOI。
4. **不要把会议误标为一区。** ICLR、NeurIPS、IROS、ICANN 是会议，JCR/中科院期刊分区不适用。
5. **研究缺口要限定范围。** 推荐写“within the reviewed literature”或“在本文检索范围内未发现”，不直接写“the first model/method”。
6. **分区具有年份和数据库依赖。** 本文档中的 Q1 仅作选刊和阅读优先级提示，不应替代投稿当年的 JCR、中科院或学校认定结果。

## 7. 分区核验入口

- EJOR：[EURO 官方期刊页面](https://www.euro-online.org/web/pages/106/publications)
- *Computers & Operations Research*：[2024 JCR/SJR 信息](https://www.iit.comillas.edu/publicacion/info_revista/en/282/Computers_%26_Operations_Research)
- *Journal of Cleaner Production*：[2024 JCR/SJR 信息](https://www.iit.comillas.edu/publicacion/info_revista/en/434/Journal_of_Cleaner_Production)
- *Transportation Research Part E*：[期刊指标入口](https://library.wur.nl/WebQuery/wda/1619981)
- *Journal of Industrial & Management Optimization*：[期刊指标入口](https://topj.lib.whu.edu.cn/show.asp?cat=esi&id=7021)
- *Neurocomputing*：[2024 JCR/SJR 信息](https://www.iit.comillas.edu/publicacion/info_revista/en/58/Neurocomputing)
- *IEEE Transactions on Industrial Informatics*：[2024 JCR/SJR 信息](https://www.iit.comillas.edu/publicacion/info_revista/en/801/IEEE_Transactions_on_Industrial_Informatics)
- *Journal of Manufacturing Systems*：[2024 JCR 信息](https://topj.lib.whu.edu.cn/search.asp?cat=sci&ordertype=desc&page=1&q=&xk=310)
- *Expert Systems with Applications*：[2024 JCR/SJR 信息](https://www.iit.comillas.edu/publicacion/info_revista/en/60/Expert_Systems_with_Applications)

---

**核验状态：** 20/20 篇均已核对作者、年份、题名、载体、卷期/页码或文章号、DOI 及原始网页。内容摘要只写入可由论文摘要、出版社页面或正式论文确认的信息；没有以二手综述替代原始文献元数据。

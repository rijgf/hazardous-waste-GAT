# 危险废物多周期库存—路径模型与 PPO–Transformer 方法：英文核心参考文献

> 更新与核验日期：2026-08-25  
> 文献数量：20 篇（与原清单完全一致）  
> 引用格式：APA 7  
> 选刊口径：优先管理科学、运筹、物流、运输及可持续运营领域期刊；经典算法与关键会议论文例外  
> 分区口径：本文所称 Q1 为 2024 JCR 期刊分区口径，不等同于中科院一区、FT50 或 UTD24

## 1. 更新结论

本清单按当前实际方法 **PPO + Transformer** 重建，不再把 GAT、Edge-GAT 或一般 GNN 作为本文方法的理论来源。文献总量严格保持为 20 篇：前 8 篇对应危险废物、多周期库存—路径、兼容混装和成本—风险管理问题，后 12 篇对应 PPO、Transformer、强化学习组合优化、物流供应链应用和改进型神经搜索。

相较原清单，本版保留了 12 篇高匹配文献，并以 8 篇新文献替换原有的 GAT/GNN 方法文献。被替换的研究包括 GAT、EGAT、GIN/GNN–PPO、Residual Edge-GAT–PPO 及工业调度中的 GAT–PPO；新增文献包括 Transformer 原始论文、管理/运筹领域的机器学习与强化学习组合优化综述、物流供应链强化学习综述，以及直接采用 Transformer + PPO 的组合优化与物流研究。

更新后，20 篇中有 **14 篇期刊论文发表于按 2024 JCR 口径为 Q1 的期刊**；1 篇为问题高度匹配但非 Q1 的管理优化期刊论文；其余 5 篇为 PPO、Transformer、注意力路由、PPO–Transformer 改进策略及偏好多目标学习的经典或关键会议/预印本文献。会议论文不适用 JCR 或中科院期刊分区。

### 1.1 核心判断

在本次核验范围内，没有发现单篇论文同时覆盖“危险废物逆向物流 + 多周期双端库存 + 兼容混装及成对附加风险 + 成本—风险偏好 + PPO–Transformer 完整解局部改进”。因此，较稳妥的创新定位是：将已经分别得到验证的管理问题建模、Transformer 状态表示、PPO 策略更新、偏好条件化和可行解局部改进机制，集成为面向危险废物多周期库存—路径约束的求解框架，而不是直接声称首次提出 PPO 或 Transformer。

### 1.2 建议优先阅读

1. **总体管理问题最接近：** Rabbani et al. (2019)。
2. **混装风险与兼容性最接近：** Paredes-Belmar et al. (2017)；Rabbani et al. (2018)。
3. **多周期库存—路径最接近：** Elbek and Wøhlk (2016)；Ma and Li (2021)。
4. **PPO 与 Transformer 基础：** Schulman et al. (2017)；Vaswani et al. (2017)。
5. **Transformer + PPO 改进型路径策略最接近：** Ma et al. (2021)；Guan et al. (2025)。
6. **物流管理中的 Transformer + PPO 直接应用：** Tian et al. (2025)。
7. **物流供应链强化学习综述：** Yan et al. (2022)。
8. **成本—风险偏好条件模型：** Lin et al. (2022)。

## 2. 数学模型文献对照

符号说明：✓ = 明确覆盖；△ = 部分覆盖或语义相近；— = 未覆盖或不能从原始来源确认。

| # | 文献 | 危废/逆向 | 多周期 | 车辆路径 | 库存联动 | 混装/兼容 | 成本—风险 | 主要作用 |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | Alumur & Kara (2007) | ✓ | — | △ | — | △ | ✓ | 危险废物选址—路径与处理技术兼容的经典基础 |
| 2 | Sheu (2007) | ✓ | △ | △ | △ | — | ✓ | 多源危废逆向物流、储存/处理/运输风险 |
| 3 | Elbek & Wøhlk (2016) | △ | ✓ | ✓ | ✓ | △ | — | 多周期累积库存—收集路径与滚动计划 |
| 4 | Paredes-Belmar et al. (2017) | ✓ | — | ✓ | — | ✓ | ✓ | 多产品同车装载及载荷组合风险 |
| 5 | Rabbani et al. (2018) | ✓ | — | ✓ | — | ✓ | ✓ | 不相容废物、异质车辆与多目标选址—路径 |
| 6 | Rabbani et al. (2019) | ✓ | ✓ | ✓ | ✓ | — | ✓ | 多周期危废选址—路径—库存总体对照 |
| 7 | Ma & Li (2021) | ✓ | ✓ | ✓ | △ | — | ✓ | 延迟收集、库存结转与跨期风险稳定 |
| 8 | Wu, Ma, et al. (2022) | △ | △ | △ | ✓ | ✓ | ✓ | 多类别危险品库存与叠加风险 |

## 3. 方法文献对照

| # | 文献 | 表示/策略网络 | 训练方法 | 管理或组合优化情境 | 完整解改进 | 偏好条件 | 在本文中的作用 |
|---:|---|---|---|---|:---:|:---:|---|
| 9 | Schulman et al. (2017) | Actor–critic 通用框架 | PPO | 通用强化学习 | — | — | clipped surrogate objective 与多轮小批量更新的算法源头 |
| 10 | Vaswani et al. (2017) | Transformer、多头自注意力 | 监督学习 | 经典基础方法 | — | — | Transformer 状态编码的原始来源；不是 GAT |
| 11 | Bengio et al. (2021) | 多类 ML 表示与决策模块 | 多类 ML | 组合优化/运筹 | △ | — | 机器学习与组合优化融合的方法论框架 |
| 12 | Mazyavkina et al. (2021) | 多类状态编码器 | 多类 RL | 组合优化/运筹 | △ | — | RL 求解组合优化的分类、比较与研究边界 |
| 13 | Yan et al. (2022) | 多类 RL 架构 | 多类 RL | 物流与供应链管理 | △ | — | 管理类应用综述及动态业务决策依据 |
| 14 | Kool et al. (2019) | 多头注意力编码—解码器 | REINFORCE | TSP/CVRP 等路由 | — | — | 注意力路由和动态可行性掩码的经典依据 |
| 15 | Wu, Song, et al. (2022) | 自注意力策略网络 | 深度 RL | TSP/CVRP | ✓ | — | 从完整解出发学习改进启发式 |
| 16 | Ma et al. (2021) | Dual-Aspect Collaborative Transformer | PPO | TSP/CVRP | ✓ | — | 直接支撑 Transformer + PPO 的迭代改进路线 |
| 17 | Lin et al. (2022) | 偏好条件注意力模型 | 多目标 RL | 多目标 TSP/VRP/背包 | — | ✓ | 单一模型覆盖不同成本—风险偏好的直接依据 |
| 18 | Que et al. (2023) | Transformer 策略网络 | PPO | 三维装箱/运营优化 | — | — | 直接的 Transformer + PPO 组合优化应用 |
| 19 | Tian et al. (2025) | Transformer + Bayesian network | PPO | 物流服务组合 | — | — | 管理/物流场景中的 Transformer + PPO 直接证据 |
| 20 | Guan et al. (2025) | Synergetic Attention-driven Transformer | PPO | TSP/CVRP | ✓ | — | 与“完整可行解 + 局部修改 + PPO”最接近 |

## 4. 完整参考文献（20 篇，APA 7）

### 4.1 危险废物、多周期库存—路径与风险管理

1. Alumur, S., & Kara, B. Y. (2007). A new model for the hazardous waste location-routing problem. *Computers & Operations Research, 34*(5), 1406–1423. https://doi.org/10.1016/j.cor.2005.06.012

2. Sheu, J.-B. (2007). A coordinated reverse logistics system for regional management of multi-source hazardous wastes. *Computers & Operations Research, 34*(5), 1442–1462. https://doi.org/10.1016/j.cor.2005.06.009

3. Elbek, M., & Wøhlk, S. (2016). A variable neighborhood search for the multi-period collection of recyclable materials. *European Journal of Operational Research, 249*(2), 540–550. https://doi.org/10.1016/j.ejor.2015.08.035

4. Paredes-Belmar, G., Bronfman, A., Marianov, V., & Latorre-Núñez, G. (2017). Hazardous materials collection with multiple-product loading. *Journal of Cleaner Production, 141*, 909–919. https://doi.org/10.1016/j.jclepro.2016.09.163

5. Rabbani, M., Heidari, R., Farrokhi-Asl, H., & Rahimi, N. (2018). Using metaheuristic algorithms to solve a multi-objective industrial hazardous waste location-routing problem considering incompatible waste types. *Journal of Cleaner Production, 170*, 227–241. https://doi.org/10.1016/j.jclepro.2017.09.029

6. Rabbani, M., Heidari, R., & Yazdanparast, R. (2019). A stochastic multi-period industrial hazardous waste location-routing problem: Integrating NSGA-II and Monte Carlo simulation. *European Journal of Operational Research, 272*(3), 945–961. https://doi.org/10.1016/j.ejor.2018.07.024

7. Ma, H., & Li, X. (2021). Multi-period hazardous waste collection planning with consideration of risk stability. *Journal of Industrial & Management Optimization, 17*(1), 393–408. https://doi.org/10.3934/jimo.2019117

8. Wu, W., Ma, J., Liu, R., & Jin, W. (2022). Multi-class hazmat distribution network design with inventory and superimposed risks. *Transportation Research Part E: Logistics and Transportation Review, 161*, Article 102693. https://doi.org/10.1016/j.tre.2022.102693

### 4.2 PPO、Transformer、组合优化与物流管理

9. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1707.06347

10. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 5998–6008). Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

11. Bengio, Y., Lodi, A., & Prouvost, A. (2021). Machine learning for combinatorial optimization: A methodological tour d’horizon. *European Journal of Operational Research, 290*(2), 405–421. https://doi.org/10.1016/j.ejor.2020.07.063

12. Mazyavkina, N., Sviridov, S., Ivanov, S., & Burnaev, E. (2021). Reinforcement learning for combinatorial optimization: A survey. *Computers & Operations Research, 134*, Article 105400. https://doi.org/10.1016/j.cor.2021.105400

13. Yan, Y., Chow, A. H. F., Ho, C. P., Kuo, Y.-H., Wu, Q., & Ying, C. (2022). Reinforcement learning for logistics and supply chain management: Methodologies, state of the art, and future opportunities. *Transportation Research Part E: Logistics and Transportation Review, 162*, Article 102712. https://doi.org/10.1016/j.tre.2022.102712

14. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, learn to solve routing problems! *International Conference on Learning Representations*. https://openreview.net/forum?id=ByxBFsRqYm

15. Wu, Y., Song, W., Cao, Z., Zhang, J., & Lim, A. (2022). Learning improvement heuristics for solving routing problems. *IEEE Transactions on Neural Networks and Learning Systems, 33*(9), 5057–5069. https://doi.org/10.1109/TNNLS.2021.3068828

16. Ma, Y., Li, J., Cao, Z., Song, W., Zhang, L., Chen, Z., & Tang, J. (2021). Learning to iteratively solve routing problems with dual-aspect collaborative transformer. In *Advances in Neural Information Processing Systems* (Vol. 34, pp. 11096–11107). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2021/hash/5c53292c032b6cb8510041c54274e65f-Abstract.html

17. Lin, X., Yang, Z., & Zhang, Q. (2022). Pareto set learning for neural multi-objective combinatorial optimization. *International Conference on Learning Representations*. https://openreview.net/forum?id=QuObT9BTWo

18. Que, Q., Yang, F., & Zhang, D. (2023). Solving 3D packing problem using Transformer network and reinforcement learning. *Expert Systems with Applications, 214*, Article 119153. https://doi.org/10.1016/j.eswa.2022.119153

19. Tian, R., Chang, L., Sun, Z., Zhao, G., & Lu, X. (2025). PTB: A deep reinforcement learning method for flexible logistics service combination problem with spatial-temporal constraint. *Transportation Research Part E: Logistics and Transportation Review, 195*, Article 103978. https://doi.org/10.1016/j.tre.2025.103978

20. Guan, Q., Cao, H., Jia, L., Yan, D., & Chen, B. (2025). Synergetic attention-driven transformer: A deep reinforcement learning approach for vehicle routing problems. *Expert Systems with Applications, 274*, Article 126961. https://doi.org/10.1016/j.eswa.2025.126961

## 5. 可直接用于论文的综述组织方式

### 5.1 管理问题综述

危险废物物流研究已将处理技术兼容、设施决策、运输成本和事故风险纳入联合优化（Alumur & Kara, 2007），并进一步从区域协同逆向物流角度统筹收集、储存、处理、运输和最终处置风险（Sheu, 2007）。多周期研究则把随时间变化的废物产生量、延期收集、库存结转和车辆路径联系起来（Elbek & Wøhlk, 2016; Rabbani et al., 2019; Ma & Li, 2021）。此外，多产品同车装载、不相容废物和多类别危险品叠加风险分别得到研究（Paredes-Belmar et al., 2017; Rabbani et al., 2018; Wu, Ma, et al., 2022）。然而，在本次检索范围内，上述要素尚未被同时纳入一个包含产废端与处理端库存、成对共载附加风险以及规划期末库存清零要求的多周期车辆路径模型。

### 5.2 方法综述

PPO 通过裁剪代理目标控制策略更新幅度（Schulman et al., 2017），Transformer 则以多头自注意力学习序列元素之间的全局依赖（Vaswani et al., 2017）。机器学习和强化学习已被系统用于替代或增强组合优化中的手工启发式（Bengio et al., 2021; Mazyavkina et al., 2021），并逐步进入物流和供应链管理决策（Yan et al., 2022）。在路由问题中，注意力策略和动态掩码能够处理节点关系与可行动作（Kool et al., 2019），自注意力网络也可从完整解出发学习改进启发式（Wu, Song, et al., 2022）。更直接地，DACT 和 SAT 均采用 Transformer 表示完整路径解，并以 PPO 训练迭代改进策略（Ma et al., 2021; Guan et al., 2025）；Transformer + PPO 还被用于三维装箱和具有时空约束的物流服务组合（Que et al., 2023; Tian et al., 2025）。偏好条件模型进一步表明，单一策略可以覆盖不同多目标权衡（Lin et al., 2022）。这些研究共同支持本文采用偏好条件 PPO–Transformer 改进器，但均未直接处理危险废物多周期库存、处理技术、兼容混装和期末清零约束。

## 6. 推荐的文内引用优先级

### A 级：建议必引

- 管理问题：Alumur and Kara (2007)、Paredes-Belmar et al. (2017)、Rabbani et al. (2018, 2019)、Ma and Li (2021)、Wu, Ma, et al. (2022)。
- 方法基础：Schulman et al. (2017)、Vaswani et al. (2017)、Yan et al. (2022)。
- 方法匹配：Ma et al. (2021)、Lin et al. (2022)、Tian et al. (2025)、Guan et al. (2025)。

### B 级：按段落选引

- 多周期库存与逆向物流：Sheu (2007)、Elbek and Wøhlk (2016)。
- 组合优化方法论：Bengio et al. (2021)、Mazyavkina et al. (2021)。
- 注意力路由与改进启发式：Kool et al. (2019)、Wu, Song, et al. (2022)、Que et al. (2023)。

## 7. 引用与表述边界

1. **本文方法应统一写为 PPO–Transformer，不写 GAT–PPO。** Transformer 对完整状态/解序列做全局自注意力编码；GAT 是基于图邻域消息聚合的另一类结构。
2. **不要把注意力论文一律写成 PPO。** Kool et al. (2019) 使用 REINFORCE；Lin et al. (2022) 使用多目标强化学习；Wu, Song, et al. (2022) 支撑自注意力改进启发式。直接采用 Transformer + PPO 的核心依据是 Ma et al. (2021)、Que et al. (2023)、Tian et al. (2025) 和 Guan et al. (2025)。
3. **不要把会议误标为一区。** NeurIPS、ICLR 不适用 JCR 或中科院期刊分区。
4. **保留 Ma and Li (2021) 作为主题例外。** 虽然该期刊不是本清单口径下的 Q1，但它对多周期危废延期收集和跨期风险的匹配度高，替换为泛化 Q1 文献会削弱问题依据。
5. **研究缺口需限定范围。** 建议使用“在本文检索范围内未发现”或 “within the reviewed literature”，不直接写 “the first model/method”。
6. **正式投稿格式需另行转换。** 当前清单沿用 APA 7 便于作者年份式综述；若投稿《供应链管理》，应按该刊要求转换为顺序编码和 GB/T 7714—2015，并检查正文—文末一一对应。

---

**核验状态：** 20/20 篇均已核对作者、年份、题名、载体、卷期与页码/文章号、DOI 或会议官方页面。新增条目优先采用出版社、会议官方页面、权威交通文献数据库或作者机构正式存档；未使用无法追溯来源的生成式条目。

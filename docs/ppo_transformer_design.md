# 多周期多目标危险废物运输的统一 GAT-PPO 求解器设计

> 状态：文献调研后的更新设计稿，尚未实现 GAT 训练代码。
> 更新日期：2026-08-23。
> 文件名因历史链接保留为 `ppo_transformer_design.md`；当前设计不再使用 Transformer。
> 配套调研：[多周期多目标逆向物流与 GAT 神经求解器文献调研](literature_review_gat_reverse_logistics.md)。

## 0. 已确认的设计决定

本文以用户的两个核心要求为约束：

1. 使用 GAT；MoE 可以使用，但不是必选项；
2. 使用同一个网络结构、同一个基础检查点处理不同规模实例和不同成本—风险偏好，重点验证泛化性。

据此，首版采用以下决定：

| 项目 | 首版决定 |
|---|---|
| 图编码器 | 边特征增强、关系感知的 GATv2 |
| 时间表示 | 双期详细窗口 + 剩余时域摘要节点 |
| 偏好处理 | 共享 GAT + 连续偏好条件化 FiLM/门控解码器 |
| MoE | 默认关闭；仅作为后续受控消融 |
| 动作 | 动态节点/边指针，不使用固定对象编号槽位 |
| 价值函数 | 偏好条件化的成本/风险双分量 Critic，加可行性辅助头 |
| 可行性 | 动作硬掩码 + 局部修复 + 剩余时域可行性 oracle |
| 终期库存 | 硬清零；末期小型 MILP/确定性补全；失败回退 incumbent |
| 泛化主结果 | 一个冻结检查点零样本求解所有测试规模与偏好 |
| 测试时适应 | 可作为补充实验，但不能计入零样本泛化主结果 |

首版正式名称建议为：

> **面向多规模与连续偏好的边增强时序 GAT-PPO 局部改进器**

英文可写为：

> **A Size-Generalizable Preference-Conditioned Temporal GAT-PPO Improver**

## 1. 问题定义

### 1.1 集合

| 符号 | 含义 |
|---|---|
| $I$ | 危险废物产生者集合 |
| $S$ | 废物类型集合 |
| $J$ | 处理设施集合 |
| $K$ | 车辆集合 |
| $T=\{1,\ldots,H\}$ | 离散时期集合 |
| $N=I\times S$ | 产生者—废物取货实体 |
| $A$ | 允许通行的有向弧集合 |

网络必须允许 $|I|,|S|,|J|,|K|,H$ 在实例之间变化，不设训练规模专属的实体槽位。

### 1.2 目标

对成本和风险分别使用稳定的实例内尺度 $b_c,b_r$：

$$
\widehat f_c=\frac{f_c}{b_c},
\qquad
\widehat f_r=\frac{f_r}{b_r}.
$$

用户偏好为：

$$
\boldsymbol\alpha=(\alpha_c,\alpha_r),
\qquad
\alpha_c+\alpha_r=1,
\qquad
\alpha_c,\alpha_r\ge 0.
$$

在可行解集合中比较：

$$
J_{\boldsymbol\alpha}
=\alpha_c\widehat f_c+\alpha_r\widehat f_r.
$$

成本包括车辆启用、运输距离和处理成本。风险包括运输事故、混装、产生者库存和设施库存风险。所有训练轨迹保留成本、风险两个原始分量，不能只保存提前加权后的单个标量。

### 1.3 硬约束

- 路线从可行设施出发并返回设施；
- 同一车辆同一时期最多一条路线；
- 车辆容量不超限；
- 废物共载满足兼容矩阵；
- 路线终点设施具备全部所需处理技术；
- 设施存储量和处理量不超限；
- 产生者、设施库存逐期守恒；
- 已在当前 sweep 提交的时期不可修改；
- `require_terminal_clear=true` 时，最终产生者和设施库存严格为零。

可行性约束先于成本—风险偏好。任何偏好都不能用较低成本或风险补偿终期库存违反。

## 2. 方法定位与总体流程

GAT-PPO 不从空白开始直接构造全部多周期路线，而是在一个完整可行解上选择局部改进动作。这样可以复用现有 MILP、GA、启发式和统一评价环境，并始终保留可行 incumbent。

```text
实例参数 + 成本风险偏好
          │
          ▼
启发式 / GA / MILP 暖启动生成完整可行解
          │
          ▼
保存 best_feasible_incumbent
          │
          ▼
窗口 {t,t+1} 详细图 + 远期摘要图
          │
          ▼
共享边增强关系 GATv2
          │
          ▼
偏好条件化共享 Actor + 向量 Critic
          │
          ▼
动态指针动作 + 硬可行性掩码
          │
          ▼
局部修改与确定性修复
          │
          ▼
剩余时域可承接性预检 / 可行补全
          │
     ┌────┴────┐
     │可提交    │不可提交
     ▼          ▼
提交当前期     拒绝动作，状态不变
并前移窗口
          │
          ▼
最终窗口：严格清零 + 完整约束检查
          │
          ▼
只接受更优的完整可行方案，否则回退 incumbent
```

## 3. 时间建模：双期详细窗口与远期摘要

### 3.1 详细窗口

第 $\tau$ 个窗口为：

$$
\mathcal W_\tau=\{t_\tau,t_{\tau+1}\}.
$$

当 $H=1$ 时退化为单期窗口。每次完成窗口搜索后，只提交并冻结 $t_\tau$，把 $t_{\tau+1}$ 作为下一窗口的热启动。最后一个窗口同时处理并提交 $H-1,H$。

双期窗口的作用是使单次图规模主要依赖实体数而不是总时期数；它不是对长期信息的完全截断。

### 3.2 左边界

窗口输入必须包含上一期结转：

$$
I^{P,\mathrm{in}}_{i,s,\tau}=I^P_{i,s,t_\tau-1},
\qquad
I^{F,\mathrm{in}}_{j,s,\tau}=I^F_{j,s,t_\tau-1}.
$$

第一窗口使用初始库存。

### 3.3 远期摘要不是一个固定长向量

为了允许废物类型数和设施数变化，未来信息不拼接成依赖 $|S|$ 或 $|J|$ 的固定向量，而构造成可变数量的摘要节点：

- 每种废物一个 `future_waste` 节点；
- 每个可处理设施—废物组合一个 `future_facility_capacity` 节点；
- 一个全局 `horizon_context` 节点。

废物类型 $s$ 的远期义务节点包含：

```text
[future_generation,
 producer_inventory_at_boundary,
 facility_inventory_at_boundary,
 remaining_pickup_capacity_ratio,
 remaining_processing_capacity_ratio,
 minimum_capacity_slack,
 latest_service_urgency,
 remaining_period_ratio]
```

设施—废物未来能力节点包含：

```text
[technology_available,
 remaining_total_processing_capacity,
 remaining_total_storage_headroom,
 earliest_capacity_bottleneck,
 normalized_processing_cost,
 normalized_inventory_risk]
```

这些节点只保存未来义务与能力，不包含窗口外具体路线，因而不会把完整时域重新变成长序列。

### 3.4 多轮正向 sweep

一次从 $\{1,2\}$ 到 $\{H-1,H\}$ 的遍历只形成一个候选。外层可执行多轮正向 sweep：

$$
\{1,2\}\rightarrow\{2,3\}\rightarrow\cdots\rightarrow\{H-1,H\}.
$$

每轮以历史最佳完整可行解为热启动。轮内冻结已提交时期，轮末重新解锁。只有候选完整可行且目标改善时才更新 incumbent。

## 4. 可变规模异构时序图

### 4.1 节点类型

| 类型 | 数量 | 作用 |
|---|---:|---|
| `horizon_context` | 1 | 规模、窗口位置、进度、整体松弛 |
| `period` | 1 或 2 | 当前/前瞻期统计 |
| `pickup_event` | $|\mathcal W||I||S|$ | 某产生者—废物在某期的服务和库存 |
| `vehicle_period` | $|\mathcal W||K|$ | 某车辆在某期的路线状态 |
| `facility_waste_period` | $|\mathcal W||J||S|$ | 某设施处理某废物的当期状态 |
| `route_edge` | 当前路线边数 | 允许动作直接指向插入位置 |
| `future_waste` | $|S|$ | 边界后该废物的总义务与能力 |
| `future_facility_capacity` | 可行 $(j,s)$ 数 | 远期设施处理能力分解 |

物理产生者与某期服务事件必须区分：

$$
u_{i,s,t}\ne u_{i,s,t+1}.
$$

同一物理节点可以跨期被访问多次；提前收集不能运输尚未产生的废物。

### 4.2 不使用绝对实体 ID

禁止使用仅在训练集有意义的 `producer_id_embedding`、`vehicle_id_embedding` 或固定对象位置。节点身份通过以下信息表达：

- 节点类型；
- 当前期/前瞻期角色；
- 坐标或相对几何；
- 废物属性与处理技术；
- 当前路线和库存关系；
- 在当前图中的局部连接。

如需区分同一图内的车辆，使用其容量、使用状态和结构位置，而不是跨实例共享的车辆编号语义。

### 4.3 无量纲节点特征

主要比例特征包括：

$$
\frac{I^P_{i,s,t}}{C_i^P},
\quad
\frac{I^F_{j,s,t}}{C_j^F},
\quad
\frac{\mathrm{load}_{k,t}}{Q_k},
\quad
\frac{P_{j,s,t}^{\mathrm{used}}}{P_{j,s,t}^{\max}},
$$

$$
\frac{d_{ij}}{d_{\mathrm{scale}}},
\quad
\frac{\mathrm{remaining\ periods}}{H},
\quad
\frac{\mathrm{cost}}{b_c},
\quad
\frac{\mathrm{risk}}{b_r}.
$$

实例规模通过 $\log(1+|I|)$、$\log(1+|S|)$、$\log(1+|J|)$、$\log(1+|K|)$ 和 $\log(1+H)$ 提供给上下文节点。显式规模信息允许策略校准不同图大小下的动作熵和资源紧度，但不改变网络层形状。

### 4.4 关系与边特征

| 关系 | 主要边特征 |
|---|---|
| `belongs_to_period` | 当前/前瞻标志 |
| `assigned_to_vehicle` | 取货量、路线位置、车辆剩余容量 |
| `route_next` | 距离、事故概率、累计载荷、路径风险 |
| `candidate_insert` | 增量距离、增量风险、兼容性 |
| `served_by_facility` | 技术可行性、处理成本、能力松弛 |
| `compatible_with` | 共载兼容性与共载风险 |
| `same_entity_next_period` | 库存变化、服务变化 |
| `future_obligation` | 边界库存、未来产生量、最迟服务紧迫度 |
| `future_capacity` | 剩余处理/运输能力、技术可用性 |
| `context_link` | 节点类型和规模上下文 |

### 4.5 稀疏候选边

候选图取以下集合的并集：当前实际路径边、距离 $k_d$ 近邻、风险 $k_r$ 近邻、全部技术可行设施边、暖启动边和少量探索边。候选图在动作后动态更新。

必须记录 `candidate_recall`：基准优质方案所用边中有多少出现在候选图。若该值低，优先修正图生成规则，而不是盲目扩大网络。

## 5. 边增强关系 GATv2

### 5.1 分类型输入投影

$$
h_v^{(0)}
=\mathrm{MLP}_{\phi(v)}(x_v)
+e_{\mathrm{type}(v)}
+e_{\mathrm{period\ role}(v)}.
$$

所有同类型节点共享投影参数。建议初始隐藏维度 $d=96$，以 3 层、4 头作为起点。

### 5.2 关系和边共同参与注意力

对关系 $r$ 的边 $i\rightarrow j$：

$$
e_{ij}^{r,h}
=a_{r,h}^{\top}
\mathrm{LeakyReLU}
\left(
W_{r,h}^{q}h_j
+W_{r,h}^{k}h_i
+U_{r,h}z_{ij}
\right).
$$

$$
\alpha_{ij}^{r,h}
=\mathrm{softmax}_{i\in\mathcal N_r(j)}e_{ij}^{r,h}.
$$

不同关系先独立归一化，再经关系门控汇总，避免数量很多的候选边淹没稀少但关键的跨期或设施能力边。

节点更新使用残差、LayerNorm 和 FFN：

$$
h_j^{(l+1)}
=\mathrm{LayerNorm}
\left[
h_j^{(l)}
+\mathrm{Dropout}(\mathrm{FFN}(m_j^{(l)}))
\right].
$$

### 5.3 图级池化

先对每种节点类型分别计算 mean、max 和注意力池化，再拼接上下文节点与规模特征：

$$
g=\mathrm{MLP}_g
\left[
h_{\mathrm{context}}
\Vert
\mathop{\Vert}_{q\in\mathcal Q}
(\mathrm{mean}_q\Vert\mathrm{max}_q\Vert\mathrm{attn}_q)
\Vert c_{\mathrm{size}}
\right].
$$

类型内池化避免某类节点数量增加时完全改变其他类型的权重。mean/max 提供对规模变化更稳定的统计，显式规模特征补回纯 mean 丢失的计数信息。

## 6. 连续偏好条件化：默认不使用 MoE

### 6.1 偏好编码

偏好编码为：

$$
p_\alpha
=\mathrm{MLP}_\alpha
\left[
\alpha_c,\alpha_r,
\sin(\pi\alpha_c),\cos(\pi\alpha_c),
\sin(2\pi\alpha_c),\cos(2\pi\alpha_c)
\right].
$$

共享 GAT 先生成结构表示，再在 Actor 和 Critic 中使用 FiLM：

$$
\widetilde h_v^{\alpha}
=\mathrm{LayerNorm}
\left[
h_v+\gamma(p_\alpha)\odot h_v+\beta(p_\alpha)
\right],
$$

$$
\widetilde g^{\alpha}
=\mathrm{LayerNorm}
\left[
g+\gamma_g(p_\alpha)\odot g+\beta_g(p_\alpha)
\right].
$$

这样所有偏好共享同一套 GAT 和解码器权重，但动作分布可随偏好连续变化。

### 6.2 向量 Critic 也必须知道偏好

虽然 Critic 输出成本和风险两个分量，但未来动作由偏好条件策略决定，因此价值函数应写为：

$$
V_\theta(s,\boldsymbol\alpha)
=\left[V_c(s,\boldsymbol\alpha),V_r(s,\boldsymbol\alpha)\right].
$$

另设辅助可行性头：

$$
q_{\mathrm{complete}}(s)
=P(\text{剩余时域可完成}\mid s).
$$

可行性头用于候选排序和表示学习，不替代确定性 oracle。

### 6.3 MoE 的可选位置

只有非 MoE 模型出现以下证据时才启用 MoE：

- 极端成本和极端风险性能无法同时提高；
- 中间偏好策略忽略偏好或 Pareto 覆盖坍缩；
- 不同目标梯度长期强冲突；
- 在相同参数/训练预算下，增加共享 MLP 宽度仍无效。

可选 MoE 只放在 GAT 后的 Actor 解码器：

$$
z_m=\sum_{e=1}^{E}g_e(g,p_\alpha)E_e(g),
\qquad
\sum_e g_e=1.
$$

建议 $E=2$ 或 $4$、连续软路由、共享动作头。不得为不同规模训练不同 GAT 专家，否则削弱“同一共享结构模型”的主张。MoE 与非 MoE 必须在同一数据、墙钟时间、参数量或 FLOPs 对齐条件下比较。

## 7. 动态动作空间

动作分解为：

$$
a=(o,u,v,q),
$$

- $o$：算子；
- $u$：源服务事件或车辆；
- $v$：目标服务事件、车辆或设施；
- $q$：插入边或第二切分位置。

联合概率为：

$$
\pi(a\mid G,\alpha)
=\pi(o\mid G,\alpha)
\pi(u\mid o,G,\alpha)
\pi(v\mid o,u,G,\alpha)
\pi(q\mid o,u,v,G,\alpha).
$$

每个对象头在当前候选节点或边上计算指针分数，输出维度随图变化。

### 7.1 首版算子

| 算子 | 作用 |
|---|---|
| `relocate_in_period` | 把服务事件移到同一期另一车辆/位置 |
| `swap_in_period` | 交换同一期两个服务事件 |
| `two_opt_route` | 反转一条路线的局部片段 |
| `change_facility` | 改变路线起终点处理设施 |
| `reschedule_between_periods` | 在相邻两期移动单个服务事件 |

跨期动作操作服务事件而不是整条路线。把 $t+1$ 的任务提前到 $t$ 时，只能收集截至 $t$ 已产生的数量；$t+1$ 新产生量仍需保留后续服务。

### 7.2 分层硬掩码

1. 算子掩码：是否存在合法候选；
2. 源掩码：节点类型、时期、冻结状态；
3. 目标掩码：容量、技术、兼容性；
4. 位置掩码：允许弧、重复访问和插入可行性；
5. 终端掩码：是否破坏最迟服务期或明显的剩余能力必要条件。

非法动作 logits 在 softmax 前设为 $-\infty$。如果全部动作被屏蔽，则结束当前窗口，不通过重复采样消耗步骤。

## 8. 终期库存的硬保障机制

### 8.1 正式终端条件

当 `require_terminal_clear=true`：

$$
I^P_{i,s,H}=0,\quad\forall i,s,
$$

$$
I^F_{j,s,H}=0,\quad\forall j,s.
$$

这与当前 MILP 和统一检查器的定义保持一致。

### 8.2 第一层：逐期最迟服务期

根据当前库存、未来产生量和产生者容量，用反向累计扫描计算每个“产生者—废物”义务的最迟可服务期 $d_{i,s}$：如果继续延后一次服务将使任一中间期库存超过 $C_i^P$，或使剩余车辆/处理能力不足以在 $H$ 前清零，则当前期就是该义务的截止期。该日期由离散状态转移直接计算，不使用易产生期序歧义的近似闭式公式。任何把任务延后到 $d_{i,s}$ 之后的动作直接屏蔽。

### 8.3 第二层：剩余能力必要条件

在窗口边界 $b$，对每类废物：

$$
R_{s,b}
=\sum_i I^P_{i,s,b}
+\sum_j I^F_{j,s,b}
+\sum_i\sum_{t>b}g_{i,s,t}.
$$

要求：

$$
R_{s,b}
\le
\sum_{j:\tau_{j,s}=1}\sum_{t>b}P_{j,s,t}^{\max}.
$$

待收集部分还必须满足剩余车辆容量的聚合条件。除此之外，构建一个小型时间扩展流：

```text
产生者—废物义务
→ 可服务时期
→ 车辆期容量
→ 技术可行设施—废物—时期
→ 处理容量
```

流模型不处理完整路径次序，作为快速必要条件 oracle。流不可行则动作一定不可提交。

### 8.4 第三层：可行补全证明

必要条件通过后，提交窗口前尝试补全未冻结时期：

1. 固定已经提交的时期；
2. 保留当前未来路线作为热启动；
3. 对受影响义务执行确定性插入和设施分配；
4. 传播全部产生者和设施库存；
5. 验证最后一期库存清零。

若启发式补全失败，在受影响节点和最后 $K=2$ 或 $3$ 期上调用小型 MILP 可行性/修复子问题。MILP 只负责补全与可行性，不替代 GAT 对搜索方向的学习。

只有存在完整可行补全时，窗口才可提交。这保证策略不会在早期把问题推入已知无法恢复的状态。

### 8.5 第四层：最终窗口修复与回退

最后窗口执行：

```text
反向计算未清义务
→ 优先拉回 H-2、H-1、H 共同消化
→ 启用空闲车辆并重新分配设施
→ 必要时调用末期 MILP repair
→ 完整 check_solution
```

不得只把全部剩余库存塞入第 $H$ 期，因为车辆和处理能力可能已无余量。

若最终仍不可行：

- 当前 rollout 标记为终端失败；
- 不把伪可行方案写入结果；
- 返回历史最佳严格可行 incumbent；
- 训练时给予独立终端失败信号。

### 8.6 终端势函数只是辅助

定义：

$$
\Phi_b
=\lambda_P\widehat I^P_b
+\lambda_F\widehat I^F_b
+\lambda_C[{-s^{\mathrm{proc}}_b}]_+
+\lambda_V[{-s^{\mathrm{veh}}_b}]_+
+\lambda_D\widehat{\mathrm{deadline\ violations}}_b.
$$

奖励使用 $\Phi_{\mathrm{before}}-\Phi_{\mathrm{after}}$。势函数帮助减少短视延后，但不能取代硬 oracle、补全和终端检查。

## 9. PPO Actor-Critic

### 9.1 二维即时回报

$$
r_t^c
=\widehat f_{c,\mathrm{before}}
-\widehat f_{c,\mathrm{after}},
$$

$$
r_t^r
=\widehat f_{r,\mathrm{before}}
-\widehat f_{r,\mathrm{after}}
+\eta(\Phi_{\mathrm{before}}-\Phi_{\mathrm{after}}).
$$

无效或补全失败使用独立的可行性惩罚字段，避免把硬约束语义混入成本或风险定义。给定偏好后：

$$
r_t^\alpha=\alpha_cr_t^c+\alpha_rr_t^r-P_t^{\mathrm{feas}}.
$$

### 9.2 双分量优势

$$
A_t^c=\mathrm{GAE}(r_t^c,V_c),
\qquad
A_t^r=\mathrm{GAE}(r_t^r,V_r),
$$

$$
A_t^\alpha=\alpha_cA_t^c+\alpha_rA_t^r.
$$

PPO clipped loss使用 $A_t^\alpha$；价值损失分别拟合两个目标。可行性头使用补全 oracle 的标签做辅助二分类损失。

### 9.3 轨迹边界

窗口前移不截断 GAE；完成全部剩余窗口、达到 sweep 停止条件或终端失败才形成 episode 终止。轮末可加入完整成本和风险改善奖励，使局部改进与完整方案质量对齐。

## 10. 面向泛化的训练设计

### 10.1 一个检查点的含义

“同一个模型”在本文中严格定义为：

- 同一网络结构；
- 同一组基础参数；
- 不因规模、偏好或实例重新初始化；
- 主结果推理时冻结参数；
- 仅图节点数、边数、掩码和偏好条件发生变化。

### 10.2 联合实例分布

每个 episode 同时随机采样：

| 维度 | 范围设计 |
|---|---|
| 实体规模 | 产生者、废物类型、设施、车辆数量 |
| 时间规模 | $H$ 及产生高峰位置 |
| 空间分布 | 均匀、聚类、偏心、现实坐标扰动 |
| 容量紧度 | 宽松、中等、临界、局部瓶颈 |
| 技术与兼容 | 稠密/稀疏处理矩阵和共载矩阵 |
| 风险结构 | 距离相关、区域热点、废物后果差异 |
| 初始解质量 | 贪心、GA、扰动可行解、小规模 MILP |
| 偏好 | 端点、端点邻域和连续中间值 |

批次按实例等权。大型图不能仅因节点多而在损失中占更高权重。

### 10.3 偏好采样

建议初始分层：

```text
15%: (1, 0)
15%: (0, 1)
20%: αc ~ Beta(0.3, 0.3)   # 端点邻域
50%: αc ~ Uniform(0, 1)
αr = 1 - αc
```

同一实例在一个训练批次内配对多个偏好，有利于模型学习偏好变化而不是把实例差异误当成偏好差异。

### 10.4 课程与回放

从宽松小/中规模提高可行轨迹比例，再逐渐增加规模和约束紧度；每个阶段持续回放旧规模、端点偏好和终端困难实例，避免遗忘。验证集按实例族划分，不能只随机打散同一分布样本。

### 10.5 可选模仿预热

小规模 MILP、GA 和确定性修复器可生成状态—可行动作对，用于 Actor 行为克隆和可行性头预热。随后再用 PPO 优化。教师数据只用于提高样本效率；最终比较必须仍以共同目标和完整约束评价。

## 11. 推理协议

默认零样本推理：

1. 载入一个冻结检查点；
2. 由启发式或 GA 生成严格可行初始解；
3. 对给定偏好执行多轮正向 sweep；
4. 每步从掩码后的策略采样/选取若干候选；
5. 使用局部评价、终端势和可行性 oracle 排序；
6. 只提交可补全动作；
7. 始终维护最佳完整可行 incumbent；
8. 达到统一墙钟预算或无改进阈值后返回 incumbent。

测试时适应、latent search 或在线更新仅作为单独增强模式。其时间必须计入总预算，且结果标为 `GAT+adaptation`，不能与冻结的 `GAT-zero-shot` 混报。

## 12. 泛化与多目标评估

### 12.1 数据切分

| 测试层 | 定义 |
|---|---|
| IID | 训练范围内的未见实例 |
| 规模内插 | 训练规模点之间的未见组合 |
| 规模外推 | 超过训练实体数或时期数 |
| 参数偏移 | 产生、容量、成本、风险分布改变 |
| 空间偏移 | 坐标分布改变或现实地理实例 |
| 结构偏移 | 技术、兼容和允许弧结构改变 |
| 偏好网格 | 稠密且包含训练未直接采样的偏好 |

### 12.2 基线

- 贪心初始解；
- 遗传算法；
- 小规模 MILP 加权和；
- 小规模增强 ε-约束 Pareto 基线；
- 可选 VNS/滚动时域启发式；
- GAT 非 MoE 主模型；
- GAT + MoE 消融。

旧 Transformer 代码和结果已经删除，不再作为当前仓库的可执行基线。若论文需要比较，应以独立归档版本或重新实现的明确基线进行，不能引用已删除结果。

### 12.3 指标

- 严格可行率；
- 终期产生者库存清零率；
- 终期设施库存清零率；
- 成本、风险和各偏好加权目标；
- Hypervolume、IGD、Pareto 覆盖宽度；
- 小规模相对最优 gap；
- 相对 GA/VNS 的改进；
- 首个改进解时间与最终 wall-clock；
- 峰值显存和每步推理时间；
- `candidate_recall`、动作掩码率、修复/补全失败率；
- 随规模增长的性能退化斜率；
- 同一实例上策略对偏好变化的敏感度和单调性诊断。

所有方法使用相同实例、目标归一化、约束检查器和时间预算。

## 13. 关键消融

| 变体 | 验证问题 |
|---|---|
| `gat_shared_preference_film` | 非 MoE 主模型 |
| `gat_decoder_moe` | MoE 是否在公平预算下有增益 |
| `gat_no_preference_condition` | 模型是否真正使用偏好 |
| `gat_scalar_critic` | 向量 Critic 的作用 |
| `gat_original_attention` | GATv2 动态注意力的作用 |
| `gat_no_edge_features` | 距离和风险边属性的作用 |
| `gat_one_period` | 双期前瞻的作用 |
| `gat_no_future_summary` | 远期摘要对长期与终端可行性的作用 |
| `gat_no_temporal_edges` | 跨期库存边的作用 |
| `gat_no_flow_oracle` | 剩余时域必要条件的作用 |
| `gat_no_completion_check` | 可行补全对终期清零率的作用 |
| `gat_penalty_only_terminal` | 证明只用终端惩罚是否不足 |
| `train_fixed_scale` | 多规模联合训练的贡献 |
| `train_single_distribution` | 域随机化的贡献 |
| `absolute_features` | 无量纲特征的贡献 |
| `fixed_object_slots` | 动态指针动作的贡献 |

## 14. 建议初始配置

```json
{
  "model_type": "temporal_edge_gat_ppo",
  "window_size": 2,
  "hidden_dim": 96,
  "gat_layers": 3,
  "attention_heads": 4,
  "edge_dim": 32,
  "dropout": 0.1,
  "distance_neighbors": 8,
  "risk_neighbors": 4,
  "exploration_edges": 2,
  "typewise_pooling": ["mean", "max", "attention"],
  "preference_conditioning": "film",
  "vector_critic": true,
  "completion_feasibility_head": true,
  "use_moe": false,
  "moe_experts": 2,
  "hard_action_masks": true,
  "flow_feasibility_oracle": true,
  "completion_check_before_commit": true,
  "terminal_repair_periods": 3,
  "terminal_repair_use_milp": true,
  "return_best_feasible_incumbent": true,
  "window_search_steps": 16,
  "window_patience": 6,
  "max_global_sweeps": 10,
  "sweep_patience": 3
}
```

PPO 初始值：

```json
{
  "gamma": 0.97,
  "gae_lambda": 0.95,
  "clip_ratio": 0.2,
  "learning_rate": 0.0003,
  "entropy_coef": 0.01,
  "value_coef": 0.5,
  "feasibility_aux_coef": 0.2,
  "invalid_action_penalty": 0.1,
  "completion_failure_penalty": 1.0,
  "terminal_failure_penalty": 5.0
}
```

数值仅作为首轮实验起点，必须通过验证集和消融确定。

## 15. 实现模块与现有实验接口

建议新增：

| 文件 | 职责 |
|---|---|
| `src/gat_types.py` | 图、动作、轨迹和结果数据结构 |
| `src/window_graph.py` | 双期图、远期摘要、候选边和掩码 |
| `src/gat_policy.py` | 边增强关系 GATv2、偏好调制、Actor/Critic |
| `src/window_operators.py` | 五类服务事件级局部算子 |
| `src/terminal_feasibility.py` | 最迟服务期、流 oracle、补全和末期 MILP repair |
| `src/gat_environment.py` | 窗口推进、库存传播、奖励和 incumbent |
| `src/gat_trainer.py` | 多规模多偏好 PPO 训练与检查点 |
| `src/gat_experiment_adapter.py` | 对接统一实验入口 |
| `configs/gat_config.json` | 图、策略、训练和终端保障配置 |
| `run_gat_ablation.py` | 泛化与结构消融 |

当前 `run_experiments.py` 已预留：

```python
class GATExperimentAdapter(Protocol):
    def prepare(...) -> float: ...
    def solve(...) -> GATRunResult: ...

def build_gat_adapter(...) -> GATExperimentAdapter | None:
    ...
```

实现完成后，`build_gat_adapter` 只负责导入并返回具体 adapter。实验入口继续使用共同的 `evaluate_solution` 和约束检查逻辑，保证 MILP、GA 与 GAT 结果口径一致。

## 16. 最低测试要求

### 图与尺寸

- 节点顺序置换不改变图级输出和对应动作分数；
- 不同实体数量无需重建网络层；
- batch 中节点/边指针不跨图；
- 不存在固定对象槽位和绝对 ID 依赖；
- 候选图始终包含当前路径边。

### 跨期与库存

- 库存逐期守恒；
- 提前服务不能收集未来产生量；
- 延后动作不能越过最迟服务期；
- 已冻结时期不被修复器修改；
- 流 oracle 对已知容量不足实例返回不可行；
- 最终产生者和设施库存严格为零；
- 末期修复失败时返回 incumbent 而非伪可行解。

### 偏好

- $(1,0)$ 与 $(0,1)$ 端点可运行；
- 中间偏好连续变化不会导致张量形状变化；
- Critic 两分量与独立成本/风险回报对齐；
- 相同实例不同偏好产生可检测的策略或结果差异；
- 无偏好条件消融能够揭示偏好输入的实际贡献。

### 实验公平性

- 单一冻结检查点完成泛化主测试；
- 各方法使用相同 wall-clock 和硬件口径；
- 所有结果经过同一完整约束检查器；
- 任何测试时更新单独标注并计时；
- 删除或失败的历史 Transformer 结果不进入新比较。

## 17. 风险与止损标准

| 风险 | 诊断 | 优先处理 |
|---|---|---|
| 候选图漏边 | `candidate_recall` 低 | 扩展候选规则，不先加深网络 |
| 规模外推退化 | 大图 gap/熵突变 | 混合规模训练、规模特征、熵校准 |
| 忽略偏好 | 不同 $\alpha$ 输出近似相同 | 成对偏好训练、条件调制、向量损失 |
| 终期失败 | 最后窗口大量 repair 失败 | 提前最迟期约束、流 oracle、跨 3 期修复 |
| GAT 过压缩 | 远期义务信息失真 | 摘要节点、类型池化、残差，避免盲目加层 |
| MoE 专家坍缩 | 路由/专家输出无差异 | 保持非 MoE 主线；仅在有证据时启用 |
| 训练被大图主导 | 大图梯度/样本权重过大 | 按图归一化损失和规模分层采样 |
| 看似泛化实为微调 | 每实例更新基础权重 | 冻结主测试，适应结果单列 |

若非 MoE 主模型已经达到目标，则不增加 MoE。若严格可行率未达到要求，则先修复环境、掩码和 terminal oracle，不通过扩大终端惩罚掩盖问题。

## 18. 论文表述边界

在完成实验前可以表述：

> 本文设计了尺寸无关的异构时序图、共享边增强 GAT 和偏好条件化指针策略，使同一网络结构能够接收不同规模实例与连续成本—风险偏好。

只有在冻结单检查点通过规模与分布外测试后，才可以进一步表述：

> 该模型在所测试的规模和分布范围内表现出跨规模、跨偏好的经验泛化能力。

不应表述为：

> GAT 天然保证任意规模泛化，或该方法保证得到 Pareto 最优解。

神经求解器仍是近似方法；小规模最优性由 MILP/ε-约束基线验证，大规模质量由 GA、滚动启发式、可行率和 Pareto 指标共同评估。

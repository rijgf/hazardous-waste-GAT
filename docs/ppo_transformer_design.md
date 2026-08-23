# 危险废物运输优化的滑动双周期 PPO+GAT-MoE 改进器设计

> 状态：待确认设计稿  
> 本次变更范围：仅修改方法说明文档，尚未修改任何实现代码或配置文件。  
> 当前代码仍实现 PPO+Transformer；本文描述的是后续拟采用的 PPO+GAT-MoE 方法。

## 摘要

本文提出一种面向多周期危险废物运输优化的**滑动双周期 PPO+GAT-MoE 局部改进方法**。与原方法把完整多周期方案编码为长 token 序列不同，新方法在任意决策时刻只向策略网络提供相邻两个时期

$$
\mathcal{W}_\tau=\{t_\tau,t_{\tau+1}\}
$$

的局部图状态。历史时期被冻结，远期状态不以完整路线或完整解的形式输入，只保留保证可行性所需的紧凑边界摘要。窗口内的产生者—废物、车辆、处理设施和当前路径关系被构造成异构时序图，通过关系感知的图注意力网络（Graph Attention Network, GAT）学习偏好中立的共享结构表示。GAT 后接两个 MLP 专家，分别学习成本导向与风险导向的策略变换；受成本—风险偏好约束的路由器融合专家输出，再由 PPO Actor-Critic 策略选择受可行性掩码约束的局部搜索动作。

该设计的目标是：

1. 将网络输入规模由随总周期数线性增长，改为主要取决于固定的两期窗口；
2. 利用图结构直接表达任务分配、路径相邻、设施能力和跨期库存传递；
3. 避免固定 128 维对象编号对不同算例规模的限制，改用图节点指针式动作选择；
4. 通过“轮内冻结过去、优化两期、提交第一期、窗口前移、轮末重新解锁”的多轮滚动机制持续改进完整方案；
5. 通过边界价值、剩余容量约束和终端修复降低短视窗口导致的不可行风险。
6. 通过成本专家、风险专家和受约束偏好路由提高不同风险偏好下策略的可分性，同时保留共享图表示。
7. 通过多规模、多分布预训练获得通用基础策略，并在新实例上优先零样本求解、按需进行实例级轻量适应。

---

## 1. 问题定义

### 1.1 集合与索引

| 符号 | 含义 |
|---|---|
| $I$ | 危险废物产生者集合，$i\in I$ |
| $S$ | 废物类型集合，$s\in S$ |
| $J$ | 处理设施集合，$j\in J$ |
| $K$ | 车辆集合，$k\in K$ |
| $T=\{1,\ldots,H\}$ | 离散时期集合，$t\in T$ |
| $N$ | 取货节点集合，节点 $n=(i,s)$ |
| $A$ | 允许通行的有向运输弧集合 |

代码中的取货节点采用 `n::{producer}::{waste_type}` 命名。完整问题参数由 `ModelParams` 保存，当前路径方案使用

```python
RoutePlan = Dict[Tuple[vehicle, period], List[node]]
```

表示。

### 1.2 优化目标

模型同时最小化经济成本与环境风险：

$$
\min J=\alpha_c\widehat f_{\mathrm{cost}}+\alpha_r\widehat f_{\mathrm{risk}},
\qquad
\alpha_c+\alpha_r=1.
$$

其中 $\widehat f_{\mathrm{cost}}$ 和 $\widehat f_{\mathrm{risk}}$ 分别为归一化后的成本与风险。

成本包括：

$$
f_{\mathrm{cost}}
=f_{\mathrm{vehicle}}
+f_{\mathrm{distance}}
+f_{\mathrm{processing}}.
$$

风险包括：

$$
f_{\mathrm{risk}}
=f_{\mathrm{transport}}
+f_{\mathrm{coload}}
+f_{\mathrm{producer\ inventory}}
+f_{\mathrm{facility\ inventory}}.
$$

### 1.3 核心约束

方法必须保持以下约束：

- 每条启用路线从处理设施出发并返回兼容设施；
- 同一车辆同一时期最多执行一条路线；
- 车辆装载量不得超过车辆容量；
- 不兼容废物不得共载；
- 设施必须具备对应废物处理技术；
- 设施库存和处理量不得超过容量；
- 产生者库存、设施库存按时期守恒；
- 过去已经提交的时期不得被后续窗口修改；
- 当 `require_terminal_clear=true` 时，最后一期结束后的产生者和设施库存必须清零。

---

## 2. 方法总览

### 2.1 整体流程

新方法不把完整解直接输入神经网络，而是执行如下滚动优化：

```text
贪心初始方案 / 当前方案
        │
        ▼
选择窗口 Wτ = {tτ, tτ+1}
        │
        ▼
构造两期异构时序图 Gτ
  ├─ 两期局部路线与库存
  ├─ 上一期结转状态
  └─ 远期紧凑边界摘要
        │
        ▼
偏好中立的关系感知多头 GAT 编码
        │
        ▼
成本 MLP 专家 Ec ─┐
                  ├─ 受约束偏好路由 → 融合图/节点表示
风险 MLP 专家 Er ─┘
        │
        ▼
PPO Actor-Critic
  ├─ 算子选择
  ├─ 源节点选择
  ├─ 目标节点选择
  └─ 辅助位置选择
        │
        ▼
动作掩码 → 局部算子 → 窗口修复
        │
        ├─ 未收敛：继续优化当前窗口
        └─ 收敛：提交 tτ，窗口前移一时期
                         │
                         ▼
              完成一次全时域 sweep
                ├─ 有改进：保存全局最优，解锁全部时期，
                │          从 {1,2} 开始下一轮
                └─ 无改进达到 patience / 达到轮数上限：停止
```

### 2.2 模块与接口

建议把图构造和窗口约束收拢为一个深模块，其外部接口可概括为：

```python
WindowState build_window_state(
    params,
    current_plan,
    window_start,
    preference,
    search_progress,
)
```

调用者只需要知道当前方案、窗口起点和偏好权重。两期子图、边界摘要、候选边、动作掩码和节点映射均属于该模块的内部实现。偏好权重不写入 GAT 节点特征，而是作为独立输入交给 MoE 路由与标量化目标。这样 GAT 编码器、MoE 路由、PPO 训练器和推理流程共享同一个状态接口，避免在多个调用点重复实现窗口逻辑。

窗口动作模块的建议接口为：

```python
WindowTransition apply_window_action(
    window_state,
    operator,
    source,
    target,
    auxiliary,
)
```

返回新窗口状态、动作是否合法、局部目标变化、修复结果和拒绝原因。测试也只需通过这两个接口验证行为。

---

## 3. 双周期滑动上下文

### 3.1 窗口定义

对 $H$ 个时期，依次建立：

$$
\mathcal{W}_1=\{1,2\},\quad
\mathcal{W}_2=\{2,3\},\quad\ldots,\quad
\mathcal{W}_{H-1}=\{H-1,H\}.
$$

当 $H=1$ 时使用单期退化窗口 $\mathcal{W}_1=\{1\}$。

每个窗口包含两个角色不同的时期：

- **当前期 $t_\tau$**：窗口收敛后在本轮 sweep 内提交并冻结；
- **前瞻期 $t_{\tau+1}$**：提供一步前瞻，可在下一窗口中再次成为当前期。

对于最后一个窗口 $\{H-1,H\}$，窗口结束后同时提交两期，并执行终端可行性检查。

### 3.2 可见信息

策略网络可以看到：

- 两期内各产生者—废物节点的期初库存、当期产生量、收集量和期末库存；
- 两期内各车辆的使用状态、载荷、路线长度和任务序列；
- 两期内各设施—废物节点的接收量、处理量和库存；
- 两期内当前方案的分配关系、路径相邻关系和运输弧属性；
- 成本—风险偏好权重作为独立路由输入；它不写入 GAT 图节点特征；
- 当前窗口位置、窗口内搜索进度和连续无改进次数；
- 窗口开始前的结转库存；
- 窗口结束后的紧凑剩余义务与剩余能力摘要。

策略网络不可看到：

- $t_{\tau+2}$ 以后逐期的详细路线；
- $t_{\tau+2}$ 以后每辆车的具体任务分配；
- 完整多周期路径序列；
- 未来窗口的逐节点动作历史。

因此，“只输入两个时期”指的是不输入窗口外的详细解。为避免截断上下文破坏库存可行性，允许输入少量聚合边界量，但这些边界量不能还原未来完整方案。

### 3.3 左边界状态

历史时期已经冻结，只通过窗口开始时的结转量进入当前状态：

$$
I^{P,\mathrm{in}}_{i,s,\tau}
=I^P_{i,s,t_\tau-1},
\qquad
I^{F,\mathrm{in}}_{j,s,\tau}
=I^F_{j,s,t_\tau-1}.
$$

对于第一个窗口，左边界使用初始库存。

### 3.4 右边界摘要

在 $t_{\tau+1}$ 之后不提供详细解，只为每种废物保留以下聚合量：

$$
D^{\mathrm{future}}_{s,\tau}
=\sum_{i\in I}\sum_{t>t_{\tau+1}} g_{i,s,t},
$$

$$
C^{\mathrm{proc,future}}_{s,\tau}
=\sum_{j\in J}\sum_{t>t_{\tau+1}}
\tau_{j,s}P_{j,s,t},
$$

$$
C^{\mathrm{veh,future}}_{\tau}
=|K|\,(H-t_{\tau+1})Q.
$$

还可加入窗口末剩余库存、终端剩余时期数和能力松弛量：

$$
\mathrm{slack}^{\mathrm{proc}}_{s,\tau}
=C^{\mathrm{proc,future}}_{s,\tau}
-D^{\mathrm{future}}_{s,\tau}
-I^{F,\mathrm{out}}_{s,\tau}.
$$

这些量只描述未来能否承接窗口末遗留义务，不包含未来具体路径。

### 3.5 提交与冻结

窗口内允许同时修改 $t_\tau$ 和 $t_{\tau+1}$。达到最大窗口步数、连续无改进阈值或不存在合法动作时：

1. 对两期方案执行窗口可行性检查；
2. 提交并冻结 $t_\tau$；
3. 将 $t_\tau$ 的期末库存作为下一窗口的左边界；
4. 保留 $t_{\tau+1}$ 的方案作为下一窗口热启动；
5. 窗口前移到 $\{t_{\tau+1},t_{\tau+2}\}$。

同一轮 sweep 的后续窗口不得修改已冻结路线，从而保证库存按时间正向传播。完成最后一个窗口后，本轮形成一个完整可行候选解；进入下一轮 sweep 时重新解锁全部时期，再从首个窗口开始优化。

### 3.6 多轮全时域 Sweep

仅从 $\{1,2\}$ 到 $\{H-1,H\}$ 遍历一次通常只能得到一次序贯局部改进，不能视为策略已经收敛。外层必须重复遍历整个时域：

$$
\mathrm{sweep}\ m:
\quad
\{1,2\}\rightarrow\{2,3\}\rightarrow\cdots\rightarrow\{H-1,H\},
\qquad m=1,\ldots,M.
$$

第 $m$ 轮以当前全局最优完整方案 $P^{*(m-1)}$ 为热启动。轮内仍按正向顺序冻结已经提交的时期；轮末得到候选完整方案 $\widetilde P^{(m)}$，执行完整可行性和目标检查：

$$
P^{*(m)}
=
\begin{cases}
\widetilde P^{(m)},&
\widetilde P^{(m)}\text{ 可行且 }
J(\widetilde P^{(m)})<J(P^{*(m-1)})-\varepsilon,\\
P^{*(m-1)},&\text{否则}.
\end{cases}
$$

若候选解没有改善，则回退到轮初全局最优方案，避免局部窗口改进累计成全局退化。随后重新解锁全部时期，并从 $\{1,2\}$ 开始下一轮。

停止条件为：

- 达到最大 sweep 数 $M$；
- 连续 `sweep_patience` 轮全局目标没有改善；
- 相对改善小于 `sweep_improvement_tol`；
- 或连续多轮没有产生与全局最优方案不同的可行候选。

采用重复正向 sweep，而不是简单反向遍历，是因为库存和废物产生具有明确的时间因果方向。多轮重新解锁已经足以让早期窗口根据上一轮形成的后期结果继续调整。

---

## 4. 两期异构时序图

### 4.1 图定义

对窗口 $\mathcal{W}_\tau$ 构造有向异构图：

$$
\mathcal{G}_\tau=(\mathcal{V}_\tau,\mathcal{E}_\tau,\mathcal{R}),
$$

其中 $\mathcal{R}$ 为关系类型集合。图只包含两个时期的实体副本。

### 4.2 节点类型

| 节点类型 | 记号 | 数量 | 含义 |
|---|---:|---:|---|
| 上下文节点 | $c_\tau$ | 1 | 窗口位置、搜索进度、右边界摘要；不含偏好权重 |
| 时期节点 | $p_t$ | 1 或 2 | 每期全局统计与时期角色 |
| 取货节点 | $u_{i,s,t}$ | $2|I||S|$ | 产生者—废物在某期的库存与服务状态 |
| 车辆节点 | $v_{k,t}$ | $2|K|$ | 车辆在某期的路线和负载状态 |
| 设施节点 | $f_{j,s,t}$ | $2|J||S|$ | 设施对某废物在某期的处理状态 |

大规模配置 $|I|=20,|S|=4,|J|=3,|K|=8$ 时，双期窗口约包含：

$$
1+2+2(80+8+12)=203
$$

个节点，不再随总周期数 $H$ 增长。

### 4.3 节点特征

不同节点类型使用不同的输入投影，不强迫同一槽位在不同实体间共享语义。

#### 上下文节点

建议特征（不包含 $\alpha_c,\alpha_r$）：

```text
[window_start/H, remaining_periods/H,
 search_step/max_steps, no_improve/patience,
 current_window_cost_norm, current_window_risk_norm,
 producer_carry_in_norm, facility_carry_in_norm,
 future_generation_by_type,
 future_processing_slack_by_type,
 future_vehicle_slack]
```

不同废物类型的未来摘要可先经过共享 MLP 聚合，再写入上下文节点，避免特征维度随 $|S|$ 无界增长。

#### 时期节点

```text
[is_current, is_lookahead, period/H,
 total_generation, total_collected,
 producer_inventory_end, facility_inventory_end,
 processed_amount, used_vehicle_ratio,
 vehicle_capacity_slack, processing_capacity_slack]
```

#### 取货节点

```text
[producer_id_embedding, waste_type_embedding,
 is_current, is_lookahead,
 carry_in, generation, available,
 collected, remaining,
 assigned, assigned_vehicle_load_ratio,
 producer_capacity_slack,
 waste_consequence, producer_inventory_risk]
```

#### 车辆节点

```text
[vehicle_id_embedding,
 is_current, is_lookahead,
 used, task_count, load_ratio,
 route_length_norm, route_risk_norm,
 compatible_type_ratio,
 remaining_capacity_ratio]
```

#### 设施节点

```text
[facility_id_embedding, waste_type_embedding,
 is_current, is_lookahead,
 technology, carry_in, received,
 before_processing_inventory,
 processing_capacity, processed,
 after_processing_inventory,
 capacity_slack,
 processing_cost, facility_inventory_risk]
```

所有连续量使用基于问题容量、距离、风险参考值的稳定尺度归一化。训练集和测试集必须使用同一套归一化规则。

### 4.4 边与关系类型

| 关系 | 方向 | 含义 | 主要边特征 |
|---|---|---|---|
| `belongs_to_period` | 实体 $\leftrightarrow$ 时期 | 实体属于哪个时期 | 当前/前瞻标志 |
| `context_link` | 上下文 $\leftrightarrow$ 所有节点 | 广播窗口与边界信息 | 节点类型 |
| `assigned_to` | 取货 $\leftrightarrow$ 车辆 | 当前服务分配 | 路线位置、取货量 |
| `served_by` | 取货 $\leftrightarrow$ 设施 | 当前路线终点设施 | 技术可行性、处理成本 |
| `route_next` | 取货节点 $\rightarrow$ 后继取货节点 | 当前路线中的取货顺序；首末段通过车辆/设施关系表示 | 距离、事故概率、累计载荷 |
| `candidate_arc` | 候选节点 $\rightarrow$ 候选节点 | 可用于插入/换序的候选运输弧 | 距离、事故概率 |
| `same_entity_next_period` | $x_t\leftrightarrow x_{t+1}$ | 同一实体跨期连接 | 库存/负载变化 |
| `compatible_with` | 取货 $\leftrightarrow$ 取货 | 废物共载兼容性 | compatibility、coload risk |
| `facility_capability` | 取货 $\leftrightarrow$ 设施 | 可选处理关系 | technology、剩余能力 |

为控制边数，`candidate_arc` 不构造完全图。每个取货节点只连接：

- 当前路线中的前驱与后继；
- 距离最近的 $k_{\mathrm{nn}}$ 个可行取货节点；
- 具备对应处理技术的设施；
- 当前和前瞻期内具有剩余容量的车辆。

实际路径边始终保留，候选边按距离、技术、容量和兼容性预筛选。

---

## 5. 关系感知图注意力网络

### 5.1 分类型输入投影

对类型为 $\phi(v)$ 的节点 $v$：

$$
\mathbf h_v^{(0)}
=\mathrm{MLP}_{\phi(v)}(\mathbf x_v)
+\mathbf e_{\mathrm{type}}^{\phi(v)}
+\mathbf e_{\mathrm{period}}^{v}.
$$

建议隐藏维度 $d=64$。节点类型和时期角色使用可学习嵌入。

### 5.2 带边特征的关系注意力

对关系 $r$ 下从节点 $i$ 指向节点 $j$ 的边，注意力打分定义为：

$$
e_{ij}^{r,h}
=\mathrm{LeakyReLU}\left(
(\mathbf a_{r,h})^\top
\left[
\mathbf W_{r,h}^{s}\mathbf h_i
\Vert
\mathbf W_{r,h}^{d}\mathbf h_j
\Vert
\mathbf U_{r,h}\mathbf z_{ij}
\right]\right),
$$

其中 $\mathbf z_{ij}$ 是距离、风险、载荷、兼容性等边特征，$h$ 为注意力头。

归一化注意力为：

$$
\alpha_{ij}^{r,h}
=
\frac{\exp(e_{ij}^{r,h})}
{\sum_{q\in\mathcal N_r(j)}\exp(e_{qj}^{r,h})}.
$$

节点更新为：

$$
\widetilde{\mathbf h}_j^{(l+1)}
=
\mathop{\Vert}_{h=1}^{H_a}
\sum_{r\in\mathcal R}
\sum_{i\in\mathcal N_r(j)}
\alpha_{ij}^{r,h}
\mathbf W_{r,h}^{v}\mathbf h_i^{(l)},
$$

$$
\mathbf h_j^{(l+1)}
=
\mathrm{LayerNorm}
\left(
\mathbf h_j^{(l)}
+\mathrm{Dropout}
\left[
\mathrm{FFN}
\left(
\widetilde{\mathbf h}_j^{(l+1)}
\right)
\right]
\right).
$$

建议使用 3 层、4 个注意力头。三层传播可覆盖“取货节点 → 车辆 → 设施 → 时期/上下文”等多跳依赖，同时双期图规模仍较小。

### 5.3 图级状态表示

保留上下文节点最终表示 $\mathbf h_c$，并对全部有效节点执行注意力池化：

$$
\beta_v
=
\mathrm{softmax}_v
\left(
\mathbf w_p^\top\tanh(\mathbf W_p\mathbf h_v)
\right),
$$

$$
\mathbf h_{\mathrm{pool}}
=\sum_{v\in\mathcal V_\tau}\beta_v\mathbf h_v.
$$

最终图状态：

$$
\mathbf g_\tau
=\mathrm{MLP}_g
\left[
\mathbf h_c
\Vert
\mathbf h_{\mathrm{pool}}
\Vert
\mathbf h_{p_{t_\tau}}
\Vert
\mathbf h_{p_{t_{\tau+1}}}
\right].
$$

GAT 输出的节点表示和图级表示均为**偏好中立共享表示**。它们先进入双专家 MoE，再用于选择算子和具体对象；价值估计采用独立的双分量 Critic。

### 5.4 GAT 后是否需要 MLP

需要。GAT 负责关系传播和图结构编码，但不能直接替代完整的 Actor-Critic 输出模块。GAT 后至少需要：

- 图级非线性变换，用于算子选择；
- 节点级非线性变换，用于源、目标和插入位置指针；
- 价值函数变换，用于 PPO 的优势估计。

原始方案可理解为“GAT + 单个共享 MLP”。本设计将该单 MLP 替换为**两个稠密激活的 MLP 专家**：

- 成本专家 $E_c$：学习降低车辆、距离和处理成本所需的表示变换；
- 风险专家 $E_r$：学习降低运输、共载和库存风险所需的表示变换。

两个专家都参与中间偏好推理，不采用稀疏 top-1 路由。这里只有两个小专家，稠密加权的额外计算有限，并能保持偏好变化的连续性。

### 5.5 双专家表示

对图级表示：

$$
\mathbf z_c=E_c^g(\mathbf g_\tau),
\qquad
\mathbf z_r=E_r^g(\mathbf g_\tau).
$$

对每个节点表示：

$$
\mathbf q_{v,c}
=
\mathbf h_v+E_c^n(\mathbf h_v),
\qquad
\mathbf q_{v,r}
=
\mathbf h_v+E_r^n(\mathbf h_v).
$$

图专家与节点专家可以共享专家标识嵌入，但建议保留不同参数，因为图级算子选择与节点级对象选择的语义不同。残差形式保留 GAT 的共享结构信息，避免专家 MLP 完全覆盖基础表示。

### 5.6 受约束偏好路由

直接令专家权重等于 $(\alpha_c,\alpha_r)$ 是一个清晰的基线，但隐含了两个过强假设：

1. 成本专家与风险专家的输出尺度已经完全校准；
2. 任意中间偏好的最优策略都能由两个极端专家线性插值得到。

第二个假设对离散组合动作尤其不稳。成本最优路线和风险最优路线之间的折中可能需要第三种动作序列，而不是逐步随机混合两个极端策略。

因此采用“**偏好主导、状态残差修正**”的受约束路由。路由器读取独立偏好向量和共享图状态：

$$
\boldsymbol\delta_\tau
=
\rho\tanh
\left(
R\left[\mathrm{sg}(\mathbf g_\tau)\Vert\alpha_c\Vert\alpha_r\right]
\right),
$$

其中 $\mathrm{sg}$ 表示在专家预训练阶段停止路由梯度回传到 GAT，$\rho$ 限制状态修正幅度。路由权重定义为：

$$
g_j
=
\frac{(\alpha_j+\epsilon)\exp(\delta_j)}
{\sum_{k\in\{c,r\}}(\alpha_k+\epsilon)\exp(\delta_k)},
\qquad j\in\{c,r\}.
$$

端点采用硬约束：

$$
(\alpha_c,\alpha_r)=(1,0)\Rightarrow(g_c,g_r)=(1,0),
$$

$$
(\alpha_c,\alpha_r)=(0,1)\Rightarrow(g_c,g_r)=(0,1).
$$

中间偏好下，路由器只能在给定偏好附近做有限的状态相关修正。使用正则项：

$$
\mathcal L_{\mathrm{gate}}
=
D_{\mathrm{KL}}
\left(
\mathbf g_\tau^{\mathrm{route}}
\Vert
\boldsymbol\alpha
\right),
$$

保证用户偏好仍是专家权重的主要决定因素。若实验表明状态修正没有收益，可令 $\rho=0$，退化为完全由风险偏好直接控制。

### 5.7 专家融合位置

融合发生在专家表示之后、动作解码之前：

$$
\mathbf z_{\mathrm{mix}}
=g_c\mathbf z_c+g_r\mathbf z_r,
$$

$$
\mathbf q_{v,\mathrm{mix}}
=g_c\mathbf q_{v,c}+g_r\mathbf q_{v,r}.
$$

随后由一个共享的算子头和共享指针解码器产生动作分布。选择“融合表示后统一解码”，而不是分别生成两个完整策略再混合概率，原因是：

- 所有动作因子共享同一次路由，避免算子来自成本专家、对象却来自风险专家；
- 统一应用同一套可行性掩码；
- 避免两个专家 logits 尺度不同导致的不可控插值；
- 对中间偏好保留非线性组合能力。

### 5.8 改进后的分阶段训练

用户提出的“随机偏好预训练共享 GAT → 冻结 GAT → 分别训练全成本/全风险专家 → 整合”总体合理，但最后一步不能只是无训练拼接。建议改为四阶段。

#### 阶段 A：偏好覆盖的共享 GAT 预训练

使用临时共享 Actor 头和双分量 Critic，训练 GAT 学习通用图结构。偏好采样采用分层分布，而不是只用连续均匀随机：

```text
20%: (αc, αr) = (1, 0)
20%: (αc, αr) = (0, 1)
60%: αc ~ Uniform(0, 1), αr = 1 - αc
```

这样端点能获得足够样本，中间偏好也不会缺失。GAT 节点输入不包含偏好，防止共享编码器把极端偏好提前纠缠进结构表示。

#### 阶段 B：冻结 GAT，训练极端专家

从阶段 A 的共享 MLP 权重复制初始化两个专家：

- 固定 $(\alpha_c,\alpha_r)=(1,0)$，训练成本专家；
- 固定 $(\alpha_c,\alpha_r)=(0,1)$，训练风险专家；
- GAT、图池化和另一专家保持冻结；
- 两个专家使用各自的目标优势，而不是已经标量化的混合优势。

该阶段保留用户提出的专家专门化思路。复制共享头比随机初始化更稳定，也减少冻结 GAT 后的表示错配。

#### 阶段 C：冻结 GAT，联合偏好校准

装配两个专家、受约束路由器和共享动作解码器，再次使用分层随机偏好训练。该阶段必须存在，因为两个极端专家在各自轨迹分布上训练，直接组合后会遇到中间偏好状态分布偏移。

联合损失建议为：

$$
\mathcal L_{\mathrm{stageC}}
=
\mathcal L_{\mathrm{PPO,mix}}
+\lambda_c\mathcal L_{\mathrm{expert},c}
+\lambda_r\mathcal L_{\mathrm{expert},r}
+\lambda_g\mathcal L_{\mathrm{gate}}
+\lambda_d\mathcal L_{\mathrm{div}}.
$$

其中：

- $\mathcal L_{\mathrm{PPO,mix}}$：按当前偏好训练融合策略；
- $\mathcal L_{\mathrm{expert},c/r}$：在端点或相应目标优势下维持专家专门化；
- $\mathcal L_{\mathrm{gate}}$：使路由保持接近用户偏好；
- $\mathcal L_{\mathrm{div}}$：仅在成本与风险优势明显冲突的状态上鼓励专家策略具有差异，防止专家坍缩为同一策略。

不建议使用传统 MoE 的强制均匀负载均衡。两个专家的使用率本来就应由训练偏好分布决定；只需保证训练偏好采样平衡。

#### 阶段 D：小学习率联合微调

阶段 C 收敛后，解冻最后一层 GAT、图池化层和 LayerNorm，使用原学习率的 $0.05\sim0.1$ 倍进行短程联合微调。前两层 GAT 保持冻结。

完全永久冻结 GAT 的问题是：阶段 A 的共享表示未必包含极端专家和中间融合真正需要的全部判别特征。只解冻最后一层可修复表示瓶颈，同时降低两个专家反向干扰共享结构编码的风险。

---

## 6. 动作空间与算子适配

### 6.1 为什么不能直接沿用原算子

原方法使用固定长度对象编号，并允许把整条路线移动到任意时期。双期滑动图下存在三个问题：

1. 网络只看两个时期，不能安全地选择窗口外目标；
2. 不同规模图的实体数不同，固定 `object_count=128` 会截断或浪费动作；
3. 把整条路线提前或延后可能导致车辆冲突，也可能错误地处理尚未产生的废物。

因此，新方法把动作对象改为图节点指针，并把跨期操作缩小到**服务事件**粒度。

### 6.2 分解动作

动作写为：

$$
a=(o,u,v,q),
$$

其中：

- $o$：算子；
- $u$：源节点或源服务事件；
- $v$：目标车辆、目标设施或第二服务事件；
- $q$：插入边、路径位置或第二切分点。

联合策略分解为：

$$
\pi_\theta(a\mid\mathcal G_\tau)
=
\pi_\theta(o\mid\mathbf z_{\mathrm{mix}})
\cdot\pi_\theta(u\mid o,\mathcal G_\tau)
\cdot\pi_\theta(v\mid o,u,\mathcal G_\tau)
\cdot\pi_\theta(q\mid o,u,v,\mathcal G_\tau).
$$

训练时对各有效动作因子的对数概率求和。

### 6.3 指针式对象选择

源节点得分示例：

$$
s_u
=
\mathbf w_u^\top
\tanh
\left(
\mathbf W_g\mathbf z_{\mathrm{mix}}
+\mathbf W_n\mathbf q_{u,\mathrm{mix}}
+\mathbf e_o
\right).
$$

在 softmax 前将不属于当前算子候选集的节点设为 $-\infty$。目标节点和辅助位置采用相同方式，并条件依赖已选择对象。

### 6.4 五类窗口算子

#### O1：期内重定位 `relocate_in_period`

把服务事件 $(n,t)$ 从当前车辆路线移到同一时期另一辆车路线的指定插入边。

动作对象：

```text
source = pickup-period node
target = vehicle-period node
auxiliary = target route insertion edge
```

适用掩码：

- 只能选择窗口内已服务的取货节点；
- 源时期和目标车辆时期必须相同；
- 目标车辆必须有足够容量；
- 目标路线废物类型必须兼容；
- 路线终点设施必须能处理该废物，或允许随后由 O4 调整设施。

#### O2：期内交换 `swap_in_period`

交换同一时期两个服务事件在各自路线中的位置。

动作对象：

```text
source = first pickup-period node
target = second pickup-period node
auxiliary = unused
```

默认限制为同一时期，避免该算子与跨期重调度语义重叠。交换后重新计算两条受影响路线的容量、共载兼容性和设施能力。

#### O3：路径内 2-opt `two_opt_route`

在选定的车辆—时期路线中反转两个切分点之间的取货序列。

动作对象：

```text
source = vehicle-period node
target = first route edge
auxiliary = second route edge
```

该算子不改变服务时期、车辆负载和处理设施，主要优化距离与运输风险。

#### O4：处理设施重分配 `change_facility`

将某条车辆—时期路线的起终点设施改为另一可行设施。

动作对象：

```text
source = vehicle-period node
target = facility node in the same period
auxiliary = unused
```

目标设施必须：

- 对路线中全部废物具有处理技术；
- 在该时期及剩余时期具有足够处理能力；
- 不造成设施存储容量违反。

#### O5：双期服务重调度 `reschedule_between_periods`

在 $t_\tau$ 与 $t_{\tau+1}$ 之间调整单个产生者—废物服务事件，而不是移动整条路线。

动作对象：

```text
source = pickup-period node
target = vehicle node in the other period
auxiliary = target route insertion edge
```

包含两种方向：

**延后 $t_\tau\rightarrow t_{\tau+1}$**

- 删除当前期服务事件；
- 废物作为库存结转到前瞻期；
- 在前瞻期目标路线插入该节点；
- 检查产生者存储容量和库存风险。

**提前 $t_{\tau+1}\rightarrow t_\tau$**

- 当前期只能收集截至 $t_\tau$ 已经产生的数量；
- 若 $t_{\tau+1}$ 仍有新增废物或残余库存，必须保留前瞻期服务事件；
- 因此“提前”可能产生两次跨期访问，不能简单地从 $t_{\tau+1}$ 删除节点；
- 检查当前期车辆容量和设施处理能力。

这是相对于原 `delay_or_advance_period` 最重要的变化。按服务事件操作可以避免把尚未产生的废物错误地提前运输。

### 6.5 动作掩码

掩码分四级生成：

1. **算子掩码**：当前图中是否存在该算子的合法候选；
2. **源节点掩码**：节点类型、时期角色和冻结状态是否合法；
3. **目标节点掩码**：容量、兼容性、设施技术和时期是否匹配；
4. **位置掩码**：插入后是否存在有效运输弧、是否形成重复非法访问。

掩码应在采样前生效，而不是先采样再依赖高额惩罚。若全部算子均无合法动作，则结束当前窗口并提交。

---

## 7. 窗口修复与可行性

### 7.1 局部修复原则

每次动作后只修复受影响的两期子方案，不重建完整多周期方案。修复范围包括：

- 删除空路线；
- 确保每个车辆—时期最多一条路线；
- 修复插入位置和设施起终点；
- 对受影响路线重新检查容量与共载兼容性；
- 更新两期产生者库存；
- 更新两期设施接收、处理和库存；
- 更新窗口末结转量。

禁止修复器静默修改已冻结时期。

### 7.2 服务事件与取货量

新方法应区分“物理取货节点”与“某期服务事件”：

$$
u_{i,s,t}\neq u_{i,s,t+1}.
$$

同一个产生者—废物节点可以在两个时期均被访问。默认取货规则为：访问时收集截至该时期可用的全部库存。这样提前服务不会消除下一时期新产生废物的后续服务需求。

### 7.3 右边界可承接性

窗口末不能只检查两期内部可行，还必须检查剩余时期是否有能力承接遗留库存。至少满足必要条件：

$$
\sum_{i}I^P_{i,s,t_{\tau+1}}
+\sum_j I^F_{j,s,t_{\tau+1}}
+D^{\mathrm{future}}_{s,\tau}
\le
C^{\mathrm{proc,future}}_{s,\tau}
+\mathrm{slack}^{\mathrm{allowed}}_s,
$$

以及：

$$
\sum_{i,s}I^P_{i,s,t_{\tau+1}}
+\sum_sD^{\mathrm{future}}_{s,\tau}
\le
C^{\mathrm{veh,future}}_\tau
+\mathrm{slack}^{\mathrm{allowed}}_{\mathrm{veh}}.
$$

训练和正式推理默认令允许松弛为 0；调试模式可以保留松弛并施加高额惩罚。

### 7.4 最终窗口

最后窗口必须执行严格终端检查：

$$
I^P_{i,s,H}=0,\qquad I^F_{j,s,H}=0.
$$

若普通窗口算子未能得到终端可行解，执行确定性的末期修复：

1. 将未收集库存插入具有剩余容量的末期路线；
2. 必要时启用空闲车辆；
3. 选择具有处理技术和剩余处理能力的设施；
4. 若仍不可行，则判定该轨迹失败并给予终端惩罚，而不是返回伪可行方案。

---

## 8. PPO Actor-Critic

### 8.1 策略头

算子头使用 MoE 融合后的图级状态：

$$
\boldsymbol\ell_o=\mathrm{MLP}_{\mathrm{op}}(\mathbf z_{\mathrm{mix}})\in\mathbb R^5.
$$

源、目标和辅助对象头使用 $\mathbf z_{\mathrm{mix}}$ 与 $\mathbf q_{v,\mathrm{mix}}$ 计算指针分数。不同算子共享 GAT 编码器和最终动作解码器，但使用独立的候选掩码和条件嵌入。

### 8.2 双分量价值头

$$
\mathbf V_\theta(\mathcal G_\tau)
=
\left[
V_c(\mathcal G_\tau),
V_r(\mathcal G_\tau)
\right]
=\mathrm{MLP}_V(\mathbf g_\tau).
$$

Critic 直接预测成本回报与风险回报两个分量，不经过 MoE 路由。对给定偏好：

$$
V_{\boldsymbol\alpha}
=\alpha_cV_c+\alpha_rV_r.
$$

分别对两个目标计算 GAE：

$$
A_t^c=\mathrm{GAE}(r_t^c,V_c),
\qquad
A_t^r=\mathrm{GAE}(r_t^r,V_r),
$$

再在 PPO 策略更新时组合：

$$
A_t^{\boldsymbol\alpha}
=\alpha_cA_t^c+\alpha_rA_t^r.
$$

这样成本和风险学习信号在 Critic 与优势估计阶段保持分离，避免过早标量化造成梯度干扰。两个价值分量都估计从当前两期窗口到完成全部剩余窗口和 sweep 的期望累计回报。

### 8.3 PPO 目标

概率比：

$$
r_t(\theta)
=
\exp
\left[
\log\pi_\theta(a_t\mid\mathcal G_t)
-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid\mathcal G_t)
\right].
$$

策略损失：

$$
\mathcal L_{\mathrm{policy}}
=-\mathbb E
\left[
\min
\left(
r_tA_t,
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
$$

总损失：

$$
\mathcal L
=
\mathcal L_{\mathrm{policy}}
+c_v
\left(
\mathcal L_{\mathrm{value},c}
+\mathcal L_{\mathrm{value},r}
\right)
-c_e\mathcal H(\pi_\theta).
$$

对分解动作，联合对数概率和熵由各有效动作因子求和。某算子不需要的辅助动作不进入联合概率。在 MoE 联合校准阶段，式中还需加入专家端点损失、路由锚定损失和小权重的条件多样性损失。

---

## 9. 局部目标与奖励

### 9.1 两期局部目标

网络不接收完整解，但环境仍可对当前方案进行确定性核算。窗口局部目标为：

$$
J_\tau^{\mathrm{window}}
=
\alpha_c\widehat f_{\mathrm{cost}}^{\,t_\tau:t_{\tau+1}}
+\alpha_r\widehat f_{\mathrm{risk}}^{\,t_\tau:t_{\tau+1}}.
$$

它只计算两期内受动作影响的车辆、运输、处理和库存项。

### 9.2 边界势函数

仅优化两期目标容易把大量库存推到窗口外。为此定义窗口末势函数：

$$
\Phi_\tau
=
\lambda_P\sum_{i,s}\widehat I^P_{i,s,t_{\tau+1}}
+\lambda_F\sum_{j,s}\widehat I^F_{j,s,t_{\tau+1}}
+\lambda_C\sum_s
\max(0,-\widehat{\mathrm{slack}}^{\mathrm{proc}}_{s,\tau})
+\lambda_V
\max(0,-\widehat{\mathrm{slack}}^{\mathrm{veh}}_\tau).
$$

### 9.3 单步奖励

训练轨迹先保存二维奖励，而不是立即只保存标量：

$$
\mathbf r_t
=
\left[
r_t^c,r_t^r
\right],
$$

$$
r_t^c
=
\left(
\widehat f_{\mathrm{cost,before}}^{\,\mathrm{window}}
-\widehat f_{\mathrm{cost,after}}^{\,\mathrm{window}}
\right)
-P_{\mathrm{invalid}}^c
-P_{\mathrm{repair}}^c,
$$

$$
r_t^r
=
\left(
\widehat f_{\mathrm{risk,before}}^{\,\mathrm{window}}
-\widehat f_{\mathrm{risk,after}}^{\,\mathrm{window}}
\right)
+\eta
\left(
\Phi_{\mathrm{before}}
-\Phi_{\mathrm{after}}
\right)
-P_{\mathrm{invalid}}^r
-P_{\mathrm{repair}}^r.
$$

给定偏好后，融合策略的标量奖励为：

$$
r_t^{\boldsymbol\alpha}
=\alpha_cr_t^c+\alpha_rr_t^r.
$$

建议：

- 合法且改善局部目标：正奖励；
- 局部目标略差但显著改善边界可承接性：允许获得正净奖励；
- 合法但无改善：小惩罚；
- 被动作掩码排除的动作不参与采样；
- 修复失败或破坏右边界必要条件：较大惩罚；
- 最终窗口无法清零库存：终端大惩罚。

势函数采用差分形式，减少奖励塑形改变最优策略的风险。

---

## 10. 训练流程

完整训练按照第 5.8 节的 A—D 四阶段执行。以下 episode 与窗口流程是各阶段共享的环境交互骨架；不同阶段只改变哪些参数可训练、偏好采样分布和使用的损失项。

### 10.1 Episode 初始化

每个 episode：

1. 随机选择训练算例；
2. 随机采样偏好 $(\alpha_c,\alpha_r)$；
3. 由贪心启发式产生完整初始方案；
4. 设置首个窗口 $\{1,2\}$；
5. 初始化冻结集合为空；
6. 初始化 sweep 编号、全局最优方案和无改进 sweep 计数器。

### 10.2 窗口内搜索

每个窗口最多执行 $L_{\mathrm{window}}$ 个 PPO 动作：

1. 构造当前两期图；
2. 批量送入 GAT；
3. 应用分层动作掩码；
4. 采样算子和对象；
5. 执行窗口算子与局部修复；
6. 计算局部目标、边界势函数和奖励；
7. 保存图、偏好、路由权重、动作、掩码、对数概率、双分量价值和二维奖励；
8. 达到 patience 或无合法动作时提前结束。

### 10.3 窗口推进

当前窗口结束后，在本轮 sweep 内提交第一期并前移窗口。窗口推进不是策略动作，而是确定性控制逻辑，避免策略通过无限停留在容易优化的时期获取奖励。

完成最后窗口后：

1. 组合得到本轮候选完整方案；
2. 执行完整约束检查和全局目标评估；
3. 若候选可行且优于全局最优，则更新全局最优；
4. 否则回退到轮初全局最优方案；
5. 清空轮内冻结集合；
6. 若未满足停止条件，则从窗口 $\{1,2\}$ 开始下一轮 sweep。

### 10.4 轨迹与 GAE

一个 episode 的轨迹跨越所有窗口以及多轮 sweep。成本与风险分别计算 GAE，使当前窗口价值能够包含后续窗口和后续 sweep 的结果：

$$
\delta_t^j=r_t^j+\gamma V_j(s_{t+1})-V_j(s_t),
\qquad j\in\{c,r\},
$$

$$
A_t^j=\delta_t^j+\gamma\lambda A_{t+1}^j.
$$

融合策略使用 $A_t^{\boldsymbol\alpha}=\alpha_cA_t^c+\alpha_rA_t^r$。窗口切换和 sweep 切换均不截断 GAE；只有达到全局停止条件、完成最大 sweep 数或轨迹失败才形成终止状态。轮末若全局目标获得改善，可向两个奖励分量分别增加与完整成本、风险改善对应的 sweep 奖励；若候选被回退，则给予小惩罚，使策略学习“局部改善必须转化为全局改善”。

### 10.5 图批处理

多个 episode 当前窗口的图使用不相交图批处理。节点索引在 batch 内偏移，候选掩码按图保存，禁止跨图选择对象。

由于每张图只包含两个时期，显存主要随 batch 中实体数和候选边数增长，而不随总周期数直接增长。

---

## 11. 推理流程

推理时采用**多轮滚动双周期局部搜索**：

1. 从贪心方案或其他可行方案开始；
2. 保存该方案为全局最优并开始第 1 轮 sweep；
3. 对窗口 $\{1,2\}$ 构图；
4. 每步生成若干合法候选动作；
5. 根据当前偏好计算受约束专家权重；第一个候选使用掩码后 argmax，其余候选从融合策略分布采样；
6. 对候选执行局部修复并选择局部目标加边界势函数最优者；
7. 达到窗口停止条件后，在本轮内提交第一期；
8. 窗口向前滑动一时期，直至完成最后窗口；
9. 执行严格终端修复、完整解校验和全局目标评估；
10. 若本轮候选更优则更新全局最优，否则回退；
11. 重新解锁全部时期，从 $\{1,2\}$ 开始下一轮 sweep；
12. 达到最大轮数或连续多轮无改进后，返回全局最优方案、逐窗口历史和逐 sweep 改进历史。

推理过程中神经网络始终只接收当前双期图；完整方案仅由环境保存，用于轮内组合、轮末比较、回退和最终校验。多轮遍历不会扩大单次网络输入。

---

## 12. 计算复杂度

原完整时域 token 编码的实体数量近似为：

$$
O\left(H(|I||S|+|K|+|J||S|)+|\mathrm{route\ arcs}|\right).
$$

双期图的节点数量为：

$$
O\left(2(|I||S|+|K|+|J||S|)\right).
$$

若每个节点只保留 $k_{\mathrm{nn}}$ 条候选边，则边数量近似为：

$$
O\left(|\mathcal V|k_{\mathrm{nn}}+|\mathcal E_{\mathrm{actual}}|\right).
$$

单层稀疏 GAT 的主要复杂度为：

$$
O(|\mathcal E|d+|\mathcal V|d^2),
$$

避免标准全局自注意力的 $O(L^2d)$ token 两两交互。

若每轮包含 $H-1$ 个窗口、每个窗口最多执行 $L_{\mathrm{window}}$ 个动作、最多执行 $M$ 轮 sweep，则总体推理复杂度近似为：

$$
O\left(
M(H-1)L_{\mathrm{window}}
\left(|\mathcal E|d+|\mathcal V|d^2\right)
\right).
$$

其中 $M$ 由全局无改进停止条件限制。单次网络输入仍固定为两期图。

---

## 13. 通用预训练、泛化与在线求解

### 13.1 修正“完整解包含参数即可泛化”的假设

完整参数化输入是跨算例泛化的必要条件之一，但不是充分条件，需要区分三件事：

1. 当前 `RoutePlan` 和 `solution["raw"]` 主要保存路线与决策变量，不包含全部外生参数；
2. 完整外生参数位于 `ModelParams`，包括未经过运输弧的距离与事故概率、技术矩阵、兼容矩阵、处理能力和未来产生量；
3. 本方法的神经网络只接收双期窗口图和远期摘要，并不会看到窗口外完整路线与全部逐期参数。

因此，准确表述应为：

> 窗口图从当前解与问题参数中提取与双期决策有关的特征；尺寸无关的图表示使模型具备跨算例迁移的结构基础，但实际泛化范围仍由预训练任务分布、归一化、候选图覆盖和训练方法共同决定。

即使把全部参数都输入网络，模型也可能因以下原因在新规模或新分布上失效：

- 训练数据没有覆盖新的规模、约束紧度或空间分布；
- 绝对数值尺度发生变化；
- 候选边剪枝漏掉关键路线；
- 消息传递深度在更大图上不足；
- 动作分布和修复失败模式发生变化；
- 新算例包含训练中未出现的参数相关性。

所以本文不使用“输入包含全部参数，因此必然泛化”的结论。

### 13.2 模型定位

建议把该模型称为：

**面向危险废物运输实例族的通用预训练神经求解器**。

它采用类似基础模型的使用方式：

- 在广泛的合成与历史实例族上一次预训练；
- 一个基础检查点支持多种规模、参数和成本—风险偏好；
- 新实例先执行零样本求解；
- 必要时在固定测试时预算内进行轻量实例适应。

但在尚未通过跨规模、跨参数分布和跨约束组合实验前，不直接称为“危险废物运输大模型”。模型参数量大并不等于具有基础模型式泛化；统一表示、预训练覆盖度和可适应性更关键。

### 13.3 跨规模表示要求

为支持不同 $|I|,|S|,|J|,|K|,H$，必须保持：

- GAT 使用动态节点和稀疏边，不设置固定最大实体编号；
- 动作通过节点指针和掩码选择，不使用固定 128 维对象槽位；
- 聚合使用 sum/mean/attention pooling，不依赖固定图大小；
- 生产者、车辆和设施 ID 只表示节点类型与相对身份，不学习训练集专属绝对 ID；
- 连续量优先使用无量纲特征；
- 训练与推理使用相同的候选边生成规则；
- 双期窗口保持固定，使单次网络输入不随总时期数线性增长。

推荐的无量纲特征包括：

$$
\frac{\mathrm{inventory}}{\mathrm{producer\ capacity}},
\quad
\frac{\mathrm{route\ load}}{Q},
\quad
\frac{\mathrm{processing\ demand}}{\mathrm{remaining\ processing\ capacity}},
$$

$$
\frac{\mathrm{distance}}{\mathrm{instance\ distance\ scale}},
\quad
\frac{\mathrm{remaining\ periods}}{H},
\quad
\frac{\mathrm{risk}}{\mathrm{instance\ risk\ reference}}.
$$

成本和风险优势也必须按实例参考值归一化，防止大规模实例仅因绝对目标值较大而主导梯度。

### 13.4 预训练任务分布

四阶段训练不应只随机成本—风险偏好，还要对完整实例族做联合域随机化。每个训练 episode 随机采样：

| 维度 | 建议变化 |
|---|---|
| 规模 | 产生者、废物类型、设施、车辆和时期数 |
| 空间 | 均匀、聚类、中心—边缘、偏斜和历史地理分布 |
| 产生 | 总量、时间集中度、废物类型不平衡和初始库存 |
| 运力 | 车辆容量、车辆数和容量紧度 |
| 设施 | 技术稀疏度、处理能力、存储能力和处理成本 |
| 风险 | 事故概率、废物后果、共载风险和库存风险 |
| 约束 | 废物兼容性、终端清零难度和候选边密度 |
| 初始解 | 贪心、随机可行、GA/MILP 暖启动及不同质量水平 |
| 偏好 | 两个端点、中间均匀偏好和端点邻域偏好 |

训练规模采用混合批次，而不是先把小规模完全训完再只训练大规模。可以使用“由易到难”的采样概率课程，但始终保留小规模与端点任务的回放比例，避免后期遗忘。

训练、验证和测试必须按实例随机种子和参数分布分开。历史真实实例只能出现在一个数据分区。

### 13.5 四阶段预训练的泛化版本

#### 阶段 A：多规模、多分布共享预训练

在上述联合任务分布上训练偏好中立 GAT、临时共享 Actor 和双分量 Critic。偏好、规模和参数同时随机化，使 GAT 学习可迁移的库存—车辆—设施关系。

#### 阶段 B：跨实例极端专家训练

冻结 GAT，在同样广泛的实例分布上分别训练成本专家和风险专家。不能只在单一规模上训练极端专家，否则 MoE 会在规模变化时重新出现专家分布偏移。

#### 阶段 C：跨偏好联合校准

冻结 GAT，以端点和中间偏好联合训练专家、受约束路由与共享动作解码器。校准批次必须混合多种规模和参数分布。

#### 阶段 D：末层微调与适应接口训练

以小学习率解冻最后一层 GAT，完成短程联合微调。同时加入一个小型**实例适配器**：

$$
\mathbf z_{\mathrm{adapt}}
=
\mathbf z_{\mathrm{mix}}
+A_{\phi}(\mathbf z_{\mathrm{mix}},\mathbf c_{\mathrm{inst}}),
$$

其中 $\phi$ 仅包含少量参数，$\mathbf c_{\mathrm{inst}}$ 是当前实例统计。离线阶段模拟测试时适应：为每个训练实例复制适配器初始化，用 support rollouts 执行 1～3 次内层更新，再用 query rollouts 评价更新后的策略，并以一阶元学习近似优化适配器初始化及允许微调的末层参数。这样训练目标直接包含“少量实例级更新后仍能改善”的要求。基础 GAT、两个专家和共享解码器在部署时默认冻结。

### 13.6 在线求解的三种模式

必须区分以下三种机制。

#### 模式 Z：零样本多轮求解

这是默认模式，不更新任何网络参数：

1. 从基础检查点载入冻结模型；
2. 对新实例执行多轮双期 sweep；
3. 使用策略采样、不同随机种子和候选动作产生多条轨迹；
4. 始终保存历史最佳严格可行解；
5. 达到时间或 sweep 预算后返回 incumbent。

同分布或轻度分布偏移实例优先使用该模式，因为速度快、可重复且不会过拟合单个实例。

#### 模式 L：低维条件搜索

在不修改主网络的情况下，为当前实例搜索低维条件向量 $\mathbf c_{\mathrm{inst}}$：

$$
\mathbf c_{\mathrm{inst}}^*
=
\arg\min_{\mathbf c}
J\left(P_{\theta}(\mathcal I,\boldsymbol\alpha,\mathbf c)\right).
$$

可使用随机搜索、交叉熵法或小规模进化搜索。该模式类似在预训练策略族中寻找适合当前实例的策略，比梯度更新更稳定，适合作为零样本与微调之间的中间层。

#### 模式 A：实例级轻量主动适应

仅在零样本搜索停滞、检测到明显 OOD 且仍有计算预算时启用。每轮交替执行：

```text
冻结基础 GAT、两个专家主体和双分量 Critic
        ↓
用当前实例级参数执行若干完整 rollout
        ↓
以真实归一化成本、风险和可行性评价轨迹
        ↓
更新实例 adapter / 路由残差 / 最终动作头偏置
        ↓
继续求解并更新历史最佳可行解
```

建议在线可训练参数按优先级限制为：

1. 实例条件向量；
2. 实例 adapter；
3. 路由残差的小型 MLP；
4. 最终动作头偏置或 LayerNorm 仿射参数。

不在线更新共享 GAT、成本/风险专家主体和 Critic 主体。

### 13.7 在线适应目标与保护

对当前实例 $\mathcal I$ 和固定偏好 $\boldsymbol\alpha$，在线适应只优化实例副本 $\phi_\mathcal I$：

$$
\min_{\phi_\mathcal I}
\mathbb E
\left[
J_{\boldsymbol\alpha}
\right]
+\lambda_{\mathrm{KL}}
D_{\mathrm{KL}}
\left(
\pi_{\theta,\phi_\mathcal I}
\Vert
\pi_{\theta,0}
\right)
+\lambda_{\phi}\|\phi_\mathcal I\|_2^2.
$$

其中 $\pi_{\theta,0}$ 是冻结基础策略。KL 与参数正则限制单实例更新偏离预训练策略过远。

在线求解必须满足：

- **incumbent 单调保护**：只返回历史最佳可行方案，不返回最后一次更新结果；
- **基础权重只读**：每个新实例从同一基础检查点开始；
- **实例更新不继承**：默认不把 $\phi_\mathcal I$ 传给下一个实例；
- **预算固定**：零样本、低维搜索和适应均计入相同墙钟时间；
- **可回退**：若连续适应轮没有改善，恢复最佳实例参数或回到零适应状态；
- **可行性优先**：不可行轨迹不能因目标值低而成为 incumbent。

在线更新可以使用小批量 PPO，但单实例样本高度相关。首版实现更建议使用带自批评基线的 REINFORCE/active search 更新小型适配参数；待样本量足够后再比较 PPO。

### 13.8 何时启用在线适应

不把微调设为所有实例的必经步骤。满足任一条件时才触发：

- 实例规模超出预训练范围；
- 归一化参数明显超出训练分位区间；
- 技术或兼容拓扑属于未见组合；
- 连续 `adapt_trigger_sweeps` 轮没有全局改善；
- 修复失败率或策略熵明显异常；
- 零样本结果相对贪心/GA 基线差距超过阈值。

OOD 检测只决定是否分配更多测试时计算，不能证明适应一定有效。

### 13.9 不采用的在线方案

以下做法从正式方法中删除：

- 不在单实例求解过程中全量更新 GAT、两个专家和 Critic；
- 不把一个实例的临时权重直接用于下一个无关实例；
- 不以最后一次在线更新产生的方案替代历史最佳可行方案；
- 不声称在线微调必然优于零样本；
- 不把训练分布外的失败归因于“模型还不够大”；
- 不同时改变偏好路由与基础表示而缺少 KL、回退和时间预算控制。

若未来面对连续到达且确认同分布的业务实例，可另行训练域 adapter；该持续学习过程需要回放数据、冻结验证集和版本化检查点，不属于本文的单实例在线求解。

### 13.10 泛化与在线适应实验

测试集至少分为：

| 测试层 | 内容 |
|---|---|
| IID | 未见随机种子、规模与参数仍在训练范围 |
| 规模内插 | 训练规模之间的未见规模组合 |
| 规模外推 | 更大的产生者、车辆、设施或时期数 |
| 参数偏移 | 产生量、容量、成本和风险分布改变 |
| 空间偏移 | 均匀训练、聚类或现实地理测试 |
| 结构偏移 | 新技术矩阵、兼容矩阵和约束紧度 |
| 现实实例 | 与合成训练分开的真实案例 |

每层在相同时间预算下比较：

- 零样本多轮求解；
- 低维条件搜索；
- 实例 adapter 主动适应；
- 全网络 active search，仅作为高成本对照；
- 贪心、GA、MILP 或其他传统求解基线。

除可行率、成本、风险和 Pareto 指标外，还报告：

- 首个可行解时间；
- incumbent 随时间的改进曲线；
- 在线更新耗时与有效函数评价次数；
- 适应前后策略 KL；
- 不同规模下的最优差距；
- 适应失败或回退比例；
- 零样本与适应模式分别达到的结果。

只有这些实验通过后，才能声称模型具有跨规模泛化或在线适应能力。

---

## 14. 建议配置

以下配置仅为待确认初始值，尚未写入 `configs/network_config.json`：

```json
{
  "network_type": "sliding_window_hetero_gat_moe",
  "window_size": 2,
  "node_embedding_dim": 64,
  "gat_layers": 3,
  "attention_heads": 4,
  "edge_embedding_dim": 16,
  "ff_hidden_dim": 128,
  "dropout": 0.1,
  "candidate_neighbors": 8,
  "use_edge_features": true,
  "use_boundary_summary": true,
  "pointer_action_heads": true,
  "expert_count": 2,
  "expert_hidden_dim": 128,
  "dense_expert_mixing": true,
  "preference_neutral_gat": true,
  "router_state_residual": true,
  "router_residual_bound": 0.5,
  "router_preference_kl_coef": 0.05,
  "expert_diversity_coef": 0.005,
  "hard_route_at_extremes": true,
  "vector_critic": true,
  "partial_unfreeze_lr_ratio": 0.1,
  "instance_condition_dim": 16,
  "instance_adapter_dim": 32,
  "freeze_base_model_at_inference": true,
  "operator_count": 5,
  "window_search_steps": 16,
  "window_patience": 6,
  "max_global_sweeps": 10,
  "sweep_patience": 3,
  "sweep_improvement_tol": 0.0001
}
```

PPO 初始建议：

```json
{
  "gamma": 0.97,
  "gae_lambda": 0.95,
  "clip_ratio": 0.2,
  "learning_rate": 0.0003,
  "entropy_coef": 0.01,
  "value_coef": 0.5,
  "invalid_action_penalty": 1.0,
  "repair_failure_penalty": 1.0,
  "terminal_infeasible_penalty": 5.0
}
```

在线求解初始建议：

```json
{
  "default_mode": "zero_shot",
  "enable_latent_condition_search": true,
  "enable_instance_adaptation": true,
  "adapt_trigger_sweeps": 3,
  "max_adapt_rounds": 8,
  "adapt_rollouts_per_round": 32,
  "adapt_learning_rate": 0.0001,
  "adapt_kl_coef": 0.05,
  "adapt_l2_coef": 0.0001,
  "adapt_patience": 3,
  "reset_instance_state": true,
  "return_best_feasible_incumbent": true,
  "online_trainable": [
    "instance_condition",
    "instance_adapter",
    "router_residual",
    "action_head_bias"
  ]
}
```

---

## 15. 建议消融实验

| 变体 | 目的 |
|---|---|
| `gat_moe_two_period_full` | 完整双期 GAT-MoE 方法 |
| `gat_single_mlp` | 双专家 MoE 与单共享 MLP 对比 |
| `moe_direct_preference_gate` | 完全直接权重与受约束状态残差路由对比 |
| `moe_no_joint_calibration` | 验证极端专家训练后直接整合的问题 |
| `moe_gat_frozen_forever` | 验证阶段 D 小学习率解冻最后一层 GAT 的作用 |
| `moe_scalar_critic` | 双分量 Critic 与提前标量化 Critic 对比 |
| `moe_no_diversity_loss` | 检查专家是否出现表示或策略坍缩 |
| `gat_one_period` | 验证第二期前瞻的贡献 |
| `gat_no_temporal_edges` | 验证同实体跨期边的贡献 |
| `gat_no_boundary_summary` | 验证远期紧凑摘要对终端可行性的贡献 |
| `gat_no_edge_features` | 验证距离、风险、载荷边特征的贡献 |
| `gat_fixed_object_ids` | 指针动作与固定对象编号对比 |
| `gat_no_potential_reward` | 验证边界势函数是否抑制库存后推 |
| `transformer_full_horizon` | 与原完整时域 Transformer 基线比较 |
| `pretrain_fixed_scale` | 固定规模训练与混合规模预训练对比 |
| `pretrain_single_distribution` | 单一参数分布与联合域随机化对比 |
| `raw_absolute_features` | 绝对值特征与无量纲特征对比 |
| `solve_zero_shot` | 冻结模型的零样本多轮搜索 |
| `solve_latent_search` | 低维实例条件搜索 |
| `solve_instance_adapter` | 轻量实例 adapter 主动适应 |
| `solve_full_active_search` | 全网络逐实例更新，仅作高成本对照 |

评价指标除成本、风险和加权目标外，还应包括：

- 最终可行率；
- 终端库存清零率；
- 单窗口平均改进幅度；
- 动作修复失败率；
- 每步推理时间；
- 峰值显存；
- 不同偏好下专家路由权重及其偏离输入偏好的幅度；
- 专家策略差异与专家坍缩率；
- 成本—风险 Pareto 前沿的 hypervolume 和偏好覆盖度；
- 对未见时期数和未见算例规模的泛化性能。

---

## 16. 后续实现映射

本文确认后，预计需要修改或新增以下实现；当前尚未执行：

| 文件/模块 | 预计职责 |
|---|---|
| `src/window_graph.py` | 双期窗口、边界摘要、异构图和动作掩码 |
| `src/gat_policy.py` | 偏好中立的关系感知 GAT 与图池化 |
| `src/moe_policy.py` | 成本/风险 MLP 专家、受约束偏好路由、共享指针动作头和双分量价值头 |
| `src/online_adaptation.py` | 零样本、条件搜索、实例 adapter、预算控制和 incumbent 管理 |
| `src/window_operators.py` | 五类窗口算子和局部修复 |
| `src/ppo_improver.py` | 训练与推理流程改为跨窗口图轨迹 |
| `src/solution_utils.py` | 支持同一物理取货节点在多期形成独立服务事件 |
| `src/instance_generator.py` | 多规模、多参数分布和压力场景联合采样 |
| `configs/network_config.json` | GAT、MoE、路由和窗口配置 |
| `run_ablation_experiments.py` | 新消融变体 |

建议保留旧 Transformer 实现作为基线，待 GAT 版本通过可行性测试后再决定是否替换。

---

## 17. 待确认设计决策

在进入代码实现前，需要最终确认以下决策：

1. **窗口宽度固定为 2**：当前文档按两个相邻时期设计；
2. **轮内冻结、轮间解锁**：同一轮 sweep 中提交后的时期不可回滚；下一轮开始时重新解锁；
3. **未来信息边界**：允许未来产生量和剩余能力的聚合摘要，但不允许未来详细路线；
4. **跨期算子粒度**：采用单个服务事件重调度，不移动整条路线；
5. **提前服务语义**：提前收集只处理当期可用库存，下一期若有新增废物则保留下一期访问；
6. **动作选择方式**：使用图节点指针与可行性掩码，不再使用固定 128 维对象编号；
7. **窗口推进方式**：由确定性 patience/步数规则控制，不设为策略动作；
8. **终端保障**：最后窗口必须通过严格库存清零检查，失败轨迹不得作为可行结果返回；
9. **多轮全时域优化**：重复正向遍历所有窗口，轮末只接受可行且全局目标更优的方案；
10. **MoE 位置**：GAT 后使用成本/风险两个 MLP 专家，先融合专家表示，再由共享动作头解码；
11. **偏好控制**：偏好权重主导路由，状态只允许有界残差修正；两个极端偏好采用硬路由；
12. **训练阶段**：共享预训练、冻结 GAT 的极端专家训练、冻结 GAT 的联合校准、最后一层 GAT 小学习率微调；
13. **价值学习**：Critic 分别预测成本和风险价值，PPO 更新时再按偏好组合优势；
14. **泛化表述**：参数化图输入只提供泛化基础，不宣称自动或必然跨规模泛化；
15. **预训练范围**：四阶段训练同时覆盖多规模、多参数分布、多初始解质量和多偏好；
16. **在线默认模式**：首先使用冻结基础模型做零样本多轮求解；
17. **适应范围**：按需只更新实例条件、adapter、路由残差或动作头偏置，不在线更新 GAT 和专家主体；
18. **实例隔离**：实例更新不回写基础模型，最终始终返回历史最佳可行解；
19. **实现策略**：先并行保留 Transformer 基线，再新增 GAT-MoE 通用预训练求解器。

以上十九项确认后，本文可作为后续代码改造、测试和消融实验的设计依据。

---

## 18. 技术依据

本设计主要参考以下原始研究：

1. Ma et al., [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://doi.org/10.1145/3219819.3220007), KDD 2018。其核心启发是共享表示之后由多个专家学习不同任务关系，并由门控决定组合方式。
2. Abdolmaleki et al., [A Distributional View on Multi-Objective Policy Optimization](https://proceedings.mlr.press/v119/abdolmaleki20a.html), ICML 2020。该工作分别学习各目标的动作分布并在偏好层组合，支持本设计保留目标分量、延后融合的思路。
3. Reymond et al., [Pareto Conditioned Networks](https://arxiv.org/abs/2204.05036), AAMAS 2022。该工作说明单一条件策略可以覆盖多个非支配策略，但需要显式条件和完整偏好覆盖训练。
4. Liu et al., [Pareto Set Learning for Multi-Objective Reinforcement Learning](https://doi.org/10.1609/AAAI.V39I18.34068), AAAI 2025。该工作使用偏好生成不同策略参数，说明仅训练少量离散偏好策略并不足以自然获得高质量连续偏好覆盖。
5. Ambadkar et al., [Preference Conditioned Multi-Objective Reinforcement Learning: Decomposed, Diversity-Driven Policy Optimization](https://arxiv.org/abs/2602.07764), 2026。该工作强调过早标量化造成的目标梯度干扰和偏好表示坍缩，支持本设计采用二维奖励、双分量 Critic、端点专家损失和条件多样性正则。
6. Hottung et al., [Efficient Active Search for Combinatorial Optimization Problems](https://arxiv.org/abs/2106.05126), ICLR 2022。该工作表明逐实例只更新少量参数比全模型 active search 更高效，并可改善跨规模求解。
7. Chalumeau et al., [Combinatorial Optimization with Policy Adaptation using Latent Space Search](https://proceedings.neurips.cc/paper_files/paper/2023/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html), NeurIPS 2023。COMPASS 在推理时搜索预训练策略潜空间，不必持续重训主网络，支持本文的低维条件搜索模式。
8. Chen et al., [Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b1efde53be364a73914f58805a001731-Paper-Conference.pdf), NeurIPS 2023。该工作先训练多目标元模型，再以少量步骤适配偏好子问题，支持“广泛预训练 + 少量适配”的整体路线。
9. Joshi et al., [Learning the Travelling Salesperson Problem Requires Rethinking Generalization](https://doi.org/10.1007/s10601-022-09327-y), Constraints 2022。该研究指出小规模训练后的零样本尺寸外推并不自动成立，需要专门设计表示、学习范式和评估协议。
10. Luo et al., [Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization](https://arxiv.org/abs/2310.07985), NeurIPS 2023。该工作展示架构与训练机制可以显著改善跨规模迁移，但其结论依赖特定问题表示，不能直接推导为任意参数化输入都能泛化。
11. Zhou et al., [Omni-Generalizable Neural Methods for Vehicle Routing Problems](https://proceedings.mlr.press/v202/zhou23o.html), ICML 2023。该工作将固定规模、固定分布训练视为泛化瓶颈，并采用面向新规模与新分布的元学习初始化。
12. Drakulic et al., [BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract.html), NeurIPS 2023。BQ-NCO 利用约简 MDP 状态和组合对称性改善尺寸迁移，支持本文的固定窗口、指针动作和尺寸无关表示。
13. Berto et al., [RouteFinder: Towards Foundation Models for Vehicle Routing Problems](https://arxiv.org/abs/2406.15007), TMLR 2025。RouteFinder 使用统一属性表示、混合任务批次、跨变体奖励归一化和高效 adapter，支持本文把“基础求解模型”建立在统一表示与广泛预训练之上，而不是仅增加参数量。
14. Zhou et al., [MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts](https://proceedings.mlr.press/v235/zhou24c.html), ICML 2024。MVMoE 展示了 MoE 统一求解多个 VRP 变体并进行零样本、少样本迁移的可行性，但其结果来自多任务预训练，不能由两个专家数量本身推出。
15. Bi et al., [Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ca70528fb11dc8086c6a623da9f3fee6-Abstract-Conference.html), NeurIPS 2022。AMDKD 使用多分布教师和自适应蒸馏改善未见分布表现，进一步说明跨分布覆盖需要显式训练机制。

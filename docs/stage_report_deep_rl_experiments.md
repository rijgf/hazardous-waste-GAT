# 深度强化学习阶段性实验说明

## 目标

本阶段围绕危险废物收运-处置优化问题，建立一套可复现实验流程，用于后续阶段性成果汇报：

- 小规模实例使用 MILP 精确求解作为最优性基准。
- 大规模实例使用启发式、GA-light 和 PPO 改进器进行扩展性对比。
- 多目标优化采用成本-风险双目标偏好。
- 网络结构参考《众包模式下基于并行生成策略的强化学习路径规划模型》，并结合本文“完整可行解 + 算子改进”的特点进行改造。
- 消融实验验证网络结构、preference token 和 route-arc token。

## 文献启发

### 中文《众包》文献

该文献的核心启发是“多类信息 token 化”：

- 将任务取货点、任务送货点、车辆起点、车辆终点等原始信息分别经过全连接/线性投影，形成多个 token。
- token 输入 Transformer 编码。
- 路径生成阶段通过车辆轮换网络和路径生成网络并行规划多车辆路径。

本文借鉴其 token 化思想，但任务不同：

- 《众包》是从空路径逐步构造路径。
- 本文 PPO 是在完整可行解上执行邻域算子改进。

因此本文额外加入 `route-arc token`，显式描述当前解中已经使用的路径边。

### 英文多目标 DRL 文献

英文文献的核心启发是 preference-driven multi-objective DRL：

- 将多目标偏好纳入神经网络输入。
- 不同偏好对应不同目标权衡。
- 一个训练框架服务多种偏好。

本文采用成本-风险双目标：

```text
preference = (w_cost, w_risk)
w_cost + w_risk = 1
```

训练时随机采样偏好，评估时固定五组偏好：

```text
(1.0, 0.0)
(0.75, 0.25)
(0.5, 0.5)
(0.25, 0.75)
(0.0, 1.0)
```

## 问题定义

所有算法必须遵守 MILP 的目标函数和约束定义。MILP 是问题定义基准，其他算法只是求解方法。

目标函数拆成：

```text
Cost = vehicle fixed cost + distance cost + processing cost
Risk = transport accident risk + coload risk + producer inventory risk + facility inventory risk
Weighted objective_raw = w_cost * Cost + w_risk * Risk
```

统一评价函数位于：

```text
src/solution_utils.py
```

由于 `Cost` 和 `Risk` 的数量级不同，当前实验不再使用裸目标直接比较，而是采用启发式初始解作为实例级参考值进行归一化：

```text
cost_ref = heuristic_cost
risk_ref = heuristic_risk
norm_cost = Cost / cost_ref
norm_risk = Risk / risk_ref
Weighted objective_normalized = w_cost * norm_cost + w_risk * norm_risk
```

表格中同时保留原始 `cost`、`risk`、`weighted_objective_raw`，但算法 gap 和主图使用 `weighted_objective_normalized`。

随机场景生成位于：

```text
src/instance_generator.py
```

## PPO 改进框架

本文当前采用：

```text
随机/配置生成场景
→ 简单启发式生成完整可行解
→ PPO 根据当前解状态选择改进算子
→ repair 修复
→ 用 MILP 目标口径计算改进量奖励
```

奖励函数：

```text
reward = J_old - J_new
J = w_cost * normalized_cost + w_risk * normalized_risk
```

若 repair 失败或动作无效，则给惩罚。

## 状态 Token 定义

当前主模型使用：

- `preference token`：由 `(w_cost, w_risk)` 经过全连接层生成。
- `global solution token`：当前解整体统计信息。
- `vehicle tokens`：每辆车一个 token。
- `producer-waste-period task tokens`：产废点-废物类型-周期相关信息。
- `facility-waste-period tokens`：设施处理能力、处理成本、库存风险等。
- `route-arc tokens`：当前完整解中已经使用的路径边。

`route-arc token` 不是《众包》原文做法，而是本文针对“算子改进完整解”增加的状态表达。

## 网络结构

主模型：

```text
Full Transformer + preference token + route-arc token
```

消融模型：

```text
Hybrid Transformer-FC
No preference token
No route-arc tokens
```

PPO 动作采用分头输出：

```text
operator_head
object_head_1
object_head_2
object_head_3
```

算子集合：

```text
relocate
swap
two_opt
change_facility
delay_or_advance_period
```

## 配置文件

三类配置文件：

```text
configs/model_config.json
configs/algorithm_config.json
configs/network_config.json
```

其中：

- `model_config.json` 控制随机场景规模和参数范围。
- `algorithm_config.json` 控制 MILP、GA、PPO、偏好和实验流程。
- `network_config.json` 控制 Transformer 和 token 使用开关。

## 已完成实验

主实验输出：

```text
outputs/20260707_083004
```

包括：

- `results.csv`
- `results_with_gap.csv`
- `summary.md`
- `training_history.csv`
- `models/`
- `figures/`
- `instances/`
- `solutions/`

消融实验输出：

```text
outputs/ablation_20260707_083702
```

包括：

- `ablation_results.csv`
- `ablation_summary.md`
- `ablation_training_history.csv`
- `models/`
- `figures/`
- `instances/`

## 当前结果解读

主实验中，所有输出解均满足当前统一检查下的可行性要求。

小规模实验中，MILP 在大部分偏好下返回 optimal，可用于计算 `gap_to_milp_percent`。纯风险偏好 `(0, 1)` 在 60 秒内未返回 optimal，因此 gap 统计自动排除了该点。

大规模实验使用 `20P-4W-3F-8V-4T`。由于无精确最优解，当前以 PPO 为基准计算 `gap_to_ppo_percent`。归一化后，PPO 在大规模平均归一化加权目标上优于启发式和 GA-light。

消融实验中，小规模 `hybrid_transformer_fc` 出现了相对启发式的改进；其他变体在当前短训练设置下多与启发式持平。该现象说明后续需要：

- 增加 PPO 训练迭代。
- 加强动作 mask 和对象选择。
- 提高算子有效动作比例。
- 增加多实例训练而非单实例训练。

## 后续建议

1. 将 `algorithm_config.json` 中 PPO 训练迭代从阶段版逐步提高。
2. 对大规模 GA 单独优化 decoder，避免当前 GA-light 过弱。
3. 增加动作有效率统计，判断 PPO 是否频繁选择无效或无改进动作。
4. 加入 sensitivity analysis，优先分析车辆容量、事故概率、风险权重、处理能力。
5. 将输出图表整理成 PPT 使用的中文图题和表题。

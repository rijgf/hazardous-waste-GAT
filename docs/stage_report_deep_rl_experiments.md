# 修复版 PPO-Transformer 实验与复现说明

> 状态：方法、复现协议与正式实验结果均已更新。正式运行 `20260825_131529_507270` 已完成；修复 GA 解码器后的定向重算也已登记到同一 manifest。
>
> 本文描述当前可执行的 PPO-Transformer 基线。GAT-PPO 仍处于设计阶段，见 [统一 GAT-PPO 求解器设计](ppo_transformer_design.md)，不得把本基线写成 GAT 实现或 GAT 泛化结果。

## 1. 本轮修复的目的与结论边界

本轮工作首先修复历史 PPO-Transformer 的时间维度错误，然后才重新进行小规模、大规模及五种成本—风险偏好的实验。历史实现中的 `repair_plan` 会按产生者—废物节点全局去重，并把访问统一重建到最后一期。这会产生两个直接后果：

- 同一产生者—废物节点在不同周期的合法重复服务被删除；
- `delay_or_advance_period` 所做的时期调整在 repair 后被抵消。

因此，旧输出目录中的 PPO 数值不能代表修复后的算法，也不能继续作为正式对比结论。本说明不抄录旧结果；第 10 节只报告修复后的正式运行。

当前实验的模型作用域为 `instance-specific`：每个实例单独训练一个 PPO-Transformer 检查点，同一实例的五种偏好和多次推理重启共享该检查点。小规模与大规模之间仍会重新训练，因而本轮实验验证的是“时间修复后的可执行基线和偏好响应”，不是单冻结检查点的跨实例或跨规模零样本泛化。

## 2. 统一问题与评价口径

所有方法使用同一 `ModelParams`、目标分解和可行性语义。成本与风险定义为：

```text
Cost = vehicle fixed cost + distance cost + processing cost
Risk = transport risk + coload risk
       + producer inventory risk + facility inventory risk
```

为避免成本与风险数量级直接支配加权目标，每个实例使用同一严格可行的启发式初始解建立参考尺度：

```text
cost_ref = initial_solution_cost
risk_ref = initial_solution_risk
normalized_cost = Cost / cost_ref
normalized_risk = Risk / risk_ref
J(w) = w_cost * normalized_cost + w_risk * normalized_risk
```

评价入口为 `src/solution_utils.py::evaluate_solution`。主表保留原始成本、原始风险、两类归一化指标、风险分项、运行时间及可行性；算法 gap 只对 MILP 已证明 `optimal` 的实验单元计算。

固定评估偏好为：

```text
(1.00, 0.00)
(0.75, 0.25)
(0.50, 0.50)
(0.25, 0.75)
(0.00, 1.00)
```

## 3. 跨期方案与 repair

修复实现位于 `src/operators.py`。路线方案仍以 `(vehicle, period)` 为键，但 repair 中的服务访问身份改为：

```text
(pickup_node, service_period)
```

这意味着同一物理取货实体可以在不同周期重复出现，但同一周期内的重复访问会被去重。repair 的主要步骤是：

1. 按周期读取现有服务访问，并保留合法的跨期重复服务；
2. 为尚无末期服务的取货实体补充末期访问，以保证末期新产生量有机会清运；
3. 按已有跨期访问传播库存，计算每个 `(node, period)` 真正可收取的当期数量，而不是使用全时域总量；
4. 在车辆容量、处理技术、废物兼容性和允许弧约束下确定性重建各期路线；
5. 将重建方案转换为完整解，并通过统一严格验证器后才提交。

提前服务只收集该期已经产生和累积的数量，不会预先收集未来产生量。若后续仍有新产生量，末期服务仍会保留。repair 本身不依赖随机遍历顺序，便于基于 plan hash 进行局部回放。

## 4. 八类邻域算子

当前动作词表在 `src/operators.py::OPERATORS` 中定义，共八类：

| 算子 | 作用 |
|---|---|
| `relocate` | 在同一现有计划中移动一个服务访问及其插入位置 |
| `swap` | 交换两个服务访问 |
| `two_opt` | 反转单条路线中的一个访问片段 |
| `change_facility` | 在处理技术允许时更换路线起终点设施 |
| `move_visit_period` | 将一个访问移动到目标周期和目标车辆 |
| `add_early_service` | 为节点增加一个末期之前的服务访问，降低生产端库存暴露 |
| `split_by_waste_type` | 把指定废物类型的整项服务访问拆到同周期另一车辆 |
| `split_route` | 把单个完整服务访问拆到同周期另一车辆 |

`split_by_waste_type` 和 `split_route` 移动的是完整服务访问，不把同一 `node-period` 的收运量拆给多辆车；这与当前 MILP 中“同一节点同一期至多访问一次且访问即全量收取”的定义保持一致。每个候选动作都必须经过 repair 和严格验证。失败时返回原方案，并记录明确的 `no_change` 或 `repair_failed` 原因。

## 5. PPO-Transformer 方法

### 5.1 状态表示

状态编码位于 `src/ppo_improver.py::StateEncoder`，包含：

- 全局方案 token；
- 连续成本—风险偏好 token；
- period token；
- vehicle-period token；
- producer-waste-period token；
- facility-waste-period token；
- 当前路线 arc token。

这些 token 经分类型线性投影后进入 Transformer 编码器。当前 Actor 仍使用四个分头输出：

```text
operator_head
object1_head
object2_head
object3_head
```

这是恢复基线的固定对象槽实现，不是 GAT 设计中的动态指针解码器。配置会在实例所需对象数超过 `object_count` 时显式失败，避免静默取模掩盖规模问题。

### 5.2 成本/风险双 Critic

历史单标量 Critic 已改为双分量输出：

```text
V(s, w) = [V_cost(s, w), V_risk(s, w)]
```

每一步分别构造归一化成本改善和风险改善：

```text
r_vector = [cost_old / cost_scale - cost_new / cost_scale,
            risk_old / risk_scale - risk_new / risk_scale]
```

两个分量分别计算 GAE，再按当前偏好合成为 Actor 优势：

```text
A(w) = w_cost * A_cost + w_risk * A_risk
```

Critic 的价值损失同时拟合成本和风险两列回报，避免稀疏的风险改善信号被单一标量价值目标完全掩盖。

### 5.3 端点增强偏好采样

训练偏好由 `PPOImprover._sample_preference` 生成。默认配置为：

```text
10%: (1, 0)
10%: (0, 1)
80%: w_risk ~ Beta(0.5, 0.5), w_cost = 1 - w_risk
```

Beta 分布加强端点邻域覆盖，两个精确端点又分别获得固定概率。这样纯成本和纯风险不再是连续均匀采样下概率为零的事件。

### 5.4 推理与 incumbent

每个偏好默认执行多次独立推理重启。每一步包含一个贪心动作和若干策略采样动作，选择当前候选中加权目标最好的可行方案继续搜索，同时单独维护历史最佳严格可行 incumbent。最终只返回该 incumbent。

推理轨迹记录：

- step、算子和三个对象编号；
- evaluation seed 与 operator seed；
- 是否接受候选；
- 候选成本、风险和归一化加权目标；
- 动作前、动作后及 incumbent 的 plan SHA-256。

## 6. 严格可行性验证

统一验证入口为 `src/solution_utils.py::validate_solution`，`enhanced_check_solution` 是其兼容别名。该验证器首先拒绝空的或缺少 `raw` 决策变量的伪解，并在原约束检查基础上增加：

- 数值有限性与非负性；
- 每期生产者库存的期初、收运和期末质量守恒；
- 每期设施接收、处理和期末库存质量守恒；
- 收运量与设施接收量按废物类型守恒；
- 处理能力和处理技术检查；
- 车辆容量、兼容性、节点访问、末期生产者库存清零和末期设施库存清零。

启发式初始解、GA、PPO-Transformer 最终解和 MILP 解均通过同一评价入口。PPO 候选只有在 repair 内通过严格验证后才进入搜索状态。历史“缺失变量按零读取，空解也可能通过”的评价漏洞不再作为正式实验口径。

## 7. 实验矩阵

实验入口为：

```powershell
py run_experiments.py
```

快速冒烟测试为：

```powershell
py run_experiments.py --quick
```

正式配置来自：

```text
configs/model_config.json
configs/algorithm_config.json
configs/network_config.json
```

对每个规模和实例，运行顺序为：

```text
生成并保存实例
→ 生成、严格验证并保存 canonical 初始方案
→ 按实例训练一个 PPO-Transformer 检查点
→ 对五种偏好分别运行 Heuristic、GA、PPO-Transformer
→ 小规模额外运行 MILP
```

正式配置下 PPO-Transformer 对每个偏好使用多个独立 evaluation restart。结果表中的每个 restart 保留独立记录，汇总时不得把训练时间混入推理时间；训练时间另存为 `preparation_seconds`。

## 8. Manifest、seed、hash 与产物

### 8.1 Seed 派生

唯一人工入口为 `configs/algorithm_config.json` 中的 `random_seed`。`src/reproducibility.py::derive_seed` 使用与调用顺序无关的 `sha256-seed-v1` 方案，分别派生：

- instance seed；
- initial-solution seed；
- PPO training seed；
- GA evaluation seed；
- PPO evaluation seed，命名空间包含实例、偏好和 restart。

禁止使用 Python 进程随机化的 `hash()` 派生 seed。实验入口同时启用 PyTorch 确定性算法提示、关闭 cuDNN benchmark，并设置 cuDNN deterministic。即便如此，跨 GPU、驱动或依赖版本仍可能存在浮点差异，因此环境快照和最终 hash 必须与 seed 一起报告。

### 8.2 Manifest

每次运行在 `outputs/<run_id>/manifest.json` 保存：

- schema、run id、时间、原始命令和 quick/full 模式；
- master seed、seed 派生方案和 `training_scope=instance-specific`；
- Git commit、分支、工作区状态及 tracked diff hash；
- Python、平台、PyTorch、NumPy、SciPy、pandas 等环境版本；
- algorithm/network config SHA-256；
- 每个实例的 seed、文件、维度和文件 SHA-256；
- canonical 初始方案、初始方案 seed、plan SHA-256 和目标参考值；
- PPO training seed、checkpoint 文件、checkpoint SHA-256 和训练时间；
- 每个偏好/restart 的 evaluation seed、trace 文件和 solution 文件；
- 最终 CSV、summary 的文件 SHA-256。

长实验每完成一个偏好单元就增量写入 `results.csv`、`results_with_gap.csv` 和 manifest，意外中断时已经完成的单元仍可检查。

### 8.3 目录结构

```text
outputs/<run_id>/
├── manifest.json
├── algorithm_config.json
├── network_config.json
├── model_config_small.json
├── model_config_large.json
├── instances/
├── initial_solutions/
├── ppo_transformer/<instance_id>/
├── solutions/
├── traces/<instance_id>/<preference>/restart_<n>.json
├── results.csv
├── results_with_gap.csv
├── summary.md
└── figures/
```

## 9. 单实验单元 replay

`replay_experiment.py` 从 manifest 定位实例、canonical 初始方案、训练检查点、目标参考尺度和指定 restart 的 evaluation seed。示例：

```powershell
py replay_experiment.py `
  --run-dir outputs/<run_id> `
  --instance small_0 `
  --cost-weight 0.0 `
  --risk-weight 1.0 `
  --restart 0
```

回放输出包括指标、结果 plan SHA-256 和完整 step trace。首先比较以下字段：

```text
initial_solution_sha256
checkpoint_sha256
evaluation_seed
result_plan_sha256
```

前三项不一致说明回放输入不相同；前三项一致而结果 hash 不一致时，再结合 manifest 的环境和 Git diff 快照定位数值或实现差异。

可选 `--steps` 只用于改变搜索步数的诊断实验；若与原配置不同，所得结果不应被称为原实验的精确回放。

## 10. 正式实验结果

### 10.1 运行边界与复现标识

本节只使用正式运行 `20260825_131529_507270`。原始主流程于 2026-08-25 13:15:29 开始，13:41:29 完成，manifest 已写入 `finished_at`。随后修复 GA 固定首设施解码器，并在不重跑 PPO、MILP、启发式、实例或初始解的前提下，使用原实例和原 GA evaluation seed 定向重算 10 个 GA 单元。当前可审计时间线为：14:39:38 写入首个 GA 修复记录；14:57:47 对全部 55 行回填方案文件与 canonical plan hash，并从已保存路线无损重建 5 个 MILP plan；15:15:10 在 append-only 审计语义和完整严格校验下再次确定性复算 GA，写入 `posthoc_updates[2]`，并通过 `supersedes_posthoc_update.index=0` 指明它取代首个 GA 记录而不删除历史。14:34 曾执行过一次预备刷新，但当时旧 `--force` 语义会覆盖同类记录，因此它不在现存 manifest 中，也不作为可审计最终结果。当前 `last_updated_at` 为 15:15:10，结果文件仍包含 55 行：小规模 30 行、大规模 25 行。产物位于：

- [manifest.json](../outputs/20260825_131529_507270/manifest.json)：完整 seed、环境、Git 状态和文件 hash；
- [results.csv](../outputs/20260825_131529_507270/results.csv)：全部逐次结果；
- [results_with_gap.csv](../outputs/20260825_131529_507270/results_with_gap.csv)：只对已证明 optimal 的 MILP 单元计算 gap；
- [summary.md](../outputs/20260825_131529_507270/summary.md)：方法级汇总；
- `outputs/20260825_131529_507270/initial_solutions/`：canonical 初始方案；
- `outputs/20260825_131529_507270/traces/`：30 条 PPO 推理轨迹；
- `outputs/20260825_131529_507270/solutions/`：方案和最终 plan hash。

GA 定向重算由 `refresh_ga_results.py` 完成，并更新了两个 CSV、summary、10 个 GA solution、artifact hashes 和逐单元 plan hash。最终复算所用源文件 SHA-256 保存在 `manifest.posthoc_updates[2].source_files_sha256`。`posthoc_updates[1]` 记录身份回填的 55 个单元和 5 个重建 MILP plan；当前 55/55 行都有 `solution_file` 与 `result_plan_sha256`。因此不能把更新后的 GA 数值与 13:41 时尚未修复的旧 GA 单元混用，也不能把身份回填误写成算法重新求解。

本次只生成了每个规模各一个实例，因此结果是两个固定实例上的算法比较，不是多实例统计检验。规模、训练和初始解标识如下。

运行时 Git 基准提交为 `57e9c54afabffdc6fbf3fc8278666d43287648c1`，工作区为 dirty；manifest 记录的 tracked diff SHA-256 为 `28a5465996b09586bc72bd769be8745fc702a03807e7031dacaf9ac2de2b16ed`。算法配置 hash 为 `ac520ca06dce7e1f1b3974176f38633a83b003c26bfa086130d2ebbb7f109b4d`，网络配置 hash 为 `84576a0b84afa7fa7613a1decd01f225adc347ec40ae0a2c6fe08a7877021988`。运行环境为 Python 3.11.5、PyTorch 2.5.1+cu121 和 NVIDIA GeForce RTX 4070 Ti；复现时不能只核对 base commit，必须同时核对 manifest 中的工作区 diff 与配置 hash。

| 项目 | 小规模 `small_0` | 大规模 `large_0` |
|---|---:|---:|
| 维度（产生者/废物/设施/车辆/周期） | 3/2/2/3/2 | 20/4/3/8/4 |
| instance seed | 3723524230 | 3720328856 |
| initial-solution seed | 3949079119 | 1294583929 |
| initial plan SHA-256 | `5fdfb1130823c21856db79670ab76cbbbac336b68ac2548cdf8c3b81d1a791f3` | `0ec3560fdcbf0063d01e489770c4fa5e750271ecd25a70b69cc30c400c1a9385` |
| PPO training seed | 3865117416 | 3358807615 |
| checkpoint SHA-256 | `30fc3be248b3b46e0c2dc009740c9cb5c25467b3930b4aeedab052746328d717` | `17e313ded6e960dfb655ac8bdee657d8451c617e5f189093fdaed9ea36386370` |
| PPO 训练时间（秒） | 185.028 | 375.185 |
| 每个偏好的推理重启数 | 3 | 3 |

master seed 为 42，派生方案为 `sha256-seed-v1`。三个 PPO restart 的 evaluation seed 和定向重算沿用的 GA evaluation seed 如下；PPO seed 顺序为 restart 0/1/2。

| 偏好 `(w_cost,w_risk)` | small PPO seeds | small GA seed | large PPO seeds | large GA seed |
|---|---|---:|---|---:|
| (1.00, 0.00) | 2571384110 / 1602506532 / 3447525709 | 542777990 | 3346291864 / 1772781090 / 1888076806 | 3299486188 |
| (0.75, 0.25) | 3229999428 / 4248913066 / 2951447865 | 3539073322 | 1566233914 / 4176533620 / 291534844 | 1432058215 |
| (0.50, 0.50) | 4084462614 / 794579791 / 1716606784 | 1107549096 | 3364835019 / 2630032059 / 1989548917 | 1809596992 |
| (0.25, 0.75) | 2790166943 / 14408268 / 1540054241 | 3693694119 | 3083789735 / 828343201 / 2191856209 | 220786744 |
| (0.00, 1.00) | 1809039277 / 4194498594 / 1142480955 | 3248652927 | 1489647236 / 3711604445 / 1899429313 | 248456519 |

### 10.2 小规模：PPO 与 MILP

启发式初始解为 `Cost=1384.992, Risk=10.8765`，所以五个偏好下的归一化加权目标均为 1。修复固定首设施解码器并定向重算后，小规模五个 GA 单元都返回严格可行的非回退解。下表中的“PPO 相对 GA”以 PPO 三重启的平均 `J` 计算；MILP gap 只在已证明 optimal 的前三个偏好下给出。

| 偏好 | GA Cost | GA Risk | GA `J` | PPO 平均 `J` 相对 GA | GA 相对 optimal MILP gap |
|---|---:|---:|---:|---:|---:|
| (1.00, 0.00) | 1165.515 | 12.2166 | 0.841532 | -2.32% | 2.372% |
| (0.75, 0.25) | 1165.515 | 9.7517 | 0.855296 | -0.61% | 1.860% |
| (0.50, 0.50) | 1165.515 | 9.7517 | 0.869059 | -6.97% | 7.487% |
| (0.25, 0.75) | 1377.874 | 8.8623 | 0.859826 | -22.92% | — |
| (0.00, 1.00) | 1551.230 | 9.2081 | 0.846603 | -59.72% | — |

PPO 列报告三次 restart 的均值 ± 样本标准差，方括号内为三次中最小的归一化加权目标 `J`。所有 PPO 和 MILP 记录均通过严格可行性验证。

| 偏好 | PPO Cost | PPO Risk | PPO `J` [best] | MILP Cost | MILP Risk | MILP `J` | MILP 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| (1.00, 0.00) | 1138.513 ± 0.000 | 10.7957 ± 0.0000 | 0.822036 ± 0.000000 [0.822036] | 1138.513 | 10.5697 | 0.822036 | `optimal`，6.34 s |
| (0.75, 0.25) | 1156.515 ± 15.590 | 9.7373 ± 0.0249 | 0.850091 ± 0.009015 [0.839681] | 1138.513 | 9.7086 | 0.839681 | `optimal`，46.22 s |
| (0.50, 0.50) | 1257.611 ± 0.000 | 7.7118 ± 0.0000 | 0.808528 ± 0.000000 [0.808528] | 1257.611 | 7.7118 | 0.808528 | `optimal`，176.34 s |
| (0.25, 0.75) | 1909.944 ± 0.000 | 4.6119 ± 0.0000 | 0.662776 ± 0.000000 [0.662776] | 1909.944 | 4.6119 | 0.662776 | `solver_status_1`，180.05 s，限时可行 incumbent |
| (0.00, 1.00) | 2299.022 ± 84.677 | 3.7087 ± 0.0843 | 0.340978 ± 0.007747 [0.332033] | 2552.649 | 3.9690 | 0.364915 | `solver_status_1`，180.05 s，限时可行 incumbent |

前三个偏好下 MILP 已证明 `optimal`。PPO 的 best restart 在数值精度内分别达到 0%、0%、0% gap；三重启均值 gap 分别约为 0%、1.240%、0%。后两个偏好的 MILP 在 180 秒结束时没有最优性证明，因此 `results_with_gap.csv` 将 gap 留空。尤其是纯风险偏好下，PPO best 的风险 3.6114 比 MILP 限时 incumbent 的 3.9690 低 9.01%；这说明 PPO 找到了更好的可行 incumbent，不能据此把任一结果称为精确最优。

### 10.3 大规模：PPO 与启发式、GA

大规模没有运行 MILP，不存在精确最优基准。启发式初始解为 `Cost=29336.011, Risk=2110.0161`，各偏好 `J=1`。下表的 PPO 数值仍为三重启均值 ± 样本标准差，方括号为 best restart；“相对 GA”只表示同一归一化加权目标的下降比例。

| 偏好 | GA Cost | GA Risk | GA `J` | PPO Cost | PPO Risk | PPO `J` [best] | PPO `J` 相对 GA |
|---|---:|---:|---:|---:|---:|---:|---:|
| (1.00, 0.00) | 20959.942 | 1271.9923 | 0.714478 | 16210.584 ± 580.886 | 1025.7687 ± 28.8928 | 0.552583 ± 0.019801 [0.534324] | -22.66% |
| (0.75, 0.25) | 20017.333 | 1131.4928 | 0.645822 | 16440.549 ± 243.051 | 810.5894 ± 69.3002 | 0.516357 ± 0.006494 [0.509188] | -20.05% |
| (0.50, 0.50) | 22727.467 | 1201.2759 | 0.672025 | 17324.625 ± 496.128 | 668.8390 ± 10.7633 | 0.453771 ± 0.009650 [0.443211] | -32.48% |
| (0.25, 0.75) | 24459.733 | 1309.8811 | 0.674039 | 21150.130 ± 535.843 | 629.1773 ± 33.1046 | 0.403880 ± 0.012736 [0.394718] | -40.08% |
| (0.00, 1.00) | 26463.505 | 1223.8862 | 0.580036 | 36399.345 ± 2803.890 | 560.2705 ± 22.0056 | 0.265529 ± 0.010429 [0.258509] | -54.22% |

全部 15 个大规模 PPO restart、五个定向重算后的 GA 结果和五个启发式结果均为严格可行解。PPO 的平均推理时间为每 restart 约 21.79 秒；最终严格校验下 GA 平均约 13.66 秒。PPO 训练时间不计入该推理时间，而是作为一次性的实例准备成本单列。

修复后的 GA 已不再退回启发式，但它在两个固定实例上的偏好响应仍不单调：例如 small 的 `(0.25,0.75)` 风险 8.8623 低于纯风险解的 9.2081，large 的中间偏好风险也有反复。因此，本轮只能说 GA 解码器可行性问题已修复，不能据此认为 GA 已充分收敛；相较之下，PPO 的 best-restart 成本—风险序列在五个偏好上呈单调权衡。

### 10.4 更新后的方法汇总与 gap

下表与当前 [summary.md](../outputs/20260825_131529_507270/summary.md) 一致。`J`、Cost 和 Risk 是各方法在五个偏好上的描述性均值；PPO 每个偏好含三个 restart，其他方法每个偏好一条记录。

| 规模 | 方法 | 平均 `J` | 平均 Cost | 平均 Risk | 平均运行时间（秒） | 可行率 |
|---|---|---:|---:|---:|---:|---:|
| small | Heuristic | 1.000000 | 1384.992 | 10.8765 | 0.0002 | 100% |
| small | GA | 0.854463 | 1285.130 | 9.9581 | 0.8196 | 100% |
| small | PPO-Transformer | 0.696882 | 1552.321 | 7.3131 | 4.8041 | 100% |
| small | MILP | 0.699587 | 1599.446 | 7.3142 | 117.8000 | 100% |
| large | Heuristic | 1.000000 | 29336.011 | 2110.0161 | 0.0120 | 100% |
| large | GA | 0.657280 | 22925.596 | 1227.7057 | 13.6603 | 100% |
| large | PPO-Transformer | 0.438424 | 21505.047 | 738.9290 | 21.7876 | 100% |

这个跨偏好方法均值不能解释为单一优化问题上的排序，尤其不能用 PPO 的 0.696882 与 MILP 的 0.699587 推断 PPO 优于精确方法：MILP 后两个偏好只是限时 incumbent，且不同偏好的权重目标并不相同。对前三个已证明 optimal 的小规模单元，`results_with_gap.csv` 中的平均 gap 为：GA 3.9059%、Heuristic 21.4745%、PPO-Transformer 0.4132%、MILP 0%。后两个偏好仍被排除在 gap 汇总之外。

### 10.5 风险分项与偏好响应

下表选取每个偏好下 `J` 最小的 PPO restart。该运行中设施库存风险均为 0，因此不再单列。

| 规模 | 偏好 | restart | Cost | Risk | 运输风险 | 混装风险 | 生产端库存风险 |
|---|---|---:|---:|---:|---:|---:|---:|
| small | (1.00, 0.00) | 0 | 1138.513 | 10.7957 | 6.0974 | 0.6024 | 4.0959 |
| small | (0.75, 0.25) | 0 | 1138.513 | 9.7086 | 5.0736 | 0.5391 | 4.0959 |
| small | (0.50, 0.50) | 0 | 1257.611 | 7.7118 | 3.6159 | 0.0000 | 4.0959 |
| small | (0.25, 0.75) | 0 | 1909.944 | 4.6119 | 3.6114 | 0.0000 | 1.0005 |
| small | (0.00, 1.00) | 2 | 2396.798 | 3.6114 | 3.6114 | 0.0000 | 0.0000 |
| large | (1.00, 0.00) | 0 | 15674.935 | 998.1422 | 582.9704 | 168.1727 | 246.9991 |
| large | (0.75, 0.25) | 0 | 16314.915 | 777.1907 | 418.2603 | 111.9313 | 246.9991 |
| large | (0.50, 0.50) | 2 | 16843.867 | 658.8551 | 343.3276 | 70.9287 | 244.5988 |
| large | (0.25, 0.75) | 0 | 21449.313 | 596.2287 | 296.6522 | 67.8576 | 231.7189 |
| large | (0.00, 1.00) | 2 | 33181.330 | 545.4586 | 301.3185 | 53.8185 | 190.3215 |

按 best restart 观察，两个规模都呈现清晰的成本—风险响应：风险权重由 0 增至 1 时，成本逐步升高、总风险逐步降低。纯风险方案相对各自初始解的风险下降为小规模 66.80%、大规模 74.15%。小规模风险改善先消除混装风险，再把生产端库存风险从 4.0959 降至 0；大规模则同时降低运输、混装和生产端库存风险。这说明端点偏好采样、双 Critic 与跨期邻域修复后，风险侧已经能产生方向正确且幅度显著的方案变化。

### 10.6 结论边界

- 本轮结论支持“在这两个固定实例上，修复版 PPO-Transformer 对五种偏好均能生成严格可行解，并在风险权重提高时作出合理响应”。
- GA 数值来自 15:15 的最终定向复算，十个单元均为严格可行且未触发 fallback；PPO、MILP、启发式、实例和初始解仍是 13:41 完成的原始主流程产物，14:57 的身份回填没有重新求解它们。
- 小规模只有前三个偏好具有 MILP 最优性证明；后两个偏好只能与 180 秒限时 incumbent 比较。大规模没有 MILP，不能宣称接近全局最优。
- 两个规模分别训练了独立 checkpoint，`training_scope=instance-specific`。本轮不是用同一个大规模权重零样本求解小规模问题，也不构成跨规模泛化证据。
- 原正式 checkpoint 使用 PyTorch deterministic algorithms（`warn_only`）、`cudnn_deterministic=true` 并关闭 cuDNN benchmark；其纯风险单元已用保存的 checkpoint、实例、配置、初始解和 seed 精确回放到相同 plan hash。当前代码又在训练阶段关闭 flash/memory-efficient SDP、启用 math SDP；两个独立 quick 运行得到逐字节相同的大小规模 checkpoint，且除耗时和 run id 外 14 行结果完全一致。GPU 型号、驱动、CUDA、cuDNN 或 PyTorch 版本改变时仍不能保证跨硬件位级一致，应同时比较可行性、指标容差和轨迹 hash。
- 当前只有每个规模一个实例。要形成统计性结论，仍需扩展到多个独立 instance seed，并把“独立实例”与“同实例多次 inference restart”分开汇总。

## 11. 已知边界与后续工作

- 当前 PPO-Transformer 仍按实例训练，不能据此宣称跨实例泛化；
- 固定对象槽 Actor 只是恢复基线，未来 GAT 应改为动态候选与指针动作；
- 当前 repair 以严格可行为首要目标，后续可进一步减少全局重建对局部动作的扰动；
- 后续训练预算调整应建立新的 run id，并与本轮固定结果分开报告，避免一边改方法一边选择性报告结果；
- GAT 的实现、混合规模训练、冻结单检查点和 OOD 评估继续遵循 `ppo_transformer_design.md`，与本轮基线实验分开报告。

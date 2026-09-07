# 正式补充实验稳定事实与来源审计

> 用途：作为最终论文回填前的内部事实清单。本文只记录可由冻结协议、冻结模型记录、锁定 schedule、初始化环境快照及其哈希绑定源码支持的事实；不记录正式运行的中间完成量或中间结果。
>
> 证据根目录均相对于仓库根目录。正式运行目录为 `outputs/supplementary_experiment_v1/`。最终结果、批次计时和失败计数只有在 `manifest.json` 达到最终验证状态后方可回填。

## 1. 证据文件与身份绑定

- 冻结协议：`outputs/supplementary_experiment_v1/protocol_config.json`
  - `protocol_id = supplementary-experiment-v1`
  - 规范化协议 SHA256：`24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba`
  - 协议文件原始字节 SHA256：`74edd93e5866d583eef7ffa17a004c9e92070a165260712761d549a691253257`
  - 原始字节哈希与规范化 JSON 哈希因序列化形式不同而不同；跨文件协议绑定使用规范化哈希。
- 运行清单与初始化环境快照：`outputs/supplementary_experiment_v1/manifest.json`
- 锁定任务表：`outputs/supplementary_experiment_v1/schedule.json`
  - 原始字节 SHA256：`7312dae1eaccf33b8f28b1421822be32a3e7f6310dc2751a90696db69d3c286a`
  - schedule 记录的协议哈希与上述规范化协议哈希一致。
- 冻结模型记录：
  - `outputs/supplementary_experiment_v1/models/Train-S.record.json`
  - `outputs/supplementary_experiment_v1/models/Train-L.record.json`
- 初始化时源代码包 SHA256：`50ff1564043fcec7bc573184f4f7aed02b6df99883440795c3bbb62e929fff6e`。该值同时出现在 manifest 和两份模型记录中。
- 初始化时 Git 提交为 `0a71b0bb1b2445ce90989d453cfba357d0d12458`，但工作树记录为 `dirty=true`；复现和论文溯源必须以协议、source bundle 与逐文件哈希为准，不能只引用 Git commit。

## 2. 协议、规模与任务计数

- 根种子：`42`；种子方案：`sha256-seed-v1`，训练实例、训练算法、测试实例、初始解和各求解方法使用分离的命名空间。
- 两份固定模型：
  - Train-S：训练规模为 Test-1，检查点为 `models/small_model.pt`。
  - Train-L：训练规模为 Test-4，检查点为 `models/large_model.pt`。
- 规模五元组顺序为“产废节点数/废物类型数/处理处置节点数/车辆数/周期数”：
  - Test-1：`3/2/2/3/2`，50 个测试实例；
  - Test-2：`6/2/2/4/3`，50 个测试实例；
  - Test-3：`10/3/3/5/3`，50 个测试实例；
  - Test-4：`20/4/3/8/4`，50 个测试实例。
- 五组固定偏好权重：`(1,0)`、`(0.75,0.25)`、`(0.5,0.5)`、`(0.25,0.75)`、`(0,1)`。
- 重启与方法范围：
  - PPO：Train-S、Train-L × Test-1—Test-4，每个实例—偏好 3 次重启，共 6000 个 cell；
  - GA：Test-1—Test-4，每个实例—偏好 3 次重启，共 3000 个 cell；
  - Heuristic：Test-1 和 Test-4，每个实例—偏好 1 次，共 500 个 cell；
  - MILP：仅 Test-1，每个实例—偏好 1 次，共 250 个 cell；
  - 合计 9750 个 cell。
- schedule 中 cell ID 与 logical ID 均为 9750 个唯一值；PPO/GA 的重启索引为 `0,1,2`，Heuristic/MILP 为 `0`。

来源：`outputs/supplementary_experiment_v1/protocol_config.json`、`outputs/supplementary_experiment_v1/schedule.json`、`outputs/supplementary_experiment_v1/manifest.json`。

## 3. 四种规模共同的实例生成参数

- 废物产生量：`U[0.5, 2.0]`。
- 坐标：`U[0,100]`；距离按欧氏距离生成。
- 车辆容量系数：`1.8`；产废节点容量系数：`2.5`；处置节点容量系数：`3.0`；处理能力系数：`2.0`。
- 事故概率：`U[0.0005,0.003]`。
- 车辆固定成本：`50`；单位距离成本：`3`；处理处置成本：`U[4,20]`。
- 废物后果系数：`U[1,4]`。
- 产废端库存风险：`U[0.2,1.0]`；处置端库存风险：`U[0.1,0.8]`；共载风险：`U[0.05,0.5]`。
- 兼容概率：`1.0`；技术可用概率：`0.85`，并修复保证每类废物至少有一个技术可用的处置节点。
- 同类废物的共载风险对角项为 `0`。
- 产废端与处置端初始库存均为 `0`；`require_terminal_clear = true`。
- 四种测试规模使用同一个 `generator_profile`。Test-2、Test-3 没有另行改变非规模生成参数。
- 冻结生成配置中没有单独的“库存成本”参数；库存项是风险系数，不得写成库存成本。

来源：`outputs/supplementary_experiment_v1/protocol_config.json`、`src/instance_generator.py`。`src/instance_generator.py` 的文件哈希由 manifest 的初始化 source bundle 锁定。

## 4. 算法与网络参数

### 4.1 PPO-Transformer

- 固定训练迭代：20；每次迭代 24 个并行 episode，每个 episode 24 步。
- 由冻结配置推导：每次迭代 576 个轨迹步，每模型共 11520 个轨迹步。以上两个数是配置乘积，不是另一次运行测量。
- update epochs：24；折扣因子 `0.95`；GAE `lambda=0.9`；clip `0.2`。
- 学习率 `0.0003`；熵系数 `0.02`；价值损失系数 `0.5`。
- repair failure penalty：`0.2`；no-change penalty：`0.01`。
- 推理步数：60；每步候选数：32；每个正式 PPO cell 设计为 3 次重启。
- 偏好采样：两个端点各以 `0.1` 概率抽取，非端点使用浓度参数 `0.5` 的 Beta 分布。
- 训练固定完成 20 次迭代；没有验证集、验证早停或基于验证集的检查点选择。保存的是最后一次迭代结束后的最终模型状态。

### 4.2 GA

- Test-1/Test-2：种群 30、60 代、交叉率 `0.85`、变异率 `0.20`、精英数 2、锦标赛规模 3。
- Test-3/Test-4：种群 20、40 代、交叉率 `0.90`、变异率 `0.25`、精英数 4、锦标赛规模 3。
- 每个实例—偏好 3 次重启；冻结配置为固定代数，没有早停参数。

### 4.3 MILP

- 时限：180 秒。
- 调用 `scipy.optimize.milp`；`mip_rel_gap=0.0`。
- 只有求解器明确返回成功的 cell 才标记为 `optimal`。限时可行 incumbent 不是已证明最优解，不能作为 `J*` 或 Gap 分母。
- 冻结配置和调用代码没有设置 MILP 线程数；环境记录也没有底层 HiGHS 的精确版本，不得补写或推测。

### 4.4 网络结构与容量预检

- 架构：`full_transformer`；嵌入维度 48；Transformer 层数 1；注意力头数 4；前馈层维度 96；dropout `0`。
- 使用 preference token；不把偏好拼接到 global 特征。
- 使用 route-arc token 和 facility token。
- `max_tokens=896`；`operator_count=8`；`object_count=512`。
- Test-1—Test-4 的预检需求依次为 48/12、113/36、242/90、758/320（token/object），均不超过 896/512 的网络上限。

来源：`outputs/supplementary_experiment_v1/protocol_config.json`、`src/ppo_improver.py`、`src/hazardous_waste_model.py`、`src/supplementary_experiment.py`。上述源码文件均属于 manifest 锁定的 source bundle。

## 5. 两份模型的冻结训练记录

### 5.1 Train-S

- 训练规模：Test-1。
- 训练实例生成种子：`4247481937`；训练算法种子：`3280000028`。
- 训练耗时：`778.8089902 s`，约 `12.9802 min`。
- 训练实例文件 SHA256：`45fd7055c32c424dd792e70cfe77010431d3125703f818fbf559f0b96779f0bc`。
- 训练实例规范化 SHA256：`05b8c4195311b90f055887034a9806f1f8da8f6ee98a6ef79cc9c182cee2a934`。
- 检查点 SHA256：`b489641a0420eae4df2bf1cbf90bb468da4f26a5091382e137d30dfdfe22a4c6`。
- policy-state SHA256：`e955092df13012d7ffaf98fa933097e90bf3a04a91b55a6cdaffdc2cda020e83`。
- history SHA256：`bd80961d5c9aa805d5d6a4831b76b266a3c1af4b2f99d19d5e3fbb7231c39d7e`。
- 冻结时间：`2026-09-04T17:35:57.824245+08:00`。
- 模型记录文件 SHA256：`86e5983b1061ce9ee4c5c26c112927ede2b75ff35ee736fa55095e95163f8414`。

### 5.2 Train-L

- 训练规模：Test-4。
- 训练实例生成种子：`4057074506`；训练算法种子：`3416884852`。
- 训练耗时：`1233.1614100 s`，约 `20.5527 min`。
- 训练实例文件 SHA256：`73d040d129ef1349536f05045ea5f1093c2eff1a85073520472e574c36b37def`。
- 训练实例规范化 SHA256：`46b6f19488c956dc8aebfdd43edd0736c01b681e403642f113122547ea570a4f`。
- 检查点 SHA256：`1b5fc32e6336d85054e2b1d41c9bf54e709f61f10b31b98c35f220e901be7fef`。
- policy-state SHA256：`f3f4ce573cb34b6c64e4eac245e3e93a329e0a12214ab5dac95718e2c56c7dc3`。
- history SHA256：`7a7b2b983adbfe57aa21532fe369840152cd1c563f9ffefb4f5273ec6ba6aee6`。
- 冻结时间：`2026-09-04T17:56:31.035713+08:00`。
- 模型记录文件 SHA256：`91f989ae46c5e8f6d439176b01b4f7ba9941012d531a7259718864398e9ede75`。

### 5.3 合计

- 两模型训练时间合计：`2011.9704002 s`，约 `33.5328 min`（约 `0.5589 h`）。
- 两份 record 中的协议哈希和 source bundle 哈希均与正式运行 manifest 的初始化记录一致；训练实例、history 与 checkpoint 的记录哈希均与对应文件一致。

来源：`outputs/supplementary_experiment_v1/models/Train-S.record.json`、`outputs/supplementary_experiment_v1/models/Train-L.record.json`、`outputs/supplementary_experiment_v1/manifest.json`。

## 6. 训练—测试隔离与共同测试条件

- manifest 的训练范围记录为 `one-instance-per-model; frozen held-out evaluation`。
- 两个训练实例种子、两个训练算法种子、200 个测试实例种子、200 个初始解种子以及 PPO/GA/Heuristic/MILP 的全部正式求解种子两两无交集；上述八类种子合计 10154 个，全部唯一。
- 200 个测试实例的规范化实例哈希全部唯一，且与两个训练实例的规范化哈希无重合。
- 两份模型先冻结，随后锁定测试集，再生成 schedule；测试阶段不得更新模型权重、优化器状态或根据测试结果重选检查点。
- schedule 中 Train-S 的 PPO cell 全部绑定 `models/small_model.pt` 及其记录哈希；Train-L 的 PPO cell 全部绑定 `models/large_model.pt` 及其记录哈希。
- 对每个“测试规模—实例”组合，schedule 在模型和方法间绑定同一个测试实例文件、初始解文件、初始方案、`initialization_seconds` 和目标归一化引用；200 个组合均未发现绑定冲突。
- PPO 求解前后校验冻结 policy state；若推理过程改变权重则拒绝结果。

来源：`outputs/supplementary_experiment_v1/manifest.json`、`outputs/supplementary_experiment_v1/schedule.json`、两份模型 record、`src/supplementary_experiment.py`。

## 7. 环境信息：可支持与不可支持的论文表述

### 7.1 冻结环境快照可支持

- 平台记录：Windows release `10`、version `10.0.26200`、AMD64。
- CPU 字符串：`Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`。
- CPython `3.11.5`。
- NumPy `1.25.2`、Pandas `2.1.0`、SciPy `1.16.0`、Matplotlib `3.7.2`、PyTorch `2.5.1+cu121`。
- CUDA 可用；CUDA runtime `12.1`；cuDNN 原始版本值 `90100`。
- 1 块 `NVIDIA GeForce RTX 4070 Ti`。
- 初始化环境快照记录 `deterministic_algorithms=false`、`cudnn_benchmark=false`、`cudnn_deterministic=false`。因此不得笼统写成“整个正式环境启用了全局确定性算法”。

### 7.2 当前冻结证据不可支持

- Windows 的市场版本名称。
- CPU 商品型号、物理/逻辑核心数。
- 主机内存容量。
- GPU 显存容量和驱动版本。
- 底层 HiGHS 精确版本。
- MILP 求解器线程数。
- 各 PPO 进程的显式 GPU 绑定和单进程显存占用。
- 尚未完成并最终核验的 GA、MILP 或其他方法批次 worker/设备设置。

以上不可支持字段必须留空，或从另行保存且纳入最终哈希闭环的环境记录回填，不得根据当前机器、软件常识或产品规格推测。

来源：`outputs/supplementary_experiment_v1/manifest.json` 的初始化 `environment` 快照。

## 8. PPO 求解段计时边界与并发解释

- 单 cell 的 `runtime_seconds` 在模型、实例和初始解加载完成后开始。
- 开始计时前执行 CUDA 同步；`improve` 返回后再次 CUDA 同步，再结束计时。
- 计时段包含固定 60 步改进求解、解码、算法内部算子和内部修复。
- 计时段不包含模型加载、实例读取、外部质量/可行性复核、checkpoint 后验校验和结果提交/落盘。
- 多进程共享单 GPU 时，单 cell 观测时间可能包含 GPU 排队、进程调度、内存带宽和其他资源竞争，不能称为隔离环境下的“纯模型推理延迟”。
- 批次墙钟、完成数与吞吐描述整个实际批次，不能冒充某一个 cell 的平均求解时间，也不能在不同资源条件间作纯算法速度归因。
- 模型训练时间是一次性准备成本，必须与正式求解段时间分列。
- Heuristic 的 `initialization_seconds` 是锁定测试集时构造共同初始解的时间，不是评价阶段重新运行启发式算法的求解时间。
- **进度性批次占位：** 实际批次数、worker 数、每批方法范围、完成/失败数、墙钟、吞吐、是否排除历史批次及最终混合计时口径，统一以达到最终验证状态后的 `outputs/supplementary_experiment_v1/manifest.json` 回填；本文不固化任何中间批次数字。

来源：`src/supplementary_experiment.py`、`outputs/supplementary_experiment_v1/manifest.json`。计时代码的文件哈希由初始化 source bundle 锁定。

## 9. 旧稿禁用数字与禁用表述

下列数字或写法不得从旧稿直接带入最终论文：

1. 旧设计估算的训练总耗时 `0.16 h`。正式冻结记录给出的两模型合计训练时间为约 `0.5589 h`。
2. “首 60 个 cell 使用 4 进程、其余 5940 个 cell 均使用 8 进程完成”及由此派生的任何整体吞吐或平均时间。最终只能按最终 manifest 的实际批次逐行回填；不得把计划、部分进度或未完成任务写成既成事实。
3. 当前运行过程中的任何中间完成数、失败数、墙钟、吞吐或阶段性均值。这些值不属于稳定论文事实。
4. 正式验证前生成的任何质量、可行率、Gap、风险分解或泛化数字，以及历史单实例实验的对应数字。
5. 用非 `optimal` 的 MILP incumbent 充当 `J*` 或计算 Gap 得到的数字。
6. 删除失败、不可行或缺失 cell 后形成的可行子集均值，不得替代主质量字段或用于主方法排序；主表必须同时保留有效样本量及失败/不可行/缺失披露。
7. 把三次重启当作独立实例形成 `n=150` 的推断性样本量。主汇总先在实例内处理重启，再对 50 个独立实例报告均值与样本标准差。
8. “库存成本”数值。冻结生成器没有独立库存成本参数，只有库存风险系数。
9. 验证集规模、验证种子、早停轮次或“最佳验证检查点”等数字。本轮没有验证集或基于验证集的模型选择。
10. 未被环境快照支持的 CPU 型号/核数、内存、GPU 显存/驱动、HiGHS 版本、MILP 线程数或进程显存数字。

## 10. 最终回填门槛

- 只从最终验证通过的正式 run 回填论文数值。
- 回填前重新核对协议哈希、schedule 绑定、模型 record、最终 manifest、验证记录和最终表清单的哈希闭环。
- 所有主质量指标使用主字段；缺失时留空，不以 conditional/可行子集字段补写。
- Gap 仅使用 MILP 状态为 `optimal` 的匹配单元。
- 正式批次与并发披露以最终 manifest 为唯一运行进度来源，并保留实际批次分层。

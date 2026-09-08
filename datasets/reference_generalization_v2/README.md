# 固定参考尺度数据集 v2

本数据集用于多实例PPO训练和四规模冻结模型泛化测试。

后续使用约定（2026-09-08）：用户要求新的实验先用单实例，不默认批量测试。每个实验所需规模先固定使用`test/<规模>/000.json`；仅在需要更换时另选实例并保留选择过程和已运行结果。现有四规模各10个实例及下述历史全量实验不变，作为备选数据和复现记录保留。单实例结果用于探索/案例分析，不替代多实例泛化验证。具体约定见仓库根目录`AGENTS.md`。

| 数据集 | 规模 | 产废节点/品类/设施/车辆/周期 | 实例数 |
| --- | --- | --- | ---: |
| train | Test-1（小规模） | 3/2/2/3/2 | 24 |
| train | Test-4（大规模） | 20/4/3/8/4 | 24 |
| test | Test-1 | 3/2/2/3/2 | 10 |
| test | Test-2 | 6/2/2/4/3 | 10 |
| test | Test-3 | 10/3/3/5/3 | 10 |
| test | Test-4 | 20/4/3/8/4 | 10 |

训练参数沿用原配置：20轮，每轮24个episode，每个episode 24步，每轮24次策略更新。每份模型有480个episode；24个实例每轮各参与一次、顺序打乱，因此每个实例参与20次，共480个交互步。策略更新重复利用已收集轨迹，不计为新增训练样本。

选择24个训练实例是依据现有预算的折中，而不是宣称数据量最优：每轮可以均衡覆盖全部实例，且每个实例有20次不同偏好和动作轨迹的训练机会；若扩为48或96个实例而保持训练参数不变，每个实例分别只有10或5个episode。每份模型共收集11,520个环境交互步，执行480次优化器更新；每轮的24次更新是对该轮轨迹的重复利用，不能据此把独立实例数放大24倍。训练集总计48个独立实例，测试集总计40个独立实例。

## 每个实例文件

- `params`：完整模型参数。
- `reference_plan`：固定、严格可行的启发式参考解。
- `b_C`、`b_R`：该参考解的成本与风险；始终固定，不随偏好、搜索起点、重启或搜索过程中最优解更新而改变。
- `instance_seed`、`reference_seed`：实例及参考解生成种子。
- `instance_sha256`、`reference_plan_sha256`：内容哈希。
- `reference_metrics`：参考解原始成本、风险、可行性等指标。

`manifest.json`记录所有实例文件哈希、原配置快照、输入编码模式、源文件哈希及数据划分审计。训练与测试实例内容哈希交集为0。数据集在训练前一次性锁定，测试结果不用于选择检查点。

## 使用约定

新模型使用`objective_normalization=instance_reference`，编码器明确接收该实例的`(b_C,b_R)`，目标相关输入为`C/b_C`、`R/b_R`和归一化加权目标。训练奖励与求解择优使用同一组尺度。新检查点必须加载目标实例自己的固定参考尺度；未提供参考值时明确报错。

旧检查点保持原始固定常数输入编码，不能被当作新归一化模型使用。本轮将旧、新两组模型放在相同的新测试集上，使用相同初始解与评估种子进行对照；该比较同时包含训练数据与编码方式变化，不单独识别任一因素的因果效应。

新模型的推荐加载方式如下。请使用带语义检查的冻结加载入口，不要只向旧编码器加载权重。若更换搜索初始方案，仍沿用实例文件的`refs`，不能从新的搜索起点重新计算b。

```python
from run_reference_generalization import load_manifest, load_instance, RUN
from src.ppo_improver import PPOImprover

manifest = load_manifest()
record = next(r for r in manifest["records"]
              if r["split"] == "test" and r["scale"] == "Test-1")
data, params, reference_plan, refs = load_instance(record)
model = PPOImprover.from_frozen_checkpoint(
    params, RUN / "models/small_model.pt", objective_refs=refs)
result = model.improve((0.5, 0.5), seed=123, initial_plan=reference_plan)
```

运行入口为仓库根目录`run_reference_generalization.py`，依次执行`prepare`、`train`、`evaluate`、`report`，或执行`all`。结果和新模型保存于`outputs/reference_generalization_v2/`。正式评估包含新模型1200次、旧模型1200次及GA 600次，共3000次求解，未安排MILP或重新调参。

求解时间是并发条件下的单次求解段观测时间，训练时间单列。各实例先平均3次重启，再等权平均5组偏好；最终以每个测试规模10个实例报告均值和样本标准差。

本轮在GPU训练，在CPU正式评估（8个进程，每进程1个PyTorch线程）。由于多进程共享GPU的推理调用开销较大，使用训练实例进行完整求解预算的CPU速度检查后，统一切换评估设备；此前118条GPU试跑记录保存在结果目录的`gpu_concurrency_pilot/`，不进入正式统计。正式3000次求解全部重新运行；CUDA与CPU的随机数流不同，不能跨设备直接复用试跑轨迹。执行评估时在当前进程设置`CUDA_VISIBLE_DEVICES=-1`，先运行`evaluate --workers 8`，再运行`report`；完整设备说明见结果目录`execution_notes.json`。

# PPO 完成后的正式评估执行方案

## 1. 适用范围与硬前置条件

本方案只适用于冻结目录 `outputs/supplementary_experiment_v1` 中 PPO 正式评估全部结束之后的 GA、Heuristic 和 MILP 评估。不得在 PPO 主调用仍未返回或任何属于该调用的子进程仍在运行时启动本方案。

开始前必须同时满足：

1. PPO `evaluate` 命令已正常返回，父进程及其 8 个 PPO 子进程均已退出；不能只根据 cell 文件已经出现 6000 个来推断进程已退出。
2. 正式 run 中 PPO 恰有 6000 个 `complete` cell，且没有当前状态为 `failed` 的 PPO cell。
3. 不存在另一个针对同一 run-dir 的 `evaluate`、`summarize` 或 `verify` 进程。
4. PPO 结束后重新检查可用物理内存、系统 commit 余量和 CPU 占用；若资源没有明显回落，应先查明仍在运行的进程，不要直接启动 MILP。
5. 从开始 GA 到最终 verify 完成期间，不修改 `SOURCE_FILES`、冻结配置、manifest、schedule、测试实例、模型或已提交 cell。

只读分析时的主机快照为：13th Gen Intel Core i9-13900K、32 个逻辑处理器；PPO 运行期间可用物理内存约 34.62 GiB，系统 committed/commit limit 约为 48.10/67.75 GiB，9 个 Python 进程的私有提交合计约 25.03 GiB、工作集约 7.98 GiB。该数值是 PPO 运行中的瞬时状态，不是后续方法的正式环境记录；PPO 进程退出后必须重测。

## 2. 并发建议

| 方法 | 正式 cell 数 | 外层任务组 | 每组串行内容 | 建议 workers |
|---|---:|---:|---|---:|
| GA | 3000 | 200 | 同一实例的 5 偏好 × 3 重启 | 16 |
| Heuristic | 500 | 100 | 同一实例的 5 偏好 | 8 |
| MILP | 250 | 50 | 同一实例的 5 偏好，每次限时 180 秒 | 8 |

三个方法必须用三个独立、串行的 `evaluate` 调用；不要使用 `--methods all`，也不要同时从多个终端调用同一个 run-dir。

### GA 为何使用 16

`ClassicGeneticAlgorithm` 是单进程、纯 Python 的串行遗传算法，没有内部进程池或线程池。16 个外层进程可利用 32 个逻辑处理器，同时给操作系统、文件提交和其他服务留下余量。提高到 24 或 32 容易在混合 P/E 核主机上增加调度和写盘争用，收益不稳定。

### Heuristic 为何使用 8

Heuristic 不重新求解，而是复用 test-lock 阶段锁定的初始解。cell 中的 `runtime_seconds` 是该实例已记录的 `initialization_seconds`，不是本次 `evaluate` 的墙钟耗时。本阶段主要工作是质量校验及 solution/trace/cell 三个 JSON 的原子提交；8 workers 足够，16 workers 只会放大 `fsync`、杀毒扫描及目录元数据争用。

### MILP 为何使用 8 而不是 16

本机 SciPy 1.16.0 捆绑 HiGHS 1.8.0。只读检查得到 HiGHS 默认选项为 `threads=0`、`parallel=choose`；当前 `hazardous_waste_model.py` 只向 `scipy.optimize.milp` 传递 `time_limit` 与 `mip_rel_gap`，没有显式锁定 HiGHS 内部线程数。HiGHS 官方说明 `threads=0` 为自动线程数，并可能使用机器可用线程的一部分：<https://ergo-code.github.io/HiGHS/stable/parallel/>。

因此，16 个外层 MILP 进程存在以下风险：

- HiGHS 内部线程与外层 `ProcessPoolExecutor` 叠加，造成 CPU 过度订阅；
- 分支定界的内存峰值随实例而变，16 个进程会明显压缩系统 commit 余量；
- 180 秒是求解器墙钟限时，资源争用可能使更多实例在限时内无法证明 optimal，从而改变可作为 \(J^*\) 的基准数量，而不只是让计时变慢；
- 多进程并行求解和自动内部线程可能降低非 optimal incumbent 的重复稳定性。

在冻结源码不能再加入 `threads=1` 的前提下，8 workers 是吞吐、内存和基准质量之间的保守折中。不得在同一正式 MILP 批次的恢复过程中改成 16 workers。若下一版协议希望严格单线程 MILP，应在初始化新协议之前显式将 `threads=1`（并视需要设 `parallel=off`）纳入求解器调用和冻结配置；本轮不能这样修改。

## 3. 数值库线程环境变量

每个启动 Python 的 PowerShell 进程都应在导入 NumPy/SciPy 前设置：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OMP_THREAD_LIMIT = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
```

本机 SciPy 链接的是 OpenBLAS，因此 `OPENBLAS_NUM_THREADS=1` 直接相关；同时设置 OMP、MKL 和 NUMEXPR 可避免其他依赖引入嵌套并发。这些变量是必要的防护，但不能据此声称 HiGHS 的 `HighsOptions.threads` 已被严格设为 1。当前 manifest 不自动记录这些变量，执行时应保留终端输出，并在方法说明中如实披露外层 workers 与线程变量口径。

## 4. 墙钟预算

Smoke GA 使用人口 6、迭代 2。以 smoke 单 cell 时间按正式人口×迭代数线性外推：

| 规模 | Smoke 秒/cell | 正式工作量倍率 | 估算正式秒/cell |
|---|---:|---:|---:|
| Test-1 | 0.00631 | 150 | 0.95 |
| Test-2 | 0.01546 | 150 | 2.32 |
| Test-3 | 0.04510 | 66.7 | 3.01 |
| Test-4 | 0.25988 | 66.7 | 17.33 |

由于每个实例的 15 个 GA cells 被绑定在同一任务组中，50 个实例在 16 workers 下每个规模至少形成 4 个波次；据此理想值约 23.6 分钟，建议预留 25–50 分钟。

| 阶段 | 建议 workers | 预计墙钟 | 计划预算 |
|---|---:|---:|---:|
| GA | 16 | 约 25–50 分钟 | 约 1 小时 |
| Heuristic | 8 | 约 1–5 分钟 | 约 10 分钟 |
| MILP | 8 | 约 30–120 分钟 | 约 2 小时 |
| summarize + verify | 单进程 | 约 5–15 分钟 | 约 20 分钟 |

MILP 若 250 个 cells 全部触及 180 秒限时，则单组最多约 `5 × 180 = 900` 秒；50 组在 8 workers 下为 `ceil(50/8) = 7` 个波次，即仅求解限时部分约 105 分钟，另加建模、进程启动、校验和写盘。Smoke 的唯一 MILP cell 在 5.046 秒结束且状态为限时，故应按偏保守区间安排。PPO 之后整体建议预留约 2–3 小时。

## 5. 可直接执行的 PowerShell

以下命令只可在第 1 节的 PPO 退出门满足后执行。入口统一使用本轮实际使用的 `py`。初次正式调用不加 `--retry-failed`、`--max-cells` 或 `--allow-concurrent-ppo`。

```powershell
Set-Location -LiteralPath 'C:\Users\Yangfeifei\Documents\ChatGPT\危废品（transformer）'

$run = (Resolve-Path -LiteralPath '.\outputs\supplementary_experiment_v1').Path
$ErrorActionPreference = 'Stop'

$env:OMP_NUM_THREADS = '1'
$env:OMP_THREAD_LIMIT = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

function Assert-MethodComplete {
    param(
        [string]$RunDir,
        [string]$Method,
        [int]$Expected
    )

    $records = @(
        Get-ChildItem -LiteralPath (Join-Path $RunDir 'cells') -Filter '*.json' -File |
        ForEach-Object {
            Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
        }
    )
    $selected = @($records | Where-Object { $_.cell.method -eq $Method })
    $complete = @($selected | Where-Object { $_.status -eq 'complete' })
    $failed = @($selected | Where-Object { $_.status -eq 'failed' })

    [pscustomobject]@{
        method   = $Method
        total    = $selected.Count
        complete = $complete.Count
        failed   = $failed.Count
        expected = $Expected
    } | Format-List

    if ($selected.Count -ne $Expected -or
        $complete.Count -ne $Expected -or
        $failed.Count -ne 0) {
        throw "$Method count/status gate failed"
    }
}

# PPO 完成门。还必须人工确认启动 PPO 的命令已经返回、其进程树已退出。
Assert-MethodComplete -RunDir $run -Method 'ppo' -Expected 6000

# 可用内存和 commit 余量必须在 PPO 退出后重新观察。
Get-Counter '\Memory\Available Bytes','\Memory\Committed Bytes','\Memory\Commit Limit' |
    ForEach-Object CounterSamples |
    Select-Object Path,@{N='GiB';E={[math]::Round($_.CookedValue/1GB,2)}} |
    Format-Table -AutoSize

# GA：单一正式批次，固定 16 workers。
py -u '.\run_supplementary_experiments.py' `
    --run-dir $run evaluate `
    --methods 'ga' --models 'GA' `
    --scales 'Test-1,Test-2,Test-3,Test-4' `
    --workers 16
if ($LASTEXITCODE -ne 0) { throw 'GA evaluate process failed' }
Assert-MethodComplete -RunDir $run -Method 'ga' -Expected 3000

# Heuristic：单一正式批次，固定 8 workers。
py -u '.\run_supplementary_experiments.py' `
    --run-dir $run evaluate `
    --methods 'heuristic' --models 'Heuristic' `
    --scales 'Test-1,Test-4' `
    --workers 8
if ($LASTEXITCODE -ne 0) { throw 'Heuristic evaluate process failed' }
Assert-MethodComplete -RunDir $run -Method 'heuristic' -Expected 500

# MILP：单一正式批次，固定 8 workers。
py -u '.\run_supplementary_experiments.py' `
    --run-dir $run evaluate `
    --methods 'milp' --models 'MILP' `
    --scales 'Test-1' `
    --workers 8
if ($LASTEXITCODE -ne 0) { throw 'MILP evaluate process failed' }
Assert-MethodComplete -RunDir $run -Method 'milp' -Expected 250
```

单 cell 的异常会在 `_execute_nonppo_group` 内被捕获并提交为 failed；因此整个 Python 进程仍可能返回退出码 0。每个方法后的 `Assert-MethodComplete` 是硬门，不能只检查 `$LASTEXITCODE`。

若命令被外部中断，已完整提交的 cell 可由相同命令幂等跳过；恢复时必须保持同一方法、同一 workers。若存在 failed cell，不加 `--retry-failed` 会跳过它。应先诊断失败原因；仅在确认是瞬态问题且未修改冻结源码后，才以相同 workers 加 `--retry-failed` 重试。不要为加速恢复而切换 workers。

## 6. 逐方法计数门

每个方法调用后必须满足：

| 方法 | complete 总数 | 更细粒度计数 |
|---|---:|---|
| PPO | 6000 | 每个模型×规模 750；每个模型×规模×偏好 150 |
| GA | 3000 | 每个规模 750；每个规模×偏好 150 |
| Heuristic | 500 | Test-1/Test-4 各 250；每个规模×偏好 50 |
| MILP | 250 | 每个偏好 50 |

任一阶段当前 cell 状态为 failed 的数量必须为 0。完成全部方法后应恰有 9750 个 scheduled cell 对应的 complete cell，missing=0、failed=0。

以下命令用于检查完整细分计数与 MILP 状态，不把非 optimal 错当作技术失败：

```powershell
$records = @(
    Get-ChildItem -LiteralPath (Join-Path $run 'cells') -Filter '*.json' -File |
    ForEach-Object {
        Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
    }
)

$records |
    Group-Object { '{0}|{1}' -f $_.cell.method, $_.status } |
    Sort-Object Name |
    Select-Object Name,Count |
    Format-Table -AutoSize

$records |
    Where-Object status -eq 'complete' |
    Group-Object {
        '{0}|{1}|{2}|{3}' -f `
            $_.cell.method,$_.cell.model_id,$_.cell.test_scale,$_.cell.preference_id
    } |
    Sort-Object Name |
    Select-Object Name,Count |
    Format-Table -AutoSize

$milp = @($records | Where-Object { $_.cell.method -eq 'milp' })
$milp |
    Group-Object solver_status |
    Sort-Object Name |
    Select-Object Name,Count |
    Format-Table -AutoSize

[pscustomobject]@{
    all_cells             = $records.Count
    all_complete          = @($records | Where-Object status -eq 'complete').Count
    all_failed            = @($records | Where-Object status -eq 'failed').Count
    milp_total            = $milp.Count
    milp_optimal          = @($milp | Where-Object solver_status -eq 'optimal').Count
    milp_quality_included = @(
        $milp | Where-Object { $_.metrics.quality_included -eq $true }
    ).Count
} | Format-List
```

MILP 各 `solver_status` 计数之和必须为 250。`solver_status` 非 optimal 不等于管线技术失败；但这些行不能作为 \(J^*\)，也不能进入 Gap 基准。应分别保留 status、是否存在 incumbent、严格可行性及 `quality_included` 的披露。

## 7. 汇总与最终 verify 门

只有第 6 节全部计数门通过后，才能顺序执行：

```powershell
py -u '.\run_supplementary_experiments.py' --run-dir $run summarize
if ($LASTEXITCODE -ne 0) { throw 'summarize failed' }

py -u '.\run_supplementary_experiments.py' --run-dir $run verify --replay-per-stratum 1
if ($LASTEXITCODE -ne 0) { throw 'verify failed' }
```

最终 `verification/verification_report.json` 和 manifest 必须共同满足：

- manifest `stage = verified`，且 manifest 中 verification 文件哈希与实际文件一致；
- `passed = true`、`count_match = true`、`summary_ledger_match = true`；
- expected counts 为 PPO 6000、GA 3000、Heuristic 500、MILP 250、总计 9750；observed complete counts 与之相同；
- `missing_count = 0`、`failed_count = 0`、`integrity_error_count = 0`；
- `artifact_error_count = 0`、`solver_replay_error_count = 0`；
- `table_integrity_errors` 为空，`source_changes_since_init` 为空；
- `model_count = 2`、`checkpoint_file_count = 2`、`test_instance_count = 200`；
- 使用 `--replay-per-stratum 1` 时，`artifact_rechecked_count = 75`、`solver_replayed_count = 15`。

MILP 非 optimal 重放差异允许被单独记录在 `milp_nonoptimal_replay_differences`，不能因此将其伪装为 optimal；只要验证报告的 `passed` 门成立且上述错误计数为零，后续才能构建论文表格。

## 8. 实现依据

- 外层进程池、Windows `spawn`、实例级分组及 manifest 单点写入：`src/supplementary_experiment.py`。
- GA/Heuristic/MILP 的组内串行执行与计时口径：`src/supplementary_experiment.py::_execute_nonppo_group`。
- 三联工件的临时文件、`fsync` 与原子替换：`src/supplementary_experiment.py::atomic_write_json` 和 `_commit_success`。
- GA 无内部并发：`遗传算法/genetic_algorithm.py::ClassicGeneticAlgorithm.run`。
- SciPy/HiGHS 调用及 180 秒限时：`hazardous_waste_model.py::HazardousWasteMILP.solve` 和冻结协议 `configs/supplementary_experiment.json`。
- 正式计数：冻结 `outputs/supplementary_experiment_v1/schedule.json`，总计 9750，分解为 6000/3000/500/250。
- Smoke 单元时间：`outputs/supplementary_experiment_smoke_ultra_v2/tables/raw_results.csv`。

本文件是执行前内部计划，不是正式结果，也不固化任何正在变化的 PPO 完成进度。

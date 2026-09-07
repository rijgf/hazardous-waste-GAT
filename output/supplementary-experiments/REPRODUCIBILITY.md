# 补充实验复现与审计说明

## 1. 材料定位

本项目区分“紧凑审计归档”和“可执行完整运行”两类材料。

| 材料 | 用途 | 能做什么 | 不能单独做什么 |
|---|---|---|---|
| `output/supplementary-experiments/replication/` | 适合纳入 Git 的紧凑可审计结果归档 | 核对协议、schedule、两份冻结 checkpoint、训练历史、10 张正式源表、verification report 及 inventory 哈希 | 不能单独执行 `verify`，不能重建 9,750 个 cell，不能从头运行求解 |
| 仓库源码与 `configs/supplementary_experiment.json` | 可执行实现与冻结协议 | 在兼容环境中重新生成合成实例、训练模型并运行完整实验 | 不包含本次正式运行产生的全部逐 cell 文件 |
| `outputs/supplementary_experiment_v1/` | 本机完整正式运行及逐 cell 证据 | 复核 test/initial inputs、cells、solutions、traces、attempts、汇总表和验证重放 | 该目录被 `.gitignore` 的 `outputs/` 规则排除，普通 Git clone 不会取得它；需要单独的不可变归档或在本机保留 |

因此，`replication/` 的准确名称是“**紧凑可审计结果归档**”，不是“单目录可执行复现包”。审计时该目录有 18 个 inventory payload，连同 inventory 共 19 个文件、27,799,907 字节；完整正式运行有 29,672 个文件、1,099,073,679 字节。紧凑归档省略了 `training_instances/`、`test_instances/`、`initial_solutions/`、`cells/`、`solutions/`、`traces/` 和源码，包内 manifest 对这些路径的引用只描述原正式运行，不表示文件已随包携带。

本研究使用代码生成的合成实例，不依赖外部受限数据。第三方若只取得 Git 仓库，可以按第 3 节从头重跑；若要逐字节审计本次已经完成的 9,750 个单元，则还需要本机完整正式运行或其另行保存的不可变归档。

## 2. 复现前提与证据优先级

1. 使用包含本说明及正式源码的仓库版本，并确认 13 个运行源码文件与正式 manifest 的 `source_files_sha256_at_init` 一致。初始化时工作树为 dirty，因此 `manifest.git_at_init.commit` 不能单独恢复实际执行源码；规范身份是逐文件 SHA-256 及 `source_bundle_sha256_at_init`。
2. Python 与包版本参考 `environment-freeze-2026-09-05.txt`。该文件是运行完成后的环境快照，并非求解时自动生成的锁文件；它支持近似重建，不证明所有进程的传递依赖完全相同。
3. 建议使用 Windows、CPython 3.11.5、PyTorch 2.5.1+cu121、CUDA 12.1、cuDNN 90100 和 NVIDIA GeForce RTX 4070 Ti 或兼容环境。2026-09-05 运行后查询的 NVIDIA driver 为 591.86。
4. 为保留正式材料，重跑时使用新的空目录。`export` 会拒绝覆盖已有目标；不要把重跑输出写到已发布的 `outputs/supplementary_experiment_v1/` 或 `output/supplementary-experiments/replication/`。

正式结果的证据优先级如下：锁定协议与源码哈希 → manifest/schedule/checkpoint/input 哈希 → 已提交 cell/solution/trace 哈希 → cell ledger 与 10 张源表 → verification report → paper-table inventory 和论文稿。进度日志只记录过程，不覆盖最终 `manifest.stage=verified` 和 verification 的 `passed=true`。

## 3. 完整命令链

以下 PowerShell 命令在仓库根目录执行。目录名刻意与正式目录不同，避免覆盖原证据。

```powershell
$RunDir = 'outputs/supplementary_experiment_reproduction_20260905'
$ReplicationDir = 'output/supplementary-experiments/replication-reproduced-20260905'
$PaperTableDir = 'output/supplementary-experiments/tables-reproduced-20260905'

py .\run_supplementary_experiments.py `
  --config .\configs\supplementary_experiment.json `
  --run-dir $RunDir init

py .\run_supplementary_experiments.py --run-dir $RunDir train
py .\run_supplementary_experiments.py --run-dir $RunDir lock-tests
py .\run_supplementary_experiments.py --run-dir $RunDir schedule

py .\run_supplementary_experiments.py --run-dir $RunDir evaluate `
  --methods ppo --workers 4 --max-cells 60 --allow-concurrent-ppo
py .\run_supplementary_experiments.py --run-dir $RunDir evaluate `
  --methods ppo --workers 8 --max-cells 120 --allow-concurrent-ppo
py .\run_supplementary_experiments.py --run-dir $RunDir evaluate `
  --methods ppo --workers 8 --allow-concurrent-ppo

py .\run_supplementary_experiments.py --run-dir $RunDir evaluate `
  --methods ga --models GA --scales Test-1,Test-2,Test-3,Test-4 --workers 16
py .\run_supplementary_experiments.py --run-dir $RunDir evaluate `
  --methods heuristic --models Heuristic --scales Test-1,Test-4 --workers 8
py .\run_supplementary_experiments.py --run-dir $RunDir evaluate `
  --methods milp --models MILP --scales Test-1 --workers 8

py .\run_supplementary_experiments.py --run-dir $RunDir summarize
py .\run_supplementary_experiments.py --run-dir $RunDir verify --replay-per-stratum 1
py .\run_supplementary_experiments.py --run-dir $RunDir export $ReplicationDir

py .\output\supplementary-experiments\scripts\build_paper_tables.py `
  $RunDir --output-dir $PaperTableDir
```

该顺序固定为 `init → train → lock-tests → schedule → evaluate → summarize → verify → export → builder`。`lock-tests` 只有在两份模型均冻结后才允许执行；`schedule` 将协议、实例、初始解、源码、checkpoint 与策略状态哈希绑定到唯一 cell。`evaluate` 可恢复执行：已存在且哈希有效的完成单元会被复用；除非明确处理失败单元，不应使用 `--retry-failed` 改变正式尝试历史。

上面三段 PPO 命令复刻本次正式 manifest 的批次边界：4 workers 完成 60 个、8 workers 完成 120 个、8 workers 完成其余 5,820 个。若只关心重新计算而不复刻批次计时元数据，可以在空运行目录中用一个完整 PPO 调用，但所得 `evaluation_runs` 将与正式 manifest 不同。

## 4. 资源、并发与预计耗时

本次正式运行在一块 RTX 4070 Ti 上记录的耗时如下。这些是本机观测值，不是服务级保证。

| 阶段 | 正式记录 | 约计 |
|---|---:|---:|
| Train-S | 778.809 s | 12 分 59 秒 |
| Train-L | 1,233.161 s | 20 分 33 秒 |
| PPO 三批合计 | 18,914.641 s | 5 小时 15 分 15 秒 |
| GA | 1,447.723 s | 24 分 8 秒 |
| 启发式 | 3.081 s | 3 秒 |
| MILP | 3,651.461 s | 1 小时 0 分 51 秒 |
| 两次训练与全部 evaluation 合计 | 26,028.877 s | 约 7 小时 14 分 |

还应为初始化、测试集锁定、汇总、75 个 artifact 分层复核、15 个 solver replay、导出和表格构建预留时间。并发 PPO 会共享同一 GPU，GA/MILP 会共享 CPU；worker 数、GPU/CPU竞争、散热和未记录的 solver thread 设置都会改变墙钟及单元延迟。若设备内存不足，应减少 workers，但必须在新的 manifest 中如实保留不同批次设置，不能把新计时冒充正式计时。

## 5. 正式哈希锚点

| 对象 | SHA-256／状态 |
|---|---|
| 协议规范 canonical hash | `24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba` |
| 正式协议文件 | `74edd93e5866d583eef7ffa17a004c9e92070a165260712761d549a691253257` |
| 初始化源码 bundle | `50ff1564043fcec7bc573184f4f7aed02b6df99883440795c3bbb62e929fff6e` |
| 正式 schedule | `7312dae1eaccf33b8f28b1421822be32a3e7f6310dc2751a90696db69d3c286a` |
| 汇总输入 cell ledger canonical hash | `bff5643b687e9ef7467ef6fa61d77d104c972bc7e6da215a5d44af576e41008d` |
| 正式 verification report | `e7799970170b264ac0f5e55aec9c4ab45228008f69d5f7fb485180adc3d50f88`；`passed=true` |
| 正式 manifest | `43f9d52058a7b338b1081bfd5e617aed3f505000531e3cdee2aaf88a62244d9e`；`stage=verified` |
| 紧凑归档 inventory | `7e1d236784b1558fb682f6d98e8c33a72ff5f27f3033c3457f1247b17bc0cfdc` |
| 论文表 paper inventory | `76edf5c7f05035676b97c6d1dc12f09195b2d2000f87aa5af3788b9abf16e0f3` |

`output/supplementary-experiments/tables/paper-table-inventory.json` 当前对 14 个 builder 输入、18 项声明的非自身输出和 10 个 CSV 行数逐项登记；新增覆盖表 5 的 CSV 与 Markdown。inventory 本身为避免递归而不自哈希。builder 的路径修订已经完成：`source_run_dir="outputs/supplementary_experiment_v1"`，`source_run_dir_path_kind="repository-relative"`。该相对路径用于仓库内定位；判断同一正式运行仍应以本节哈希及 inventory 内的逐项哈希为准。

可用下列只读命令核对主要文件：

```powershell
Get-FileHash -Algorithm SHA256 .\outputs\supplementary_experiment_v1\manifest.json
Get-FileHash -Algorithm SHA256 .\outputs\supplementary_experiment_v1\schedule.json
Get-FileHash -Algorithm SHA256 .\outputs\supplementary_experiment_v1\verification\verification_report.json
Get-FileHash -Algorithm SHA256 .\output\supplementary-experiments\replication\inventory.json
py -m json.tool .\output\supplementary-experiments\audits\seed-audit.json *> $null
```

## 6. 种子与冻结性

根种子为 42，方案为 `sha256-seed-v1`。八个 namespace 使用互斥高位 lane；单元 seed 还绑定模型、测试规模、实例、偏好、restart 和初始解身份。独立审计结果见 `audits/seed-audit.json` 与 `audits/seed-audit.md`：10,154 个种子记录全部唯一；其中 9,750 个 evaluation seed 全部唯一；派生偏差为 0；训练与测试／评估 seed 交集为 0；训练与测试实例内容哈希交集为 0。

两份正式模型各有一个已提交训练记录、一个 checkpoint 和一份训练历史。冻结推理使用 `trainable=False`、无 optimizer、`requires_grad_(False)` 和 inference mode；6,000 个 PPO cell 中登记的策略状态 before/after 及 checkpoint after 均与 manifest 锁定值一致。这里的“各一次训练”指每个模型各有一份正式提交训练结果；现有账本不主张能够证明操作系统进程层面从未发生过提交前中断。

## 7. 确定性边界

种子、协议、输入、检查点和输出哈希使本次运行可追溯，但不构成跨硬件位级一致保证，原因包括：

- CUDA、cuDNN、GPU driver、浮点归约顺序和具体硬件可能改变末位数值或离散选择；
- 训练阶段使用 `torch.use_deterministic_algorithms(True, warn_only=True)`，`warn_only=True` 并不承诺所有算子均有严格确定性实现；
- `CUBLAS_WORKSPACE_CONFIG` 的实际进程值未写入 manifest，程序只在变量未预设时提供默认值；
- `environment-freeze-2026-09-05.txt` 是运行后快照，不是传递依赖锁；
- SciPy 版本有记录，但正式运行未单独记录所捆绑 HiGHS 的精确版本，MILP solver thread 数也未记录；
- 并发 workers 会影响计时，计时不应解释为隔离条件下的纯算法速度。

因此，第三方复现的合理目标是：协议与 seed 身份相同、计划计数相同、全部单元可完成、哈希和统计链可重新建立、主要描述性方向可核对。除非同时锁定完整软件栈、驱动与硬件，不能承诺新机器生成与本次所有二进制／浮点产物逐字节相同。

## 8. 最终验证门

本次正式 verification 已确认：PPO 6,000/6,000、GA 3,000/3,000、启发式 500/500、MILP 250/250，总计 9,750/9,750；模型 2、checkpoint 2、测试实例 200；missing 0、failed 0、integrity error 0；artifact 分层复核 75/错误 0；solver replay 15/错误 0；源码变化 0；最终 `passed=true`。

75 和 15 是分层复核数量，不代表重新执行全部 9,750 个单元。紧凑归档中的 verification report 是对完整正式运行的已锁定证明；若完整运行目录不可用，只能验证报告及其哈希链，不能从紧凑目录再次执行同一 `verify`。

## 9. 文档版本说明

早期 `scripts/script-index.md`、`logs/process-log-scholar-analyze-2026-09-04.md` 和 `logs/formal-provenance-facts.md` 保留了“待正式结果”“进行中”或批次占位语句，它们是历史过程记录。关于当前完成状态、复现用途边界和命令顺序，以本文件、正式 manifest、verification report、seed audit 及 paper-table inventory 为准。

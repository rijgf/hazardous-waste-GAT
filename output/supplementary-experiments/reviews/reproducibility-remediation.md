# Reviewer 5 复现性整改记录

- 整改日期：2026-09-05（Asia/Shanghai）
- 整改范围：新增复现说明、运行后环境快照和 seed 审计材料，并在 builder 路径修订完成后同步复现文档
- 正式实验重跑：否
- 正式 locked run／核心源码／replication payload 与 inventory／论文稿修改：否
- 整改状态：**COMPLETE（在获准的文档级范围内）**

## 1. Reviewer 5 问题落实

| 原问题 | 落实结果 | 新证据 | 剩余边界 |
|---|---|---|---|
| I-1 紧凑 replication 不能独立执行 | 已明确把 `replication/` 定义为“紧凑可审计结果归档”，区分 Git 源码、ignored 完整正式运行与紧凑归档，并给出 `init → train → lock-tests → schedule → evaluate → summarize → verify → export → builder` 的完整 PowerShell 命令链、恢复语义、并发批次和预计耗时 | `output/supplementary-experiments/REPRODUCIBILITY.md` 第 1–5 节 | 没有改变紧凑归档的 payload；若要逐字节重验本次 9,750 个 cell，仍需保留或另行发布完整正式运行 |
| I-2 环境／确定性说明不足 | 已保存 2026-09-05 当前 `py -m pip freeze` 的全部 37 行，记录 Python、OS、GPU、driver 591.86、PyTorch、CUDA、cuDNN、SciPy/milp，并明确 HiGHS 精确版本、solver threads 与实际 CUBLAS 进程值未记录；区分 init、训练和冻结推理阶段开关 | `environment-freeze-2026-09-05.txt`；`REPRODUCIBILITY.md` 第 2、4、7 节 | 该文件是运行后快照而非求解时自动锁，不承诺跨硬件位级一致；未知版本未推断或伪造 |
| I-3 verify 未把 seed 唯一性设为显式硬门 | 已从协议、manifest、schedule 全量重派生 10,154 个 seed，并保存机器可读 JSON 与解释文档；本次结果 PASS | `audits/seed-audit.json`；`audits/seed-audit.md` | 为保持正式源码和验证报告哈希，本轮没有修改 `verify_run`；下一版协议再把相同断言并入正式硬门 |
| M-1 “只训练一次”的证据边界 | 已说明“两次”指两份正式提交训练记录，不声称能证明提交前从无进程中断 | `REPRODUCIBILITY.md` 第 6 节 | 无 |
| M-2 paper inventory 绝对路径 | builder 修订已实际完成并重新发布 inventory：`source_run_dir="outputs/supplementary_experiment_v1"`，`source_run_dir_path_kind="repository-relative"`；复现说明已同步为当前事实 | `tables/paper-table-inventory.json`；`REPRODUCIBILITY.md` 第 5 节 | 无；本次限定同步未再次修改 builder、paper inventory 或结果 payload |
| M-3 旧说明保留中间状态 | 已把早期 script index/process log/provenance facts 标为历史过程记录，并声明当前状态证据优先级 | `REPRODUCIBILITY.md` 第 2、9 节 | 本轮按要求未改写历史日志 |

本次整改通过“准确限定能力”处理复现包问题，不把紧凑归档虚构成完整可执行包，也不把运行后环境快照描述成求解时锁文件。

builder 后续修订现已完成：当前 paper inventory 登记 14 个输入、18 项声明的非自身输出和 10 个 CSV 行数，包含表 5 的 CSV/Markdown；inventory SHA-256 为 `76edf5c7f05035676b97c6d1dc12f09195b2d2000f87aa5af3788b9abf16e0f3`。旧版计数与路径描述已全部替换为当前事实。

## 2. Seed 审计复算结果

独立校验脚本再次解析已写入的 `seed-audit.json`，并从正式协议重新计算每个训练实例、训练算法、测试实例、初始解和 evaluation cell seed。断言结果如下：

- JSON 解析：PASS；`status=PASS`。
- 协议 canonical hash 同时匹配 manifest 与 schedule。
- schedule 实际文件哈希匹配 manifest 声明。
- 全部 seed：10,154 条，唯一值 10,154，重复 0。
- evaluation seed：9,750 条，唯一值 9,750。
- seed 派生偏差：0。
- evaluation cell identity 派生偏差：0。
- 训练 seed 与测试／评估 seed 交集：0。
- 两份训练实例与 200 个测试实例内容哈希交集：0。
- manifest 状态：`verified`；manifest verification：`passed=true`。

分类计数为：训练实例 2、训练算法 2、测试实例 200、初始解 200、PPO 6,000、GA 3,000、启发式 500、MILP 250，总计 10,154。

## 3. 环境快照复核

- `py -m pip freeze` 当前输出 37 行。
- 从 `environment-freeze-2026-09-05.txt` 排除注释与空行后得到 37 行。
- 两者逐字符、逐行比较：**完全一致**。
- 当前查询：CPython 3.11.5；Windows 10.0.26200 AMD64；RTX 4070 Ti；NVIDIA driver 591.86；PyTorch 2.5.1+cu121；CUDA 12.1；cuDNN 90100；SciPy 1.16.0；`scipy.optimize.milp` 可用。
- 未发现可据此可靠声明的独立 HiGHS 精确版本，因此快照明确记录为 unknown；正式 manifest 未记录 solver thread 数。

## 4. 新增文件与哈希

| 文件 | SHA-256 |
|---|---|
| `output/supplementary-experiments/REPRODUCIBILITY.md` | `19ea01f7dc9b13ce1a7c2ccdb6e7ff9761e7b5a213a09c79cc2b392c102238c0` |
| `output/supplementary-experiments/environment-freeze-2026-09-05.txt` | `1351ec6e69c8cec1934185e41dc8d950480e81571720d047a1a87171bcb54026` |
| `output/supplementary-experiments/audits/seed-audit.json` | `d79a0de13114bac0d69d3a2672595c0c2941f528727140ac9bf9ef17ab68d06e` |
| `output/supplementary-experiments/audits/seed-audit.md` | `eb2a3738a7eaf192ecf0390229b009682c140344680d6b5696414aa84f421a07` |

本整改记录自身不做递归自哈希。原 Reviewer 5 报告保持未编辑；复核时其 SHA-256 为 `7c2a045c094b41bb1f542f283bbbd99c1f8e3d1bd5dbc0395b700a570f1e07f5`。

## 5. 正式锚点保持

整改后重新读取的正式锚点仍为：

- manifest：`43f9d52058a7b338b1081bfd5e617aed3f505000531e3cdee2aaf88a62244d9e`；
- schedule：`7312dae1eaccf33b8f28b1421822be32a3e7f6310dc2751a90696db69d3c286a`；
- verification report：`e7799970170b264ac0f5e55aec9c4ab45228008f69d5f7fb485180adc3d50f88`；
- replication inventory：`7e1d236784b1558fb682f6d98e8c33a72ff5f27f3033c3457f1247b17bc0cfdc`；
- paper-table inventory：`76edf5c7f05035676b97c6d1dc12f09195b2d2000f87aa5af3788b9abf16e0f3`。

formal run 与 replication 的四个锚点和 Reviewer 5 原审查一致；paper-table inventory 的新锚点对应已经完成的 builder 修订。本次限定同步不修改任何正式结果数值、checkpoint、schedule、source bundle、replication payload/inventory、论文表 payload 或论文稿。

## 6. 结论

**整改完成并已同步 builder 当前状态。** Reviewer 5 提出的无需重跑事项已在授权范围内形成可执行命令说明、诚实的材料用途边界、运行后环境快照和全量机器可读 seed 审计；paper inventory 的相对路径修订已经落地。新增证据提升了第三方审计与从头重跑的可操作性，同时保留了“紧凑归档不等于完整运行”“运行后快照不等于环境锁”“跨硬件不保证位级一致”三项必要限定。

# v11 独立中期核验：Test-1 / Test-2

日期：2026-09-10。角色：独立交付完整性智能体，scholar-verify completeness 路线。结论为 **INTERIM_PASS_NOT_FINAL**。本记录不替代尚未完成的全规模、敏感性、正文图表和 Git 交付终验。

## 核验范围与方法

只读核验 Test-1、Test-2 两固定实例，两份冻结 v9 模型各 3 次新 PPO 前沿（共 12），及各实例 3 份复用 NSGA-II B1 档案（共 6）。PPO 上限 252000/前沿，NSGA-II B1 为 40320，实例及绑定 b 未改；不能称为等预算对照。

独立审阅脚本通过内存命令执行，不保存或修改实验实现，不进行搜索：逐前沿校验实例、参考起点、模型及协议；逐保存方案重新计算原始 C/R 与对应偏好 J；用 NumPy 广播独立检查归一化容差 1e-8 下的内部支配/目标重合和两个覆盖方向，不调用报告的覆盖率函数。完整约束阶段调用已审阅的 full_model_replay，将这些方案重构为原 MILP 变量，执行原代数约束、变量界和整数性检查，未调用求解器。

252 个新 PPO 偏好日志均完整解压至 EOF，压缩哈希和 CRC 通过；检查候选计数、顺序、批次、同输入不重复、有限记录概率及停止原因。此项复用当前 audit.check_log，不冒称独立重算全部神经网络概率。每个偏好种子另核对 derive_seed(task.seed, 'preference', index, bits=32)。12 个新前沿均交叉检查 completed、actual_execution 与 launcher，核对 PID、CPU 单线程、执行 ID、完成哈希及可解析且先后有序的 UTC 时间。

## 数值和合法性结果

| 实例 | 去重保存方案（含末次方案） | 原 MILP 变量数 | 约束行数 | 违规行 | 最大行残差 | 最大 C 重算差 | 最大 R 重算差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test-1 | 187 | 2194 | 6248 | 0 | 8.881784197001252e-16 | 9.094947017729282e-13 | 7.105427357601002e-15 |
| Test-2 | 987 | 14688 | 44152 | 0 | 1.7763568394002505e-15 | 5.4569682106375694e-12 | 3.552713678800501e-14 |

两实例变量界及整数性最大残差均为 0。共同评价器相对档案原 C/R 的最大差分别为 3.637978807091713e-12、3.552713678800501e-15；全部严格可行、无违反记录，所有前沿内部无被支配点或容差内目标重复。1039 个独立方案文件被读取，连同保存在前沿内的 252 个末次方案，按实例去重后共有上表 1174 个方案；两种计数对象不同。

252 个偏好全部以正概率动作空间耗尽结束；实际候选总数 **524869**，总上限 3024000。未使用名额未伪计为执行次数。小规模有限动作空间使提高上限未必增加实际搜索，不能仅据预算上限解释性能。

## 独立覆盖率结果

N→P 为 NSGA-II 覆盖 PPO 的比例，从 PPO 角度越小越好；P→N 反之，越大越好。下表每组 3 次，标准差 ddof=1。

| PPO 模型 | 实例 | N→P 均值 ± SD | P→N 均值 ± SD |
| --- | --- | --- | --- |
| Train-S | Test-1 | 0.6416433240 ± 0.0564419399 | 0.6990740741 ± 0.1265973048 |
| Train-L | Test-1 | 0.6638888889 ± 0.0375770813 | 0.5717592593 ± 0.1811333348 |
| Train-S | Test-2 | 0.1271864068 ± 0.2109500198 | 0.8225117177 ± 0.2023601065 |
| Train-L | Test-2 | 0.0092592593 ± 0.0160375075 | 0.9541632203 ± 0.0634698669 |

可复算的逐次分子/分母（按 repeat 0、1、2）：

| 模型 / 实例 | N→P 三次 | P→N 三次 |
| --- | --- | --- |
| Train-S / Test-1 | 12/17，13/21，12/20 | 9/16，13/16，13/18 |
| Train-L / Test-1 | 10/16，14/20，14/21 | 8/16，7/16，14/18 |
| Train-S / Test-2 | 43/116，0/124，1/92 | 56/93，87/87，45/52 |
| Train-L / Test-2 | 3/108，0/117，0/123 | 82/93，87/87，51/52 |

这些数值仅描述本轮两案例已保存近似解集，不证明真实前沿完整或跨实例统计优势。Test-1 两模型覆盖值与历史 v9 一致；这不是复用了旧 PPO 档案，而是本轮新动作记录和实际有限搜索结果，需与提前耗尽共同解释。

## 前次问题修复复核

report.verify_preflight 已适配实际 v11 键名及 large 检查点，不再读取缺失旧字段；parameter_revision_evidence 明确 models_retrained_this_round=false、historical_v9_models_retrained=true。audit.check_small_metadata 已补实例、b、参考方案、重复编号、种子、偏好和模型；check_front_seed 已补派生种子核对。9 项 v11 回归测试由本智能体实际运行，通过，其中包含这些新增负例。冻结计算入口仍为原哈希，报告/审核修复没有改动运行算法。

## 本次文件锁

核验开始和结束时下列源码及 18 个前沿哈希一致。全部路径相对项目根；前沿路径前缀为 output/pareto-ppo-budget-v11/fronts/，后缀为 .json。

| 文件 | SHA256 |
| --- | --- |
| output/pareto-ppo-budget-v11/protocol.json | db90a62fa5dab3d2180e1f640239fba195fe6d9470382f6c91ecdcee96e68ce5 |
| run_ppo_budget_v11.py | 8a17359f2251bfa241b42886601b4405749b88b65f45b2f91de6a5dbb73b8bcf |
| report_ppo_budget_v11.py | 10d46521768e780c8c47aec0be822fd29865ec67b67910ecb4ffe0f5a0e98410 |
| audit_ppo_budget_v11.py | 538e6afddd3bbdbe2c8fe03a47edd1dcb04941cec3cab593bb42bb3e3d40aebc |
| audit_scalar_advantage_v5.py | a1f9ae38c0984f3622df97790101af3e5504bee3537d03ecebfd78ca6786d13c |
| src/solution_utils.py | 0e987e27da0e77be61530e9b71b536966c5da3f9648f9786fa62b6901eaab79f |
| hazardous_waste_model.py | 021d3ebf56ed7b3952f8d18a852092ddb9abe20a64cf41621151d15951d081a6 |
| NSGA-Test-1-r0-B40320 | 638e395aaff64ed19f64f791d47e6565b57bfb1a6736fc7c0a590b2baedd5d33 |
| NSGA-Test-1-r1-B40320 | bc8c42bcc88813cf6b545f5e6f56dff89c16c2e8422cf15f005a74f1a4b7d24a |
| NSGA-Test-1-r2-B40320 | 76cf95d2b44d9f22234a5d08633ec72d300c07fe7f7d53f1b9be7bf015b5111f |
| PPO-Test-1-small-r0-B252000 | 311863ecc5268519bb3074d814a17eb78bd77b6f49342a375e3aede4cc0d12c3 |
| PPO-Test-1-small-r1-B252000 | 805ab8ec46430713aa385907eee25cd1c0822eae28389282efbd2ee750a06609 |
| PPO-Test-1-small-r2-B252000 | 34f10924c0bae7e0f85ce16fc681ae721c65b88d4094162cfd9cc876b0939d99 |
| PPO-Test-1-large-r0-B252000 | 2e89c5f1cdb913d313075d1aa05749b5a3f1c0f5d9b4b0c8ec04ba5d3358ba83 |
| PPO-Test-1-large-r1-B252000 | 6f83756a845b390dd678824adfc5d3d47a092da86bfd2ecb1f35daf785fbb2ec |
| PPO-Test-1-large-r2-B252000 | c4bced79066eb25be07602140e2c751a57908711c32d937827a304ff77e849b6 |
| NSGA-Test-2-r0-B40320 | 4c8f89a1c4ca1e813758a1d03a1210b1b2b26a964fce6bdc43fbc1587d4ebfc1 |
| NSGA-Test-2-r1-B40320 | 32cda204bbf1acce790cc2c3b878914142aa470b066f6e5a6cb4bc826160be20 |
| NSGA-Test-2-r2-B40320 | e6968c5852536bc4e422c9fed91596067a886bbbb65569b05d56042036dfcac4 |
| PPO-Test-2-small-r0-B252000 | 780cc0358b72de6b01ded81ad4463a427797e50776bad842627829bd06792557 |
| PPO-Test-2-small-r1-B252000 | c9a952f18f0e7d6abcd1e309aa3efcb3ccc67675f26925fa726c421a7780f8c1 |
| PPO-Test-2-small-r2-B252000 | 981007fe50bef1eab3e03a4068af7d00553eadaa11e0770fb66b7b4723d76166 |
| PPO-Test-2-large-r0-B252000 | 69fb3f7e4fbdb68d6dfb068d94eaa83356b118f159733afd40e272e193eb8ab5 |
| PPO-Test-2-large-r1-B252000 | 5dedf045af69b52ad46fff45398fde7c7ece8453bd5bd23363044315740a18ce |
| PPO-Test-2-large-r2-B252000 | 39981d276828e951bc91e0b701228129397b16597e4435c9fddb9c985a18a7a6 |

补充映射摘要：252 条日志核验记录 JSON（sort_keys=true）SHA256 为 42ea006bc9d576f35a54e9cb98550409e655f5c0a679063085b4be0850a087d3；1039 个方案相对路径→文件哈希映射摘要为 5b83b626c9b92fa0849d308b46b4cc989e631426d7bfcf13c39bb94123d67ca0；12 个 task→实际执行文件哈希映射摘要为 fd9eadc5eee989a340d35e04daa50e63bc35cfc21b9bcbcfb2a477b1253001ef。原逐日志哈希可从上述锁定前沿的 final_solutions 追溯，不以聚合摘要替代原始来源。

## 后续仍需完成

其余 36 个新完整 PPO 前沿、全 15 次小算例、全 15 份 NSGA-II 档案、5 组复用 MILP、27 对覆盖、9 个敏感性情景、三版图、正式正文及当前文件哈希的全量末验。此报告未验证未完成任务，不作全交付完成或推送成功声明。

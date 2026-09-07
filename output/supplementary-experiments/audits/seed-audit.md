# 正式补充实验 seed 审计

- 审计时间：2026-09-05T01:46:11.863640+08:00
- 机器可读结果：`seed-audit.json`
- JSON SHA-256：`d79a0de13114bac0d69d3a2672595c0c2941f528727140ac9bf9ef17ab68d06e`
- 结论：**PASS**

## 核查结果

审计程序读取锁定的协议、正式 manifest 和完整 schedule，用 `src.supplementary_protocol.derive_protocol_seed` 对每个已登记 seed 重新派生，并重建全部 evaluation `logical_cell_id` 与 compact `cell_id`。本次只读核查没有训练模型、运行求解器或修改正式目录。

| 类别 | 记录数 | 唯一 seed 数 | 结果 |
|---|---:|---:|---|
| 训练实例 | 2 | 2 | PASS |
| 训练算法 | 2 | 2 | PASS |
| 测试实例 | 200 | 200 | PASS |
| 初始解 | 200 | 200 | PASS |
| PPO 推理 | 6,000 | 6,000 | PASS |
| GA restart | 3,000 | 3,000 | PASS |
| 启发式 | 500 | 500 | PASS |
| MILP | 250 | 250 | PASS |
| 全部 | 10,154 | 10,154 | PASS |

补充硬检查均通过：9,750 个 evaluation seed 全部唯一；9,750 个 `cell_id` 和 9,750 个 `logical_cell_id` 均唯一；10,154 个 seed 的派生偏差为 0；9,750 个 evaluation 身份重建偏差为 0；训练 seed 与测试／评估 seed 交集为 0；两份训练实例与 200 个测试实例的内容哈希交集为 0。

八个 namespace 通过排序后的高 3 位 lane 隔离，lane 0–7 依次为 `ga_restart`、`heuristic`、`initial_solution`、`milp`、`ppo_inference`、`test_instance`、`training_algorithm`、`training_instance`。每个 lane 内使用 SHA-256 canonical payload 的低 29 位；本次正式配置未发生 lane 内碰撞。

## 哈希锚点

| 对象 | SHA-256 |
|---|---|
| 协议 canonical hash | `24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba` |
| 源协议文件 | `db0313f644df5b93385bebfb00ae30edef6f1f0a66571edf4b5ec0fe48a18c43` |
| 正式协议文件 | `74edd93e5866d583eef7ffa17a004c9e92070a165260712761d549a691253257` |
| 正式 schedule | `7312dae1eaccf33b8f28b1421822be32a3e7f6310dc2751a90696db69d3c286a` |
| 正式 manifest | `43f9d52058a7b338b1081bfd5e617aed3f505000531e3cdee2aaf88a62244d9e` |

源协议与正式协议的 JSON 内容相等；canonical protocol hash 同时匹配 manifest 和 schedule；schedule 文件哈希匹配 manifest；manifest 为 `stage=verified` 且 verification 节点为 `passed=true`。

## 现有硬门缺口

正式 `verify_run` 会检查 cell 身份、输入与结果哈希、冻结策略状态、运行计数、表格、artifact 分层复核和 solver replay，但当前版本没有把“所有 seed 唯一”和“训练—测试 seed／实例内容互斥”作为显式机器门。该缺口不影响本次结果，因为本审计已对正式全量记录给出 PASS；`seed-audit.json` 是本次发布的补充机器可读证据。下一版协议可把相同断言并入正式 verify，在修改锁定源码前不应回写或伪造本次已完成的 verification report。

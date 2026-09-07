# Process Log: /scholar-analyze
- **Date**: 2026-09-04
- **Time started**: 16:00 CST
- **Arguments**: 完成《供应链管理写作/补充实验.md》规定的多实例统计与跨规模泛化实验，汇总至《数值实验与结果分析_论文稿.md》，参考本地众包论文的实验写法，并提交推送 GitHub。
- **Working Directory**: C:/Users/Yangfeifei/Documents/ChatGPT/危废品（transformer）
- **Design Type**: predictive-ML（优化算法的冻结模型留出测试与基线比较）
- **Safety Status**: CLEARED（实验实例由仓库代码合成生成，不读取用户提供的行级研究数据）

## Steps

| # | Timestamp | Step | Action | Output | Status |
|---|-----------|------|--------|--------|--------|
| 0 | 16:00 | Safety Gate | 输入为代码生成的合成算例，无用户行级数据；按公开/合成数据路径继续 | — | ✓ |
| A0 | 16:00 | Setup | 读取实验要求，选择 predictive-ML 路径并创建输出目录 | output/supplementary-experiments/ | ✓ |
| A1 | 16:18 | Protocol audit | 核对原入口、配置、求解器与复现工具；确认原入口为逐实例训练且冻结适配器会忽略测试实例 | run_experiments.py；src/ppo_improver.py | ✓ |
| A2 | 16:31 | Feasibility audit | 核对 CUDA、CPU、内存、磁盘、SciPy/HiGHS 与四档编码容量；估算去重后 9,750 个求解单元 | 内部审计记录 | ✓ |
| A3 | 16:38 | Reference style | 对众包参考论文实验章节进行文本提取与页面渲染核验，提炼“目的—口径—表格—关键数字—机制—边界”的写法 | 参考文献/众包模式下…pdf | ✓ |
| A4 | 16:45 | Implementation gate | 保留历史入口，新增冻结模型批量评估协议、两阶段checkpoint提交、原子cell账本、全量指标复算与计算回放 | `run_supplementary_experiments.py`、`src/supplementary_*`、测试 | ✓ |
| A4b | 18:24 | Paper-table pipeline | 新增经verify门控的论文表格、结果登记与裁决日志生成器；完成完整formal协议哈希门、实例级语义复算、optimal-only Gap、失败分类、全方法/MILP状态附表和原子inventory整改 | `output/supplementary-experiments/scripts/build_paper_tables.py`；`tests/test_build_paper_tables.py` | ✓ |
| A5 | 17:22 | Integration gate | 完成46项单元测试及15-cell全流程冒烟；再次执行时15/15跳过，verify实际重跑15/15且通过，原子export通过 | `outputs/supplementary_experiment_smoke_ultra_v2/` | ✓ |
| A6 | 17:23 | Formal training | 以冻结协议训练Train-S与Train-L各一次；训练耗时分别为778.809 s与1233.161 s，生成且仅生成2份checkpoint | `outputs/supplementary_experiment_v1/models/` | ✓ |
| A7 | 17:56 | Held-out lock | 在两模型冻结后生成并锁定Test-1—Test-4各50个未见实例及共同初始解；形成9,750个去重cell | `outputs/supplementary_experiment_v1/schedule.json` | ✓ |
| A8 | 17:57 | Formal evaluation | 先完成60-cell（4 worker）与120-cell（8 worker）正式PPO吞吐基准，均0失败；其余PPO按8 worker可恢复执行 | `outputs/supplementary_experiment_v1/cells/` | 进行中 |

## Review Gates

- Spec复核：协议计数、冻结推理、失败/缺失统计、同一初始解与真实计算回放均已通过；剩余事项仅为完成正式实验与论文交付。
- Standards复核：46项测试通过；两阶段checkpoint、Windows中断、源码bundle、全量指标重算、cell ledger及verify/export哈希链均无阻塞问题。
- 下游表格独立复核：无阻断项。核心聚合顺序、样本SD、PPO相对GA改善率、MILP optimal-only Gap、2×4宏平均与风险分解计算正确；formal完整配置哈希、全量cell完整性与分层回放计数、实例级主质量/可行率/配对改善率复算、失败抑制、三方法统一Gap门、历史并发批次、启发式/MILP透明度、全方法附表及原子inventory均已核验。专项测试12/12、全套测试58/58、真实15-cell smoke CLI与16项输出哈希闭环均通过。
- 图形计划：本轮核心关系由精确的5偏好表和2×4矩阵更清楚地表达，暂不新增图形；若后续需要视觉摘要，再单独确认图形计划。

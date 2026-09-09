# 单实例近似帕累托实验 v3

本目录是本轮新实验，不能与 `reference_generalization_v2` 或旧五偏好单目标结果混用。当前源协议已锁定，正式状态以 `completed/` 与独立审计报告为准，不以本说明文档代替完成证据。

## 固定对象

- 表4：保留历史小案例，实例种子3723524230，参考方案种子3949079119；不是新Test-1案例。
- 表5与表6：Test-1至Test-4各使用已锁定测试库的000号一个实例；L20与Test-4同例。
- 图2：Test-4基础网络的9个单因素参数版本（共用基准），不是9个独立随机实例。
- PPO只加载登记的New-S/New-L冻结模型，不训练、微调、换模型或按测试成绩改配置。每实例/参数版本固定自身b，输入和选解同尺度。

## 正式运行清单

| 部分 | 完整前沿／求解数量 | 说明 |
| --- | ---: | --- |
| 两模型四规模PPO | 24 | 2模型×4案例×3完整运行 |
| 敏感性新增PPO | 24 | 8个非基准版本×3运行，基准复用 |
| NSGA-II | 12 | 4案例×3运行，Test-4在B1后继续B2 |
| 小案例PPO | 15 | 5偏好×3运行，表中取同一最小J实际方案 |
| 小案例MILP | 5 | 每偏好1次，3600秒上限 |

一次PPO前沿为21组权重×60步×32候选=40320候选尝试。NSGA-II B1=40320，B2=120960；失败及重复也消耗尝试。随机种子、源哈希、数据文件哈希、模型身份详见 `protocol.json`。MILP有独立 `milp/protocol.json`，实例与固定b与主协议逐字段核验。

## 文件索引

- `instances/`：固定参数、绑定b、完整参考方案、身份哈希。
- `fronts/`：每次完整前沿的原始指标、候选计数、计时、模型与方案路径。`preflight-B64.json`仅预检，不进入正式汇总。
- `solutions/`：每个交付前沿点的完整收运计划，按实例与方案SHA256寻址；严格回放可重得C、R。
- `small/`：15次冻结PPO小案例实际方案和计时。
- `milp/`：5个MILP完整原始变量、约束验证、日志、时限与最优性状态。
- `milp/validated/`：五组原始返回向量按统一数值容差规范化后的派生记录；保存原文件及规范化源码SHA256、每个变化量、完整派生向量、约束残差和原目标差，不覆盖原始记录。
- `completed/`、`progress/`、`failures/`：正式完成标记、在运行进度、技术失败证据（若有）。
- `protocol-drafts/`：正式运行前预检修订的旧草稿，保留而不混入正式结果。
- `tables/`：逐点、逐前沿、逐配对、表4选解及敏感性端点CSV。
- `figures/`：图2的可编辑SVG和350 dpi PNG。
- `audit/`：独立数值核验结果。

## 复现入口

在仓库根目录运行：

```text
py -B run_pareto_experiments.py prepare
py -B run_pareto_experiments.py preflight
py -B -u run_pareto_experiments.py run --workers 16
py -B -u run_pareto_milp.py
py -B normalize_pareto_milp.py
py -B report_pareto_experiments.py
py -B audit_pareto_delivery.py --require-report
```

正式结果存在时入口按任务复用；源码、实例或冻结模型不匹配时停止，不覆盖已锁定结果。若重新设计协议，应另建新版本，不删除不利结果后补跑。本轮未设置任何自动重训入口。

只重建表图和论文时执行`py -B report_pareto_experiments.py`，不运行求解或训练。最后执行`py -B audit_pareto_delivery.py --require-report`完成独立验收，不能把仅有原始运行的部分审计状态当成交付通过。

Git克隆后应保留锁定源码、JSON、Markdown、CSV和模型文件的原始字节及换行；仓库`.gitattributes`为SHA锁禁止这些文件的自动文本换行转换。若哈希不一致，先检查克隆和传输是否改变字节，不自动重算manifest或源哈希来掩盖差异。分发包必须同时包含两份冻结pt、对应训练记录、完整`datasets/reference_generalization_v2/manifest.json`与实例数据，不能只上传模型登记而漏掉模型本体。

## 统计和计时边界

覆盖率为弱支配（含相等），分母为第二个解集点数，容差为固定b归一化后1e-8。各档案分别去重并非支配筛选；按3个运行编号配对后求均值和样本标准差，不能将21偏好或前沿点数当样本数。空前沿记失败/NA，不代填0或1。

前沿搜索段时间覆盖本次全部权重或整个演化过程及严格档案维护，不包括模型加载、公共准备和最终导出；这些投入另有任务总时间及准备时间。NSGA-II B2时间为从该次运行起累计至B2，不是增加预算段的增量。小案例PPO时间按3次搜索累加，其他表的前沿时间按3次运行求均值±标准差。16进程共享CPU的墙钟时间不代表隔离运行的纯算法速度。

MILP表值统一读取`milp/validated/`中的同一完整变量方案。全部五组按既有10⁻⁵可行容差舍入近整数变量、将绝对值小于10⁻⁵的连续量置零，然后重查完整原模型矩阵、变量界、整数性及共同评价器，且与原求解器目标的差满足10⁻⁷绝对/相对容差。原始p3/p4的目标不一致标志、原始指标和完整向量仍保留，没有追加求解调用、增加预算或改变最优状态。表4 MILP时间沿用原始求解运行墙钟，不包括离线规范化与审计；规范化自身时间另存`cleanup_seconds`。

`candidate_attempts`是尝试数；`candidate_objective_evaluations + ppo_internal_objective_evaluations + return_validation_evaluations`是搜索段目标评价器调用次数（不含公共参考准备或离线表图核验）。最后一项是PPO每偏好返回方案的额外严格验证，每完整前沿21次；NSGA-II为0。该项按已保存的`final_solutions`记录数汇总，不修改原始观察器计数或尝试预算。缓存不免费增加尝试；不可行尝试、不可行新评价和动作/修复返回失败分别记录。

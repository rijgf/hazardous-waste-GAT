# Standards 轴代码审查

- 固定点：`0a71b0bb1b2445ce90989d453cfba357d0d12458`（当前HEAD与固定点相同；范围为未提交工作树及新增文件）。
- 仓库未发现适用于本项目的 `AGENTS.md`、`CONTRIBUTING.md` 或 `CODING_STANDARDS.md`；以下依据现有代码约定与代码异味基线。
- 验证：`py -m unittest discover -s tests -v`，60/60通过。`py -m pytest -q` 无法运行，因为当前解释器未安装pytest；测试本身均为unittest。

## 可操作问题

### [P1] 无操作的重复命令会撤销已通过的正式验证

`src/supplementary_experiment.py:557,569-575,701-706` 在确认所有checkpoint均可复用前就删除verification，随后即使没有训练任何模型也把stage降为`models_frozen`；`src/supplementary_experiment.py:716,725-742,799-801` 对已锁定测试集重复调用时同样降级；`src/supplementary_experiment.py:1311-1343,1405-1424` 在`pending=[]`时仍删除verification并写成`evaluation_in_progress`。因此，对一个`verified` run执行常见的恢复/幂等命令会使有效证明失效，尽管任何cell、模型或测试集都没变。应只在首次实际写入/重试产物时失效验证；纯校验或全量skip应保持原manifest状态，并增加三个verified-run重复调用回归测试。

### [P2] PPO并发门禁按独立过滤域判断，误拒绝不含PPO的组合

`src/supplementary_experiment.py:1287-1310` 用`method_filter or domains["methods"]`判断是否选择PPO，却没有与`models/scales`过滤后的真实条目求交。例如`--models GA --workers 2`只会选择GA，但因未传`--methods`仍被要求`--allow-concurrent-ppo`；错误地传入不相容的`--methods ga --models Train-S`又会落入上一个无操作降级问题。应先形成过滤后的entry集合，再对其中是否实际含PPO执行门禁，并测试仅model过滤、空交集和已完成PPO三种情况。

### [P2] manifest登记的训练历史在正常验证与导出路径上没有完整性检查

模型记录保存`history_file/history_sha256`（`src/supplementary_experiment.py:665-678`），但已登记模型的快速路径只核对checkpoint（`:569-575`），`lock_test_sets`也只核对checkpoint（`:721-724`），`verify_run`不核对训练历史，`export_replication_summary`则直接复制目录文件（`:2337-2345`）。训练历史被删除或篡改后，验证仍可PASS，导出甚至可能生成与内嵌manifest矛盾但自哈希一致的复现包。应抽取统一的模型产物校验器，核对checkpoint、policy identity、history文件及hash，并在train复用、lock、verify、export入口调用；增加删除/篡改history的失败测试。

### [P2] publication发布会保留旧版本的未声明成员

`output/supplementary-experiments/scripts/build_paper_tables.py:3171-3189` 只替换本次staging中的文件并最后发布inventory，没有拒绝或清理输出目录里的旧成员。未来删除/重命名一种表后，旧文件会与新inventory并存；inventory虽自洽，但目录不再是注释所称的“one coherent publication set”。应在发布前将现有成员与旧inventory及新成员集合对账，并采用目录级交换，或对无法安全归属的额外文件直接失败；新增“预置未声明旧文件后重建”的测试。

## 汇总

P0：0；P1：1；P2：3；P3：0。存在1项应在合并前处理的状态机阻断问题；未发现P0数据破坏或安全问题。

影响界定：上述4项不改变本次formal run按其实际执行顺序已经生成的9,750个执行结果、当前汇总统计或现有验证哈希链；它们分别影响verified run之后重复调用阶段命令的幂等恢复、组合过滤的易用性、训练历史遭后续删除或篡改时的复现防护，以及未来publication产物集合变更时的目录闭合性。该范围界定不改变上述优先级。

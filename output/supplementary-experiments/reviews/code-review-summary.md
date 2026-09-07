# 补充实验实现双轴代码审查汇总

- 固定点：`0a71b0bb1b2445ce90989d453cfba357d0d12458`
- 规格源：`供应链管理写作/补充实验.md`
- 审查对象：固定点之后的当前工作树、新增实验产物及本机正式运行目录

## Standards

完整报告见 `code-review-standards.md`。审查发现 P0=0、P1=1、P2=3、P3=0：

1. P1：对已验证运行重复调用 `train`、`lock-tests` 或无待处理单元的 `evaluate` 时，纯 no-op 仍会撤销验证状态。
2. P2：并发 PPO 门禁没有先按 method/model/scale 的真实交集形成待执行集合，可能误拒绝只含非 PPO 的过滤组合。
3. P2：训练历史虽在 manifest 登记哈希，但正常 verify/export 路径尚未统一执行历史文件完整性检查。
4. P2：publication builder 原子发布声明成员，但未来删表或改名时尚未拒绝输出目录中的旧版未声明成员。

这四项不改变本次 formal run 按实际执行顺序生成的 9,750 个结果、当前汇总统计或既有验证哈希链；影响集中在验证后幂等恢复、过滤易用性、训练历史的后续篡改防护和未来发布集合变更。全量 `unittest` 为 60/60 通过。

为避免在不重跑正式实验的情况下破坏 `source_changes_since_init={}`，本轮处置及下一协议版本的验收项记录在 `code-review-disposition.md`；不对已验证 source bundle 作追溯性修改。

## Spec

完整报告见 `code-review-spec.md`。审查结果为 PASS，P0=P1=P2=P3=0。两模型冻结测试、四组各 50 个未见实例、2×4 交叉设计、3/3/1/1 重启、实例级分母、MILP optimal-only Gap、9,750 个唯一执行单元、75 项 artifact metric recomputation、15 项 solver replay、表 3—7 和无敏感性/消融边界均与规格一致。

正式目标稿尚未替换属于 `scholar-analyze` 的用户验收门控动作，不计为规格缺陷；用户验收后应以审定稿覆盖目标并再次核对哈希、占位符和 Git 提交内容。

## 一行结论

Standards 轴共 4 项发现，最严重为验证后 no-op 命令撤销已验证状态（P1）；Spec 轴 0 项发现，无规格偏差。

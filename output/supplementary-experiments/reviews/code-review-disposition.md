# 双轴代码审查发现的处置决定

## 决定

Standards 轴的 1 项 P1 与 3 项 P2 均登记为下一协议版本的维护项，本轮不修改已生成正式结果所绑定的 13 个 source files。原因不是忽略缺陷，而是当前 verification 已证明 `source_changes_since_init={}`；在没有重跑 9,750 个正式求解单元的情况下追溯修改这些文件，会破坏正式结果与执行源码之间的哈希对应关系。

Spec 轴为 PASS，未发现实验结果或论文交付与 `供应链管理写作/补充实验.md` 不一致。

## 对本轮证据的影响

| 发现 | 本轮影响判定 | 当前控制 | 下一版本验收项 |
|---|---|---|---|
| verified run 后的纯 no-op 命令会降级状态 | 不改变已执行单元、表格或当前 verification；影响后续重复调用的幂等性 | 不再对正式目录执行无必要的 `train`、`lock-tests` 或空 `evaluate`；如误触，须重新运行 `verify` | 三类 verified-run no-op 回归测试；无实际写入时保持 manifest 和 verification 不变 |
| PPO 并发门禁未按组合过滤的真实交集判断 | 不影响本次六个正式 evaluation run；影响部分 CLI 过滤组合的易用性 | 正式命令显式给出 method/model/scale 与并发声明 | 先形成真实选中 entries，再判断是否含 PPO；覆盖仅 model、空交集与已完成组合 |
| 核心 verify/export 未统一核验训练 history 哈希 | 未发现本次历史缺失或篡改；replication inventory 与独立审计已登记当前文件哈希 | 保留 manifest、训练历史与紧凑归档哈希；不把现有检查夸大为 core 硬门 | 统一模型产物校验器覆盖 checkpoint、policy identity、history file/hash，并加删除与篡改负向测试 |
| publication 未来改名/删表时可能保留旧成员 | 当前目录已经独立核对，无漏列或多列，18项声明输出与实际非inventory文件完全一致 | 以当前 paper inventory 和 `formal-artifact-audit.md` 为发布闭合证据 | 发布前核对旧/新成员集合，目录级交换或对未声明旧成员失败，并加预置旧文件测试 |

## 边界

此处置只适用于已经完成并验证的 `supplementary-experiment-v1`。下一协议版本若修改核心源码，必须重新锁定 source bundle、重新生成正式运行证据并再次执行完整验证；不得把本轮 verification 继续用于修改后的实现。

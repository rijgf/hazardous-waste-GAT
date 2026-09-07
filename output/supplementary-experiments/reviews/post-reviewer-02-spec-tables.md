# 审定稿复审 2：规格与表格产物

## 结论

- 判定：**PASS**
- 评分：**100/100**
- 问题：阻断 0，重要 0，轻微 0
- 审查对象：`output/supplementary-experiments/drafts/数值实验与结果分析_审定稿.md`
- 审查对象 SHA-256：`c6b1adf87b488d49820f73fcc73ade402e643c80b91b3c4818d7ab2da702721c`
- 规格源：`供应链管理写作/补充实验.md`

未发现需要修订的问题。以下为逐项复核证据。

## 1. 表 3–7 与三张生成附表

对 8 份 builder Markdown 去除各自一级标题及其后的空行后，将剩余表头、分隔行、全部数据行、空行和表注作为连续文本，与审定稿逐行、区分大小写比较。每一份均在稿件中精确命中 1 次，差异为 0：

| 生成文件 | 源正文行数 | 稿件位置 | 精确命中 | 判定 |
|---|---:|---|---:|---|
| `output/supplementary-experiments/tables/table3_small_scale.md` | 9 | `审定稿.md:49-57` | 1 | PASS |
| `output/supplementary-experiments/tables/table4_large_scale.md` | 9 | `审定稿.md:79-87` | 1 | PASS |
| `output/supplementary-experiments/tables/table5_scale_design.md` | 10 | `审定稿.md:105-114` | 1 | PASS |
| `output/supplementary-experiments/tables/table6_generalization_matrix.md` | 6 | `审定稿.md:120-125` | 1 | PASS |
| `output/supplementary-experiments/tables/table7_generalization_operational.md` | 12 | `审定稿.md:133-144` | 1 | PASS |
| `output/supplementary-experiments/tables/appendix_all_methods.md` | 39 | `审定稿.md:206-244` | 1 | PASS |
| `output/supplementary-experiments/tables/appendix_risk_components.md` | 39 | `审定稿.md:248-286` | 1 | PASS |
| `output/supplementary-experiments/tables/appendix_milp_status.md` | 13 | `审定稿.md:290-302` | 1 | PASS |

主表标题文字也与生成稿一致，仅由生成文件的 Markdown 一级标题改为论文稿的加粗表题（`审定稿.md:47,77,103,118,131`）。三张生成附表的源标题本来是无编号的通用“附表”，稿件依连续编号要求分别规范为表 A2、A3、A4；稿件已明确披露只规范化标题、表体与表注不变（`审定稿.md:202-204,246,288`），不构成内容差异。

上述 8 个源文件的实际 SHA-256 还与稿件表 A9 的登记逐项一致（`审定稿.md:369-380`），分别为：

- 表 3：`15ed52b9aff43ef9cda76518bc61c51d9fb86398a02354fa04cf3f4b59dc3995`
- 表 4：`9b758cba04451832f7332db5deacd28b5dbdea54eb040ba849dba1b6e311b647`
- 表 5：`d10fcc7bee2a4bcd686918c71483a7d0512699761656f4033171ae0a86acf1a1`
- 表 6：`08c797c0891500258a60fa9652300a354f271f80b8c9866ad4e9606d651fe746`
- 表 7：`789c07234c517a05b95f6f0eededfc33b02577b7f3dd640f7a0d6193623f8d8d`
- 全方法附表：`ecd1b7f9bf7450d40e46d8e02612d9eb1681ec32f7c4ae4a5a99a3e53e771ae7`
- 风险构成附表：`0cdaf4e799587819c229931923358cd913cc3848feb85da0b155d95e40d5d740`
- MILP 状态附表：`ed96bee58e5cc5d35c9ff370634d51a4b4557264a66f93088cdc1e3149f4ec1d`

## 2. 表 5 的协议派生

表 5 的 6 个对象行全部可从锁定正式协议直接派生，6/6 精确一致、差异 0：

- 两个训练行来自 `outputs/supplementary_experiment_v1/protocol_config.json` 的 `models`：Train-S→Test-1→`small_model.pt`、Train-L→Test-4→`large_model.pt`；源配置同样见 `configs/supplementary_experiment.json:41-51`。
- 四个测试行来自 `test_scales`，规模向量依次为 3/2/2/3/2、6/2/2/4/3、10/3/3/5/3、20/4/3/8/4，实例数均为 50（`configs/supplementary_experiment.json:53-94`）。
- builder 从模型的 `train_scale`、checkpoint、五个规模字段和协议实例数构造训练/测试行（`output/supplementary-experiments/scripts/build_paper_tables.py:1889-1986`）；稿件表 5 的 6 行与生成结果一致（`审定稿.md:107-112`）。
- `py -B -m unittest tests.test_build_paper_tables -v`：14 项全部通过，其中 `test_table5_is_derived_from_the_locked_formal_protocol` 通过。

## 3. A1–A9 编号与交叉引用

附表定义连续且各出现一次：A1（`审定稿.md:184`）、A2（`:204`）、A3（`:246`）、A4（`:288`）、A5（`:306`）、A6（`:320`）、A7（`:337`）、A8（`:352`）、A9（`:371`）。附录小节 A.1–A.7 也连续且无重复（`审定稿.md:176,180,200,304,318,335,350`）。

全部“表 3–7／表 A1–A9”引用均能解析到唯一表定义，无孤立引用或重复定义。重点语义引用正确：

- 表 A2 被用于启发式和全方法结果说明（`审定稿.md:61,73,91`），对应“E1/E2 全方法结果”。
- 表 A3 被用于 E1/E2 风险分项及偏好响应说明（`审定稿.md:65,93,160`），对应“风险构成”。
- 正文对附录 A.5 和 A.6 的环境、并发计时引用分别指向正确小节（`审定稿.md:35-37,318-350`）。
- 表 3、4、5、6、7 的表前口径和表后分析分别位于其正确实验段落，未发生旧稿表号串用（`审定稿.md:39-154`）。

## 4. E1、E2 与 G 设计

设计与 `供应链管理写作/补充实验.md:20-30,51-67,113-176` 一致：

- E1：固定 Train-S，Test-1 的 50 个未见实例；PPO/GA 各 3 个求解种子，启发式/MILP 各 1 次；MILP 时限 180 s；只对 status=optimal 的逐实例基准计算 Gap并报告有效 n（`审定稿.md:9-11,27-33,39-67`）。
- E2：固定 Train-L，Test-4 的 50 个未见实例；PPO/GA 各 3 个种子，启发式 1 次，不运行 MILP、不报告 Gap（`审定稿.md:69-95`）。
- G：Train-S 与 Train-L 分别交叉 Test-1…Test-4，形成 2×4 共 8 个组合；每规模 50 个实例、5 个偏好、3 个 PPO 种子；GA 每规模只建一套基线，端点复用 E1/E2 cell 而非重复计数（`审定稿.md:97-154`）。
- 表 6 先在实例—偏好内汇总 3 个种子，再在实例内等权汇总 5 个偏好，最后以 50 个独立实例计算均值与样本标准差；表 7 保留完整分母和异常计数（`审定稿.md:116-150`）。
- 稿件明确本轮没有敏感性分析或消融实验（`审定稿.md:154,168`）。

## 5. 目标替换验收门（非问题）

审查时 `供应链管理写作/数值实验与结果分析_论文稿.md` 仍是验收前旧稿。委托者已明确这是 scholar-analyze 流程的用户验收门，不是审定稿或实验实现缺陷，因此不扣分、不列为 REVISE 项。

验收通过后应将本次已复核的审定稿写入该目标路径；若届时仍提交旧目标稿，才会构成交付缺口。本复审没有修改审定稿或目标稿。

## 问题清单

无。

**最终判定：PASS，100/100。**

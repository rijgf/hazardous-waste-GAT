# Reviewer 1：计算与统计聚合复核

## 审稿结论

**REVISE（93/100）。** 未发现会改变表3、表4、表6、表7或主要结论方向的计算错误；当前稿件的公式、重启聚合顺序、样本SD、有效$n$、MILP Gap限定、风险端点、可行性与时间数字均与正式数据一致。但是，实验G尚应显式补足400/400和2000/2000的总分母，并披露逐“实例—偏好”配对中存在19个实质负值；否则“所有组合的均值均为正”容易被误读为逐单元支配。

- 阻断问题：0项。
- 重要问题：1项。
- 轻微问题：3项（均为口径/溯源措辞，不改变结果）。

## 终审快照与证据范围

- 稿件：output/supplementary-experiments/drafts/数值实验与结果分析_正式结果初稿.md，378行，56078 bytes。
- 终审SHA-256：3fee329fe38c80beeafad049cb89f8c6c2860906fef22506dec44795b3fee912。
- 唯一数据源：outputs/supplementary_experiment_v1 的正式run及其复制到 output/supplementary-experiments/replication/tables/ 的正式CSV；未使用中途输出或历史结果。
- 正式身份：manifest.stage=verified、smoke=false；protocol_sha256=24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba，source_bundle_sha256_at_init=50ff1564043fcec7bc573184f4f7aed02b6df99883440795c3bbb62e929fff6e。
- verification_report.json：passed=true、count_match=true、summary_ledger_match=true；missing_count=failed_count=integrity_error_count=artifact_error_count=solver_replay_error_count=0。
- paper-table-inventory.json登记的16/16个输出文件哈希和9/9个CSV行数与当前文件一致。表3、表4、表6、表7及三份附表的Markdown主体均原样出现于稿件（7/7），所列数字未发现手工转录差异。

## 独立复算结果

### 1. “3次重启→实例→50实例”的聚合顺序正确

raw_results.csv共9750行：PPO 6000、GA 3000、启发式500、MILP 250。以 method, model_id, test_scale, instance_index, preference_id 为键重组后，PPO和GA每键均恰有3行，restart_index={0,1,2}；启发式和MILP每键1行，restart_index=0。instance_level_results.csv共3750行，其 expected_restarts 与 observed_restarts 对PPO/GA均为3，对启发式/MILP均为1。

对 runtime_seconds、cost、risk、weighted_objective_normalized 及四项风险分量等14个字段，独立从原始行按上述键取重启均值，与实例层表的最大绝对差介于0和1.09e-11（纯浮点舍入）。再按偏好在50个实例上求均值和 ddof=1 样本标准差，可逐项复现表3和表4。G中则在此基础上先对每个实例的5组偏好等权平均，再对50个实例求均值和样本标准差，可逐项复现表6和表7。稿件第11、33、43、45、73、75、101、116、127和142行的统计单位表述与此一致。

### 2. 所有“±”为样本SD，未将SE误作SD

正文和附表的“均值±”均可以实例层 ddof=1 复现。例如，E1的$(1,0)$ PPO $J$为 mean=0.8044305160、SD=0.1279547753、n=50，因而表3正确显示0.8044 ± 0.1280。results-registry.csv中同一指标的 metric_se=0.01809553786=SD/sqrt(50)，该SE没有被写入表内“±”位置。稿件第33、45、57、75、87、116、125、142、242和284行的“样本SD”标注正确。

### 3. $\Delta J$、Gap与$L$的口径和数值正确

- $\Delta J$在同一实例—偏好上以已完成重启均值的 ppo_j 和 ga_j 按 (ga_j-ppo_j)/ga_j×100 计算，再向上聚合。E1五组均值为4.286288721、4.858595686、6.418182536、22.645576275和66.751941506；先在实例内对五组偏好等权平均后为20.992116945 ± 7.102537109，n=50。E2对应五值为18.807454120、19.311527618、21.072267647、27.534160811和46.220897869，宏平均为26.589261613 ± 1.551140263，n=50。稿件第21—25、61和91行及表3/4/7均正确。
- Gap只使用 milp_gap_by_instance.csv 中 milp_status=optimal 的同实例、同偏好配对。MILP五组 optimal n 为50、49、26、2、2，合计129/250；非optimal为121/250。PPO Gap五组为0.958081848、0.942860136、4.062381973、2.494200835、1.141974173，GA为6.230080495、6.908406130、12.672432057、10.632428489、113.475463245，并使用上述变化的有效$n$。稿件第27—33、45、51—55、63、300、312和378行数字正确，且没有用非optimal incumbent作$J^*$。
- $L$按同一测试规模下两份模型较低的宏平均$J$为分母计算，Train-L在Test-1至Test-4上的$L$为1.392088420%、1.014424307%、0.806034353%和1.295684096%。第116、125和146行明确其不是MILP Gap或严格泛化损失，口径正确。

### 4. 有效$n$、可行性、期末清零和异常分母

9750个原始单元的 status 全为 complete，feasible、terminal_inventory_clear 和 quality_included 全为true，violation_count全为0。3750个实例—偏好层行的 feasible_rate 和 terminal_clear_rate 全为1，technical_failure_rate 和 missing_rate 全为0。因此，E1/E2每个偏好的质量n=50、可行率100%和异常0正确；G每行50/50宏平均实例、250/250质量/配对单元、100%可行率和100%清零率也正确。

G的8个模型—规模组合合计应报告400/400个完整宏平均实例（8×50）和2000/2000个同实例—偏好质量/PPO–GA配对单元（8×50×5）。以1e-9为浮点容差，2000个 delta_j_percent 中1871个为正、110个持平、19个为负，取值范围为-5.911326629%至89.413743845%。但先在实例内对五偏好等权平均后，400个宏平均改善率全为正，范围9.112791616%至45.074516838%；40个“模型—规模—偏好”组均值也全为正，范围3.781953677%至73.766089239%。因而“八个组合的宏平均均为正”成立，“2000个配对均为正”不成立。

### 5. 风险端点、偏好响应与时间数字

- E1 PPO成本1233.839972→2782.642668，总风险10.016731→2.667511；运输风险5.132098→2.495935，共载风险0.256402→0.063372，产废端库存风险4.628231→0.108203，处理处置端库存风险全为0。GA的产废端库存风险在五组偏好上均为4.6282312。第65、154和156行的数字与方向正确。
- E2 PPO成本11927.205842→24882.723501，总风险859.488420→488.022494；运输、共载和产废端库存风险的端点降幅分别为191.358539、72.661679和107.445708，因此“运输风险绝对变化最大”正确；处理处置端库存风险全为0。GA的产废端库存风险在五组偏好上均为356.19750792。第93、154和156行正确。
- E1和E2的PPO五点均值的成本均单调不降、总风险均单调不升，第154和160行的离散偏好表述成立；稿件没有将该结果外推到连续偏好空间。
- E2 PPO单元观测时间的五偏好均值范围为36.2957—38.0994 s，GA为18.8027—18.9804 s；G中Train-S为20.4506—30.4859 s，Train-L为21.9795—36.9065 s。前三个PPO evaluation run的墙钟之和为18914.6408278 s，稿件正确显示18914.641 s；GA和MILP run墙钟分别为1447.7234791 s和3651.4613948 s。第35、57、87、91、142、148、168和331—340行正确区分了cell求解段、方法级run墙钟、一次性训练与启发式初始化时间，也未作隔离资源下的速度因果排序。

### 6. 其他正文数字全量核对

- 第9—11行的训练时间、50实例、5偏好、3/1次重启、180 s时限和6000+3000+500+250=9750均与protocol/manifest/schedule一致。
- 第37行与表A3的操作系统、Python、NumPy、pandas、SciPy、PyTorch、CUDA和GPU版本/数量均与manifest一致。
- 第43、73和105—114行的四种规模数量与protocol一致。
- 第146和148行的8个宏平均$J$、4个$L$、两模型的改善率区间与时间区间均可从 generalization_operational.csv 以50个宏平均实例复现。
- 第176行的生成参数、容量系数、风险系数、概率、种子根与派生方案均与protocol一致。独立收集2个训练实例种子、2个训练算法种子、200个测试实例种子、200个初始解种子以及9750个求解种子，合计10154个，全部唯一且组间无交集，因而“训练与测试种子集互斥”成立。
- 第180行的网络与PPO训练/评估参数、第188—191行的训练种子/时间和第196行的GA/MILP参数均与protocol及manifest一致。
- 第306—314和376—378行的计数与verification字段一致：无missing/failed/integrity error，MILP为129 optimal和121 nonoptimal，artifact_rechecked_count=75、solver_replayed_count=15，对应错误均为0。
- 第346—370行所列protocol、配置、schedule、source bundle、manifest、verification、训练实例、检查点、训练历史与7份Markdown产物哈希均与对应登记值一致；但第354行的 bff564… 是cell账本行的canonical JSON hash，不是最终CSV文件哈希，见下方轻微问题2。

## 需修订问题

### 重要-1：实验G缺总分母和逐配对方向异质性披露

- **位置：** 第127、148行；第170行结论也应与修订后口径一致。
- **现文：** 每行报告50/50和250/250，并说“所有组合的均值均为正”。这些数字本身正确，但未给出8行合计分母，也未提示2000个逐实例—偏好配对并非全部为正。
- **证据字段：** table7_generalization_operational.csv::{expected_instances, valid_macro_j_instances, expected_paired_instance_preference_cells, paired_instance_preference_cells}；paired_ppo_vs_ga.csv::{model_id, test_scale, instance_index, preference_id, delta_j_percent}。
- **建议在第148行改为/补入的精确句子：**

> 八个模型—测试规模组合合计保留400/400个完整宏平均实例和2000/2000个同实例—偏好PPO–GA配对单元。40个“模型—规模—偏好”组的平均$\Delta J$均为正；但在2000个逐实例—偏好配对中，以$10^{-9}$为数值容差，1871个为正、110个持平、19个为负；而先对每个实例的五组偏好等权平均后，400个实例宏平均改善率均为正。因此，表7支持平均表现的正向差异，不表示PPO在每个配对单元上都优于GA。

### 轻微-1：表3将所有非optimal incumbent统称为“限时解”

- **位置：** 第57行。
- **问题：** 正式MILP状态只能确认129个 optimal 与121个 solver_status_1。solver_status_1不足以单独区分限时与其他中止；第300行“包括限时或其他中止状态”反而更准确。
- **证据字段：** appendix_milp_status.csv::{solver_status, status_n, eligible_as_j_star}；instance_level_results.csv::solver_status。
- **建议修文：** 将“非最优限时解不进入Gap”改为“任何非optimal incumbent（包括限时或其他中止）均不进入Gap”。

### 轻微-2：cell账本的canonical hash与CSV文件哈希应显式区分

- **位置：** 第354行。
- **问题：** bff5643b…正确对应manifest/verification的 summary_input_cell_ledger_sha256，即由 cell_id、status、cell_file_sha256 行构成的canonical JSON hash；当前 tables/cell_file_ledger.csv 的实际文件哈希是67b0ade133ee104a244b7e92a06905013a529ed4ac7039e3ff8a424691c61e05，并已由verification和inventory登记。现有“汇总输入cell账本”可被理解为前者，但表A5的列名是“SHA-256或状态”，存在被误读为CSV文件哈希的可能。
- **证据字段：** manifest.summary_input_cell_ledger_sha256；verification_report.summary_input_cell_ledger_sha256；verification_report.table_files_sha256['cell_file_ledger.csv']；paper-table-inventory.input_files_sha256['tables/cell_file_ledger.csv']。
- **建议修文：** 将对象名改为“汇总输入cell账本（canonical row hash）”；若需同时披露文件哈希，再单列67b0ade1…。

### 轻微-3：75与15的验证对象可表述得更精确

- **位置：** 第376行。
- **问题：** artifact_rechecked_count=75计数的是成功重新读取方案并复算可行性、plan hash和质量指标的分层抽样cell；它不是“75个trace的算法语义重放”。trace文件的哈希则在全量cell完整性检查中核验。solver_replayed_count=15才是solver重放计数。
- **证据字段：** verification_report::{artifact_rechecked_count, artifact_error_count, solver_replayed_count, solver_replay_error_count, integrity_error_count}。
- **建议修文：**

> 在全量9750个cell的身份与文件完整性核验基础上，验证流程另对75个分层抽样cell重新计算方案质量指标，并重放15个solver cell；对应错误均为0。

## 评分依据

- 聚合顺序与样本统计：25/25。
- $\Delta J$/Gap/$L$公式、配对与有效$n$：25/25。
- 主表、附表、风险端点与正文数字：25/25。
- 分母、方向异质性与验证口径披露：18/25。

合计：**93/100**。完成上述1项重要修订和3项轻微口径修订后，本评审可转为PASS。现稿没有宣称统计显著性、因果效应、逐单元支配或未知规模的普遍泛化，这一结果边界应保留。

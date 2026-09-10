"""Rebuild only NSGA-dependent manuscript results; retain frozen PPO/MILP."""
import argparse
import csv
import json
import statistics as st

from run_nsga_red_budget import ROOT, BASE, OUT, PAPER, read, write, verify
from run_nsga_parameter_probe import hv
from src.pareto_experiment import coverage_pair
from src.reproducibility import file_sha256


def ms(values,digits=4):
    return f'{st.mean(values):.{digits}f} ± {st.stdev(values):.{digits}f}'


def front(root,name):
    d=read(root/f'fronts/{name}.json')
    return {k:d[k] for k in ('points','task','budget','seconds','counts','instance_id','instance_sha256','b_C','b_R','reference_plan_sha256','archive_id')}


def data():
    protocol=verify();nsga={};ppo={};pairs=[];timing=[];diagnostics=[]
    for task in protocol['tasks']:
        done=read(OUT/f"completed/{task['id']}.json")
        for path,h in done['files'].items(): assert file_sha256(OUT/path)==h
        for budget in task['budgets']:
            nsga[task['instance'],task['repeat'],budget]=front(OUT,f"{task['id']}-B{budget}")
    assert len(nsga)==15
    for scale in range(1,5):
        name=f'Test-{scale}'
        for model in ('small','large'):
            group=[]
            for repeat in range(3):
                p=front(BASE,f'PPO-{name}-{model}-r{repeat}-B252000');ppo[name,model,repeat]=p;group.append(p)
                for budget in ((40320,120960) if scale==4 and model=='large' else (40320,)):
                    n=nsga[name,repeat,budget]
                    assert (p['instance_sha256'],p['b_C'],p['b_R'])==(n['instance_sha256'],n['b_C'],n['b_R'])
                    c=coverage_pair(n['points'],p['points'],(p['b_C'],p['b_R']))
                    pairs.append(dict(instance=name,model=model,repeat=repeat,budget=budget,
                                      N_to_P=c['A_covers_B'],P_to_N=c['B_covers_A'],
                                      covered_P=c['A_covers_B_count'],covered_N=c['B_covers_A_count'],
                                      n_N=c['n_A'],n_P=c['n_B']))
            timing.append(dict(method=f"PPO（Train-{'S' if model=='small' else 'L'}）",instance=name,
                               seconds=ms([p['seconds'] for p in group],2),points=ms([len(p['points']) for p in group],2)))
        for budget in ((40320,120960) if scale==4 else (40320,)):
            group=[nsga[name,r,budget] for r in range(3)]
            label='B1' if budget==40320 else 'B2'
            timing.append(dict(method=f'NSGA-II（{label}；本轮重跑）',instance=name,
                               seconds=ms([n['seconds'] for n in group],2),points=ms([len(n['points']) for n in group],2)))
            attempts=sum(n['counts']['candidate_attempts'] for n in group)
            invalid=sum(n['counts']['invalid_candidate_attempts'] for n in group)
            nonref=[sum(p['solution_id']!=n['reference_plan_sha256'] for p in n['points']) for n in group]
            diagnostics.append(dict(instance=name,budget=budget,label=label,attempts=attempts,invalid=invalid,
                feasible=attempts-invalid,new_feasible_evaluations=sum(n['counts']['candidate_objective_evaluations']-n['counts']['invalid_candidates'] for n in group),
                reference_only=sum(k==0 for k in nonref),non_reference_points=nonref,
                unique_points=len({p['solution_id'] for n in group for p in n['points']})))
    return nsga,ppo,pairs,timing,diagnostics


def figures(nsga,ppo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
    groups={'PPO-V9':[ppo['Test-4','large',r] for r in range(3)],
            'NSGA-II B1':[nsga['Test-4',r,40320] for r in range(3)],
            'NSGA-II B2':[nsga['Test-4',r,120960] for r in range(3)]}
    selected={};coords=[];all_hv=[]
    for method,fronts in groups.items():
        for f in fronts: all_hv.append(dict(method=method,repeat=f['task']['repeat'],hv=hv(f['points'],(f['b_C'],f['b_R']))))
        selected[method]=max(fronts,key=lambda f:(hv(f['points'],(f['b_C'],f['b_R'])),-f['task']['repeat']))
    styles=[('#c83737','o'),('#2774b5','s'),('#d58b19','^')]
    def draw(ax,chosen,label_repeat=False):
        for (method,f),(color,marker) in zip(chosen.items(),styles):
            points=sorted(f['points'],key=lambda p:(p['cost'],p['risk'],p['solution_id']))
            x,y=[p['cost'] for p in points],[p['risk'] for p in points]
            line,=ax.plot(x,y,marker=marker,color=color,markersize=3.5,linewidth=1.25,
                         label=method+(f"（第{f['task']['repeat']+1}次）" if label_repeat else ''))
            assert list(line.get_xdata())==x and list(line.get_ydata())==y
            for p in points: coords.append(dict(panel=ax.get_label(),method=method,repeat=f['task']['repeat'],C=p['cost'],R=p['risk'],solution_id=p['solution_id']))
        ax.set_xlabel('经济成本 C（模型单位）');ax.set_ylabel('系统风险 R（模型指标）');ax.grid(alpha=.18)
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,5.5));ax.set_label('best');draw(ax,selected,True)
    ax.set_title('大规模近似帕累托前沿：各方法最佳一次');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(directory/'ppo_nsga_best.png',dpi=300);fig.savefig(directory/'ppo_nsga_best.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True)
    for r,ax in enumerate(axes):
        ax.set_label(f'repeat{r}');draw(ax,{method:fs[r] for method,fs in groups.items()});ax.set_title(f'第{r+1}次配对')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(directory/'ppo_nsga_all_repeats.png',dpi=220);plt.close(fig)
    selection={method:dict(repeat=f['task']['repeat'],points=len(f['points']),
                         min_cost=min(p['cost'] for p in f['points']),max_cost=max(p['cost'] for p in f['points']),
                         min_risk=min(p['risk'] for p in f['points']),max_risk=max(p['risk'] for p in f['points']),
                         archive_id=f['archive_id']) for method,f in selected.items()}
    write(OUT/'figure_selection.json',dict(hv_reference=[1.1,1.1],rule='max HV per method, tie smaller repeat; actual points only',all_hv=all_hv,selected=selection,coordinates=coords))
    return selection


def build():
    nsga,ppo,pairs,timing,diagnostics=data();selected=figures(nsga,ppo)
    (OUT/'tables').mkdir(exist_ok=True)
    for name,rows in [('coverage_pairs',pairs),('frontier_times',timing)]:
        with (OUT/f'tables/{name}.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'results.json',dict(pairs=pairs,timing=timing,nsga_search_diagnostics=diagnostics))
    text=(OUT/'before'/PAPER.name).read_text(encoding='utf8')
    def para(prefix,new):
        nonlocal text
        found=[p for p in text.split('\n\n') if p.startswith(prefix)]
        assert len(found)==1,(prefix,len(found))
        text=text.replace(found[0],new)
    def stats(instance,model,budget):
        rows=[p for p in pairs if (p['instance'],p['model'],p['budget'])==(instance,model,budget)]
        assert len(rows)==3
        return ms([p['N_to_P'] for p in rows]),ms([p['P_to_N'] for p in rows])
    para('本节依次开展', '本节依次开展固定偏好下的小规模质量验证、大规模近似帕累托前沿比较、冻结模型跨规模案例测试和关键参数敏感性分析。本次沿用原五个基础案例及八个敏感性版本、固定b、V9冻结模型及已完成的v11 PPO结果，仅按指定红线参数重新计算全部NSGA-II基线；MILP继续复用v9已完成记录，不新增训练、PPO或MILP求解。所有重复均为同一实例上的算法随机重复，不是独立实例样本。历史NSGA-II档案另存，不与新基线拼接。')
    para('历史NSGA-II搜索实际使用', '本次NSGA-II搜索与保存方案复核使用当前共同评价器（SHA256 `0e987e27da0e77be61530e9b71b536966c5da3f9648f9786fa62b6901eaab79f`），包含BG/BD服务前总容量检查；PPO沿用v11同一评价口径的已完成档案。旧v4 NSGA-II记录仅作为历史来源保留，不作为表5、表6的新基线。本次新NSGA-II档案及参与覆盖比较的PPO方案均需重新通过完整原MILP矩阵、变量界与整数性审核；PASS以当前文件哈希为准。')
    para('复用的历史NSGA-II使用', '本次NSGA-II采用范围优先探索确定的参数：种群20、交叉率0.10、个体变异率1.00，随机初始化的非末周期服务概率仍为0.50。个体变异率1表示每个子代一次单分量变异尝试，并非所有基因都变异。非支配等级—拥挤度二元锦标赛、父子代精英选择、编码、解码及确定性修复均未改变；较低交叉率使搜索更依赖变异，但不更换算法框架。每个“收运实体—周期”的基因表达服务开关、车辆、设施偏好和访问顺序，末期必访，前期服务可变。外部非支配档案仅用于记录，不反馈搜索。四个规模各3次运行至B1=40320候选尝试，Test-4同次继续至B2=120960；初始化新增19个个体、重复和修复失败均计数，半代已评价的可行候选仍纳入档案。参数曾依据Test-4低预算范围反馈选择，因此属于测试知情参数设定；本次正式预算运行前锁定，不在中途调参。')
    para('预算参照核验', '预算参照核验：本地Chen等论文的1000/3000为iterations，不等同于本仓库候选尝试。本次种群20，每个完整子代批为20个候选，初始化新增19个候选；B1/B2仍保持文稿原有40320/120960次尝试，不因种群缩小而缩减总预算。PPO仍复用每偏好12000、完整前沿上限252000的v11档案，分别为B1的6.25倍、B2的约2.0833倍。本轮12次新NSGA-II演化产生15份预算档案；候选尝试、目标评价和墙钟时间是不同投入指标，不声明与参考文献或PPO严格等投入。')
    para('新PPO计算采用', 'PPO沿用v11的CPU冻结推理及历史计时，原workers配置16、每进程1线程。本次NSGA-II新运行workers=4、每进程1线程；不同批次的资源争用条件不同，时间仅作实际投入记录，不直接声称公平加速倍数。MILP继续复用单线程原始耗时。NSGA-II计时包括搜索、严格候选核验和档案维护，B2从同次搜索起点累计并包含B1快照写盘，不是增量时间；参考方案首次评价、最终预算快照导出和离线审核不计入搜索段，任务总耗时另存。PPO原计时口径不变，模型加载、训练和本次离线报告耗时不追加进旧推理时间。')
    b1,b2=stats('Test-4','large',40320),stats('Test-4','large',120960)
    b2_rows=[p for p in pairs if (p['instance'],p['model'],p['budget'])==('Test-4','large',120960)]
    interpretation=('B2中已出现NSGA-II覆盖PPO点、且自身部分点不再被PPO覆盖的运行，说明追加预算在本次档案比较中产生了收益；但平均覆盖率仍偏向PPO。不能再沿用旧稿“两档均由PPO覆盖全部NSGA-II点”的结论。'
                    if any(p['N_to_P']>0 and p['P_to_N']<1 for p in b2_rows) else '')
    block=['## （三）大规模近似帕累托前沿比较','',
           'L20采用Test-4唯一固定案例，PPO使用冻结Train-L的v11已完成结果。每次PPO前沿与同重复编号的新NSGA-II B1、B2配对；NSGA-II沿用原正式实验种子，B2接续同次B1，不是另外独立重启。','',
           '**表5 大规模实例下PPO-Transformer与NSGA-II的双向覆盖率**','',
           '| 方法 | B1 N→P | B1 P→N | B2 N→P | B2 P→N |','| --- | --- | --- | --- | --- |',
           f'| PPO-L vs NSGA-II | {b1[0]} | {b1[1]} | {b2[0]} | {b2[1]} |','',
           '注：B1=40320、B2=120960候选尝试；每格为3次配对的均值±样本标准差。N→P表示𝒞(N,P)，P→N表示𝒞(P,N)，不是成本/风险降低比例。两档各计划3次、有效3次、失败0次；共用相同PPO档案，不增加PPO重启。时间见附录A.2，未舍入覆盖点数和分母见新coverage_pairs.csv。','',
           f'本次B1的N→P/P→N为{b1[0]}/{b1[1]}，B2为{b2[0]}/{b2[1]}。结论仅描述当前实例、编码及不同明确预算下已获解集，不证明真实完整前沿、跨实例统计优势或预算公平性。参数由此前低预算范围探索选取，不能将低预算图的跨度直接视为正式预算下的结果。','',
           interpretation,'',
           '**图1 大规模PPO与NSGA-II近似帕累托前沿对比**','',
           '![PPO与NSGA-II前沿对比](../output/pareto-nsga-red-budget/figures/ppo_nsga_best.png)','',
           '注：每种方法在3次运行中，按归一化参照点(1.1,1.1)下HV最大选取一次，并列取较小重复编号。各条线均为该次全部实际非支配方案按成本排序连接，不平均坐标、不平滑、不补入被支配点。连线中间位置不代表额外可行解。图为择优展示，表5仍使用全部3次配对，不能用择优图替代平均比较。', '',
           '本图选择：'+ '；'.join(f"{m}第{s['repeat']+1}次，{s['points']}点，成本[{s['min_cost']:.2f},{s['max_cost']:.2f}]、风险[{s['min_risk']:.2f},{s['max_risk']:.2f}]" for m,s in selected.items())+'。','',
           '三次原始配对同时保留如下，使用共同坐标范围，以展示运行间波动。','',
           '![三次前沿配对](../output/pareto-nsga-red-budget/figures/ppo_nsga_all_repeats.png)','',
           '全部交付点的严格可行率不等于候选尝试可行率，更不意味着搜索充分；候选有效性见附录A.2。旧NSGA-II结果保留于v11历史目录，不和新参数档案联合筛选。','']
    start=text.index('## （三）');end=text.index('## （四）');text=text[:start]+'\n'.join(block)+'\n'+text[end:]
    para('两个冻结模型分别用于', '两个冻结模型分别用于四个固定案例，沿用v11每组合3次的24份完整PPO前沿。各规模的两模型共用本次新参数NSGA-II的3份B1档案；Test-4的Train-L行与表5 B1完全一致，不更换基线或重复运行。实例、模型、PPO预算与原结果均未改变。Test-4曾参与PPO架构及NSGA参数探索，因此不能称为完全未见案例泛化。')
    for model,label in [('small','Train-S'),('large','Train-L')]:
        line=next(x for x in text.splitlines() if x.startswith('| '+label+' |'))
        cells=[v for i in range(1,5) for v in stats(f'Test-{i}',model,40320)]
        text=text.replace(line,'| '+label+' | '+' | '.join(cells)+' |')
    para('Test-1上Train-S覆盖', '表6仅反映两种冻结模型相对各自规模下同一新NSGA-II基线的覆盖，不是两个PPO档案彼此的支配关系。基线改变会影响覆盖数值，不能将表6相对旧稿的变化解释为PPO模型本身改进或退化；本次PPO未重新求解。')
    para('这些案例和重复只支持', '这些案例和重复只支持当前模型、生成规则与预算下的描述性结论，3次重复不是跨实例样本。本次NSGA-II为新计时，PPO/MILP为历史计时，批次和并发条件不同，不作公平速度排名。两份PPO模型、测试实例和b均未改变；NSGA-II仅采用本次运行前锁定的红线参数。PPO架构和NSGA-II参数均曾依据Test-4反馈探索，因此不能将本次表述为无测试知情调参的泛化验证。正式多实例统计、更多训练种子及真实企业数据验证需另行安排。')
    for row in timing:
        if not row['method'].startswith('NSGA'):continue
        label='B2' if 'B2' in row['method'] else 'B1'
        old=next(line for line in text.splitlines() if line.startswith(f'| NSGA-II（{label}；v4复用） | {row["instance"]} |'))
        text=text.replace(old,f"| {row['method']} | {row['instance']} | {row['seconds']} | {row['points']} | 3/3 | 100% | 100% |")
    para('注：时间和点数为', '注：时间和点数为同一案例3次运行的均值±样本标准差。NSGA-II为本次4-worker新记录，PPO为v11既有记录，不同资源争用条件下不能直接声称公平加速。B2时间累计包含B1；表4小规模PPO时间为三次之和，本表前沿时间为单次完整前沿的运行均值。候选尝试、实际评价、缓存命中、修复失败和不可行计数单独保留；不可行尝试包含缓存命中。PPO工作量及其21次偏好初始化/返回验证评价维持原记录，不因本次报告重建增加。')
    diag=[]
    for d in diagnostics:
        diag.append(f"{d['instance']}/{d['label']}：不可行尝试{d['invalid']}/{d['attempts']}（{100*d['invalid']/d['attempts']:.2f}%），严格可行尝试{d['feasible']}次，缓存未命中后的可行评价{d['new_feasible_evaluations']}次；仅参考档案{d['reference_only']}/3，非参考前沿点逐次为{'/'.join(map(str,d['non_reference_points']))}，按方案标识去重{d['unique_points']}份")
    para('NSGA-II搜索有效性', 'NSGA-II新搜索有效性按每规模/预算3次汇总如下，不将B1和接续B2相加作为独立投入：'+'；'.join(diag)+'。可行尝试包含重复，不等于唯一方案数；未保存的中间可行方案不据计数恢复。')
    # Make unchanged PPO sensitivity provenance explicit without changing values/formulas.
    text=text.replace('## （五）关键参数敏感性分析\n\n','## （五）关键参数敏感性分析\n\n本节完整复用v11阶段已完成的PPO敏感性结果，以下27次记录及三版图均未在本次NSGA-II重跑中重新求解。\n\n')
    a=text.index('### A.4 ');b=text.index('### A.5 ')
    replacement='''### A.4 文件对应与复现

本次协议为nsga-red-budget，12次新NSGA-II演化产生15份预算档案。PPO和MILP、实例、b及两份PT均复用既有v11来源，旧文件哈希逐一锁定；不覆盖历史NSGA-II结果。公式仍是可编辑LaTeX文字，不使用公式图片，本次不修改Word。

新前沿、实际方案、逐次执行元数据、覆盖点数与时间见[新NSGA-II实验目录](../output/pareto-nsga-red-budget/)。表5/6对应[tables/coverage_pairs.csv](../output/pareto-nsga-red-budget/tables/coverage_pairs.csv)，附录A.2时间对应[tables/frontier_times.csv](../output/pareto-nsga-red-budget/tables/frontier_times.csv)。图1的原始点、轮次选择和HV见[figure_selection.json](../output/pareto-nsga-red-budget/figure_selection.json)。表4、PPO工作量和敏感性仍对应[v11来源目录](../output/pareto-ppo-budget-v11/)，不是本轮新计算。

执行入口为`py -B run_nsga_red_budget.py run`，已有完成任务只在文件哈希通过后复用，未完成启动需先核查而不静默重试。报告入口`py -B report_nsga_red_budget.py`仅生成暂存稿；`--patch`生成向正式Markdown发布的补丁，不执行搜索。完整含文稿审核入口`py -B audit_nsga_red_budget.py --require-report`；整体交付PASS须与当前正式稿、图、表和原始档案哈希一致。旧v11报告入口只用于历史重现，不再用于覆盖本次新稿。

本次沿用多权重解集、双向覆盖与固定案例展示方式，不声称实现参考文献的邻域参数迁移。NSGA-II参数曾以Test-4低预算范围表现探索选择，这种测试知情设定和不等预算限制均需随结果披露。

参考：[Chen等本地论文](<../参考文献/Chen 等 - 2026 - Integrated hybrid energy and time-of-use electricity tariffs for the resource-constrained project sc.pdf>)。历史PPO运行及完整日志审核按[v11协议](../docs/ppo_budget_v11_protocol.md)保留；MILP最优性交接记录仍见[原审核记录](../output/pareto-ppo-budget-v11/milp/report_optimality_verdicts.json)。旧版报告的文稿哈希是历史值，不作为本次新稿的交付证明。

'''
    text=text[:a]+replacement+text[b:]
    (OUT/PAPER.name).write_text(text,encoding='utf8',newline='\n')
    write(OUT/'report_verification.json',dict(paper_sha256=file_sha256(OUT/PAPER.name),
          tables={p.name:file_sha256(p) for p in (OUT/'tables').glob('*.csv')},
          figures={p.name:file_sha256(p) for p in (OUT/'figures').glob('*')},
          source_sha256=file_sha256(ROOT/'report_nsga_red_budget.py')))
    print('STAGED',OUT/PAPER.name,flush=True)


def patch():
    import sys
    sys.stdout.reconfigure(encoding='utf8')
    before=PAPER.read_text(encoding='utf8');after=(OUT/PAPER.name).read_text(encoding='utf8')
    assert file_sha256(PAPER)==read(OUT/'protocol.json')['paper_before_sha256'],'Formal paper changed; inspect overlap'
    print('*** Begin Patch\n*** Update File: '+PAPER.relative_to(ROOT).as_posix()+'\n@@')
    print('\n'.join('-'+line for line in before.splitlines()))
    print('\n'.join('+'+line for line in after.splitlines()))
    print('*** End Patch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--patch',action='store_true');a=p.parse_args()
    if a.patch:patch()
    else:build()

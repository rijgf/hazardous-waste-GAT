"""Summarize all parameter trials, including unsuccessful settings."""
import json
import statistics
from pathlib import Path

from run_nsga_parameter_probe import OUT, ROOT, read, write
from src.reproducibility import file_sha256


def main():
    paths = sorted(p for p in OUT.glob('*-B*.json') if '.started.' not in p.name)
    rows = []
    for path in paths:
        d = read(path)
        counts, diag = d['result']['counts'], d['diagnostics']
        rows.append(dict(name=d['name'],seed=d['seed'],budget=d['budget'],population=d['population'],
                         crossover=d['crossover'],mutation=d['mutation'],**d['summary'],
                         seconds=d['result']['seconds'],invalid_rate=counts['invalid_candidate_attempts']/d['budget'],
                         cache_rate=counts['cache_hits']/d['budget'],
                         parent_identical_rate=diag['genome_identical_to_a_parent']/diag['children'],
                         path=path.name,sha256=file_sha256(path)))
    incomplete = [p.name for p in OUT.glob('*.started.json')
                  if not p.with_name(p.name.replace('.started.json','.json')).exists()]
    groups = []
    for budget in sorted({r['budget'] for r in rows}):
        for name in sorted({r['name'] for r in rows if r['budget']==budget}):
            group = [r for r in rows if r['budget']==budget and r['name']==name]
            groups.append(dict(name=name,budget=budget,n=len(group),seeds=[r['seed'] for r in group],
                               **{key:statistics.mean(r[key] for r in group)
                                  for key in ('hv','best_J','seconds','invalid_rate','cache_rate')},
                               cost_span=statistics.mean(r['max_cost']-r['min_cost'] for r in group),
                               risk_span=statistics.mean(r['max_risk']-r['min_risk'] for r in group)))
    audit_path = OUT/'audit.json'
    audit = read(audit_path) if audit_path.exists() else {}
    audited = (audit.get('status')=='PASS' and not incomplete and
               audit.get('hashes')=={p.name:file_sha256(p) for p in paths})
    confirmed = [g for g in groups if g['budget']==6000 and g['seeds']==[20260910,20260911,20260912]]
    best = max(confirmed,key=lambda g:g['hv']) if confirmed else None
    base = next((g for g in confirmed if g['name']=='baseline'),None)
    text = ['# NSGA-II 低预算参数探索结果', '',
            '固定 Test-4/000、公共参考方案与绑定 b；未改 PPO、历史 NSGA 档案、正式文稿或 Word。', '',
            '只调整种群规模、交叉概率、个体变异概率。编码/解码/修复、约束违反条数、排序和精英选择均未改变。', '',
            'HV 使用预设归一化参照点 (1.1,1.1)，不是此前聊天图的参照点，因此 HV 绝对数不可跨口径直接比较。跨度不是质量标准；候选预算包含无效和重复。', '',
            '## 汇总（相同预算和相同种子集合才可直接配对）', '',
            '| 配置 | 候选预算 | 重复数 | 平均HV↑ | 平均最小J↓ | 平均成本跨度 | 平均风险跨度 | 不可行比例 | 时间/s |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        text.append(f"| {g['name']} | {g['budget']} | {g['n']} | {g['hv']:.6f} | {g['best_J']:.6f} | {g['cost_span']:.2f} | {g['risk_span']:.2f} | {g['invalid_rate']:.2%} | {g['seconds']:.2f} |")
    if best and base:
        text += ['', '## 当前三种子复核结论', '',
                 f"已完成三种子配置中，平均 HV 最好的是 {best['name']}。相对同预算原参数，HV 提高 {(best['hv']/base['hv']-1)*100:.2f}%，平均最小 J 降低 {(1-best['best_J']/base['best_J'])*100:.2f}%。", '',
                 f"平均成本跨度由 {base['cost_span']:.2f} 变为 {best['cost_span']:.2f}，平均风险跨度由 {base['risk_span']:.2f} 变为 {best['risk_span']:.2f}。质量改善不等于跨度扩大；不能宣称短前沿问题已解决。", '',
                 '仅适用于本次低预算和固定实例；未证明在原40320/120960预算下同样最好，也未与PPO做新的等预算比较。', '',
                 '首批对照表明：单独提高变异率或降低交叉率并非稳定收益；缩小种群改善早期可行搜索，再提高变异率可减轻后期重复。不可行比例和缓存命中是机制线索，而非独立因果分解。']
        if audited:
            row = next(r for r in rows if r['name']==best['name'] and r['budget']==6000)
            write(ROOT/'configs/nsga_low_budget_exploratory.json',dict(
                status='exploratory_only_not_formal_default',name=best['name'],
                population=row['population'],crossover=row['crossover'],mutation=row['mutation'],
                budget=6000,seeds=best['seeds'],instance='output/pareto-parameter-revision-v4/instances/Test-4.json',
                entry='run_nsga_parameter_probe.py',command=f"py -B run_nsga_parameter_probe.py {best['name']} --seed 20260910 --budget 6000",
                selection='maximum mean HV among three-seed confirmed settings; single-instance adaptive exploration',
                unchanged='encoding, repair, nondominated sorting, crowding, tournament, elitist replacement',
                audit_sha256=file_sha256(audit_path),
                sources={p:file_sha256(ROOT/p) for p in ('run_nsga_parameter_probe.py','src/pareto_experiment.py','src/operators.py','src/solution_utils.py')}))
    text += ['', '## 全部运行（含失败方向）', '',
             '| 配置 | N/pc/pm | 种子 | 候选 | HV↑ | 最小J↓ | 点数 | 成本范围 | 风险范围 | 缓存命中 | 父代相同基因 |',
             '|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|']
    for r in rows:
        text.append(f"| [{r['name']}]({r['path']}) | {r['population']}/{r['crossover']}/{r['mutation']} | {r['seed']} | {r['budget']} | {r['hv']:.6f} | {r['best_J']:.6f} | {r['points']} | {r['min_cost']:.2f}–{r['max_cost']:.2f} | {r['min_risk']:.2f}–{r['max_risk']:.2f} | {r['cache_rate']:.2%} | {r['parent_identical_rate']:.2%} |")
    text += ['', '## 审核与局限', '',
             f"完整 MILP 审核与当前全部结果哈希一致：{'PASS' if audited else '尚未完成'}。未完成启动：{incomplete}。", '',
             '多进程并发，耗时不用于公平加速论断。固定实例反复选参属于探索；第三种子仅为随机种子复核，不是独立实例泛化。参数收益不能证明编码、修复、违反度处理已经没有问题。', '',
             '复现：`py -B run_nsga_parameter_probe.py <配置名> --seed <种子> --budget <候选数>`。结果拒绝覆盖。审核：`py -B run_nsga_parameter_probe.py audit`。']
    (OUT/'results.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    write(OUT/'summary.json',dict(rows=rows,groups=groups,incomplete=incomplete,audited=audited))
    print(json.dumps(dict(groups=groups,incomplete=incomplete,audited=audited),ensure_ascii=False))


if __name__=='__main__':
    main()

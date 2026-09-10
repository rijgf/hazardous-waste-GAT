"""Delivery checks, honest paired summaries and actual-point scientific plots."""
import json
from pathlib import Path
import statistics

import run_nsga_parameter_probe as previous
import run_nsga_spread_probe as spread
from src.reproducibility import file_sha256


def main():
    spread.report()
    out=spread.OUT
    summary=previous.read(out/'summary.json')
    audit=previous.read(out/'audit.json')
    assert not summary['incomplete']
    assert audit['status']=='PASS'
    assert audit['hashes']=={r['path']:file_sha256(out/r['path']) for r in summary['rows']}
    reused={}
    for seed in (20260910,20260911,20260912):
        name=f'baseline-s{seed}-B6000.json'
        source=previous.ROOT/'output/nsga-parameter-probe'/name
        assert source.read_bytes()==(out/name).read_bytes()
        reused[name]=dict(source=str(source.relative_to(previous.ROOT)),sha256=file_sha256(source))
    initialization=[]
    from run_nsga_initialization_probe import CONFIGS as initial_configs
    for path in sorted((out/'initialization').glob('*.json')):
        m=previous.read(path)
        result=previous.read(out/path.name.replace('.initialization.json','.json'))
        assert m['source_sha256']==file_sha256(previous.ROOT/'run_nsga_initialization_probe.py')
        assert m['probability']==initial_configs[result['name']][3]
        assert m['params']==[result[k] for k in ('population','crossover','mutation')]
        assert m['seed']==result['seed'] and m['budget']==result['budget']
        initialization.append(dict(name=result['name'],seed=m['seed'],probability=m['probability'],sha256=file_sha256(path)))
    rows=summary['rows']
    pair=[]
    for seed in (20260910,20260911,20260912):
        a=next(r for r in rows if (r['name'],r['seed'],r['budget'])==('n20_c10_m100',seed,6000))
        b=next(r for r in rows if (r['name'],r['seed'],r['budget'])==('baseline',seed,6000))
        pair.append(dict(seed=seed,cost_ratio=a['cost_span']/b['cost_span'],risk_ratio=a['risk_span']/b['risk_span'],
                         new_points=a['points'],new_cost_span=a['cost_span'],new_risk_span=a['risk_span']))
    def group(name):
        return next(g for g in summary['groups'] if g['name']==name and g['budget']==6000)
    a,b=group('n20_c10_m100'),group('baseline')
    stats=dict(cost_ratio=a['cost_span']/b['cost_span'],risk_ratio=a['risk_span']/b['risk_span'],
               hv_change_pct=100*(a['hv']/b['hv']-1),J_change_pct=100*(a['best_J']/b['best_J']-1),
               both_axes_wider_runs=sum(p['cost_ratio']>1 and p['risk_ratio']>1 for p in pair))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharex=True,sharey=True)
    for ax,seed in zip(axes,(20260910,20260911,20260912)):
        for name,label,color,marker in [('baseline','原参数 N=100, pc=0.9, pm=0.25','#2563eb','o'),
                                        ('n20_c10_m100','范围候选 N=20, pc=0.1, pm=1.0','#dc2626','s')]:
            d=previous.read(out/f'{name}-s{seed}-B6000.json')
            points=sorted(d['result']['points'],key=lambda p:(p['cost'],p['risk'],p['solution_id']))
            x,y=[p['cost'] for p in points],[p['risk'] for p in points]
            line,=ax.plot(x,y,color=color,marker=marker,markersize=3.8,linewidth=1.2,label=label)
            assert len(line.get_xdata())==len(points)
            assert list(line.get_xdata())==x and list(line.get_ydata())==y
        ax.set_title(f'种子 {seed}')
        ax.set_xlabel('经济成本 C（模型单位）')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('系统风险 R（模型指标）')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.02))
    fig.suptitle('同实例、每次6,000候选；全部实际非支配点按成本连线',y=.92,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.89))
    fig.savefig(out/'paired-fronts.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'paired-fronts.svg',bbox_inches='tight')
    plt.close(fig)
    text=['# 范围优先调参：阶段结论','',
          '只调参数；保留原 NSGA-II 算法结构及全部失败结果。正式文稿、PPO和历史档案不覆盖。','',
          '## 三次同预算配对','',
          '| 配置 | 平均成本跨度 | 平均风险跨度 | 平均HV | 平均最小J |',
          '|---|---:|---:|---:|---:|',
          f"| 原参数：100 / 0.9 / 0.25 | {b['cost_span']:.2f} | {b['risk_span']:.2f} | {b['hv']:.5f} | {b['best_J']:.5f} |",
          f"| 范围候选：20 / 0.1 / 1.0 | {a['cost_span']:.2f} | {a['risk_span']:.2f} | {a['hv']:.5f} | {a['best_J']:.5f} |",'',
          f"均为6,000候选。成本/风险平均跨度为原参数的 {stats['cost_ratio']:.2f}/{stats['risk_ratio']:.2f} 倍；HV变化 {stats['hv_change_pct']:+.2f}%，最小J均值变化 {stats['J_change_pct']:+.2f}%。三个种子中 {stats['both_axes_wider_runs']}/3 的两轴跨度同时超过原参数。",'',
          '| 种子 | 成本跨度倍数 | 风险跨度倍数 | 新点数 |','|---|---:|---:|---:|']
    for p in pair:
        text.append(f"| {p['seed']} | {p['cost_ratio']:.2f} | {p['risk_ratio']:.2f} | {p['new_points']} |")
    text+=['','![三次真实方案对照](paired-fronts.png)','',
           '连线仅作视觉引导，不表示线段中间点都是可行解。所有点保留，不做平滑、平均或跨重复合并。', '',
           '## 修改边界与失败方向','',
           '候选仍使用原初始化概率0.5，只将N=100改20、pc=.9改.1、pm=.25改1。pm=1是每个子代一次单分量变异尝试，不是每个基因都变异。低交叉率使搜索更依赖变异，但排序、拥挤距离、锦标赛、精英保留、解码和修复都未改变。','',
           '另试过初始化服务概率.8/.95：仅改变原Bernoulli参数，原生成流程不变，相关运行的实际概率如下；它们没有混入上述三次候选结果。','',
           '| 初始化试验 | 种子 | 非末周期服务概率 |','|---|---:|---:|']
    for m in initialization: text.append(f"| {m['name']} | {m['seed']} | {m['probability']} |")
    text+=['','完整参数表及不利结果见 [results.md](results.md)。初筛变异率.5曾出现成本跨度5128，但其他种子缩小，6000候选后首种子缩至406，未将其作为可靠成功。','',
           f"审核：{audit['runs']}份档案（其中3份原参数6000结果逐字节复用），{audit['unique_plans']}份去重方案通过完整原MILP核验。",'',
           '单实例反复试参，只是探索性结果；质量允许下降并不意味着被支配方案可补回前沿。本轮未证明在原40320/120960预算下仍有同样范围，也不是对真实完整Pareto前沿的证明。']
    (out/'conclusions.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    previous.write(out/'delivery.json',dict(status='PASS',audit_sha256=file_sha256(out/'audit.json'),
        runs=len(rows),new_searches=len(rows)-3,reused=reused,initialization=initialization,
        comparisons=pair,summary=stats,conclusions_sha256=file_sha256(out/'conclusions.md'),
        figure_sha256=file_sha256(out/'paired-fronts.png'),
        sources={f:file_sha256(previous.ROOT/f) for f in ('run_nsga_spread_probe.py','run_nsga_initialization_probe.py','run_nsga_parameter_probe.py','src/pareto_experiment.py','src/operators.py','src/solution_utils.py')}))
    previous.write(previous.ROOT/'configs/nsga_spread_exploratory.json',dict(
        status='exploratory_only',population=20,crossover=.1,mutation=1.,initial_service_probability=.5,
        budget=6000,seeds=[20260910,20260911,20260912],scope='Test-4/000 spread-first, quality tradeoff allowed; not formal default',
        command='py -B run_nsga_spread_probe.py n20_c10_m100 --seed 20260910 --budget 6000',
        delivery_sha256=file_sha256(out/'delivery.json')))
    print(json.dumps(dict(stats=stats,pairs=pair,runs=len(rows)),ensure_ascii=False),flush=True)


if __name__=='__main__': main()

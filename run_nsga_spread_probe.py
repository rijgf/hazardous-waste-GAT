"""Spread-first parameter-only experiments. Original NSGA-II is untouched."""
import argparse
import json
from pathlib import Path
import statistics

import run_nsga_parameter_probe as previous
from src.reproducibility import file_sha256

ROOT = previous.ROOT
OUT = ROOT/'output/nsga-spread-probe'
CONFIGS = {
    'baseline': (100,.9,.25),
    'n100_m01': (100,.9,.01),
    'n100_m05': (100,.9,.05),
    'n100_m10': (100,.9,.10),
    'n100_m50': (100,.9,.50),
    'n200_m05': (200,.9,.05),
    'n200_m50': (200,.9,.50),
    'n400_m25': (400,.9,.25),
    'n40_m05': (40,.9,.05),
    'n100_c10_m25': (100,.1,.25),
    'n100_c10_m100': (100,.1,1.),
    'n100_c100_m05': (100,1.,.05),
    'n200_c100_m100': (200,1.,1.),
    'n100_c30_m50': (100,.3,.5),
    'n100_c50_m50': (100,.5,.5),
    'n40_c10_m100': (40,.1,1.),
    'n20_c10_m100': (20,.1,1.),
    'n100_c01_m100': (100,.01,1.),
    'n100_c01_m50': (100,.01,.5),
    'n200_c10_m100': (200,.1,1.),
}


def configure():
    previous.OUT = OUT
    previous.CONFIGS = CONFIGS


def metrics(d):
    p = d['result']['points']
    c = sorted(x['cost'] for x in p)
    r = sorted(x['risk'] for x in p)
    dc, dr = c[-1]-c[0], r[-1]-r[0]
    # Both axes must grow: max-min on normalized spans, not summed width or HV.
    balance = min(dc/d['refs'][0],dr/d['refs'][1])
    gaps = [b-a for a,b in zip(c,c[1:])]
    return dict(cost_span=dc,risk_span=dr,balanced_span=balance,
                largest_cost_gap_fraction=max(gaps,default=0)/dc if dc else 0,
                cost_bin_occupancy=len({min(9,int(10*(x-c[0])/dc)) for x in c}) if dc else 1)


def run(name,seed,budget):
    configure()
    target=OUT/f'{name}-s{seed}-B{budget}.json'
    manifest=target.with_suffix('.wrapper.json')
    assert not target.exists() and not manifest.exists(),'No overwrites'
    previous.write(manifest,dict(wrapper_sha256=file_sha256(Path(__file__)),
                                 base_driver_sha256=file_sha256(ROOT/'run_nsga_parameter_probe.py'),
                                 params=CONFIGS[name],seed=seed,budget=budget))
    previous.run(name,seed,budget)
    print(json.dumps(dict(name=name,seed=seed,budget=budget,**metrics(previous.read(target)))),flush=True)


def report():
    rows=[]
    for p in sorted(OUT.glob('*-B*.json')):
        if '.started.' in p.name or '.wrapper.' in p.name:
            continue
        d=previous.read(p)
        rows.append(dict(name=d['name'],seed=d['seed'],budget=d['budget'],
                         params=[d['population'],d['crossover'],d['mutation']],
                         **d['summary'],**metrics(d),path=p.name,sha256=file_sha256(p)))
    groups=[]
    for budget,name in sorted({(r['budget'],r['name']) for r in rows}):
        group=[r for r in rows if r['budget']==budget and r['name']==name]
        groups.append(dict(name=name,budget=budget,n=len(group),seeds=[r['seed'] for r in group],
                           **{key:statistics.mean(r[key] for r in group) for key in
                              ('cost_span','risk_span','balanced_span','hv','best_J','points','cost_bin_occupancy','largest_cost_gap_fraction')}))
    groups.sort(key=lambda g:(g['budget'],-g['balanced_span']))
    incomplete=[p.name for p in OUT.glob('*.wrapper.json')
                if not p.with_name(p.name.replace('.wrapper.json','.json')).exists()]
    previous.write(OUT/'summary.json',dict(rows=rows,groups=groups,incomplete=incomplete))
    text=['# NSGA-II 范围优先的纯调参探索','',
          '固定原 Test-4/000 与绑定 b，不改算法结构。质量允许下降；只记录真实、合法的非支配方案，不将被支配点补回曲线，不跨重复合并。','',
          '排序指标为 min(成本跨度/b_C, 风险跨度/b_R)，同时报告实际范围、HV、点数、10等分成本区间占用数、最大相邻成本间隙占比。相同预算和相同种子集合才可直接配对。','',
          '## 配置汇总','',
          '| 配置 | 候选预算 | 重复 | 平均成本跨度 | 平均风险跨度 | 双轴指标↑ | HV（仅记录） | 平均点数 | 成本分箱占用/10 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        text.append(f"| {g['name']} | {g['budget']} | {g['n']} | {g['cost_span']:.2f} | {g['risk_span']:.2f} | {g['balanced_span']:.5f} | {g['hv']:.5f} | {g['points']:.1f} | {g['cost_bin_occupancy']:.1f} |")
    text+=['','## 全部运行（包括失败方向）','','| 配置 | N/pc/pm | 种子 | 预算 | 成本范围 | 风险范围 | 点数 | 双轴指标 | 最大间隙占比 |','|---|---|---:|---:|---|---|---:|---:|---:|']
    for r in rows:
        text.append(f"| [{r['name']}]({r['path']}) | {'/'.join(map(str,r['params']))} | {r['seed']} | {r['budget']} | {r['min_cost']:.2f}–{r['max_cost']:.2f} | {r['min_risk']:.2f}–{r['max_risk']:.2f} | {r['points']} | {r['balanced_span']:.5f} | {r['largest_cost_gap_fraction']:.2%} |")
    text+=['',f'未完成启动：{incomplete}。','',
           '本报告为固定实例的自适应参数探索，不是无偏泛化；所有质量损失和跨度缩小的结果均保留。完整合法性以 audit.json 为准，报告生成本身不代表审核通过。']
    (OUT/'results.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps(groups,ensure_ascii=False),flush=True)


def audit():
    # The old auditor globs all *-B*.json; keep wrapper manifests outside that glob temporarily
    # by using a local explicit iterator filter, without moving or hiding actual results.
    from unittest.mock import patch
    original_glob=Path.glob
    def glob(path,pattern):
        return (p for p in original_glob(path,pattern) if '.wrapper.' not in p.name)
    configure()
    with patch.object(Path,'glob',glob):
        previous.audit()
    for p in OUT.glob('*.wrapper.json'):
        d=previous.read(p)
        assert d['wrapper_sha256']==file_sha256(Path(__file__))
        assert d['base_driver_sha256']==file_sha256(ROOT/'run_nsga_parameter_probe.py')
    print('Wrapper provenance PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('names',nargs='+',choices=[*CONFIGS,'report','audit'])
    parser.add_argument('--seed',type=int,default=20260910)
    parser.add_argument('--budget',type=int,default=3000)
    a=parser.parse_args()
    for name in a.names:
        if name=='report': report()
        elif name=='audit': audit()
        else: run(name,a.seed,a.budget)

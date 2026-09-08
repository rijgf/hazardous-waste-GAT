"""Editable plot: actual within-scenario nondominated solutions, never mean points."""
from __future__ import annotations

import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output/pareto-single-instance-v3'


def plot():
    with (OUT/'tables/sensitivity_points.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows = list(csv.DictReader(handle))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(12.5,5.1),sharex=True,sharey=True)
    styles = {70:('#009b9f','D','−30%'),80:('#c5a300','v','−20%'),
              100:('#bd3036','s','基准'),120:('#df7f25','o','+20%'),130:('#589bbc','^','+30%')}
    for axis,parameter,title in zip(axes,['capacity','coload'],['(a) 处理处置能力','(b) 共载附加风险系数']):
        for level,(color,marker,label) in styles.items():
            selected=[row for row in rows if row['scenario']==('baseline' if level==100 else f'{parameter}-{level}')]
            axis.scatter([float(row['cost']) for row in selected],[float(row['risk']) for row in selected],
                         s=34 if level==100 else 29,marker=marker,facecolors='none',edgecolors=color,
                         linewidths=1.0,label=label,alpha=.9)
        axis.set_xlabel('经济成本 C（成本模型单位）')
        axis.set_title(title,pad=12)
        axis.grid(True,alpha=.18,linewidth=.6)
        axis.legend(frameon=True,fontsize=9,ncol=1)
    axes[0].set_ylabel('系统风险 R（模型风险指标）')
    fig.tight_layout(w_pad=2)
    path=OUT/'figures'
    path.mkdir(parents=True,exist_ok=True)
    fig.savefig(path/'sensitivity.svg',bbox_inches='tight')
    fig.savefig(path/'sensitivity.png',dpi=350,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    plot()

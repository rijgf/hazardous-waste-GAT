"""V9 line/smooth views reuse the original PCHIP, with honest singleton display."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_pareto_sensitivity_variants import STYLES, curve_points
from src.reproducibility import file_sha256

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/pareto-ppo-v9'


def visual_curve(points):
    """Do not invent a curve when an observed archive contains only one point."""
    if len(points)==1:
        point=np.asarray(points,dtype=float)
        if point.shape!=(1,2) or not np.isfinite(point).all():
            raise ValueError('Invalid single observed point')
        return point,point.copy()
    return curve_points(points)


def plot(out=OUT):
    out=Path(out)
    source=out/'tables/sensitivity_points.csv'
    before=file_sha256(source)
    with source.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    scenarios={}
    for row in rows:
        scenarios.setdefault(row['scenario'],[]).append((float(row['cost']),float(row['risk'])))
    expected={'baseline',*[f'{p}-{n}' for p in ('capacity','coload') for n in (70,80,120,130)]}
    if set(scenarios)!=expected:
        raise ValueError('Require all nine sensitivity scenarios; never silently omit one')
    curves={name:visual_curve(points) for name,points in scenarios.items()}
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    figures=out/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    for kind in ('line','smooth'):
        fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),sharex=True,sharey=True)
        for axis,parameter,title in zip(axes,('capacity','coload'),
                ('(a) 处理处置能力','(b) 共载附加风险系数')):
            for level,(color,style,label) in STYLES.items():
                name='baseline' if level==100 else f'{parameter}-{level}'
                knots,smooth=curves[name]
                points=knots if kind=='line' else smooth
                axis.plot(points[:,0],points[:,1],color=color,linestyle=style,
                    marker='o' if len(knots)==1 else None,markersize=4,
                    linewidth=1.65 if level==100 else 1.3,label=label,
                    zorder=4 if level==100 else 3)
            axis.set_xlabel('经济成本 C（成本模型单位）')
            axis.set_title(title,pad=12)
            axis.grid(True,alpha=.18,linewidth=.6)
            axis.legend(frameon=True,fontsize=9,handlelength=3.2)
        axes[0].set_ylabel('系统风险 R（模型风险指标）')
        fig.tight_layout(w_pad=2)
        fig.savefig(figures/f'sensitivity_{kind}.svg',bbox_inches='tight')
        fig.savefig(figures/f'sensitivity_{kind}.png',dpi=350,bbox_inches='tight')
        plt.close(fig)
    evidence={'source':'tables/sensitivity_points.csv','source_sha256':before,
        'plot_script_sha256':file_sha256(Path(__file__)),
        'interpolation_source_sha256':file_sha256(ROOT/'plot_pareto_sensitivity_variants.py'),
        'method':'PCHIP risk(cost); every original knot retained; no extrapolation; singleton as observed marker',
        'visual_only':True,'new_optimization_runs':0,
        'single_point_scenarios':[name for name,(knots,_) in curves.items() if len(knots)==1],
        'scenarios':{name:{'original_points':knots.tolist(),'smooth_visual_points':smooth.tolist()}
                     for name,(knots,smooth) in curves.items()},
        'artifacts':{f'sensitivity_{kind}.{ext}':file_sha256(figures/f'sensitivity_{kind}.{ext}')
                     for kind in ('line','smooth') for ext in ('png','svg')}}
    if file_sha256(source)!=before:
        raise ValueError('Source coordinates changed during plotting')
    (figures/'sensitivity_variants.json').write_text(
        json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':plot()

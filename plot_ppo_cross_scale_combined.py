"""Replot the same eight selected fronts in a 2x2 scientific figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from run_nsga_red_budget import OUT, ROOT, read, write
from src.reproducibility import file_sha256


def main():
    directory=OUT/'ppo_cross_scale_figures'
    source=directory/'provenance.json'
    data=read(source)
    for path,expected in data['sources'].items():
        assert file_sha256(ROOT/path)==expected
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(2,2,figsize=(11.6,8.5))
    for scale,ax in enumerate(axes.flat,1):
        for model,label,color,marker,style in [
            ('small','小规模训练 PPO','#c83737','o','-'),
            ('large','大规模训练 PPO','#2774b5','s','--')]:
            selected=next(s for s in data['selections'] if s['scale']==scale and s['model']==model)
            raw=read(ROOT/f"output/pareto-ppo-budget-v11/fronts/{selected['archive_id']}.json")
            points=sorted(raw['points'],key=lambda p:(p['cost'],p['risk'],p['solution_id']))
            assert [{k:p[k] for k in ('cost','risk','solution_id')} for p in points]==selected['points']
            ax.plot([p['cost'] for p in points],[p['risk'] for p in points],
                    color=color,marker=marker,linestyle=style,markersize=3.2,
                    markerfacecolor='none',linewidth=1.35,label=label)
        ax.set_title(f'({chr(96+scale)}) Test-{scale}',fontsize=12,pad=8)
        ax.set_xlabel('经济成本 C（模型单位）')
        ax.set_ylabel('系统风险 R（模型指标）')
        ax.grid(alpha=.16)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(style='plain',axis='both',useOffset=False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.5,.995),ncol=2,frameon=False,handlelength=3)
    fig.text(.5,.014,'V9 · 每偏好最多 12,000 候选 · 各模型取三次中 HV 最佳一次 · 子图坐标范围不同',
             ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.945),h_pad=1.8,w_pad=2)
    outputs={}
    for suffix in ('png','svg','pdf'):
        path=directory/f'ppo_cross_scale_combined.{suffix}'
        fig.savefig(path,dpi=300)
        outputs[path.name]=file_sha256(path)
    plt.close(fig)
    write(directory/'combined_provenance.json',dict(source_sha256=file_sha256(source),
          script_sha256=file_sha256(ROOT/'plot_ppo_cross_scale_combined.py'),
          figures=outputs,same_points_as_four_original_figures=True,layout='2x2; independent axes'))
    print('Combined figure created; eight raw point sequences verified.')


if __name__=='__main__':
    main()

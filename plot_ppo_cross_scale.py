"""Four descriptive, best-repeat PPO-only comparisons from frozen v11 archives."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_nsga_red_budget import BASE, OUT, ROOT, read, write, verify
from run_nsga_parameter_probe import hv
from src.reproducibility import file_sha256


def main():
    verify()
    registry=read(ROOT/'configs/ppo_models_frontier_v9.json')
    target=OUT/'ppo_cross_scale_figures'
    target.mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11})
    record={'model_version':'V9','candidate_cap_per_preference':12000,
            'front_cap':252000,'hv_reference':[1.1,1.1],
            'selection':'each model and scale: max HV among three runs, tie smaller repeat',
            'limitations':'Selected-repeat descriptive plots, not three-run means or unbiased generalization; segments are not additional feasible solutions.',
            'sources':{},'selections':[],'figures':{}}
    for scale in range(1,5):
        fig,ax=plt.subplots(figsize=(8,5.4))
        instance_refs=None
        for model,label,color,marker,linestyle in [
            ('small','小规模训练 PPO','#c83737','o','-'),
            ('large','大规模训练 PPO','#2774b5','s','--')]:
            checkpoint=registry['models'][model]
            assert file_sha256(ROOT/checkpoint['checkpoint'])==checkpoint['sha256']
            fronts=[]
            for repeat in range(3):
                path=BASE/f'fronts/PPO-Test-{scale}-{model}-r{repeat}-B252000.json'
                d=read(path)
                assert d['checkpoint']==checkpoint
                assert d['budget']==252000
                key=(d['instance_sha256'],d['b_C'],d['b_R'])
                instance_refs=key if instance_refs is None else instance_refs
                assert key==instance_refs
                record['sources'][path.relative_to(ROOT).as_posix()]=file_sha256(path)
                fronts.append({k:d[k] for k in ('points','task','b_C','b_R','archive_id')})
            scores=[hv(d['points'],(d['b_C'],d['b_R'])) for d in fronts]
            best=max(range(3),key=lambda r:(scores[r],-r))
            points=sorted(fronts[best]['points'],key=lambda p:(p['cost'],p['risk'],p['solution_id']))
            assert all(p['feasible'] for p in points)
            x,y=[p['cost'] for p in points],[p['risk'] for p in points]
            line,=ax.plot(x,y,color=color,marker=marker,linestyle=linestyle,
                         markersize=4,markerfacecolor='none',linewidth=1.4,
                         label=f'{label}（第{best+1}次）')
            assert list(line.get_xdata())==x and list(line.get_ydata())==y
            record['selections'].append(dict(scale=scale,model=model,repeat=best,
                all_hv=scores,archive_id=fronts[best]['archive_id'],
                points=[{k:p[k] for k in ('cost','risk','solution_id')} for p in points]))
        ax.set_title(f'Test-{scale}：两种训练规模 PPO 的前沿对比')
        ax.set_xlabel('经济成本 C（模型单位）')
        ax.set_ylabel('系统风险 R（模型指标）')
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
        ax.ticklabel_format(style='plain',axis='both',useOffset=False)
        fig.text(.5,.015,'V9；每偏好最多 12,000 候选；各模型按 HV 选取最佳一次',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.045,1,1))
        for suffix in ('png','svg'):
            path=target/f'test_{scale}_ppo_comparison.{suffix}'
            fig.savefig(path,dpi=250)
            record['figures'][path.name]=file_sha256(path)
        plt.close(fig)
        print(f'Test-{scale} plotted',flush=True)
    record['script_sha256']=file_sha256(ROOT/'plot_ppo_cross_scale.py')
    write(target/'provenance.json',record)


if __name__=='__main__':
    main()

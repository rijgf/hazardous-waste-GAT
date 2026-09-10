"""Read-only computational replay and versioned delivery of scalar exploration."""
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import zipfile

from run_ppo_scalar_exploration import ROOT, OUT, BASE, read, write, sha
from audit_scalar_advantage_v5 import full_model_replay
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan,plan_sha256
from src.solution_utils import route_plan_to_solution,evaluate_solution


def main():
    d=read(BASE/'instances/Test-4.json');params=params_from_json_data(d['params']);refs=(d['b_C'],d['b_R'])
    scalar=lambda p:.5*p['cost']/refs[0]+.5*p['risk']/refs[1]
    old=read(ROOT/'output/ppo-scalar-advantage-v5/preserved_inputs.json')
    for path,value in old.items():assert sha(ROOT/path)==value,path
    samples=[];baselines=[]
    for budget in (40320,120960):
        for repeat in range(3):
            path=BASE/f'fronts/NSGA-Test-4-r{repeat}-B{budget}.json';front=read(path)
            assert front['instance_sha256']==d['instance_sha256'] and (front['b_C'],front['b_R'])==refs
            best=min(front['points'],key=scalar);saved=read(BASE/best['solution_path']);plan=deserialize_plan(saved['plan'])
            assert plan_sha256(plan)==best['solution_id'];samples.append((plan,best))
            baselines.append({'method':'NSGA-II','budget':budget,'repeat':repeat,'C':best['cost'],
                              'R':best['risk'],'J':scalar(best),'solution_id':best['solution_id'],
                              'archive_sha256':sha(path),'historical_full_front_seconds':front['seconds']})
    threshold=min(x['J'] for x in baselines);rows=[];hashes={}
    for path in sorted((OUT/'trials').glob('*/config.json')):
        result_path=path.parent/'result.json';assert result_path.exists(),'Incomplete trial'
        config,result=read(path),read(result_path);args=config['args'];assert result['config_sha256']==sha(path)
        for source,value in config['sources'].items():
            assert sha(ROOT/source)==value,source
            assert sha(path.parent/'source'/source)==value,source
        assert config['benchmark']['target_J']==threshold
        assert config['benchmark']['instance_file_sha256']==sha(BASE/'instances/Test-4.json')
        assert config['start']=='common_reference_plan_only'
        if args['method']!='compact':
            entry=config['checkpoint'];assert sha(ROOT/entry['checkpoint'])==entry['sha256']
        plan=deserialize_plan(result['plan']);assert plan_sha256(plan)==result['solution_id']
        metric=evaluate_solution(params,route_plan_to_solution(params,plan),(.5,.5),refs)
        assert metric['feasible'],metric['violations']
        for key in ('cost','risk','weighted_objective','fixed_cost','distance_cost','processing_cost',
                    'transport_risk','coload_risk','producer_inventory_risk','facility_inventory_risk'):
            assert math.isclose(metric[key],result['metrics'][key],rel_tol=1e-11,abs_tol=1e-8),key
        actual_attempts=result['counts'].get('attempts',result['counts'].get('candidate_attempts'))
        assert actual_attempts==args['budget']==8000
        if args['method']!='legacy':
            counts=result['counts']
            assert sum(counts.get(k,0) for k in ('no_proposal','repair_failed','cache_hits','objective_evaluations'))==actual_attempts
            assert all(a['J']>b['J'] for a,b in zip(result['trace'],result['trace'][1:]))
            assert all(1<=t['attempt']<=8000 for t in result['trace'])
        success=metric['weighted_objective']<threshold-1e-8
        assert result['beats_all_existing_nsga']==success
        samples.append((plan,metric))
        rows.append({'id':args['id'],'method':args['method'],'seed':args['seed'],'budget':8000,
                     'C':metric['cost'],'R':metric['risk'],'J':scalar(metric),'seconds':result['seconds'],
                     'visits':sum(len(r)-2 for r in plan.values()),'routes':len(plan),
                     'improvement_vs_best_nsga_pct':100*(threshold-scalar(metric))/threshold,
                     'beats_best_nsga':success,'solution_id':result['solution_id'],
                     'first_better_attempt':next((t['attempt'] for t in result['trace'] if args['method']!='legacy' and t['J']<threshold-1e-8),None)})
        hashes[str(result_path.relative_to(ROOT))]=sha(result_path)
    assert len(rows)==7
    for method in ('compact','ppo_compact'):
        group=[x for x in rows if x['method']==method]
        assert {x['seed'] for x in group}=={3381415411,3078273277,958147205}
        assert all(x['beats_best_nsga'] for x in group)
    backup=ROOT/'backups/ppo-v4-before-exploration-20260909'
    assert sha(backup/'code.zip')=='d3a051b1ac3a03978097bc89e2fc46d6b6ec567f8473a7c7a151556342f9f8f9'
    with zipfile.ZipFile(backup/'code.zip') as z:
        entries={n.replace('\\','/'):hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
        assert entries['src/ppo_improver.py']==sha(ROOT/'src/ppo_improver.py')
    write(backup/'zip_contents_sha256.json',entries)
    for filename,path in [('v4_small.pt','outputs/parameter_revision_v4/models/small_model.pt'),
                          ('v4_large.pt','outputs/parameter_revision_v4/models/large_model.pt'),
                          ('v5_large_failed_comparison.pt','outputs/ppo_scalar_advantage_v5/models/large_model.pt')]:
        assert sha(backup/'models'/filename)==sha(ROOT/path)
    print('FULL MILP replay for',len(samples),'solutions',flush=True)
    matrix=full_model_replay(params,samples)
    results={'nsga_target_J':threshold,'nsga_baselines':baselines,'trials':rows,
             'scope':'one fixed large instance, preference 0.5/0.5; exploratory, not generalization',
             'means':{method:{k:statistics.mean(x[k] for x in rows if x['method']==method) for k in ('C','R','J','seconds')}
                      for method in ('compact','ppo_compact')}}
    write(OUT/'results.json',results)
    labels={'compact':'简化搜索（无PPO）','ppo_compact':'PPO算子引导＋简化搜索','legacy':'原PPO延长搜索'}
    table='\n'.join(f"| {labels[x['method']]} | {x['seed']} | {x['budget']} | {x['C']:.3f} | {x['R']:.3f} | {x['J']:.6f} | {x['seconds']:.2f} | {x['improvement_vs_best_nsga_pct']:.2f}% |" for x in rows)
    best=min(baselines,key=lambda x:x['J'])
    report=f'''# 大规模单点算法探索：已超过现有NSGA-II标量基准

## 比较边界

固定原v4 Test-4/000实例、原公共参考方案、实例自身b_C={refs[0]}、b_R={refs[1]}，只计算偏好(0.5,0.5)。目标J=0.5×C/b_C+0.5×R/b_R，越小越好。全部方法从公共参考方案独立出发，没有用NSGA-II解热启动，没有换例或重算b。

已有NSGA-II的两档预算、三次重复共六份档案均未改变。从这些档案的全部非支配点中选择该偏好的最小值，得到C={best['C']:.6f}、R={best['R']:.6f}、J={threshold:.9f}，来自B{best['budget']}重复{best['repeat']}。这是已有NSGA-II结果的最好标量值，不是重新运行NSGA-II得到的单点值，也不是其全局最优证明。

## 本轮全部七次搜索

| 方法 | 种子 | 候选尝试预算 | 成本C | 风险R | 目标J | 搜索时间/s | J相对NSGA-II最好值降低 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

每次新搜索预设8000次尝试，无变化、修复失败、缓存命中均计入预算；候选插入位置通过距离增量选取，不逐位置完整评价J。实际完整目标调用次数见各trial的counts。时间不含模型加载、结果导出及最终MILP审计；本轮存在并发运行，不作严格硬件速度归因。

NSGA-II预算为每次完整前沿40320或120960次，新搜索仅优化一个偏好。按用户要求比较该点的质量，不能把不同工作量直接换算为完整前沿的速度优势。三次新重复为原PPO该偏好对应的三个种子，未按结果选择种子。

PPO引导版三次平均J={results['means']['ppo_compact']['J']:.9f}，无PPO简化版平均J={results['means']['compact']['J']:.9f}。两种版本均三次胜过最强既有NSGA-II基准，但平均值接近，不能宣称PPO引导显著优于不使用PPO，也不能把所有收益归于学习。

本轮是一次实现、两种引导方式各三种子的探索，以及原PPO延长预算的一次对照。没有反复调参后只保留最优轮，也没有隐藏原PPO延长后仍落后的结果。但算法设计使用了此前对此测试实例的诊断，仍属测试集适配的探索性案例，不是无偏泛化结论。

另做过一次“仅保留末周期服务”的可行性探针：原公共修复器返回失败，存在车辆超载及末端库存未清空，因此没有将其作为新初始解或合格实验结果。这项失败诊断也不用于挑选测试实例。

## 改了什么，为什么有效

新分支只有六种简单动作：取消非末周期服务、同周期任务迁移、交换、路线反转、换设施、添加服务。没有增加复杂破坏重建算子，仍调用既有公共修复器。旧八算子代码及原PPO入口保留，以便历史复现。

1. **取消服务成为直接动作。** 原实现靠跨周期移动后的重复去除间接取消访问；现在不必同时猜车辆和目标周期。
2. **同质车辆去除编号冗余。** 按每周期路线内容确定规范车号，缓存不再重复区分纯粹换车号的等价方案。车辆仍有容量与数量约束，没有取消实际路线之间的任务迁移。
3. **动作对象不再由三个独立的512分类头盲猜。** PPO引导版只使用旧冻结模型的共同算子类别概率；具体对象由简单随机选择和距离最小增量插入生成，完整成本/风险评价决定接受。新增取消动作没有冒充旧网络已学习的动作。
4. **更多连续有效调整。** 在单点预算内采用逐次改进搜索，停滞后少量扰动；不再把大部分候选都花在同一状态的32路采样。它与算子/对象简化共同构成新算法，不能仅靠本对照分解每项改动的独立贡献。

无PPO对照同样获胜、原PPO延长至8000次仍落后，说明至少不需要依赖重新训练优势归一化才能解决当前案例的差距。当前证据支持优先改进搜索动作与对象生成，但不能据此说强化学习在其他实例上一概无用。

## 车辆同质问题的准确含义

仅把同一路线从K1改名为K2，或者完整交换两条路线的车号，不改变该模型的成本和风险，属于冗余。但把某个收运任务从一条路线移到另一条，会改变载量、共载组合、路程、所需车辆数，仍然是有意义的决策。因此本轮简化车号选择和重复拆分动作，不删除路线间任务分配。

## 合法性、模型与回退

修复了共同校验器漏检收运前BG总容量、处理前BD总容量的问题。回归测试先复现“处理前超容量却被判可行”，修复后拒绝该反例；没有放宽任何MILP约束。

独立终验对本轮七份结果及六个NSGA-II基准点，共13份方案补齐辅助变量，代入原MILP全部变量界、整数性与{matrix['rows']:,}条约束，成本/风险也独立重算。审核结果和实际文件哈希见[audit.json](audit.json)。历史3,325份输入文件哈希未变。

本轮没有重训PT。PPO引导版使用原v4大模型，SHA256为`61db3f4dcc6aa768f22dd36515449416facb603b934c9eaf9a5e0278838fab3a`；新方法是混合搜索，不是原始PPO网络端到端完成新动作对象选择。原小模型、v4大模型、v5失败对照均保留，不改变其他正式实验的默认模型。

回退包为`backups/ppo-v4-before-exploration-20260909/code.zip`和其models子目录；解压到独立目录再比较恢复，不直接覆盖整个工作区。每次trial另保存计算源码快照、配置、轨迹、最终方案及哈希。

运行入口：`py -B run_ppo_scalar_exploration.py --id <新的唯一编号> --method ppo_compact --budget 8000 --seed 3381415411 --audit`。无PPO消融把method改为compact。`audit_ppo_scalar_exploration.py`重算本次交付的比较与审计；本次交付固定为七个trial，新增试验应另行版本化审核范围。

本轮未覆盖原论文实验表格、未生成完整帕累托前沿、未提交或推送GitHub。
'''
    (OUT/'results.md').write_text(report,encoding='utf8')
    audit={'status':'PASS','full_milp':matrix,'historical_inputs_unchanged':len(old),'trials':len(rows),
           'no_nsga_searches':True,'trial_result_hashes':hashes,'report_sha256':sha(OUT/'results.md'),
           'results_sha256':sha(OUT/'results.json'),'auditor_sha256':sha(Path(__file__)),
           'backup_zip_sha256':sha(backup/'code.zip'),'python':platform.python_version()}
    write(OUT/'audit.json',audit)
    print(json.dumps({'audit':audit,'means':results['means']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

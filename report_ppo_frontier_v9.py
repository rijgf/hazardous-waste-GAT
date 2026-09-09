"""Rebuild the complete v9 report from immutable new-run records.

The v4 reporting calculations and prose structure are retained here as a
versioned adapter; no historical source or manuscript is changed on import.
Default output is a staged Markdown. --publish is an explicit final-delivery step.
"""
from __future__ import annotations

import argparse
import csv
import os
import json
from itertools import combinations
import math
from pathlib import Path
import statistics

from run_pareto_experiments import ROOT, read, write, checked_instance, SMALL_PREFS
from src.pareto_experiment import ParetoArchive, coverage_pair
from src.reproducibility import deserialize_plan, file_sha256, plan_sha256
from src.solution_utils import evaluate_solution, route_plan_to_solution
from hazardous_waste_model import pickup_type, display_node
from sample_params import params_to_json_data

OUT = ROOT/'output/pareto-ppo-v9'
PAPER = OUT/'数值实验与结果分析_论文稿.md'
FORMAL_PAPER = ROOT/'供应链管理写作/数值实验与结果分析_论文稿.md'
REGISTRY = ROOT/'configs/ppo_models_frontier_v9.json'
PPO_BUDGET = 40320
B1 = B2 = None
ENTRY_SCRIPT = Path(__file__).resolve()


def load_protocol():
    """Read only the new locked protocol; never substitute a historical run."""
    protocol=read(OUT/'protocol.json')
    if protocol['protocol']!='ppo-frontier-v9':
        raise ValueError('This report only accepts the new ppo-frontier-v9 protocol')
    registry=read(REGISTRY)
    if protocol['registry']!=registry:
        raise ValueError('Protocol/registry mismatch')
    for path,expected in protocol['sources'].items():
        if file_sha256(ROOT/path)!=expected:
            raise ValueError('Frozen calculation source changed: '+path)
    for name,expected in protocol['instances'].items():
        if file_sha256(OUT/f'instances/{name}.json')!=expected:
            raise ValueError('Frozen instance changed: '+name)
    for name,entry in registry['models'].items():
        if entry['checkpoint']!=f'outputs/ppo_frontier_v9/models/{name}_model.pt':
            raise ValueError('Unexpected v9 checkpoint path')
        if file_sha256(ROOT/entry['checkpoint'])!=entry['sha256']:
            raise ValueError('Checkpoint hash mismatch')
    return protocol


def execution_summary(protocol):
    records=[read(p) for p in sorted(OUT.glob('execution-*.json'))]
    if not records or any(r['device']!='cpu' or r['threads_per_worker']!=1 for r in records):
        raise ValueError('Expected recorded CPU/single-thread execution batches')
    # Reused NSGA runs belong to the historical execution, not a new v9 batch.
    task_ids={t['id'] for t in protocol['tasks'] if t['kind']=='PPO'}
    if not task_ids <= {task for r in records for task in r['tasks']}:
        raise ValueError('Missing execution provenance for formal tasks')
    workers=sorted({r['workers'] for r in records})
    reuse=protocol['nsga_reuse']
    historical_root=ROOT/reuse['source_root']
    historical_path=historical_root/'execution.json'
    if file_sha256(historical_path)!=reuse['source_execution_sha256']:
        raise ValueError('Historical NSGA execution record changed')
    historical=read(historical_path)
    if historical['protocol_sha256']!=reuse['source_protocol_sha256']:
        raise ValueError('Historical NSGA protocol/execution mismatch')
    if file_sha256(historical_root/'protocol.json')!=reuse['source_protocol_sha256']:
        raise ValueError('Historical NSGA protocol changed')
    if historical['device']!='cpu' or historical['threads_per_worker']!=1:
        raise ValueError('Unexpected historical NSGA execution device/thread count')
    return {'device':'cpu','threads_per_worker':1,'workers':'/'.join(map(str,workers)),
            'batches':len(records),'source_files':[p.name for p in sorted(OUT.glob('execution-*.json'))],
            'orchestration':orchestration_summary(protocol),
            'nsga_reused':True,'nsga_workers':historical['workers'],
            'nsga_execution_source':historical_path.relative_to(ROOT).as_posix(),
            'nsga_execution_sha256':file_sha256(historical_path)}


def orchestration_summary(protocol):
    paths=sorted(OUT.glob('orchestration/*/manifest.json'))
    if not paths:
        return None
    if len(paths)!=1:
        raise ValueError('Multiple orchestration manifests require explicit aggregation')
    path=paths[0];record=read(path)
    if record['protocol_sha256']!=file_sha256(OUT/'protocol.json'):
        raise ValueError('Auxiliary orchestration protocol mismatch')
    script=OUT/'orchestration/launch_coload_aux.ps1'
    if file_sha256(script)!=record['orchestration_source_sha256']:
        raise ValueError('Auxiliary orchestration source changed')
    ids=set(record['task_ids'])
    formal={t['id'] for t in protocol['tasks'] if t['kind']=='PPO'}
    entries=record['entries']
    if not ids<=formal or len(ids)!=len(entries) or {e['task_id'] for e in entries}!=ids:
        raise ValueError('Orchestration task identity/count mismatch')
    initial=record['existing_main_pool_worker_count'];auxiliary=record['auxiliary_worker_count_planned']
    upper=record['declared_global_worker_upper_bound']
    if initial!=3 or auxiliary!=12 or upper!=initial+auxiliary:
        raise ValueError('Unexpected phased concurrency plan')
    for entry in entries:
        if entry['workers']!=1 or entry['cpu_threads_per_worker']!=1:
            raise ValueError('Auxiliary task has changed its thread budget')
        if entry['exit_code']!=0 or not entry['ended_utc'] or not entry.get('completion_exists'):
            raise ValueError('Auxiliary orchestration has not completed cleanly')
        if (not isinstance(entry['pid'],int) or entry['pid']<=0 or not entry['started_utc']
                or entry['command']!=f"py -B run_ppo_frontier_v9.py run --id {entry['task_id']} --workers 1"):
            raise ValueError('Incomplete auxiliary command/PID/start provenance')
        completion=OUT/f"completed/{entry['task_id']}.json"
        if file_sha256(completion)!=entry['completion_sha256']:
            raise ValueError('Auxiliary completion hash mismatch')
        for stream in ('stdout','stderr'):
            name=entry[stream]
            if Path(name).name!=name or file_sha256(path.parent/name)!=entry[stream+'_sha256']:
                raise ValueError('Auxiliary orchestration log hash mismatch')
    links_path=path.parent/'execution_links.json'
    links=read(links_path)
    if links['protocol_sha256']!=record['protocol_sha256'] or {e['task_id'] for e in links['entries']}!=ids:
        raise ValueError('Auxiliary execution links do not cover the declared tasks')
    warnings=[]
    for link in links['entries']:
        if link['status']=='original_execution_found':
            execution_path=ROOT/link['execution_path']
            if file_sha256(execution_path)!=link['execution_sha256']:
                raise ValueError('Auxiliary original execution hash mismatch')
            actual=read(execution_path)
            if actual['tasks']!=[link['task_id']] or actual['workers']!=1 or actual['threads_per_worker']!=1:
                raise ValueError('Auxiliary execution is not the one-task launcher record')
        elif link['status']=='missing_original_execution_record':
            if link['execution_path'] is not None or link['execution_sha256'] is not None:
                raise ValueError('Missing metadata must not contain an invented source path/hash')
            front=read(OUT/f"fronts/{link['task_id']}-B{PPO_BUDGET}.json")
            if front['protocol_sha256']!=record['protocol_sha256'] or front['task']['id']!=link['task_id']:
                raise ValueError('Missing execution record has no matching new frontier')
            if len(front['final_solutions'])!=21:
                raise ValueError('Missing execution record has incomplete preference logs')
            for final in front['final_solutions']:
                if file_sha256(OUT/final['action_log_path'])!=final['action_log_sha256']:
                    raise ValueError('Missing execution record has mismatched action-log provenance')
            warnings.append({'code':'missing_original_execution_record','task_id':link['task_id'],
                'alternative_provenance':'exact command/PID/UTC interval, stdout/stderr, completion and 21 preference action logs',
                'cause':'not established; do not assert timestamp collision as fact'})
        else:
            raise ValueError('Unknown auxiliary execution metadata status')
    # A live launcher process is not proof that a solver task is simultaneously
    # evaluating candidates; do not sum the PID peak and main-pool capacity.
    observed=record.get('observed_global_active_solver_peak')
    if observed is not None and (not isinstance(observed,int) or not 0<=observed<=upper):
        raise ValueError('Invalid recorded global active-solver peak')
    return {'initial_worker_limit':initial,'auxiliary_launcher_count':auxiliary,
            'declared_concurrent_task_upper_bound':upper,
            'verified_peak_concurrent_tasks':observed,
            'observed_auxiliary_launcher_peak':record['observed_auxiliary_process_peak'],
            'source_path':path.relative_to(ROOT).as_posix(),'source_sha256':file_sha256(path),
            'execution_links_path':links_path.relative_to(ROOT).as_posix(),
            'execution_links_sha256':file_sha256(links_path),'metadata_warnings':warnings,
            'source_script_sha256':file_sha256(script),
            'peak_scope':'Global active solver count only; launcher PID peak is not used as a substitute'}


def concurrency_description(execution):
    orchestration=execution.get('orchestration')
    if orchestration is None:
        return (f"新PPO记录的单入口workers配置为{execution['workers']}；"
                '单入口工作进程数不直接等于全局同时求解任务数。')
    peak=orchestration.get('verified_peak_concurrent_tasks')
    peak_text=(f'调度记录核验的实际共同求解峰值为{peak}个任务。' if peak is not None
               else '未得到可精确核验的实际共同求解峰值，不将调度上界冒充实测峰值。')
    launcher_peak=orchestration.get('observed_auxiliary_launcher_peak')
    if launcher_peak is not None:
        peak_text+=f'辅助CLI存活进程峰值为{launcher_peak}，不等于同时求解任务峰值。'
    return (f"新PPO先采用{orchestration['initial_worker_limit']}任务并发，后追加独立入口并行调度，"
            f"预声明同时求解任务上界为{orchestration['declared_concurrent_task_upper_bound']}。"
            +peak_text+f"单入口workers配置为{execution['workers']}，不是全局并发数；"
            '不同任务可能经历不同程度的资源竞争。')


def orchestration_provenance_text(execution,root_link):
    evidence=execution.get('orchestration')
    if evidence is None:
        return ''
    warnings=evidence.get('metadata_warnings',[])
    text=(f"分阶段并行调度见[编排记录]({root_link}/{evidence['source_path']})及"
          f"[独立入口execution映射]({root_link}/{evidence['execution_links_path']})。")
    if warnings:
        tasks='、'.join('`'+w['task_id']+'`' for w in warnings)
        text+=(f'有{len(warnings)}份辅助入口原始execution记录缺失（{tasks}），原因未确定，'
               '不将时间戳碰撞推测当作已证实原因，也没有补造该文件。'
               '这些运行由编排记录中的精确命令、PID、UTC起止时间、标准输出/错误日志、完成文件及21偏好动作日志交叉追溯。'
               '这是执行元数据缺口，不隐瞒为来源完全齐备；方案、种子、预算和动作完整性仍须通过独立核验。'
               '主入口预排任务清单不被当作辅助入口实际启动的证据。')
    return text


def timing_description(execution, training):
    return ('新PPO计算采用CPU冻结推理，每进程1线程。'+concurrency_description(execution)+
            f"NSGA-II复用v4历史运行，其原批次并发数为{execution['nsga_workers']}，同为CPU、每进程1线程。"
            '两算法的秒数来自不同并行争用条件，不是同批、独占机器或严格等时对照，不能直接用时间比值声称公平加速倍数。'
            'MILP本轮另以单线程任务重新求解。表中时间均保留原始墙钟观测，'
            '包括一次完整前沿的搜索、严格候选核验及档案维护；新PPO计时包含动作日志及偏好进度写盘，'
            '历史NSGA-II B2累计时间包含B1中途快照写盘，不因本轮复制档案或离线回放而追加时间。'
            '模型加载、前沿公共初始方案首次评价及最后预算快照导出不计入对应搜索段；'
            '任务总耗时与准备耗时另存。'
            f"Train-S、Train-L本轮训练耗时分别为{training['small']['training_seconds']:.3f} s和"
            f"{training['large']['training_seconds']:.3f} s，不计入推理时间。")


def evaluator_version_text(protocol):
    source_root=ROOT/protocol['nsga_reuse']['source_root']
    historical=read(source_root/'protocol.json')['sources']['src/solution_utils.py']
    current=protocol['sources']['src/solution_utils.py']
    if historical==current:
        raise ValueError('Expected explicit historical/current common-evaluator version distinction')
    return (f'历史NSGA-II搜索实际使用v4版solution_utils（SHA256 `{historical}`）；'
            f'本轮版本（SHA256 `{current}`）已补强BG/BD服务前总容量检查，二者不是同字节评价器。'
            '数学模型的约束定义共用，但历史执行版本应与当前复核版本区分。'
            '本轮所有复用方案均在报告重建时由当前共同评价器重新计算；原完整MILP矩阵、变量界和整数性另由独立审核核验。'
            '完整约束通过结论必须以匹配实际交付文件哈希的本轮终验PASS为准，不将旧版源码锁声称为当前仍通过。')


def reference_utilization(data):
    params,plan,refs=checked_instance(data)
    solution=route_plan_to_solution(params,plan)
    raw=solution['raw']
    processing=max(raw.get(('p',j,s,t),0)/cap for (j,s,t),cap in params.processing_capacity.items()
                   if params.technology[j,s])
    loads=[sum(raw.get(('q',n,v,t),0) for n in params.pickup_nodes)
           for v in params.vehicles for t in params.periods]
    return {'max_processing_utilization':processing,'max_vehicle_utilization':max(loads)/params.vehicle_capacity}


def budget_levels(protocol):
    """NSGA budgets are protocol inputs, never inferred from paper iterations."""
    tasks=protocol['tasks']
    nsga=[t for t in tasks if t['kind']=='NSGA']
    large={tuple(t['budgets']) for t in nsga if t['instance']=='Test-4'}
    if len(large)!=1 or len(next(iter(large)))!=2:
        raise ValueError('Require one declared B1/B2 pair for Test-4')
    b1,b2=next(iter(large))
    if not (isinstance(b1,int) and isinstance(b2,int) and 0<b1<b2):
        raise ValueError('Budgets must be positive increasing candidate counts')
    if any(t['budgets']!=[b1] for t in nsga if t['instance']!='Test-4'):
        raise ValueError('Generalization baseline must use the same B1')
    if any(t.get('budgets',[t.get('budget',PPO_BUDGET)])!=[PPO_BUDGET] for t in tasks if t['kind']=='PPO'):
        raise ValueError('PPO cap must remain 21 times 1920')
    return b1,b2


def training_summary(name, entry):
    """Validate saved training metadata, including nonuniform preference exposure."""
    directory=ROOT/'outputs/ppo_frontier_v9/training'/name
    completed=read(directory/'completed.json')
    config=read(directory/'config.json')
    history=read(directory/'history.json')
    coverage=read(directory/'preference_coverage.json')
    for key,file in [('source_config_sha256','config.json'),('coverage_sha256','preference_coverage.json'),
                     ('history_sha256','history.json')]:
        if completed[key]!=file_sha256(directory/file):
            raise ValueError('Training record hash mismatch: '+name+'/'+file)
    if completed['checkpoint_sha256']!=entry['sha256'] or completed['checkpoint']!=entry['checkpoint']:
        raise ValueError('Training record/checkpoint mismatch')
    training=config['training']
    expected={'instances':24,'rollout_steps':32,'updates':48,'epochs':3,'minibatch':96,
              'lr':.0003,'gamma':.95,'gae_lambda':.90,'clip':.2,'entropy_coefficient':.02,
              'gradient_clip':.5,'actions':36864,'optimizer_steps':1152,'seed':20260909}
    if any(training.get(k)!=v for k,v in expected.items()):
        raise ValueError('Training contract differs from reported v9 specifications')
    if config['args']['dimension']!=96 or config['args']['layers']!=1:
        raise ValueError('Unexpected architecture')
    if len(history)!=48 or history[-1]['attempts']!=36864 or history[-1]['optimizer_steps']!=1152:
        raise ValueError('Training did not finish the predefined final update')
    if len(config['training_inputs'])!=24 or coverage['total_actions']!=36864 or coverage['optimizer_steps']!=1152:
        raise ValueError('Training input/action count mismatch')
    counts=coverage['preference_action_counts']
    if len(counts)!=21 or min(counts)<=0 or sum(counts)!=36864:
        raise ValueError('The complete preference grid was not covered')
    for path,expected_hash in config['training_inputs'].items():
        if '/train/' not in path or file_sha256(ROOT/path)!=expected_hash:
            raise ValueError('Training data provenance mismatch')
    return {'instances':24,'training_seed':training['seed'],'training_seconds':completed['seconds'],
            'training_actions':completed['training_actions'],'optimizer_steps':completed['optimizer_steps'],
            'preference_action_counts':counts,'config':config,
            'training_instance_hashes':[read(ROOT/p)['instance_sha256'] for p in config['training_inputs']],
            'completed_sha256':file_sha256(directory/'completed.json')}


def preference_budget_row(final, archive_id, task, index):
    cap=final['candidate_budget'];actual=final['actual_candidate_attempts']
    if cap!=1920 or not isinstance(actual,int) or not 0<=actual<=cap:
        raise ValueError('Invalid per-preference candidate budget')
    if final['unused_candidate_budget']!=cap-actual:
        raise ValueError('Unused candidate count mismatch')
    reason=final['stop_reason']
    if actual<cap and reason!='positive_probability_action_space_exhausted':
        raise ValueError('Incomplete search must have an explicit exhaustion reason')
    if final['ledger_counts']['candidate_attempts']!=actual:
        raise ValueError('Preference ledger/action count mismatch')
    return {'archive_id':archive_id,'instance':task['instance'],'model':task.get('model','small'),
            'repeat':task['repeat'],'preference_index':index,
            'preference':str(tuple(final['preference'])),'candidate_budget':cap,
            'actual_candidate_attempts':actual,'unused_candidate_budget':cap-actual,
            'stop_reason':reason,'batches':final['batches'],'seconds':final['seconds'],
            'final_validation_evaluations':final['final_validation_evaluations'],
            'preference_initialization_evaluations':final['preference_initialization_evaluations']}


def verify_preference_budget(record):
    finals=record['final_solutions']
    if len(finals)!=21 or record['budget']!=PPO_BUDGET:
        raise ValueError('Incomplete PPO preference grid')
    rows=[preference_budget_row(f,record['archive_id'],record['task'],i) for i,f in enumerate(finals)]
    for index,final in enumerate(finals):
        if not (math.isclose(final['preference'][0],index/20,abs_tol=1e-12) and
                math.isclose(final['preference'][1],1-index/20,abs_tol=1e-12)):
            raise ValueError('Preference order/grid differs from the complete frontier protocol')
    if sum(row['actual_candidate_attempts'] for row in rows)!=record['counts']['candidate_attempts']:
        raise ValueError('PPO frontier and per-preference attempt counts differ')
    return rows


def work_summary(rows,small):
    small_rows=[]
    for i,item in enumerate(small):
        for repeat in range(3):
            record=read(OUT/f'small/ppo-p{i}-r{repeat}.json')
            small_rows.append(preference_budget_row(record,f'small-p{i}-r{repeat}',
                {'instance':'small-fixed','model':'small','repeat':repeat},i))
    csv_write('small_ppo_work.csv',small_rows)
    groups=[]
    for label,selected in [('48次完整PPO前沿',rows),('15次小规模PPO',small_rows)]:
        groups.append({'label':label,'preference_runs':len(selected),
            'candidate_cap':sum(r['candidate_budget'] for r in selected),
            'actual_candidate_attempts':sum(r['actual_candidate_attempts'] for r in selected),
            'unused_candidate_budget':sum(r['unused_candidate_budget'] for r in selected),
            'exhausted_preference_runs':sum(r['actual_candidate_attempts']<r['candidate_budget'] for r in selected),
            'minimum_attempts':min(r['actual_candidate_attempts'] for r in selected),
            'maximum_attempts':max(r['actual_candidate_attempts'] for r in selected)})
    return {'groups':groups,'front_preference_rows':len(rows),'small_preference_rows':len(small_rows)}


def budget_reference_text(protocol):
    b1,b2=budget_levels(protocol)
    reuse_text=('用户最终确认预算不变并允许复用NSGA-II，因此本轮直接复用v4原12次演化的15份预算档案，'
                '不将它们写成新运行，不将原文迭代数强行换算为本仓库尝试数。' if protocol.get('nsga_reuse') else '')
    return ('预算参照核验：本地Chen等论文第15页第5.3节及第16页表7/8的1000/3000标为iterations（迭代次数），'
            '不是本仓库的候选尝试计数。当前NSGA-II种群为100，一个完整子代批产生100个候选，初始化新增99个个体也计数；'
            f'本轮B1={b1}、B2={b2}按锁定协议计为候选尝试，不能宣称与该文1000/3000迭代严格等预算。'
            +reuse_text+'候选尝试次数、目标评价次数及墙钟时间是不同投入指标，两种方法单次动作计算量也不同。')


def work_appendix(work,root_link,output_link):
    table=markdown_table(['方法/实验','偏好运行数','尝试上限合计','实际尝试','未使用名额','提前穷尽次数'],
        [['PPO / '+g['label'],g['preference_runs'],g['candidate_cap'],g['actual_candidate_attempts'],
          g['unused_candidate_budget'],g['exhausted_preference_runs']] for g in work['groups']])
    return f'''\n### A.5 同状态去重与实际候选预算\n\n{table}\n\n预算是求解阶段的完整动作尝试上限，不是训练更新次数，也不等于成功改变方案的次数。不可行、无变化和拒绝动作仍计入实际尝试。只有当前计划及截断后的进度输入均不变时，才保留已试完整动作的记忆；从剩余PPO联合概率中直接抽样，不补抽或额外评价。接受新方案或其他输入变化即清空记忆。正概率动作集合耗尽时如实提前停止，实际次数可能小于上限，不把未使用名额计作算子操作或转给其他偏好。\n\n逐偏好实际次数、未用名额、停止原因、批次数与秒数见[前沿工作量CSV]({output_link}/tables/ppo_preference_work.csv)和[小规模工作量CSV]({output_link}/tables/small_ppo_work.csv)。模型参数不在求解阶段更新；网络结构、完整实例与动作预算在求解前固定。\n'''


def csv_write(name, rows):
    path=OUT/'tables'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values, precision=4):
    values=[value for value in values if value is not None]
    if not values:
        return 'NA'
    mean=statistics.mean(values)
    sd=f'{statistics.stdev(values):.{precision}f}' if len(values)>1 else 'NA'
    return f'{mean:.{precision}f} ± {sd}'


def finite_mean(values):
    values=[value for value in values if value is not None]
    return statistics.mean(values) if values else None


def paired_status(rows):
    valid=sum(row['status']=='valid' for row in rows)
    return f'计划{len(rows)}，有效{valid}，失败{len(rows)-valid}'


def display_difference(value):
    return '0.000' if round(value,3)==0 else f'{value:.3f}'


def milp_optimality_verdict(original, normalized):
    """Separate solver proof from the preserved, pre-handoff source booleans.

    The report may reconcile a false source flag only after a zero-gap optimal
    solver termination and the existing normalized full-matrix/target checks.
    It never changes either input record or treats an elapsed time as proof.
    The surrounding report additionally checks raw/normalized provenance hashes;
    independent delivery audit must recheck the complete mathematical model.
    """
    feasibility_tolerance=1e-5
    objective_tolerance=1e-7
    solver=original.get('solver',{})
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    def near(a,b,tol):
        return finite(a) and finite(b) and math.isclose(a,b,rel_tol=tol,abs_tol=tol)
    def residual(key):
        value=normalized.get(key)
        return finite(value) and 0<=value<=feasibility_tolerance
    objective=normalized.get('metrics',{}).get('weighted_objective')
    solver_objective=solver.get('fun')
    difference=normalized.get('solver_objective_difference')
    checks={
        'same_solver_record':normalized.get('solver')==solver,
        'optimal_termination':solver.get('status')==0 and solver.get('success') is True,
        'zero_reported_gap':finite(solver.get('mip_gap')) and solver['mip_gap']==0.,
        'primal_dual_agreement':near(solver_objective,solver.get('mip_dual_bound'),1e-9),
        'declared_cleanup_tolerances':normalized.get('feasibility_tolerance')==feasibility_tolerance
            and normalized.get('objective_tolerance')==objective_tolerance,
        'strict_normalized_feasible':normalized.get('strict_feasible') is True
            and normalized.get('metrics',{}).get('feasible') is True,
        'normalized_objective_consistent':normalized.get('objective_consistent') is True,
        'full_matrix_within_tolerance':residual('max_matrix_residual'),
        'variable_bounds_within_tolerance':residual('max_bound_residual'),
        'integrality_within_tolerance':residual('max_integer_residual'),
        'original_linear_matches_solver':near(normalized.get('original_linear_objective'),solver_objective,objective_tolerance),
        'cleaned_linear_matches_common':near(normalized.get('cleaned_linear_objective'),objective,objective_tolerance),
        'common_matches_solver':near(objective,solver_objective,objective_tolerance),
        'difference_field_consistent':finite(objective) and finite(solver_objective)
            and near(difference,objective-solver_objective,1e-12),
        'no_additional_solver_calls':normalized.get('additional_solver_calls')==0,
        'source_flag_preserved':normalized.get('proven_optimal')==original.get('proven_optimal')
            and normalized.get('original_objective_consistent')==original.get('objective_consistent'),
    }
    derived=all(checks.values())
    source_flags={'raw_proven_optimal':original.get('proven_optimal'),
                  'raw_objective_consistent':original.get('objective_consistent'),
                  'normalized_proven_optimal':normalized.get('proven_optimal'),
                  'normalized_objective_consistent':normalized.get('objective_consistent')}
    return {'schema':'v9-report-optimality-handoff-v1','source_flags':source_flags,
            'report_proven_optimal':derived,
            'reconciled_source_flag':derived and original.get('proven_optimal') is not True,
            'checks':checks,'failed_checks':[key for key,value in checks.items() if not value],
            'solver_status':solver.get('status'),'solver_success':solver.get('success'),
            'solver_gap':solver.get('mip_gap'),'solver_objective':solver_objective,
            'solver_dual_bound':solver.get('mip_dual_bound'),'normalized_objective':objective,
            'normalized_minus_solver':difference,
            'max_matrix_residual':normalized.get('max_matrix_residual'),
            'max_bound_residual':normalized.get('max_bound_residual'),
            'max_integer_residual':normalized.get('max_integer_residual'),
            'feasibility_tolerance':feasibility_tolerance,'objective_tolerance':objective_tolerance,
            'interpretation':'Report-layer handoff of the existing solver proof, not a new solve or overwritten source flag'}


def milp_status_label(record, verdict=None):
    proven=verdict['report_proven_optimal'] if verdict is not None else record['proven_optimal']
    if proven:
        if verdict is not None and verdict['reconciled_source_flag']:
            return '已证最优（规范化核验）'
        return '已证最优'
    if not record['strict_feasible']:
        return '无严格可行解'
    if 'time limit' in record['solver']['message'].lower():
        return '限时可行（未证最优）'
    return f"可行（未证最优；状态{record['solver']['status']}）"


def milp_reference_note(nonoptimal_count):
    if nonoptimal_count:
        return ('†表示相对尚未证明最优的可行incumbent的差异，不是最优性差距。'
                '未证最优组仅比较当前incumbent质量，不将其作为全局最优基准；终止原因见状态列及原日志。')
    return ('本轮五组MILP参照均满足声明数值容差内的最优终止与交接判据，表中没有限时可行参照。'
            '最优性依据求解器证明及规范化完整约束核验，而非是否接近某个已知目标。')


def risk_component_description(point):
    values=[point[field+'_risk_at_minimum_risk'] for field in
            ('transport','coload','producer_inventory','facility_inventory')]
    if not math.isclose(sum(values),point['minimum_risk'],rel_tol=1e-10,abs_tol=1e-8):
        raise ValueError('Risk endpoint components do not sum to its total risk')
    return '运输、共载、产废端库存、设施库存风险分别为'+'、'.join(f'{value:.4f}' for value in values)


def capacity_endpoint_interpretation(scenes):
    ordered=[scenes[name] for name in ('capacity-70','capacity-80','baseline','capacity-120','capacity-130')]
    cost_text='、'.join(f"{point['minimum_cost']:.3f}" for point in ordered)
    risk_text='、'.join(f"{point['minimum_risk']:.4f}" for point in ordered)
    baseline=scenes['baseline']
    if baseline['minimum_cost']<=0 or baseline['minimum_risk']<=0:
        raise ValueError('Positive baseline endpoints required for relative response reporting')
    changes={field:[100*(point[field]/baseline[field]-1) for point in (ordered[0],ordered[-1])]
             for field in ('minimum_cost','minimum_risk')}
    formatted={field:'和'.join(f'{value:+.2f}%'.replace('-','−') for value in values)
               for field,values in changes.items()}
    share=100*baseline['coload_risk_at_minimum_risk']/baseline['minimum_risk']
    return (f'按处理能力−30%、−20%、基准、+20%、+30%的顺序，五个联合前沿的最低成本依次为{cost_text}，'
            f'最低风险依次为{risk_text}；同一情景的这两个极值通常不属于同一方案。'
            f"相对基准，能力−30%和+30%的最低成本变化分别为{formatted['minimum_cost']}，"
            f"最低风险变化分别为{formatted['minimum_risk']}。"
            f'基准最低风险方案的共载风险占总风险{share:.2f}%，这一比例是该实际方案的风险分解，不是所有方案或全系统的固定倍数。'
            '这些数值反映有限权重、固定搜索预算及情景自身参考尺度下的已获端点，不是精确最优响应或因果效应。')


def nsga_search_diagnostics(records, instances):
    """Count search attempts separately from feasibility of archived points.

    Feasible attempts and cache-miss evaluations are not counts of distinct new
    plans. Only the retained non-reference solutions can be counted uniquely
    from these immutable run records; no unrecorded discovery count is inferred.
    """
    groups={}
    for record in records:
        task=record['task']
        if task['kind']!='NSGA':
            continue
        key=task['instance'],record['budget']
        groups.setdefault(key,[]).append(record)
    summaries=[]
    for (name,budget),group in sorted(groups.items()):
        group.sort(key=lambda record:record['task']['repeat'])
        reference=instances[name]['reference_plan_sha256']
        attempts=invalid=evaluations=invalid_evaluations=operator_failures=0
        only_reference=[]
        nonreference_ids=set()
        by_run=[]
        for record in group:
            assert record['instance_id']==instances[name]['instance_id']
            assert record['reference_plan_sha256']==reference
            counts=record['counts']
            assert counts['candidate_attempts']==budget
            assert 0<=counts['invalid_candidate_attempts']<=budget
            assert 0<=counts['invalid_candidates']<=counts['candidate_objective_evaluations']<=budget
            attempts+=budget
            invalid+=counts['invalid_candidate_attempts']
            evaluations+=counts['candidate_objective_evaluations']
            invalid_evaluations+=counts['invalid_candidates']
            operator_failures+=counts['operator_failures']
            ids=[point['solution_id'] for point in record['points']]
            if ids==[reference]:
                only_reference.append(record['archive_id'])
            nonreference=[identity for identity in ids if identity!=reference]
            nonreference_ids.update(nonreference)
            by_run.append({'archive_id':record['archive_id'],'repeat':record['task']['repeat'],
                           'nonreference_front_points':len(nonreference)})
        summaries.append({'instance':name,'instance_id':instances[name]['instance_id'],'budget':budget,
            'runs':len(group),'candidate_attempts':attempts,'invalid_candidate_attempts':invalid,
            'invalid_attempt_rate':invalid/attempts if attempts else None,
            'feasible_candidate_attempts':attempts-invalid,'candidate_objective_evaluations':evaluations,
            'invalid_objective_evaluations':invalid_evaluations,
            'feasible_objective_evaluations_excluding_preparation':evaluations-invalid_evaluations,
            'operator_failures':operator_failures,'reference_only_run_count':len(only_reference),
            'reference_only_archive_ids':only_reference,
            'nonreference_front_point_occurrences':sum(row['nonreference_front_points'] for row in by_run),
            'nonreference_unique_front_solutions':len(nonreference_ids),
            'nonreference_front_points_by_run':by_run})
    return summaries


def budget_response_interpretation(pair_rows):
    matched={(row['repeat'],row['nsga_budget']):row for row in pair_rows
             if row['instance']=='Test-4' and row['model']=='large'}
    expected={(repeat,budget) for repeat in range(3) for budget in (B1,B2)}
    if set(matched)!=expected or any(row['status']!='valid' for row in matched.values()):
        return 'B1与B2存在缺失或无效配对，未据不完整覆盖记录判断追加预算收益。'
    differences=[]
    for repeat in range(3):
        before,after=matched[repeat,B1],matched[repeat,B2]
        differences.extend([after['N_covers_P']-before['N_covers_P'],
                            before['P_covers_N']-after['P_covers_N']])
    if all(abs(value)<=1e-12 for value in differences):
        return '本案例三次配对中，B1与B2的两方向覆盖率均相同，未观察到追加预算的覆盖收益；这不等于两档案的所有坐标或方案必然相同。'
    if all(value>=-1e-12 for value in differences) and any(value>1e-12 for value in differences):
        return '相对B1，B2在三次配对的两方向覆盖指标中均未出现对NSGA-II不利的变化，且至少一项严格改善；这里的追加预算收益仅指相对于同一PPO档案的覆盖表现，不证明其已覆盖真实完整前沿。'
    return '从B1到B2的两方向覆盖或重复间响应不能概括为一致改善；不能只凭B2某一方向的覆盖值判断追加预算有效。'


def large_coverage_interpretation(pair_rows):
    rows=[row for row in pair_rows if row['instance']=='Test-4' and row['model']=='large']
    expected={(repeat,budget) for repeat in range(3) for budget in (B1,B2)}
    complete=({(row['repeat'],row['nsga_budget']) for row in rows}==expected and len(rows)==6
              and all(row['status']=='valid' for row in rows))
    if complete and all(row['N_covers_P']==1. and row['P_covers_N']==0. for row in rows):
        return ('本次L20的Train-L比较中，B1与B2各自三次配对均得到𝒞(N,P)=1、𝒞(P,N)=0：'
                'NSGA-II档案均覆盖对应PPO档案的全部点，而PPO均未覆盖NSGA-II档案中的任何点。'
                '因此，当前实例、冻结模型和预算下，NSGA-II已获解集的覆盖表现优于PPO；'
                '这一结论针对保存的近似档案，不证明NSGA-II已获得真实完整帕累托前沿，也不构成跨实例算法优劣结论。')
    if complete and all(row['N_covers_P']==0. and row['P_covers_N']==1. for row in rows):
        return ('本次L20的Train-L比较中，B1与B2各自三次配对均得到𝒞(N,P)=0、𝒞(P,N)=1：'
                'PPO档案均覆盖对应NSGA-II档案的全部点，而NSGA-II均未覆盖PPO档案中的任何点。'
                '这一结果仅支持本固定案例、算法版本及不同明确预算下已获解集的覆盖比较，'
                '不证明PPO已获得真实完整帕累托前沿，也不构成跨实例统计优势或预算公平性的证明。')
    if (complete and all(row['P_covers_N']==0. for row in rows)
            and all(row['N_covers_P']==1. for row in rows if row['nsga_budget']==B2)):
        uncovered=100*(1-sum(row['N_covers_P'] for row in rows if row['nsga_budget']==B1)/3)
        return (f'本轮大规模完整偏好网格的近似前沿比较仍落后于NSGA-II。B1下，平均约{uncovered:.1f}%的PPO点未被对应NSGA-II档案覆盖，'
                '但这不代表PPO反向覆盖了NSGA-II：三次配对的𝒞(P,N)均为0；B2下，NSGA-II三次均完全覆盖对应PPO档案，反向覆盖仍为0。'
                '这一结论仅针对当前固定实例、模型及预算下保存的解集，不证明真实前沿已被完整获得，也不是跨实例统计优越性结论。'
                '此前v8在0.5/0.5单点、每次12000候选下的标量目标胜出，不能套用于本轮多偏好训练模型、每个偏好1920候选的完整前沿比较；'
                '两次实验的模型训练口径、每偏好预算和评价对象不同，当前结果尚不能将差距定量归因于其中某一项。')
    return '大规模结果应按两个方向及各次配对共同解释，结论仅限当前实例、模型和预算下保存的近似档案，不证明真实帕累托前沿的完整性或跨实例优势。'


def experiment_section(task, budget):
    if task['instance'].startswith(('capacity-','coload-')):
        return 'sensitivity'
    if task['instance']!='Test-4':
        return 'generalization'
    if task['kind']=='PPO':
        return 'large_comparison;generalization;sensitivity_baseline' if task['model']=='large' else 'generalization'
    return 'large_comparison;generalization' if budget==B1 else 'large_comparison'


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def grouped_table(first, groups, rows):
    headers=[first]+[f'{group} {direction}' for group in groups for direction in ('N→P','P→N')]
    return markdown_table(headers,rows)


def parameter_revision_evidence(protocol, parameters):
    """Validate the revised base/scenario parameters before describing them.

    This is reporting validation, not a parameter generator or a substitute for
    the independently saved instance/calibration and numerical audit records.
    Legacy v3 has no revision claim and retains its separate capacity proof.
    """
    revision=protocol.get('parameter_revision')
    if revision is None:
        return None
    if (revision['capacity_factor']!=2. or revision['risk_total_ratio_range']!=[3.,6.]
            or revision['risk_strength_range']!=[2.,5.] or not revision['models_retrained']
            or revision['training_hyperparameters_changed'] or not revision['small_case_revised']):
        raise ValueError('Report does not match the user-authorized revision protocol')
    capacities={}
    risks={}
    for name in ['Test-1','Test-2','Test-3','Test-4','small-fixed']:
        params=parameters[name]
        rows=[]
        for waste in params.waste_types:
            average=sum(params.generation[i,waste,t] for i in params.producers for t in params.periods)/len(params.periods)
            eligible=[j for j in params.facilities if params.technology[j,waste]]
            totals=[sum(params.processing_capacity[j,waste,t] for j in eligible) for t in params.periods]
            if not eligible or not all(math.isclose(value,2*average,rel_tol=1e-12,abs_tol=1e-12) for value in totals):
                raise ValueError(f'{name}/{waste}: effective capacity is not twice average period generation')
            rows.append({'waste':waste,'average_period_generation':average,
                         'eligible_facilities':eligible,'effective_capacity_each_period':totals})
        ratios=[]
        for a,b in combinations(params.waste_types,2):
            pair=tuple(sorted((a,b)))
            gamma=params.coload_risk[pair]
            if params.compatibility[pair] and gamma>0:
                ratio=1+2*gamma/(params.waste_consequence[a]+params.waste_consequence[b])
                if not 3.-1e-12<=ratio<=6.+1e-12:
                    raise ValueError(f'{name}/{pair}: same-arc equal-load transport risk ratio outside [3,6]')
                ratios.append(ratio)
        if not ratios:
            raise ValueError(f'{name}: no positive compatible co-load pair')
        capacities[name]=rows
        risks[name]={'compatible_pairs':len(ratios),'observed_total_ratio_range':[min(ratios),max(ratios)]}
    base=parameters['Test-4']
    base_json=params_to_json_data(base)
    for field,name in [('processing_capacity','capacity'),('coload_risk','coload')]:
        for factor in [.7,.8,1.2,1.3]:
            key=f'{name}-{round(factor*100)}'
            changed=parameters[key]
            changed_json=params_to_json_data(changed)
            if {k for k in base_json if base_json[k]!=changed_json[k]}!={field}:
                raise ValueError(f'{key}: sensitivity is not a single-field change')
            old_values,new_values=getattr(base,field),getattr(changed,field)
            if old_values.keys()!=new_values.keys() or not all(
                    math.isclose(new_values[k],v*factor,rel_tol=1e-12,abs_tol=1e-12) for k,v in old_values.items()):
                raise ValueError(f'{key}: sensitivity multiplier differs from protocol')
    return {'capacity_by_instance':capacities,'risk_by_instance':risks,
            'capacity_scenario_factors':[.7,.8,1.,1.2,1.3],
            'risk_total_ratio_range':[3.,6.],'risk_strength_range':[2.,5.],
            'risk_scope':'same-arc equal positive two-type loads, transport only; not whole-system risk',
            'models_retrained':True,'historical_v4_training_hyperparameters_changed':False,
            'current_v9_uses_optimized_architecture_and_training':True}


def build(publish=False):
    global B1, B2, PAPER
    PAPER = FORMAL_PAPER if publish else OUT/FORMAL_PAPER.name
    protocol=load_protocol()
    B1,B2=budget_levels(protocol)
    preference_rows=[]
    protocol_hash=file_sha256(OUT/'protocol.json')
    instance_data={name:read(OUT/f'instances/{name}.json') for name in protocol['instances']}
    revision=parameter_revision_evidence(protocol,{name:checked_instance(data)[0] for name,data in instance_data.items()})
    if revision is None or not protocol.get('nsga_reuse'):
        raise ValueError('This v9 delivery requires unchanged v4 parameters and explicit historical NSGA reuse')
    output_link=os.path.relpath(OUT,PAPER.parent).replace('\\','/')
    root_link=os.path.relpath(ROOT,PAPER.parent).replace('\\','/')
    preflight=read(OUT/'preflight.json')
    execution=execution_summary(protocol)
    assert preflight['status']=='PASS' and preflight['not_used_in_formal_results']
    assert preflight['source_sha256']==protocol['sources']
    assert preflight['model_sha256']==protocol['registry']['models']['small']['sha256']
    expected={task['id'] for task in protocol['tasks']}
    assert len(expected)==60 and sum(t['kind']=='PPO' for t in protocol['tasks'])==48
    assert sum(t['kind']=='NSGA' for t in protocol['tasks'])==12
    actual={path.stem for path in (OUT/'completed').glob('*.json')}
    if actual != expected:
        raise RuntimeError(f'No partial manuscript: completed {len(actual)}/{len(expected)}')
    fronts={}
    front_checks={}
    replayed=0
    point_rows=[]
    run_rows=[]
    for task in protocol['tasks']:
        data=read(OUT/f'instances/{task["instance"]}.json')
        params,_,refs=checked_instance(data)
        for budget in task.get('budgets',[PPO_BUDGET]):
            archive_id=f'{task["id"]}-B{budget}'
            record=read(OUT/f'fronts/{archive_id}.json')
            if task['kind']=='NSGA':
                from run_ppo_frontier_v9 import verify_reused_nsga
                verify_reused_nsga(protocol,record,OUT/f'fronts/{archive_id}.json')
            else:
                assert record['protocol_sha256']==protocol_hash
            assert record['instance_sha256']==data['instance_sha256']
            assert 0 <= record['counts']['candidate_attempts'] <= budget
            if task['kind']=='NSGA':
                assert record['counts']['candidate_attempts']==budget
            else:
                preference_rows.extend(verify_preference_budget(record))
            assert (record['b_C'],record['b_R'])==refs
            archive=ParetoArchive(refs)
            clear_points=0
            for point in record['points']:
                solution=read(OUT/point['solution_path'])
                plan=deserialize_plan(solution['plan'])
                assert plan_sha256(plan)==point['solution_id']==solution['solution_id']
                replay_solution=route_plan_to_solution(params,plan)
                metrics=evaluate_solution(params,replay_solution,(.5,.5),objective_refs=refs)
                last=params.periods[-1]
                terminal_values=[replay_solution['raw'].get(('IG',node,last),0) for node in params.pickup_nodes]
                terminal_values += [replay_solution['raw'].get(('ID',facility,waste,last),0) for facility in params.facilities for waste in params.waste_types]
                clear_points+=all(abs(value)<=1e-5 for value in terminal_values)
                assert metrics['feasible']
                assert abs(metrics['cost']-point['cost'])<1e-8 and abs(metrics['risk']-point['risk'])<1e-8
                archive.add(point)
                replayed+=1
                point_rows.append({'archive_id':archive_id,'instance':task['instance'],'instance_id':data['instance_id'],
                                   'experiment_section':experiment_section(task,budget),
                                   'model':task.get('model',''),'algorithm':task['kind'],'repeat':task['repeat'],
                                   'budget':budget,'b_C':refs[0],'b_R':refs[1],'solution_id':point['solution_id'],
                                   'cost':point['cost'],'risk':point['risk'],'feasible':True,
                                   'source':json.dumps(point['provenance'],ensure_ascii=False),'solution_path':point['solution_path']})
            assert len(archive.sorted_points())==len(record['points'])
            fronts[archive_id]=record
            front_checks[archive_id]={'points':len(record['points']),'strict_feasible_points':sum(point['feasible'] for point in record['points']),
                                      'terminal_clear_points':clear_points}
            run_rows.append({'archive_id':archive_id,'instance':task['instance'],'model':task.get('model',''),
                             'algorithm':task['kind'],'repeat':task['repeat'],'budget':budget,
                             'run_origin':'v4_reused' if task['kind']=='NSGA' else 'v9_new',
                             'execution_workers':execution['nsga_workers'] if task['kind']=='NSGA' else execution['workers'],
                             'execution_workers_scope':'historical_batch' if task['kind']=='NSGA' else 'per_entry_configuration_not_global_peak',
                             'seconds':record['seconds'],'points':len(record['points']),'status':record['status'],
                             'strict_feasible_points':front_checks[archive_id]['strict_feasible_points'],'terminal_clear_points':clear_points,
                             'preparation_seconds':record['preparation_seconds'], **record['counts'],
                             'return_validation_evaluations':len(record.get('final_solutions',[])),
                             'preference_initialization_evaluations':record.get('preference_initialization_evaluations',0),
                             'objective_evaluations_total_excluding_preparation':record['counts']['candidate_objective_evaluations']+record['counts']['ppo_internal_objective_evaluations']+len(record.get('final_solutions',[]))+record.get('preference_initialization_evaluations',0)})
    csv_write('ppo_preference_work.csv',preference_rows)
    csv_write('frontier_points.csv',point_rows)
    csv_write('frontier_runs.csv',run_rows)
    pair_rows=[]
    for scale in ('Test-1','Test-2','Test-3','Test-4'):
        for model in ('small','large'):
            for budget in ([B1,B2] if scale=='Test-4' and model=='large' else [B1]):
                for repeat in range(3):
                    p=fronts[f'PPO-{scale}-{model}-r{repeat}-B{PPO_BUDGET}']
                    n=fronts[f'NSGA-{scale}-r{repeat}-B{budget}']
                    coverage=coverage_pair(p['points'],n['points'],(p['b_C'],p['b_R']))
                    pair_rows.append({'instance':scale,'instance_id':p['instance_id'],'model':model,
                                      'repeat':repeat,'nsga_budget':budget,'P_archive':p['archive_id'],'N_archive':n['archive_id'],
                                      'n_P':coverage['n_A'],'n_N':coverage['n_B'],
                                      'b_C':p['b_C'],'b_R':p['b_R'],
                                      'N_covers_P_count':coverage['B_covers_A_count'],'P_covers_N_count':coverage['A_covers_B_count'],
                                      'N_covers_P':coverage['B_covers_A'],'P_covers_N':coverage['A_covers_B'],
                                      'tolerance':coverage['tolerance'],'status':coverage['status']})
    csv_write('coverage_pairs.csv',pair_rows)
    t5=[]
    for budget in (B1,B2):
        rows=[row for row in pair_rows if row['instance']=='Test-4' and row['model']=='large' and row['nsga_budget']==budget]
        t5.extend([stats([row['N_covers_P'] for row in rows]),stats([row['P_covers_N'] for row in rows])])
    b2_interpretation=budget_response_interpretation(pair_rows)
    t6=[]
    for model,label in [('small','Train-S'),('large','Train-L')]:
        row=[label]
        for scale in ('Test-1','Test-2','Test-3','Test-4'):
            matched=[item for item in pair_rows if item['instance']==scale and item['model']==model and item['nsga_budget']==B1]
            row += [stats([x['N_covers_P'] for x in matched]),stats([x['P_covers_N'] for x in matched])]
        t6.append(row)
    scale_comparisons=[]
    for scale in ('Test-1','Test-2','Test-3','Test-4'):
        values={model:{direction:finite_mean(item[direction] for item in pair_rows
                       if item['instance']==scale and item['model']==model and item['nsga_budget']==B1)
                       for direction in ('P_covers_N','N_covers_P')} for model in ('small','large')}
        s,l=values['small'],values['large']
        if any(value is None for value in [*s.values(),*l.values()]):
            scale_comparisons.append(f'{scale}存在空前沿或失败，未据缺失配对排序模型')
        elif abs(s['P_covers_N']-l['P_covers_N'])<1e-12 and abs(s['N_covers_P']-l['N_covers_P'])<1e-12:
            scale_comparisons.append(f'{scale}上两模型相对同一NSGA-II基线的两方向覆盖均值相同')
        elif s['P_covers_N']>=l['P_covers_N'] and s['N_covers_P']<=l['N_covers_P']:
            scale_comparisons.append(f'{scale}上Train-S覆盖NSGA-II的比例不低于Train-L，且被NSGA-II覆盖的比例不高于Train-L')
        elif l['P_covers_N']>=s['P_covers_N'] and l['N_covers_P']<=s['N_covers_P']:
            scale_comparisons.append(f'{scale}上Train-L覆盖NSGA-II的比例不低于Train-S，且被NSGA-II覆盖的比例不高于Train-S')
        else:
            scale_comparisons.append(f'{scale}上两方向覆盖指标出现交叉，不能据单一方向作模型排序')
    small=[]
    small_rows=[]
    small_data=read(OUT/'instances/small-fixed.json')
    small_params,_,small_refs=checked_instance(small_data)
    for index,preference in enumerate(SMALL_PREFS):
        records=[read(OUT/f'small/ppo-p{index}-r{repeat}.json') for repeat in range(3)]
        for record in records:
            assert record['protocol_sha256']==protocol_hash
            plan=deserialize_plan(record['plan'])
            metrics=evaluate_solution(small_params,route_plan_to_solution(small_params,plan),preference,objective_refs=small_refs)
            assert metrics['feasible'] and abs(metrics['weighted_objective']-record['metrics']['weighted_objective'])<1e-9
        selected=min(records,key=lambda record:(record['metrics']['weighted_objective'],record['repeat']))
        mip=read(OUT/f'milp/validated/p{index}.json')
        original_mip=read(OUT/f'milp/p{index}.json')
        assert mip['original_result_sha256']==file_sha256(OUT/f'milp/p{index}.json')
        assert mip['cleanup_source_sha256']==file_sha256(ROOT/'normalize_pareto_milp.py')
        assert mip['original_protocol_file_sha256']==file_sha256(OUT/'milp/protocol.json')
        assert mip['original_objective_consistent']==original_mip['objective_consistent']
        assert mip['elapsed_seconds']==original_mip['elapsed_seconds'] and mip['solver']==original_mip['solver']
        assert mip['additional_solver_calls']==0
        assert mip['instance_sha256']==small_data['instance_sha256']
        assert mip['reference_sha256']==small_data['reference_plan_sha256']
        assert tuple(mip['objective_refs'])==small_refs and tuple(mip['preference'])==preference
        assert mip['strict_feasible'] and mip['objective_consistent']
        verdict=milp_optimality_verdict(original_mip,mip)
        verdict.update(preference=list(preference),preference_index=index,
                       original_result_path=f'milp/p{index}.json',
                       original_result_sha256=file_sha256(OUT/f'milp/p{index}.json'),
                       normalized_result_path=f'milp/validated/p{index}.json',
                       normalized_result_sha256=file_sha256(OUT/f'milp/validated/p{index}.json'))
        mp=read(OUT/'milp/protocol.json')
        assert mp['solver_options']['time_limit']==3600
        j=selected['metrics']['weighted_objective']
        mj=mip['metrics']['weighted_objective']
        difference=100*(j-mj)/mj if mj else j-mj
        total=sum(record['seconds'] for record in records)
        small.append({'preference':preference,'selected_repeat':selected['repeat'],'ppo':selected,
                      'milp':mip,'milp_verdict':verdict,'difference':difference,'total_ppo_seconds':total})
        m=selected['metrics']
        small_rows.append(['PPO',str(preference),f'{m["cost"]:.3f}',f'{m["risk"]:.4f}',
                           f'{j:.6f}',display_difference(difference)+('†' if not verdict['report_proven_optimal'] else ''),f'{total:.2f}','严格可行'])
    for item in small:
        mip=item['milp'];m=mip['metrics']
        small_rows.append(['MILP',str(item['preference']),f'{m["cost"]:.3f}',f'{m["risk"]:.4f}',
                           f'{m["weighted_objective"]:.6f}','—',f'{mip["elapsed_seconds"]:.2f}',
                           milp_status_label(mip,item['milp_verdict'])])
    write(OUT/'milp/report_optimality_verdicts.json',{'source_flags_preserved':True,
        'additional_solver_calls':0,'report_source_sha256':file_sha256(Path(__file__)),
        'verdicts':[item['milp_verdict'] for item in small]})
    csv_write('table4_selected_solutions.csv',[{'preference':str(x['preference']),'selected_repeat':x['selected_repeat'],
              'solution_id':x['ppo']['solution_id'],'ppo_cost':x['ppo']['metrics']['cost'],'ppo_risk':x['ppo']['metrics']['risk'],
              'ppo_J':x['ppo']['metrics']['weighted_objective'],'milp_J':x['milp']['metrics']['weighted_objective'],
              'relative_difference_percent':x['difference'],'ppo_total_seconds':x['total_ppo_seconds'],
              'milp_seconds':x['milp']['elapsed_seconds'],'milp_proven_optimal':x['milp']['proven_optimal'],
              'milp_raw_proven_optimal':x['milp_verdict']['source_flags']['raw_proven_optimal'],
              'milp_report_proven_optimal':x['milp_verdict']['report_proven_optimal'],
              'milp_source_flag_reconciled':x['milp_verdict']['reconciled_source_flag'],
              'milp_cost':x['milp']['metrics']['cost'],'milp_risk':x['milp']['metrics']['risk'],
              'milp_original_result_path':x['milp']['original_result_path'],
              'milp_original_result_sha256':x['milp']['original_result_sha256'],
              'milp_cleanup_seconds':x['milp']['cleanup_seconds'],
              'milp_cleanup_source_sha256':x['milp']['cleanup_source_sha256'],
              'milp_solver_objective_difference':x['milp']['solver_objective_difference']} for x in small])
    scenario_rows=[]
    scene_summary=[]
    scene_sets={}
    for scene in ['baseline','capacity-70','capacity-80','capacity-120','capacity-130',
                  'coload-70','coload-80','coload-120','coload-130']:
        source='Test-4' if scene=='baseline' else scene
        data=read(OUT/f'instances/{source}.json')
        union=ParetoArchive((data['b_C'],data['b_R']))
        records=[fronts[f'PPO-{source}-large-r{repeat}-B{PPO_BUDGET}'] for repeat in range(3)]
        for record in records:
            for point in record['points']:
                union.add({**point,'archive_id':record['archive_id']})
        points=union.sorted_points()
        scene_sets[scene]=points
        scenario_params,_,_=checked_instance(data)
        peak_processing_utilization=0.
        for point in points:
            saved_plan=deserialize_plan(read(OUT/point['solution_path'])['plan'])
            saved_solution=route_plan_to_solution(scenario_params,saved_plan)
            peak_processing_utilization=max(peak_processing_utilization,
                max(saved_solution['raw'].get(('p',j,s,t),0)/capacity for (j,s,t),capacity in scenario_params.processing_capacity.items()
                    if scenario_params.technology[j,s]))
            scenario_rows.append({'scenario':scene,'instance_id':data['instance_id'],'b_C':data['b_C'],'b_R':data['b_R'],
                                  'solution_id':point['solution_id'],'cost':point['cost'],'risk':point['risk'],
                                  'archive_id':point['archive_id'],'solution_path':point['solution_path']})
        min_cost=min(points,key=lambda p:(p['cost'],p['risk'],p['solution_id']))
        min_risk=min(points,key=lambda p:(p['risk'],p['cost'],p['solution_id']))
        risk_plan=deserialize_plan(read(OUT/min_risk['solution_path'])['plan'])
        periods=list(scenario_params.periods)
        ordered_routes=sorted(risk_plan.items(),key=lambda item:(item[0][1],item[0][0]))
        mixed_routes=[item for item in ordered_routes if len({pickup_type(node) for node in item[1][1:-1]})>1]
        (example_vehicle,example_period),example_route=(mixed_routes or ordered_routes)[0]
        risk_solution=route_plan_to_solution(scenario_params,risk_plan)
        return_loads={waste:risk_solution['raw'].get(('F',example_route[-2],example_route[-1],waste,example_vehicle,example_period),0)
                      for waste in scenario_params.waste_types}
        return_loads={key:value for key,value in return_loads.items() if value>1e-9}
        scene_summary.append({'scenario':scene,'b_C':data['b_C'],'b_R':data['b_R'],'points':len(points),
                              'maximum_processing_utilization_in_union':peak_processing_utilization,
                              'time_mean':statistics.mean(r['seconds'] for r in records),
                              'minimum_cost':min_cost['cost'],'risk_at_minimum_cost':min_cost['risk'],
                              'minimum_risk':min_risk['risk'],'cost_at_minimum_risk':min_risk['cost'],
                              'minimum_cost_solution':min_cost['solution_id'],'minimum_risk_solution':min_risk['solution_id'],
                              'vehicle_period_routes_at_minimum_risk':len(risk_plan),
                              'service_visits_at_minimum_risk':sum(len(route)-2 for route in risk_plan.values()),
                              'visits_by_period_at_minimum_risk':json.dumps({period:sum(len(route)-2 for (vehicle,t),route in risk_plan.items() if t==period) for period in periods}),
                              'example_route_vehicle':example_vehicle,'example_route_period':example_period,
                              'example_route':json.dumps(example_route,ensure_ascii=False),
                              'example_route_mixed':bool(mixed_routes),'example_return_loads':json.dumps(return_loads),
                              'transport_risk_at_minimum_risk':min_risk['transport_risk'],
                              'coload_risk_at_minimum_risk':min_risk['coload_risk'],
                              'producer_inventory_risk_at_minimum_risk':min_risk['producer_inventory_risk'],
                              'facility_inventory_risk_at_minimum_risk':min_risk['facility_inventory_risk']})
    csv_write('sensitivity_points.csv',scenario_rows)
    csv_write('sensitivity_scenarios.csv',scene_summary)
    from plot_pareto_sensitivity import plot
    import plot_pareto_sensitivity as scatter
    previous_out=scatter.OUT
    try:
        scatter.OUT=OUT
        plot()
    finally:
        scatter.OUT=previous_out
    sensitivity_variants_text = ''
    if revision is not None:
        from plot_ppo_frontier_v9 import plot as plot_variants
        plot_variants(OUT)
        sensitivity_variants_text = f'''
**图2（版本二：折线版） 同一批实际方案按成本排序连线**

![关键参数敏感性前沿折线版]({output_link}/figures/sensitivity_line.png)

注：上方原图为版本一（原始散点版）。本版使用完全相同的实际方案坐标，各情景按成本递增依次连接，不删除点、不平均坐标；为减少遮挡不再逐点绘制空心标记。线段仅辅助辨认变化趋势，不表示两方案之间的成本—风险组合均可行。

**图2（版本三：平滑曲线版） 同一批实际方案的保形插值展示**

![关键参数敏感性前沿平滑曲线版]({output_link}/figures/sensitivity_smooth.png)

注：采用分段三次Hermite保形插值（PCHIP），以成本为自变量，对每个情景单独插值；曲线经过全部原始点，保持风险随成本非增且不超出相邻点的风险范围，不向观测成本区间之外外推。未做回归拟合或改变原始数值；局部陡降和长间隔仍保留，因此不应把视觉平滑理解为真实前沿连续或已求得更多方案。三版采用相同坐标尺度，展示同一组实验而非三次独立实验。

两种曲线版本的可编辑绘图脚本为[plot_ppo_frontier_v9.py]({root_link}/plot_ppo_frontier_v9.py)，复用原PCHIP计算；若某情景仅有一个实际点，则只显示该点，不虚构曲线。同时提供[折线SVG]({output_link}/figures/sensitivity_line.svg)、[平滑SVG]({output_link}/figures/sensitivity_smooth.svg)与350 dpi PNG；[绘图追溯记录]({output_link}/figures/sensitivity_variants.json)保存源数据哈希、原始节点及仅供可视化的插值坐标。
'''
    training={model:training_summary(model,entry) for model,entry in protocol['registry']['models'].items()}
    training_hashes={h for record in training.values() for h in record['training_instance_hashes']}
    if any(instance_data[name]['instance_sha256'] in training_hashes for name in ('Test-1','Test-2','Test-3','Test-4')):
        raise ValueError('Claimed train/test parameter separation does not hold')
    operations=[]
    for model,label in [('small','Train-S'),('large','Train-L')]:
        for scale in ('Test-1','Test-2','Test-3','Test-4'):
            records=[fronts[f'PPO-{scale}-{model}-r{repeat}-B{PPO_BUDGET}'] for repeat in range(3)]
            success=sum(record['status']=='success' and bool(record['points']) for record in records)
            points=[point for record in records for point in record['points']]
            feasible_rate=f"{sum(point['feasible'] for point in points)/len(points)*100:.0f}%" if points else 'NA'
            terminal_rate=f"{sum(front_checks[r['archive_id']]['terminal_clear_points'] for r in records)/len(points)*100:.0f}%" if points else 'NA'
            operations.append([f'PPO（{label}）',scale,stats([r['seconds'] for r in records],2),
                               stats([len(r['points']) for r in records],2),f'{success}/{len(records)}',feasible_rate,terminal_rate])
    for scale in ('Test-1','Test-2','Test-3','Test-4'):
        for budget in ([B1,B2] if scale=='Test-4' else [B1]):
            records=[fronts[f'NSGA-{scale}-r{repeat}-B{budget}'] for repeat in range(3)]
            success=sum(record['status']=='success' and bool(record['points']) for record in records)
            points=[point for record in records for point in record['points']]
            feasible_rate=f"{sum(point['feasible'] for point in points)/len(points)*100:.0f}%" if points else 'NA'
            terminal_rate=f"{sum(front_checks[r['archive_id']]['terminal_clear_points'] for r in records)/len(points)*100:.0f}%" if points else 'NA'
            operations.append([f'NSGA-II（B{1 if budget==B1 else 2}；v4复用）',scale,stats([r['seconds'] for r in records],2),
                               stats([len(r['points']) for r in records],2),f'{success}/{len(records)}',feasible_rate,terminal_rate])
    capacity_identical=all([(p['solution_id'],p['cost'],p['risk']) for p in scene_sets[scene]]==
                           [(p['solution_id'],p['cost'],p['risk']) for p in scene_sets['baseline']]
                           for scene in ['capacity-70','capacity-80','capacity-120','capacity-130'])
    optimum_items=[x for x in small if x['milp_verdict']['report_proven_optimal']]
    exact_matches=sum(abs(x['difference'])<1e-4 for x in optimum_items)
    limited_items=[x for x in small if not x['milp_verdict']['report_proven_optimal']]
    small_limited_text='；'.join(f"偏好{x['preference']}下，PPO的最小J为{x['ppo']['metrics']['weighted_objective']:.6f}，"
        f"MILP可行J为{x['milp']['metrics']['weighted_objective']:.6f}（{milp_status_label(x['milp'],x['milp_verdict'])}），相对该incumbent的差异为{display_difference(x['difference'])}%"
        for x in limited_items) if limited_items else ''
    paired_valid=sum(row['status']=='valid' for row in pair_rows)
    scenes={item['scenario']:item for item in scene_summary}
    baseline=scenes['baseline']
    base_data=instance_data['Test-4']
    base_params,_,_=checked_instance(base_data)
    type_totals={waste:sum(base_params.generation[producer,waste,period] for producer in base_params.producers
                           for period in base_params.periods) for waste in base_params.waste_types}
    low_params,_,_=checked_instance(read(OUT/'instances/capacity-70.json'))
    low_capacities={waste:min(low_params.processing_capacity[facility,waste,period] for facility in low_params.facilities
                               for period in low_params.periods if low_params.technology[facility,waste])
                    for waste in low_params.waste_types}
    assert all(value==0 for value in base_params.initial_producer_inventory.values())
    assert all(value==0 for value in base_params.initial_facility_inventory.values())
    if revision is None:
        assert all(low_capacities[waste]>type_totals[waste] for waste in type_totals)
    coload_interpretation=[]
    route_interpretation=[]
    for scene,label in [('baseline','基准'),('coload-70','共载系数−30%'),('coload-130','共载系数+30%')]:
        item=scenes[scene]
        route='→'.join(display_node(node) for node in json.loads(item['example_route']))
        loads='、'.join(f'{waste}={amount:.4f}' for waste,amount in json.loads(item['example_return_loads']).items())
        mixture='含兼容异类共载的首条路线' if item['example_route_mixed'] else '未发现异类共载，以下为首条单品类路线'
        route_interpretation.append(f"{label}风险端点方案`{item['minimum_risk_solution'][:12]}`：{mixture}（按周期、车辆稳定排序），"
            f"周期{item['example_route_period']}、车辆{item['example_route_vehicle']}执行{route}，返回设施前携带{loads}")
    for scene,label in [('coload-70','下调30%'),('coload-130','上调30%')]:
        item=scenes[scene]
        coload_interpretation.append(f"共载风险系数{label}时，风险端点为(C,R)=({item['cost_at_minimum_risk']:.3f},{item['minimum_risk']:.4f})，"
            f"{risk_component_description(item)}，"
            f"使用{item['vehicle_period_routes_at_minimum_risk']}条车辆—周期路线、{item['service_visits_at_minimum_risk']}次实体服务，逐期服务次数为{item['visits_by_period_at_minimum_risk']}")
    aliases={model:entry.get('experiment_alias','New-S' if model=='small' else 'New-L')
             for model,entry in protocol['registry']['models'].items()}
    training_instances={model:record['instances'] for model,record in training.items()}
    assert set(training_instances.values())=={24}
    capacity_reference=reference_utilization(instance_data['Test-4'])
    capacity_low_reference=reference_utilization(instance_data['capacity-70'])
    mismatch_count=sum(not x['milp']['original_objective_consistent'] for x in small)
    mismatch_text=(f"{mismatch_count}组原始返回解的共同评价与求解器目标在规范化前不一致；整数弧及连续载量的近零容差值可能被评价器的存在性判断放大"
                   if mismatch_count else '五组原始返回解的共同评价与求解器目标在规范化前均一致；仍统一执行相同的容差规范化与完整约束复核')
    reconciled_items=[x for x in small if x['milp_verdict']['reconciled_source_flag']]
    handoff_details='；'.join(
        f"偏好{x['preference']}的原始及规范化记录均保留proven_optimal=false，"
        f"原始objective_consistent=false；但求解器status=0、success=true、gap=0，"
        f"原目标与对偶界均为{x['milp_verdict']['solver_objective']:.17g}，"
        f"规范化J={x['milp_verdict']['normalized_objective']:.17g}，"
        f"两者绝对差为{abs(x['milp_verdict']['normalized_minus_solver']):.3g}，"
        f"完整矩阵最大残差为{x['milp_verdict']['max_matrix_residual']:.3g}，"
        f"运行时间为{x['milp']['elapsed_seconds']:.2f} s，并非限时终止"
        for x in reconciled_items)
    handoff_text=('最优状态采用独立的报告层交接判定：要求求解器正常最优终止、status=0、success=true、'
        'mip_gap=0、原目标与对偶界相符，并要求规范化方案严格可行、共同目标及线性目标与求解器目标一致，'
        '原完整矩阵、变量界和整数性残差均不超过10⁻⁵，目标交接容差为10⁻⁷。'
        '这里“已证最优”指声明数值容差内的最优终止与交接；规范化J略小于求解器对偶界时，'
        '该微小差异属于数值容差，不是找到比已证最优值更优的新解。'
        '该判定只接续原有求解器证明，不新增求解或覆盖原始布尔标志。'
        +(handoff_details+'。这些组在表中标为“已证最优（规范化核验）”，不误标为限时可行。' if handoff_details
          else '本轮没有通过规范化交接更正报告最优状态的组。'))
    base_identity=base_data['instance_id']
    scenario_records=[fronts[f'PPO-{source}-large-r{repeat}-B{PPO_BUDGET}']
                      for source in ['Test-4',*['capacity-'+str(n) for n in [70,80,120,130]],
                                     *['coload-'+str(n) for n in [70,80,120,130]]] for repeat in range(3)]
    scenario_success=sum(r['status']=='success' and bool(r['points']) for r in scenario_records)
    if revision is not None:
        case_intro='本轮逐字节复用v4参数修订后的五个基础案例及八个敏感性版本，不更换实例、不重新绑定参考尺度；重新训练大小规模PPO并重新求解PPO与MILP，NSGA-II按用户确认复用原预算下的历史结果。'
        model_intro=(f"Train-S与Train-L分别对应本轮重新训练后冻结的{aliases['small']}、{aliases['large']}。两模型分别在原v4对应规模的24个训练实例上从头训练，采用v8优化后的全对象PPO结构，并按预先固定的21偏好episode分配覆盖整张偏好网格。每实例只分配1至3个偏好，不声称每实例均见齐21偏好。训练结束仅保存预定最终检查点，随后冻结推理；旧模型、训练库与结果保留。v8结构曾依据固定Test-4反馈调整，本轮重训不抹去这种测试知情设计。")
        reference_intro=(f"来源实例生成种子为{small_data['instance_seed']}，新版实例为`{small_data['instance_id']}`。"
            '参考方案沿用v4时已完成的逐期服务、联合车辆—设施分配及确定性路线排序构造，原分配器种子为0；本轮只校验保存的方案和b，未重新构造或定标。原参考种子作为历史来源信息保留，不冒充本轮新构造的随机种子。')
        registry_path='configs/ppo_models_frontier_v9.json'
        training_completion='本轮在原v4训练库从头训练两份多偏好模型，随后冻结用于全部测试；网络及训练口径采用优化后的v8，不冒充原v4训练规格'
        cap=revision['capacity_by_instance']['Test-4']
        observed=revision['risk_by_instance']['Test-4']['observed_total_ratio_range']
        average_text=', '.join(f"{row['average_period_generation']:.4f}" for row in cap)
        capacity_text=', '.join(f"{row['effective_capacity_each_period'][0]:.4f}" for row in cap)
        revision_intro=(f"新版处理能力的口径为：每类废物在每个周期内，对技术匹配设施的有效能力求和，合计等于该类全系统平均单周期产量的2倍，匹配设施之间等额分配；技术不匹配的名义容量不计入有效合计。兼容异类对各自按固定种子独立抽取附加强度κ∼U(2,5)，共载系数为κ(h_s+h_t)/2，其中h_s、h_t为废物后果系数。只有在同一弧段、两类等正载量的条件下，基础运输风险加额外共载风险与无共载附加项的运输风险之比为1+κ∼U(3,6)，不将最大一对强制校准到5倍，也不意味着整个系统风险放大相同倍数。Test-4实际兼容异类对的这一比值范围为{observed[0]:.4f}—{observed[1]:.4f}。此区间是按本轮要求设置的合成压力情景，不是化学反应或企业数据标定系数。")
        capacity_bound=(f"本次四类平均单周期产量依次为{average_text}，"
            f"每期有效处理能力合计依次为{capacity_text}，均为对应平均单期产量的2倍。"
            '五个敏感性水平对应有效合计能力为平均单期产量的1.4、1.6、2.0、2.4、2.6倍。该条件是系统合计口径，不等于每座设施各有同样的系统能力；具体车辆—设施分配、逐期产量及库存安排仍可能使局部约束紧张。旧版以全规划期总量设置单设施能力的冗余性证明不适用于本轮，不能预先认定五种能力情景的可行域相同。实际处理利用率和端点决策用于描述已获方案，不能仅由前沿移动断言容量约束的因果贡献。')
        generator_description='新版训练库48个实例、测试库四种规模各1个实例和独立小规模验证案例，均从原锁定数据逐例派生，只更改processing_capacity和coload_risk两个参数字段；参数版本、来源哈希与各对实际系数均随实例保存。其余合成生成区间保持原样：单节点—品类—周期产生量[0.5,2.0]，坐标[0,100]，路段事故概率[0.0005,0.003]，处理成本[4,20]；车辆、产废端库存、处理处置端库存系数分别为1.8、2.5、3.0。废物后果系数[1,4]，产废端库存风险系数[0.2,1.0]，设施库存风险系数[0.1,0.8]；固定车辆启用成本50、单位距离成本3。兼容概率为1，技术匹配生成概率0.85且保障每类废物至少有可处理设施，初始库存为0并要求期末清零。新版共载系数不再沿用旧[0.05,0.5]区间，而按第（一）节的随机相对运输强度规则生成；处理能力不再以单设施相对全期总量定义。所有数值仍为合成参数，不能作为危险废物实际混合安全阈值。'
    work=work_summary(preference_rows, small)
    budgets_text=budget_reference_text(protocol)
    report_sources={'report_script':Path(__file__).relative_to(ROOT).as_posix(),
                    'report_script_sha256':file_sha256(Path(__file__)),
                    'report_entry_script':ENTRY_SCRIPT.relative_to(ROOT).as_posix(),
                    'report_entry_script_sha256':file_sha256(ENTRY_SCRIPT)}
    nsga_diagnostics=nsga_search_diagnostics(fronts.values(),instance_data) if revision is not None else []
    nsga_diagnostic_text=''
    nsga_interpretation=''
    if nsga_diagnostics:
        descriptions=[]
        for row in nsga_diagnostics:
            label=f"{row['instance']}/B{1 if row['budget']==B1 else 2}"
            per_run='/'.join(str(item['nonreference_front_points']) for item in row['nonreference_front_points_by_run'])
            descriptions.append(f"{label}：不可行尝试{row['invalid_candidate_attempts']}/{row['candidate_attempts']}"
                f"（{row['invalid_attempt_rate']*100:.2f}%），严格可行尝试{row['feasible_candidate_attempts']}次，"
                f"缓存未命中后的可行评价{row['feasible_objective_evaluations_excluding_preparation']}次；"
                f"仅参考方案的档案{row['reference_only_run_count']}/{row['runs']}次，非参考前沿点逐次为{per_run}"
                f"（合计{row['nonreference_front_point_occurrences']}点位，按方案标识去重{row['nonreference_unique_front_solutions']}份）")
        nsga_diagnostic_text=('NSGA-II搜索有效性另行核对如下，均按同一规模/预算的3次原始记录汇总，不把B1和继续运行的B2相加作为独立投入：'
            +'；'.join(descriptions)+'。严格可行尝试数等于candidate_attempts减invalid_candidate_attempts，不含公共参考的准备，但包含重复与缓存命中；缓存未命中的可行评价数等于candidate_objective_evaluations减invalid_candidates，也不是互不相同的新方案数。未保留的所有历史可行方案无法据这些计数恢复为唯一方案数，故不作此推断。仅参考档案的具体运行编号和逐次非参考前沿点数随results及report_verification中的nsga_search_diagnostics保存。')
        reference_only=sum(row['reference_only_run_count'] for row in nsga_diagnostics)
        nsga_interpretation=('交付档案点的严格可行率与候选尝试的可行率是不同指标，前者即使为100%也不表示基线进行了充分有效搜索。'
            +(f'本轮有{reference_only}个NSGA-II预算档案仅保留公共参考方案（B2包含B1，并非独立运行数）；在这些档案中没有保留下来的非参考前沿方案，不能据非空前沿的success状态掩盖输出未扩展。' if reference_only else '本轮各NSGA-II预算档案均保留了非参考方案，但这仍不能证明搜索充分或前沿完整。')
            +'所有尝试和失败按原预算保留，不依据本轮结果调整解码器或参数。方法间的覆盖优劣只针对本案例、当前编码修复及可行性限制下的已获解集，不能概括为任何一方的算法本质优势。详细候选有效性与仅参考档案计数见附录A.2。')
    write(OUT/'results.json',{'protocol_sha256':protocol_hash,'table5':t5,'table6':t6,
                             'coverage_valid':paired_valid,'coverage_planned':27,'front_records':len(fronts),
                             'front_point_replays':replayed,'sensitivity_capacity_union_identical':capacity_identical,
                             'parameter_revision_evidence':revision,**report_sources,
                             'nsga_search_diagnostics':nsga_diagnostics,
                             'nsga_budget_response_interpretation':b2_interpretation,
                             'ppo_work':work,'training_summaries':training,'execution_summary':execution,
                             'new_ppo_front_runs':48,'reused_nsga_runs':12,'reused_nsga_front_archives':15,
                             'milp_optimality_verdicts':[x['milp_verdict'] for x in small],
                             'budget_levels':{'PPO_front_cap':PPO_BUDGET,'NSGA_B1':B1,'NSGA_B2':B2},
                             'small_selected':[{'preference':x['preference'],'J':x['ppo']['metrics']['weighted_objective'],
                                               'difference':x['difference'],'repeat':x['selected_repeat']} for x in small]})
    manuscript=fr'''# 四、数值实验与结果分析

本节依次开展固定偏好下的小规模质量验证、大规模近似帕累托前沿比较、冻结模型跨规模案例测试和关键参数敏感性分析。{case_intro}所有重复均为同一实例上的算法随机重复，不是独立实例样本。历史五偏好单目标GA结果及50实例汇总不并入本轮。

## （一）实验设置与共同评价口径

### 1. 实例、模型与归一化

小规模质量验证的规模为3/2/2/3/2（依次为产废节点、废物品类、处理处置设施、车辆及周期），{reference_intro}四个跨规模测试案例分别为Test-1（3/2/2/3/2）、Test-2（6/2/2/4/3）、Test-3（10/3/3/5/3）和Test-4（20/4/3/8/4），均读取已锁定测试库的000号{'新版' if revision is not None else ''}实例。小规模质量验证案例与Test-1虽具有相同规模，但实例身份不同。大规模主实验L20及敏感性分析共用Test-4基础实例。

{model_intro}每个实例在准备阶段与一份严格可行参考方案绑定，固定正值`b_C`、`b_R`，目标为

$$
J_{{\omega,i}}(x)=\omega_c\frac{{C_i(x)}}{{b_{{C,i}}}}+\omega_r\frac{{R_i(x)}}{{b_{{R,i}}}},\qquad\omega_c+\omega_r=1.
$$

PPO的目标输入、奖励规则及冻结搜索的加权择优均读取目标实例自身的同一组b，不使用1000/100旧编码，不随偏好、重复或当前解重新定标。测试b无需等于训练b。各算法针对同一实例和数学约束，保存方案在本轮报告重建时统一由当前共同评价器重放；PPO和NSGA-II共享公共参考起点，MILP仅共享实例及固定b，未声明使用该起点暖启动。NSGA-II按原始成本C和风险R搜索双目标。敏感性版本分别在自身参数下预先绑定b，跨版本只比较原始C、R及实际方案，不以不同b下的J排序。

{evaluator_version_text(protocol)}

{revision_intro}

### 2. 搜索预算、档案与统计

小规模设置五组偏好，每组PPO独立求解3次，各次上限1920个候选动作，取最小J实际方案。其他PPO实验的一次完整前沿包含21组偏好(k/20,1−k/20)，k=0,…,20；每偏好均从同一公共起点出发，上限1920次尝试，每批8个候选，满额相当于240批，完整前沿上限{PPO_BUDGET}次。采用温度2、输入进度horizon=1024和同状态完整动作去重，未增加原数值实验文稿的每偏好尝试上限。状态输入改变即清空去重记录；仍按PPO剩余联合概率抽样，不用启发式代选对象。正概率动作集合真实耗尽时允许提前停止，不补虚构次数、不将剩余名额转移给其他偏好。外部档案保存初始方案及全部实际生成且严格可行的非支配候选，而不只保留21个末次方案；档案不反馈搜索。

复用的历史NSGA-II使用种群100、交叉率0.90、个体变异率0.25，以及非支配等级—拥挤度二元锦标赛和父子代精英环境选择。每个“收运实体—周期”的基因表达服务开关、车辆、设施偏好及访问顺序；末期必访，前期服务独立可变，统一确定性修复允许同一实体跨周期再次全量服务。NSGA-II也维护仅供记录的外部非支配档案，不用档案改变环境选择。原B1为{B1}次候选尝试，Test-4在原运行中另续至B2={B2}次；初始化新增个体、重复候选和修复失败均计入预算，半代耗尽时已评价的可行候选仍纳入档案。

{budgets_text}

主指标采用弱覆盖率，以𝒞区别于经济成本C：

$$
\mathcal{{C}}(A,B)=\frac{{\left|\left\{{b\in B:\ \exists a\in A,\ C(a)\leq C(b),\ R(a)\leq R(b)\right\}}\right|}}{{|B|}}
$$

其中，集合外的竖线表示集合基数，即点数。分子是在B中被A内至少一个方案弱支配的点数，分母为B的点数；成本和风险均不大于被比较点即计入覆盖，包括相等目标点。判定使用未经展示舍入的`C/b_C`、`R/b_R`，统一容差10⁻⁸。两个档案先分别去重并剔除严格被支配点，不先合并算法解集；目标重合时按稳定方案标识保留代表。每个重复编号配对计算两个方向覆盖率，表内为3次完整前沿配对的均值±运行间样本标准差（ddof=1）。21偏好不构成21个统计样本。

{timing_description(execution,training)}

## （二）小规模固定偏好求解质量验证

{'新版' if revision is not None else '原'}小规模实例`{small_data['instance_id']}`绑定b_C={small_refs[0]:.15g}、b_R={small_refs[1]:.17g}。本节仅比较当前冻结Train-S与MILP，不构建完整精确帕累托前沿。MILP每组单次求解，显式时限3600 s、相对gap目标0；原始变量、状态、界和墙钟时间均另存。PPO各偏好沿用已记录的3个原评估种子，模型改用登记的新冻结模型。

**表4 小规模固定偏好下PPO-Transformer与MILP的求解结果**

{markdown_table(['方法','偏好 (ωc,ωr)','成本 C','风险 R','目标 J','差异 (%)','时间 (s)','状态'],small_rows)}

注：表中PPO为PPO-Transformer的简称。PPO每行对应3次运行中J最小的同一个实际方案，成本、风险与J不分别择优；并列按运行编号。PPO时间为3次完整搜索耗时之和。MILP时间为本轮重新求解的单次实际墙钟时间，不含之后离线数值规范化或独立审计；C、R与J来自同一个经验证的实际变量解。相对差异记为Δ_J，单位为%，正值表示PPO目标值高于MILP参照值，计算式为：

$$
\Delta_J(\%)=100\,\frac{{J_{{\mathrm{{PPO}}}}-J_{{\mathrm{{MILP}}}}}}{{J_{{\mathrm{{MILP}}}}}}
$$

{milp_reference_note(len(limited_items))}

MILP数值交接采用统一离线规范化：对五组返回向量均将距整数不超过10⁻⁵的整数变量舍入，连续变量绝对值小于10⁻⁵时置零，其余连续量不变。{mismatch_text}；原始向量、指标与一致性标志完整保留在[原始MILP记录]({output_link}/milp/)中，不用重建路线替代返回解。规范化后的[五组完整变量方案]({output_link}/milp/validated/)均重新通过原完整MILP矩阵、变量界、整数性与共同严格可行检查，最大约束残差为{max(x['milp']['max_matrix_residual'] for x in small):.3g}，共同评价J与原求解器目标的最大绝对差为{max(abs(x['milp']['solver_objective_difference']) for x in small):.3g}。该步骤未新增搜索或提高最优性证明，后处理耗时另存cleanup_seconds，不计入表中求解时间；求解器终止状态、界、gap和3600 s时限均来自原记录；报告最优判定单独保存，不覆盖源文件标志。

{handoff_text}

五组偏好中，MILP在{len(optimum_items)}组完成最优性证明，PPO在这些组中的{exact_matches}组达到相同目标（数值容差内）。纯成本偏好下风险不参与目标，成本相同的不同实际方案可以具有不同风险，表中未用其他方案的较小风险替换MILP返回解；纯风险偏好下成本同样只作为伴随指标报告，不参与该次目标。{small_limited_text+'。' if small_limited_text else ''}小规模表使用三次择优结果，不能将其解释为单次平均性能；该投入也必须和三次累计时间一并考虑。

## （三）大规模近似帕累托前沿比较

L20采用Test-4唯一固定案例`{base_identity}`，PPO使用Train-L。每次新完整PPO前沿同时与相同重复编号的v4历史NSGA-II B1、B2档案比较；B2在原运行中接续B1演化，不是本轮新启动或额外一套独立重复。两方向分别统计，不能将覆盖率理解为成本或风险降低比例。

**表5 大规模实例下PPO-Transformer与NSGA-II的双向覆盖率**

{grouped_table('方法',['B1','B2'],[['PPO-L vs NSGA-II',*t5]])}

注：PPO-L表示PPO（Train-L），固定案例为L20。B1、B2分别为{B1}、{B2}次尝试。每格为同一固定案例3次完整前沿配对的均值±样本标准差。P表示PPO前沿，N表示对应预算的NSGA-II前沿；N→P表示𝒞(N,P)，P→N表示𝒞(P,N)，从PPO角度前者越小、后者越大越有利。两组比较复用同一PPO档案，B2未给PPO增加重启。B1配对：{paired_status([row for row in pair_rows if row['instance']=='Test-4' and row['model']=='large' and row['nsga_budget']==B1])}；B2配对：{paired_status([row for row in pair_rows if row['nsga_budget']==B2])}。原始覆盖点数与分母见coverage_pairs.csv。失败配对记NA，若有失败则均值为有效配对条件下的统计，少于2个有效配对时SD记NA。对应方法的完整前沿计算时间集中列于附表A2。

B1下𝒞(N,P)与𝒞(P,N)分别为{t5[0]}和{t5[1]}，B2下分别为{t5[2]}和{t5[3]}。这些数值只描述两算法在本案例、当前编码与明确预算下发现的近似解集，不证明任何一方覆盖真实完整前沿。候选尝试预算相同也不代表目标评价次数或运行时间相同，判断投入需结合附表A2的完整前沿时间。

{large_coverage_interpretation(pair_rows) if revision is not None else ''}

{b2_interpretation}

{nsga_interpretation}

## （四）冻结模型跨规模案例测试

两个冻结模型分别用于四个固定案例，每个模型—案例组合进行3次完整前沿求解，共24次PPO前沿。各规模共用同一套复用的v4历史3次NSGA-II B1记录；Test-4的Train-L行与表5 B1数值完全一致，不重复计算或更换基线。训练数据与四个测试实例的规范参数哈希无交集，但Test-4曾用于v8架构探索，因此不能称为完全未见评估。新编码使用稳定的实体—周期、车辆—周期及设施对象槽位，三个512类头接收前序选择与候选边信息；对象范围、路线对应与硬掩码由新版本预检核对，不沿用旧编码的token上限说明。

**表6 两种训练模型在四种测试规模上的双向覆盖率**

{grouped_table('PPO模型',['Test-1','Test-2','Test-3','Test-4'],t6)}

注：两行×四组规模，每组两方向，共八个数值列；N→P表示𝒞(N,P)，P→N表示𝒞(P,N)。每格仅针对一个固定实例上的3次配对运行，均值±标准差反映算法随机性，不反映实例间变异。P为该行PPO模型档案，N为相同案例NSGA-II B1档案，不是将两个PPO模型互作覆盖基准。配对状态为{paired_status([row for row in pair_rows if row['nsga_budget']==B1])}；空前沿失败配对记NA，不代填0或1。每一模型—规模及NSGA基线的完整前沿计算时间见附表A2。

{'；'.join(scale_comparisons)}。这比较的是各模型相对共同基线的覆盖表现，不等于两个PPO档案彼此的严格支配关系。

本节据四个案例观察同一冻结策略在不同任务规模上的适应情况。表中跨规模数值同时受目标景观和相应NSGA-II解集影响，不能直接据其排列规模难度，也不能把相对基线覆盖率称为严格的泛化损失。每种训练规模只有一份固定检查点、每种测试规模只有一个实例，因此不据此推断训练规模的平均效应或跨实例统计优势。

## （五）关键参数敏感性分析

以`{base_identity}`为唯一基础网络，分别将{'新版基准' if revision is not None else '现有'}处理能力或兼容异类共载风险系数乘以0.7、0.8、1.0、1.2、1.3。两个参数共用基准，总共9个参数版本；逐字段验证除指定参数外其余网络、产废量、技术匹配、兼容关系、车辆与库存容量均不变。每版本使用当前参数下预先绑定的b与同一Train-L，在固定配对种子下重新求解3个完整前沿。基准3次严格复用表5记录，其余8版本新增24次，合计27次记录。

**图2 关键参数变化下的成本—风险近似帕累托前沿**

![关键参数敏感性前沿]({output_link}/figures/sensitivity.png)

注：案例`{base_identity}`；Train-L；21组偏好，每组最多1920候选、每批8个；每情景3次运行联合获得的近似前沿。两子图各展示−30%、−20%、基准、+20%、+30%五个情景，每一点对应已保存且严格可行的实际方案。各情景分别联合、去重和非支配筛选，不平均点坐标，也不跨情景筛除点。散点不表示点间插值可行；横轴为成本模型单位，纵轴为模型风险指标。9情景共计划{len(scenario_records)}次运行，成功{scenario_success}次。
{sensitivity_variants_text}

{'处理能力五个水平的联合前沿及方案标识完全重合，按实际结果保留，不为制造差异追加参数调整或搜索预算。' if capacity_identical else '处理能力扰动下的曲线按实际求解结果展示，不通过改动未声明容量或追加搜索人为拉开差异。'} 参考方案回放中，基准参考方案最大处理能力利用率为{capacity_reference['max_processing_utilization']*100:.2f}%，处理能力下调30%版本自身参考方案的最大处理利用率为{capacity_low_reference['max_processing_utilization']*100:.2f}%；基准参考方案车辆最大利用率为{capacity_reference['max_vehicle_utilization']*100:.2f}%。上述利用率对应各版本预先固定的参考方案，不冒充所有最优方案的约束活跃性证明。

即使某项处理能力约束未活跃，设施对象中的处理能力与利用率输入仍可能随该参数改变，进而影响冻结策略的动作概率和有限预算搜索轨迹。因此，能力扰动下前沿的变化不能自动解释为可行域放松带来的系统收益，需要与约束利用率、实际方案和策略输入响应区分。

{capacity_bound}

{capacity_endpoint_interpretation(scenes) if revision is not None else ''}

联合前沿中，基准与处理能力下调30%版本的最大处理利用率分别为{scenes['baseline']['maximum_processing_utilization_in_union']*100:.2f}%与{scenes['capacity-70']['maximum_processing_utilization_in_union']*100:.2f}%。基准风险端点为(C,R)=({baseline['cost_at_minimum_risk']:.3f},{baseline['minimum_risk']:.4f})，{risk_component_description(baseline)}；使用{baseline['vehicle_period_routes_at_minimum_risk']}条车辆—周期路线、{baseline['service_visits_at_minimum_risk']}次实体服务，逐期服务次数为{baseline['visits_by_period_at_minimum_risk']}。{'；'.join(coload_interpretation)}。这些端点都是各情景实际联合档案中的单个方案；对不同方案的风险分项与服务安排作并列展示，并不能单独识别参数的因果效应。

共载风险系数变化会同时改变同一方案的风险评价和重优化时的选解，因此不能把风险坐标下降全部归因于路线策略改善。各情景使用自身固定b，相同权重表示相对该情景参考方案的偏好，不意味着跨情景绝对目标系数相同。附表A3报告实际前沿端点，补充CSV记录端点风险分项与逐期服务次数，完整路线、车辆与周期记录可按方案标识回放；所有响应均限定于本冻结PPO求解协议，不视为精确最优响应。

为核对实际共载组成，以下仅提取每个对应风险端点的一条稳定选定路线，并非把该路线当作整个方案：{'；'.join(route_interpretation)}。车辆在访问各实体时全量收取当期可用量；完整方案及其他车辆、周期的收运安排均由对应方案文件保留。

## （六）结果范围与局限

本轮使用5个固定基础实例（{'原小案例与四个测试规模案例的参数修订版' if revision is not None else '历史小案例加四个测试规模案例'}），敏感性的8个新增版本均来自同一Test-4基础网络，不是新增随机实例。小规模按固定偏好验证解质量，其他部分比较有明确尝试预算的近似解集。有限权重加权和可能遗漏非凸目标区域中的非支撑有效点，保存完整候选档案仍不构成全局帕累托完整性证明。

这些案例和重复只支持当前模型、生成规则与预算下的描述性结论；3次重复不是跨实例样本。NSGA-II为历史档案而非新计时对照，其原16任务并发条件与新PPO批次不同，时间只作原始投入记录，不据此作公平速度排名。本轮新模型只使用TRAIN数据更新参数并保存预定最终检查点，评估期间未根据结果更换实例、微调检查点或追加某一情景预算。但优化架构在此前已反复参照Test-4表现，因此本轮不能表述为无测试知情调参的泛化验证，也不据三个重复声称统计显著。正式的多实例统计验证、更多训练种子及真实企业数据验证需另行安排。

## 附录A 可复现参数与交付记录

### A.1 冻结模型和参考尺度

{markdown_table(['方法','训练实例','seed','训练时间 (s)','检查点'],[[f'PPO（Train-{label}）',training_instances[model],training[model]['training_seed'],f"{training[model]['training_seconds']:.3f}",aliases[model]] for model,label in [('small','S'),('large','L')]])}

Train-S完整SHA256：

{protocol['registry']['models']['small']['sha256']}

Train-L完整SHA256：

{protocol['registry']['models']['large']['sha256']}

每模型训练48次rollout更新×32步×24实例=36864次动作；每次rollout做3个epoch、minibatch=96，合计1152次Adam更新。学习率0.0003、γ=0.95、GAE λ=0.90、截断0.20、熵系数0.02、梯度裁剪0.50；episode horizon按实例为512、1024、1536。训练按预先固定的21偏好网格分配episode，保存每实例×偏好的实际动作计数。网络为96维嵌入、1层Transformer、4个注意力头、前馈192维、Dropout=0、六个算子和三个独立参数的条件化512类对象头。聚合上下文经过Linear(101,96)与GELU，价值输出为Linear(96,1)，不使用追加隐藏层的失败对照结构。输入、奖励和择优统一采用实例绑定b。训练使用温度1的普通条件抽样；同状态去重仅用于冻结推理。{training_completion}；两模型对应规范路径与哈希以[{registry_path}]({root_link}/{registry_path})为准。

{generator_description}

### A.2 完整前沿运行时间、点数与可行性

{markdown_table(['方法','规模','前沿时间 (s)','点数','成功/计划','严格可行率','期末清零率'],operations)}

注：时间和点数为同一案例3次运行的均值±样本标准差。NSGA-II全部时间沿用原v4记录（{execution['nsga_workers']}任务并发）；PPO为本轮新记录，其分阶段并行调度与实际共同运行峰值见第（一）节，单入口workers=1或3不代表全局同时任务数。二者并行争用条件不同，不能直接计算公平加速倍数。NSGA-II B2时间是从原同次运行开始累计至{B2}次尝试，包含B1，不是B1之后的增量。PPO小规模表4时间为三次累计，而本表前沿时间为一次完整21偏好前沿的运行均值，两者口径不同。candidate_attempts、candidate_objective_evaluations、ppo_internal_objective_evaluations、cache_hits与operator_failures分别保留；搜索段实际评价次数= candidate_objective_evaluations + ppo_internal_objective_evaluations + return_validation_evaluations + preference_initialization_evaluations。后两项分别为PPO每偏好返回方案再验证及偏好初始化评价，各21次（NSGA-II均为0），不改变{PPO_BUDGET}次尝试上限。该总数不含前沿公共参考准备或离线表图回放核验；新策略的全部候选由公共ledger核验，不把PPO内部评价计数留零误写成没有其他评价开销。invalid_candidate_attempts包含缓存命中的不可行尝试，invalid_candidates为新评价中不可行次数；动作无变化/修复返回失败与严格不可行不混为同一指标。

{nsga_diagnostic_text}

### A.3 敏感性真实端点

{markdown_table(['方法','情景','b_C','b_R','点数','最低C','对应R','最低R','对应C','时间 (s)'],[['PPO-L',x['scenario'],f"{x['b_C']:.3f}",f"{x['b_R']:.4f}",x['points'],f"{x['minimum_cost']:.3f}",f"{x['risk_at_minimum_cost']:.4f}",f"{x['minimum_risk']:.4f}",f"{x['cost_at_minimum_risk']:.3f}",f"{x['time_mean']:.2f}"] for x in scene_summary])}

注：PPO-L为PPO（Train-L）；点数为各情景3次运行联合前沿的点数，C、R分别表示成本、风险。成本端点按(C,R,方案标识)排序，风险端点按(R,C,方案标识)排序，每对坐标属于同一实际方案。最低成本点不被预先规定为最大风险点，最低风险点也不被预先规定为最大成本点。计算时间为一次完整前沿的3次运行均值，不是3次累计。详细风险分项及端点方案标识见sensitivity_scenarios.csv。

### A.4 文件对应与复现

公式采用可直接编辑的LaTeX数学块，不使用公式图片；数学排版需由支持LaTeX的Markdown预览器显示。本轮重新训练PPO并更新PPO、MILP求解，NSGA-II保留原预算并复用历史结果；公式展示仍保持可编辑的文字形式。

本轮协议为{protocol['protocol']}，协议SHA256为`{protocol_hash}`。原始实例/参数版本、公共参考方案和固定b见[instances]({output_link}/instances/)；全部前沿、方案、覆盖点计数、未舍入坐标及候选工作量保存在[实验目录]({output_link}/)中。表4对应table4_selected_solutions.csv；表5和表6共用coverage_pairs.csv；图2对应sensitivity_points.csv，可编辑脚本为[plot_pareto_sensitivity.py]({root_link}/plot_pareto_sensitivity.py)，另交付[SVG矢量图]({output_link}/figures/sensitivity.svg)及350 dpi PNG。交付包含48次新PPO完整前沿、复用的12次历史NSGA-II演化（15个预算档案）、15次新小规模PPO求解及5次新3600 s上限MILP求解；基准敏感性3次为已有Train-L/Test-4记录的复用。

{orchestration_provenance_text(execution,root_link)}

仅重建已完成结果的表图和暂存文稿使用`py -B {ENTRY_SCRIPT.name}`，最终确认后以`--publish`写入正式Markdown；该入口会核验协议并逐方案回放，不启动训练或新增搜索。报告模板、实际入口、绘图脚本与本文的SHA256另存report_verification.json。MILP的原始与规范化source_flags、报告层derived判定、全部判据及来源哈希见[最优性交接记录]({output_link}/milp/report_optimality_verdicts.json)。新训练入口为`train_ppo_frontier_v9.py`，MILP入口为`run_frontier_v9_milp.py`，其余运行与核验命令以本轮[协议文档]({root_link}/docs/ppo_frontier_v9_protocol.md)为准。报告自身回放不替代独立原MILP全约束核验；只有独立交付审核与实际文件哈希一致时才报告整体完成。源码与JSON等字节由.gitattributes保护；克隆后须核验记录的SHA，不通过自动重算manifest掩盖换行或来源变化。

方法组织参照Chen等（2026）多权重解集、双向覆盖与固定案例敏感性展示，但本研究不声称实现其邻域参数迁移。参考：[本地论文](<{root_link}/参考文献/Chen 等 - 2026 - Integrated hybrid energy and time-of-use electricity tariffs for the resource-constrained project sc.pdf>)。本轮按新PPO结果与明确复用的NSGA-II档案重新汇总并核验全部表图，不将旧五偏好均值改名为帕累托实验，也不将复用的NSGA-II档案冒充本轮新计算。
'''
    manuscript += work_appendix(work, root_link, output_link)
    PAPER.parent.mkdir(parents=True,exist_ok=True)
    PAPER.write_text(manuscript,encoding='utf8')
    write(OUT/'report_verification.json',{'passed':True,'formal_front_records':len(fronts),
          'formal_tasks':len(expected),'front_points_replayed':replayed,'coverage_pair_count':len(pair_rows),
          'coverage_valid':paired_valid,'small_ppo_runs':15,'milp_runs':5,
          'protocol_sha256':protocol_hash,'manuscript_sha256':file_sha256(PAPER),'manuscript_path':PAPER.relative_to(ROOT).as_posix(),
          'ppo_work':work,'execution_summary':execution,
          'new_ppo_front_runs':48,'reused_nsga_runs':12,'reused_nsga_front_archives':15,
          'milp_optimality_verdicts':[x['milp_verdict'] for x in small],
          **report_sources,'parameter_revision_evidence':revision,
          'nsga_search_diagnostics':nsga_diagnostics,
          'nsga_budget_response_interpretation':b2_interpretation,
          'plot_script_sha256':file_sha256(ROOT/'plot_pareto_sensitivity.py'),
          'plot_variants_script_sha256':file_sha256(ROOT/'plot_ppo_frontier_v9.py') if revision is not None else None,
          'interpolation_source_sha256':file_sha256(ROOT/'plot_pareto_sensitivity_variants.py'),
          'milp_cleanup_source_sha256':file_sha256(ROOT/'normalize_pareto_milp.py')})
    print(json.dumps(read(OUT/'results.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true',help='Write the fully verified new report to the formal Markdown')
    build(publish=parser.parse_args().publish)

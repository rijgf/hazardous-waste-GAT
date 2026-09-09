"""One controlled large-case comparison; never rerun or overwrite v4 NSGA-II.

Only the actor advantage scalarization changes. All training data, parameters,
seeds, architecture, initial plans, references and per-front budgets are reused.
Historical source locks remain historical: the exact old PPO source is archived,
not relabelled as the corrected source. No v4 report/audit is overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

ORIGINAL_CUDA = os.environ.get('CUDA_VISIBLE_DEVICES')
import run_pareto_experiments as engine
from src.pareto_experiment import CandidateLedger, coverage_pair
from src.reproducibility import (derive_seed, environment_snapshot, file_sha256,
    deserialize_plan, plan_sha256, plan_to_canonical_data)
from src.solution_utils import evaluate_solution, route_plan_to_solution

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'output/pareto-parameter-revision-v4'
OUT = ROOT / 'output/ppo-scalar-advantage-v5'
MODEL = ROOT / 'outputs/ppo_scalar_advantage_v5/models/large_model.pt'
REGISTRY = ROOT / 'configs/ppo_large_scalar_advantage_v5.json'
DATA = ROOT / 'datasets/parameter_revision_v4'
read, write = engine.read, engine.write
SOURCE_BEFORE = OUT / 'source-before/src/ppo_improver.py'


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def prepare():
    if (OUT / 'protocol.json').exists():
        return protocol()
    old = read(BASE / 'protocol.json')
    registry = read(ROOT / 'configs/ppo_models_parameter_revision_v4.json')
    assert registry == old['registry']
    assert file_sha256(SOURCE_BEFORE) == old['sources']['src/ppo_improver.py']
    for name, digest in old['sources'].items():
        if name != 'src/ppo_improver.py':
            assert file_sha256(ROOT / name) == digest, name
    manifest = read(DATA / 'manifest.json')
    assert file_sha256(DATA / 'manifest.json') == registry['dataset_manifest_sha256']
    for item in manifest['records']:
        assert file_sha256(DATA / item['path']) == item['sha256']
    for item in registry['models'].values():
        assert file_sha256(ROOT / item['checkpoint']) == item['sha256']
    preserved = {}
    for folder in (BASE, DATA, ROOT / 'outputs/parameter_revision_v4'):
        for path in folder.rglob('*'):
            if path.is_file():
                preserved[relative(path)] = file_sha256(path)
    preserved['configs/ppo_models_parameter_revision_v4.json'] = file_sha256(ROOT / 'configs/ppo_models_parameter_revision_v4.json')
    write(OUT / 'preserved_inputs.json', preserved)
    training_record = read(ROOT / registry['models']['large']['training_record'])
    value = {
        'protocol': 'ppo-scalar-advantage-v5-large-only',
        'change': 'scalarize vector GAE with each sample preference BEFORE common scalar whitening',
        'base_protocol_sha256': file_sha256(BASE / 'protocol.json'),
        'preserved_inputs_sha256': file_sha256(OUT / 'preserved_inputs.json'),
        'sources': {**{name: file_sha256(ROOT / name) for name in old['sources']},
                    relative(Path(__file__)): file_sha256(Path(__file__)),
                    relative(SOURCE_BEFORE): file_sha256(SOURCE_BEFORE)},
        'dataset_manifest_sha256': file_sha256(DATA / 'manifest.json'),
        'training_seed': training_record['training_seed'],
        'algorithm': training_record['algorithm'], 'network': training_record['network'],
        'training_instances': [item for item in manifest['records'] if item['split'] == 'train' and item['scale'] == 'Test-4'],
        'instance_path': relative(BASE / 'instances/Test-4.json'),
        'instance_sha256': file_sha256(BASE / 'instances/Test-4.json'),
        'tasks': [task for task in old['tasks'] if task['kind'] == 'PPO' and task['instance'] == 'Test-4' and task['model'] == 'large'],
        'preferences': old['preferences'], 'ppo_steps': 60, 'ppo_candidates_per_step': 32,
        'repeats': 3, 'nsga_budgets': [40320, 120960], 'tolerance': old['tolerance_normalized'],
        'execution': {'training_device': 'cuda', 'evaluation_device': 'cpu', 'threads_per_worker': 1,
                      'evaluation_workers': 3, 'timing_note': 'new run concurrency differs from historical 16-task batch; not an isolated speed comparison'},
    }
    assert len(value['training_instances']) == 24 and len(value['tasks']) == 3 and len(value['preferences']) == 21
    write(OUT / 'protocol.json', value)
    return value


def protocol():
    value = read(OUT / 'protocol.json')
    assert file_sha256(OUT / 'preserved_inputs.json') == value['preserved_inputs_sha256']
    for name, digest in value['sources'].items():
        assert file_sha256(ROOT / name) == digest, 'Experiment source changed: ' + name
    for name, digest in read(OUT / 'preserved_inputs.json').items():
        assert file_sha256(ROOT / name) == digest, 'Historical input changed: ' + name
    return value


def train():
    value = prepare()
    if (OUT / 'training_record.json').exists():
        record = read(OUT / 'training_record.json')
        assert file_sha256(MODEL) == record['checkpoint_sha256']
        assert record['protocol_sha256'] == file_sha256(OUT / 'protocol.json')
        print('REUSED frozen corrected large checkpoint', flush=True)
        return
    if MODEL.exists():
        raise RuntimeError('Unfinished checkpoint exists; preserve and investigate before resuming')
    if ORIGINAL_CUDA is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ORIGINAL_CUDA
    import torch
    from src.ppo_improver import PPOImprover, PPOTrainingInstance
    assert torch.cuda.is_available(), 'Same GPU training runtime required; do not silently train on CPU'
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    instances = []
    for item in value['training_instances']:
        data = read(DATA / item['path'])
        params, plan, refs = engine.checked_instance(data)
        instances.append(PPOTrainingInstance(data['instance_id'], params, plan, refs))
    first = instances[0]
    model = PPOImprover(first.params, {'ppo': deepcopy(value['algorithm'])}, deepcopy(value['network']),
                        seed=value['training_seed'], objective_refs=first.objective_refs)
    started = time.perf_counter()
    advantage_diagnostics = []
    original_scalarize = model._scalarize_advantages

    def monitored_scalarize(vector, preferences):
        result = original_scalarize(vector, preferences)
        stds = vector.std(dim=0).detach().cpu().tolist()
        legacy_vector = (vector-vector.mean(dim=0, keepdim=True))/vector.std(dim=0, keepdim=True).clamp_min(1e-6)
        legacy = (legacy_vector*preferences).sum(dim=-1)
        legacy = (legacy-legacy.mean())/legacy.std().clamp_min(1e-6)
        advantage_diagnostics.append({'cost_advantage_sd': stds[0], 'risk_advantage_sd': stds[1],
            'legacy_vs_fixed_sign_disagreement_fraction': float(((legacy*result)<0).float().mean().item())})
        return result

    def progress(iteration, metrics):
        record = {'iteration': iteration+1, 'elapsed_seconds': time.perf_counter()-started,
                  **metrics, 'advantage_diagnostics': advantage_diagnostics[-1]}
        write(OUT / 'training_progress.json', record)
        write(OUT / 'advantage_diagnostics.json', advantage_diagnostics)
        print('TRAIN', iteration+1, '/20', json.dumps(record), flush=True)

    with patch.object(model, '_scalarize_advantages', monitored_scalarize):
        history = model.train(MODEL, training_instances=instances, progress_callback=progress)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    assert model.training_instance_counts == {x.instance_id: 20 for x in instances}
    checkpoint = torch.load(MODEL, map_location='cpu', weights_only=False)
    assert all(float(s['step']) == 480 for s in checkpoint['optimizer_state_dict']['state'].values())
    checkpoint['training_semantics'] = 'preference_scalarization_before_advantage_normalization-v5'
    checkpoint['training_protocol_sha256'] = file_sha256(OUT / 'protocol.json')
    torch.save(checkpoint, MODEL)
    record = {'checkpoint': relative(MODEL), 'checkpoint_sha256': file_sha256(MODEL),
              'protocol_sha256': file_sha256(OUT / 'protocol.json'), 'training_seconds': elapsed,
              'training_seed': value['training_seed'], 'algorithm': value['algorithm'], 'network': value['network'],
              'instance_episode_counts': model.training_instance_counts, 'optimizer_updates': 480,
              'environment': environment_snapshot(), 'training_semantics': checkpoint['training_semantics']}
    write(OUT / 'training_history.json', history)
    write(OUT / 'training_record.json', record)
    entry = {'experiment_alias': 'New-L-v5-scalar-advantage', 'checkpoint': relative(MODEL),
             'sha256': record['checkpoint_sha256'], 'training_record': relative(OUT / 'training_record.json'),
             'training_scale': 'Test-4'}
    write(REGISTRY, {'status': 'frozen_after_final_training_iteration', 'scope': 'large Test-4 comparison only; v4 pair remains historical default for other experiments',
                     'objective_normalization': 'instance_reference', 'dataset_manifest': relative(DATA / 'manifest.json'),
                     'dataset_manifest_sha256': value['dataset_manifest_sha256'], 'models': {'large': entry}})
    protocol()
    print('TRAINING_COMPLETE', record['checkpoint_sha256'], 'seconds', elapsed, flush=True)


def run_repeat(repeat):
    value = protocol()
    task = next(t for t in value['tasks'] if t['repeat'] == repeat)
    target = OUT / f'fronts/{task["id"]}-B40320.json'
    if target.exists():
        assert read(target)['protocol_sha256'] == file_sha256(OUT / 'protocol.json')
        return 'REUSED ' + task['id']
    registry = read(REGISTRY)
    entry = registry['models']['large']
    assert file_sha256(MODEL) == entry['sha256']
    data = read(ROOT / value['instance_path'])
    params, plan, refs = engine.checked_instance(data)
    model = engine.load_model(params, refs, 'large', {'registry': registry})
    ledger = CandidateLedger(params, refs, plan)
    final_solutions = []
    started = time.perf_counter()
    for index, preference in enumerate(value['preferences']):
        seed = derive_seed(task['seed'], 'preference', index, bits=32)
        start = time.perf_counter()
        with ledger.observe_ppo(preference, seed):
            result = model.improve(tuple(preference), seed=seed, initial_plan=plan)
        metrics = evaluate_solution(params, result.solution, tuple(preference), objective_refs=refs)
        assert metrics['feasible']
        final_solutions.append({'preference': preference, 'seed': seed, 'seconds': time.perf_counter()-start,
            'plan': plan_to_canonical_data(result.plan), 'solution_id': plan_sha256(result.plan), 'metrics': metrics})
        write(OUT / f'progress/{task["id"]}.json', {'preferences_completed': index+1,
            'seconds': time.perf_counter()-started, 'counts': dict(ledger.counts)})
        print('EVAL', repeat, index+1, '/21', 'J', metrics['weighted_objective'], flush=True)
    assert ledger.counts['candidate_attempts'] == 40320
    snapshot = {'budget': 40320, 'counts': dict(ledger.counts), 'seconds': time.perf_counter()-started,
                'preparation_seconds': ledger.preparation_seconds, 'points': ledger.archive.sorted_points(),
                'final_solutions': final_solutions, 'stop_reason': 'all_21_weight_budgets_completed'}
    engine.OUT = OUT
    engine.save_front(task, data, snapshot, {'registry': registry})
    return 'DONE ' + task['id']


def evaluate():
    protocol()
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn')) as pool:
        for result in pool.map(run_repeat, range(3)):
            print(result, flush=True)


def report():
    value = protocol()
    data = read(ROOT / value['instance_path'])
    params, _, refs = engine.checked_instance(data)
    rows, comparisons, unique_verified = [], [], set()
    for repeat in range(3):
        new = read(OUT / f'fronts/PPO-Test-4-large-r{repeat}-B40320.json')
        old = read(BASE / f'fronts/PPO-Test-4-large-r{repeat}-B40320.json')
        assert new['checkpoint'] == read(REGISTRY)['models']['large']
        for directory, front in ((OUT, new), (BASE, old)):
            assert front['instance_sha256'] == data['instance_sha256']
            assert (front['b_C'], front['b_R']) == refs
            assert front['reference_plan_sha256'] == data['reference_plan_sha256']
        comparison = coverage_pair(new['points'], old['points'], refs, value['tolerance'])
        comparisons.append({'repeat': repeat, 'new_covers_old': comparison['A_covers_B'],
                            'old_covers_new': comparison['B_covers_A']})
        for budget in value['nsga_budgets']:
            nsga = read(BASE / f'fronts/NSGA-Test-4-r{repeat}-B{budget}.json')
            assert nsga['instance_sha256'] == data['instance_sha256'] and (nsga['b_C'], nsga['b_R']) == refs
            for directory, front in ((OUT, new), (BASE, old), (BASE, nsga)):
                for point in front['points']:
                    key = (str(directory), point['solution_id'])
                    if key in unique_verified:
                        continue
                    saved = read(directory / point['solution_path'])
                    plan = deserialize_plan(saved['plan'])
                    assert plan_sha256(plan) == point['solution_id']
                    actual = evaluate_solution(params, route_plan_to_solution(params, plan), objective_refs=refs)
                    assert actual['feasible']
                    for field in ('cost', 'risk', 'transport_risk', 'coload_risk', 'producer_inventory_risk', 'facility_inventory_risk'):
                        assert math.isclose(actual[field], point[field], rel_tol=1e-10, abs_tol=1e-8), field
                    unique_verified.add(key)
            pair = coverage_pair(nsga['points'], new['points'], refs, value['tolerance'])
            # Direct count, independently of the production archive filter.
            n_to_p = sum(any(all((a[k]-b[k])/r <= value['tolerance'] for k,r in zip(('cost','risk'),refs)) for a in nsga['points']) for b in new['points'])/len(new['points'])
            p_to_n = sum(any(all((a[k]-b[k])/r <= value['tolerance'] for k,r in zip(('cost','risk'),refs)) for a in new['points']) for b in nsga['points'])/len(nsga['points'])
            assert pair['A_covers_B'] == n_to_p and pair['B_covers_A'] == p_to_n
            rows.append({'repeat': repeat, 'nsga_budget': budget, 'N_covers_P': n_to_p, 'P_covers_N': p_to_n,
                         'n_P': len(new['points']), 'n_N': len(nsga['points']), 'ppo_seconds': new['seconds'],
                         'nsga_seconds_historical': nsga['seconds'], 'N_archive_sha256': file_sha256(BASE / f'fronts/NSGA-Test-4-r{repeat}-B{budget}.json')})
    summary = []
    for budget in value['nsga_budgets']:
        group = [r for r in rows if r['nsga_budget'] == budget]
        summary.append({'budget': budget, **{k: {'mean': statistics.mean(r[k] for r in group), 'sd': statistics.stdev(r[k] for r in group)} for k in ('N_covers_P','P_covers_N')},
                        'ppo_mean_seconds': statistics.mean(r['ppo_seconds'] for r in group)})
    result = {'protocol_sha256': file_sha256(OUT/'protocol.json'), 'checkpoint_sha256': file_sha256(MODEL),
              'paired_results': rows, 'summary': summary, 'new_vs_old_ppo': comparisons,
              'unique_solutions_replayed': len(unique_verified), 'historical_inputs_unchanged': True,
              'still_totally_dominated': all(r['N_covers_P']==1 and r['P_covers_N']==0 for r in rows)}
    write(OUT / 'results.json', result)
    with (OUT/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    record = read(OUT / 'training_record.json')
    table = '\n'.join(f"| PPO-L-v5 | {r['budget']} | {r['N_covers_P']['mean']:.4f} ± {r['N_covers_P']['sd']:.4f} | {r['P_covers_N']['mean']:.4f} ± {r['P_covers_N']['sd']:.4f} | {r['ppo_mean_seconds']:.2f} |" for r in summary)
    conclusion = ('修复后仍在全部三次重复、两档NSGA-II预算下被完全覆盖。该修复不足以解释或消除原性能差距。' if result['still_totally_dominated'] else '覆盖率按下表实际结果报告；不能把局部改善直接表述为全面优于NSGA-II。')
    text = f'''# PPO优势加权顺序修复：单个大规模前沿对比（v5）

本轮仅将训练更新改为“按偏好合成优势，再整体标准化”。保留旧PT、数据及NSGA-II前沿，不修复其他算法问题、不换测试实例、不追加搜索预算。

大规模训练使用原24个Test-4训练实例；20轮×24回合×24步，共11520交互步、480次优化器更新。网络、种子与全部超参数不变；训练耗时{record['training_seconds']:.3f}秒。新PT为`{relative(MODEL)}`，SHA256：`{result['checkpoint_sha256']}`。

测试仅使用原Test-4/000实例和其固定b。3次完整前沿，各21组偏好×60步×32候选，偏好种子与v4配对。原NSGA-II的B1、B2档案直接复用，未重新求解。公共参考方案、评价器和算子保持不变。

| 方法 | NSGA-II候选预算 | N→P（越小越好） | P→N（越大越好） | PPO完整前沿平均时间/s |
|---|---:|---:|---:|---:|
{table}

覆盖率为同一实例3次配对的均值±样本标准差。两档预算共用相同的3份PPO档案。时间为每次完整前沿耗时的平均值，不是三次累计；本轮3进程与历史16任务批次的并发条件不同，不作严格速度归因。

{conclusion}

与旧PPO的逐次双向覆盖、新旧原始解验证和逐次对比见[results.json](results.json)及[comparison.csv](comparison.csv)。全部旧输入哈希见[preserved_inputs.json](preserved_inputs.json)；训练中记录的优势分量标准差及旧/新处理的符号差异见[advantage_diagnostics.json](advantage_diagnostics.json)。这些差异来自新训练轨迹，不能冒充历史v4训练时的统计。

计算源码发生了经授权的版本升级。v4原始PPO源码按原字节保存在[source-before/src/ppo_improver.py](source-before/src/ppo_improver.py)；v4协议和历史审核没有重写。此文件为独立补充对照，不静默替换原论文的其他实验结果。
'''
    (OUT/'results.md').write_text(text, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','train','evaluate','report','all'])
    stage = parser.parse_args().stage
    if stage == 'prepare': prepare()
    elif stage == 'train': train()
    elif stage == 'evaluate': evaluate()
    elif stage == 'report': report()
    else:
        train(); evaluate(); report()


if __name__ == '__main__':
    main()

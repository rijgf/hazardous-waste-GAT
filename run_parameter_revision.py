"""Re-run the unchanged v3 algorithms on versioned, revised instances.

All solver functions are reused. Only instance preparation and output routing
are new. The frozen training data, models and old experimental artifacts remain
immutable. Run prepare, preflight, run, milp, normalize, report, audit in order.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import time
import os

ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
import run_pareto_experiments as engine
from sample_params import params_to_json_data
from src.parameter_revision import construct_reference, revised_parameters
from src.reproducibility import derive_seed, environment_snapshot, file_sha256, json_sha256
from src.solution_utils import evaluate_solution, route_plan_to_solution

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'output/pareto-single-instance-v3'
OUT = ROOT / 'output/pareto-parameter-revision-v4'
OLD_DATA = ROOT / 'datasets/reference_generalization_v2'
DATA = ROOT / 'datasets/parameter_revision_v4'
MODEL_RUN = ROOT / 'outputs/parameter_revision_v4'
REGISTRY = ROOT / 'configs/ppo_models_parameter_revision_v4.json'
engine.OUT = OUT
engine.DATA = DATA
engine.REGISTRY = REGISTRY
for source in ['src/parameter_revision.py', 'run_parameter_revision.py']:
    if source not in engine.SOURCES:
        engine.SOURCES.append(source)


def original_models():
    registry = engine.read(ROOT / 'configs/frozen_ppo_models.json')
    for entry in registry['models'].values():
        if file_sha256(ROOT / entry['checkpoint']) != entry['sha256']:
            raise ValueError('Original frozen checkpoint identity changed')
    if file_sha256(OLD_DATA / 'manifest.json') != registry['dataset_manifest_sha256']:
        raise ValueError('Original training dataset manifest changed')
    return registry


def revise_record(original, source):
    old_params, _, _ = engine.checked_instance(original)
    parameter_seed = derive_seed(original['instance_seed'], 'parameter-revision-v4-coload', bits=32)
    params, calibration = revised_parameters(old_params, seed=parameter_seed)
    before, after = params_to_json_data(old_params), params_to_json_data(params)
    differences = [key for key in before if before[key] != after[key]]
    if set(differences) != {'processing_capacity', 'coload_risk'}:
        raise ValueError('Unexpected revision fields: ' + str(differences))
    reference, construction = construct_reference(params)
    return engine.bind(original['instance_id'] + '--revision-v4', params, reference,
        base_instance_id=original['instance_id'], instance_seed=original['instance_seed'],
        reference_seed=original['reference_seed'], scenario_parameter='baseline', scenario_multiplier=1.,
        source_instance_path=str(source.relative_to(ROOT)).replace('\\', '/'),
        source_instance_sha256=file_sha256(source), original_params_sha256=original['instance_sha256'],
        changed_fields=differences, parameter_revision=calibration, reference_construction=construction)


def prepare_data():
    original_models()
    path = DATA / 'manifest.json'
    if path.exists():
        manifest = engine.read(path)
        if manifest['source_hashes'] != engine.source_hashes():
            raise ValueError('Preparation sources changed after dataset lock')
        for item in manifest['records']:
            if file_sha256(DATA / item['path']) != item['sha256']:
                raise ValueError('Revised dataset changed')
        return manifest
    old = engine.read(OLD_DATA / 'manifest.json')
    records = []
    selected = [r for r in old['records'] if r['split'] == 'train' or r['index'] == 0]
    for index, entry in enumerate(selected):
        source = OLD_DATA / entry['path']
        if file_sha256(source) != entry['sha256']:
            raise ValueError('Original dataset file changed')
        data = revise_record(engine.read(source), source)
        data.update(split=entry['split'], scale=entry['scale'], index=entry['index'])
        target = DATA / entry['path']
        if target.exists():
            prior = engine.read(target)
            if prior['instance_sha256'] != data['instance_sha256'] or prior['reference_plan_sha256'] != data['reference_plan_sha256']:
                raise ValueError('Non-deterministic preparation resume')
        else:
            engine.write(target, data)
        records.append({'path': entry['path'], 'sha256': file_sha256(target),
                        **{k: data[k] for k in ['instance_id', 'instance_sha256', 'split', 'scale', 'index', 'b_C', 'b_R']}})
        print(f'DATA {index+1}/{len(selected)} {entry["path"]}', flush=True)
    train = {r['instance_sha256'] for r in records if r['split'] == 'train'}
    test = {r['instance_sha256'] for r in records if r['split'] == 'test'}
    if train & test or len({r['instance_sha256'] for r in records}) != len(records):
        raise ValueError('Train/test overlap or duplicate instance')
    manifest = {'protocol': 'parameter-revision-v4', 'records': records,
                'training_instances_per_scale': 24, 'test_instances_per_scale': 1,
                'training_scales': old['training_scales'],
                'scales': {k: {**v, 'instances': 1} for k, v in old['scales'].items()},
                'algorithm': old['algorithm'], 'network': old['network'],
                'base_generator_profile': old['generator_profile'],
                'parameter_revision': {'capacity_factor': 2., 'risk_total_ratio_range': [3., 6.],
                                       'risk_strength_range': [2., 5.], 'changed_fields': ['processing_capacity', 'coload_risk']},
                'source_manifest_sha256': file_sha256(OLD_DATA / 'manifest.json'),
                'source_hashes': engine.source_hashes(), 'environment': environment_snapshot(),
                'train_test_hash_overlap': 0}
    engine.write(path, manifest)
    return manifest


def train_models():
    # The reused inference runner masks CUDA. Training must use the same GPU
    # availability policy as v2, before torch's CUDA runtime is initialized.
    if ORIGINAL_CUDA_VISIBLE_DEVICES is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ORIGINAL_CUDA_VISIBLE_DEVICES
    import torch
    from src.ppo_improver import PPOImprover, PPOTrainingInstance
    manifest = prepare_data()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    registry = {'registry_version': 1, 'status': 'retrained_for_user_parameter_revision_frozen_after_training',
                'objective_normalization': 'instance_reference', 'path_base': 'repository_root',
                'dataset_manifest': str((DATA / 'manifest.json').relative_to(ROOT)).replace('\\', '/'),
                'dataset_manifest_sha256': file_sha256(DATA / 'manifest.json'), 'models': {}}
    old_registry = original_models()
    for name, scale in [('small', 'Test-1'), ('large', 'Test-4')]:
        target = MODEL_RUN / f'models/{name}_model.pt'
        record_path = MODEL_RUN / f'models/{name}.json'
        old_record = engine.read(ROOT / old_registry['models'][name]['training_record'])
        training_seed = old_record['training_seed']
        if record_path.exists():
            record = engine.read(record_path)
            if file_sha256(target) != record['checkpoint_sha256'] or record['dataset_manifest_sha256'] != registry['dataset_manifest_sha256']:
                raise ValueError('Revised trained model identity changed')
        else:
            if target.exists():
                raise ValueError('Unfinished checkpoint without final record; preserve and investigate')
            instances = []
            for entry in manifest['records']:
                if entry['split'] == 'train' and entry['scale'] == scale:
                    data = engine.read(DATA / entry['path'])
                    params, plan, refs = engine.checked_instance(data)
                    instances.append(PPOTrainingInstance(data['instance_id'], params, plan, refs))
            first = instances[0]
            model = PPOImprover(first.params, {'ppo': manifest['algorithm']['ppo']}, manifest['network'],
                                seed=training_seed, objective_refs=first.objective_refs)
            started = time.perf_counter()

            def progress(iteration, metrics):
                elapsed = time.perf_counter() - started
                engine.write(MODEL_RUN / f'training/{name}_progress.json',
                             {'iteration': iteration + 1, 'elapsed_seconds': elapsed, **metrics})
                print(f'TRAIN {name} {iteration+1}/20 elapsed={elapsed:.1f}s', flush=True)

            history = model.train(target, training_instances=instances, progress_callback=progress)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if set(model.training_instance_counts.values()) != {20} or len(model.training_instance_counts) != 24:
                raise ValueError('Training coverage differs from unchanged 24-instance protocol')
            record = {'model': name, 'training_seed': training_seed, 'training_seconds': elapsed,
                      'checkpoint_sha256': file_sha256(target), 'dataset_manifest_sha256': registry['dataset_manifest_sha256'],
                      'instance_episode_counts': model.training_instance_counts,
                      'algorithm': manifest['algorithm']['ppo'], 'network': manifest['network'],
                      'parameter_revision': manifest['parameter_revision'], 'environment': environment_snapshot()}
            engine.write(MODEL_RUN / f'training/{name}_history.json', history)
            engine.write(record_path, record)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        registry['models'][name] = {'experiment_alias': 'New-S-v4' if name == 'small' else 'New-L-v4',
                                    'training_scale': scale, 'sha256': record['checkpoint_sha256'],
                                    'checkpoint': str(target.relative_to(ROOT)).replace('\\', '/'),
                                    'training_record': str(record_path.relative_to(ROOT)).replace('\\', '/')}
    if REGISTRY.exists() and engine.read(REGISTRY) != registry:
        raise ValueError('Revised model registry changed')
    engine.write(REGISTRY, registry)
    engine.verify_models()


def prepare(keep_small=False):
    if keep_small:
        raise ValueError('User explicitly requires the small case to be revised too')
    if (OUT / 'protocol.json').exists():
        current = engine.load_protocol()
        if current['parameter_revision']['small_case_revised'] == keep_small:
            raise ValueError('Small-case choice differs from frozen revision')
        return current
    registry = engine.verify_models()
    old_protocol = engine.read(OLD / 'protocol.json')
    for source, expected in old_protocol['sources'].items():
        if file_sha256(ROOT / source) != expected:
            raise ValueError('Core algorithm changed since v3: ' + source)
    snapshots = {str(p.relative_to(ROOT)).replace('\\', '/'): file_sha256(p)
                 for folder in [OLD, OLD_DATA]
                 for p in sorted(folder.rglob('*')) if p.is_file()}
    for entry in registry['models'].values():
        snapshots[entry['checkpoint']] = file_sha256(ROOT / entry['checkpoint'])
    engine.write(OUT / 'prior_artifact_hashes.json', snapshots)
    manuscript = ROOT / '供应链管理写作/数值实验与结果分析_论文稿.md'
    archive = OUT / 'history/数值实验与结果分析_论文稿_v3.md'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(manuscript, archive)
    instances = {}
    for name in ['Test-1', 'Test-2', 'Test-3', 'Test-4', 'small-fixed']:
        source = OLD / f'instances/{name}.json'
        original = engine.read(source)
        if file_sha256(source) != old_protocol['instances'][name]:
            raise ValueError('Original instance identity differs: ' + name)
        instances[name] = (engine.read(DATA / f'test/{name}/000.json') if name.startswith('Test-')
                           else revise_record(original, source))
    base = instances['Test-4']
    base_params, base_plan, _ = engine.checked_instance(base)
    for field, name in [('processing_capacity', 'capacity'), ('coload_risk', 'coload')]:
        for multiplier in [.7, .8, 1.2, 1.3]:
            params = replace(base_params, **{field: {k: v * multiplier for k, v in getattr(base_params, field).items()}})
            scene = f'{name}-{round(multiplier * 100)}'
            reference = base_plan
            construction = {'method': 'unchanged baseline reference', 'seconds': 0.}
            if not evaluate_solution(params, route_plan_to_solution(params, reference))['feasible']:
                reference, construction = construct_reference(params)
            instances[scene] = engine.bind(base['instance_id'] + '--' + scene, params, reference,
                base_instance_id=base['instance_id'], scenario_parameter=field, scenario_multiplier=multiplier,
                changed_fields=[field], instance_seed=base['instance_seed'], reference_seed=base['reference_seed'],
                reference_construction=construction)
    for name, data in instances.items():
        engine.checked_instance(data)
        engine.write(OUT / f'instances/{name}.json', data)
    protocol = deepcopy(old_protocol)
    protocol.update(protocol='pareto-parameter-revision-v4', sources=engine.source_hashes(),
        registry=registry, registry_sha256=file_sha256(REGISTRY), dataset_manifest_sha256=file_sha256(DATA / 'manifest.json'),
        instances={name: file_sha256(OUT / f'instances/{name}.json') for name in instances},
        parameter_revision={
            'capacity_factor': 2., 'risk_total_ratio_range': [3., 6.], 'risk_strength_range': [2., 5.],
            'models_retrained': True, 'training_hyperparameters_changed': False,
            'small_case_revised': not keep_small,
            'source_protocol_sha256': file_sha256(OLD / 'protocol.json'),
            'old_artifact_snapshot_sha256': file_sha256(OUT / 'prior_artifact_hashes.json'),
            'policy': 'new instance versions and retrained then frozen models; original networks, non-revised parameters, algorithms, budgets and algorithm seeds unchanged',
            'reference_policy': 'rebind each new instance once after deterministic strict-feasible reference preparation; all algorithms share that reference',
            'chemical_evidence': 'https://www.epa.gov/hwpermitting/method-determining-compatibility-hazardous-wastes',
            'evidence_limit': 'Supports combination-specific hazards, not a universal numeric multiplier; Uniform(3,6) equal-load total transport ratio is a synthetic experiment assumption'})
    engine.write(OUT / 'protocol.json', protocol)
    return engine.load_protocol()


def configure_milp():
    import run_pareto_milp as exact
    protocol = engine.load_protocol()
    data = engine.read(OUT / 'instances/small-fixed.json')
    params, reference, refs = engine.checked_instance(data)
    exact.DEFAULT_OUTPUT = OUT / 'milp'
    exact.INSTANCE_HASH = data['instance_sha256']
    exact.REFERENCE_HASH = data['reference_plan_sha256']
    exact.REFS = refs
    exact.SOURCE_FILES = tuple(dict.fromkeys([*exact.SOURCE_FILES, *engine.SOURCES]))
    exact.historical_case = lambda: (params, reference)
    original_protocol = exact.protocol

    def revised_protocol(time_limit):
        record = original_protocol(time_limit)
        record.update(schema='pareto-parameter-revision-milp-v4',
                      instance_id=data['instance_id'], main_protocol_sha256=file_sha256(OUT / 'protocol.json'),
                      parameter_revision=protocol['parameter_revision'])
        return record

    exact.protocol = revised_protocol
    return exact


def run_milp(index=None):
    exact = configure_milp()
    if index is not None:
        exact.run_one(index, OUT / 'milp', 3600.)
        return
    exact.bind_protocol(OUT / 'milp', 3600.)
    processes = []
    for i in range(5):
        if (OUT / f'milp/p{i}.json').exists():
            exact.run_one(i, OUT / 'milp', 3600.)
            continue
        with (OUT / f'milp/p{i}.log').open('a', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                      'milp', '--index', str(i)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        processes.append((i, child))
        print(f'MILP {i}: PID {child.pid}', flush=True)
    results = [(i, child.wait()) for i, child in processes]
    if any(code for _, code in results):
        raise RuntimeError('MILP jobs failed: ' + str(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare-data', 'train', 'prepare', 'preflight', 'run', 'small', 'milp', 'normalize', 'report', 'audit'])
    parser.add_argument('--keep-small', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--index', type=int, choices=range(5))
    args = parser.parse_args()
    if args.stage == 'prepare-data':
        prepare_data()
    elif args.stage == 'train':
        train_models()
    elif args.stage == 'prepare':
        prepare(args.keep_small)
    elif args.stage == 'preflight':
        engine.preflight()
    elif args.stage in ['run', 'small']:
        sys.argv = [sys.argv[0], args.stage, '--workers', str(args.workers)]
        engine.main()
    elif args.stage == 'milp':
        run_milp(args.index)
    elif args.stage == 'normalize':
        exact = configure_milp()
        import normalize_pareto_milp as cleanup
        cleanup.DIRECTORY = OUT / 'milp'
        cleanup.historical_case = exact.historical_case
        cleanup.main()
    elif args.stage == 'report':
        from report_parameter_revision import main as report
        report()
    else:
        from audit_parameter_revision import main as audit
        audit()


if __name__ == '__main__':
    main()

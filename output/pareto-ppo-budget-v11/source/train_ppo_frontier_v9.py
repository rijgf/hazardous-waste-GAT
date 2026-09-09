"""Train two v8-architecture PPO models on a predeclared 21-preference schedule.

No test instances, genetic solutions, or test-selected checkpoints are read.
The number of actions/updates and mixed episode horizons match v8 exactly.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan
from src.ppo_objects_v7 import SearchEnvironment, OPS, distributions
from src.ppo_objects_v8 import ObjectState
from src.ppo_objects_v8c import ObjectPolicy, sample_batch

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/ppo_frontier_v9'
PREFERENCES = [(i / 20., (20-i) / 20.) for i in range(21)]
STEPS_PER_INSTANCE = 48 * 32
SOURCES = ('train_ppo_frontier_v9.py', 'src/ppo_objects_v7.py',
           'src/ppo_objects_v7b.py', 'src/ppo_objects_v8.py', 'src/ppo_objects_v8c.py',
           'src/solution_utils.py', 'src/reproducibility.py',
           'hazardous_waste_model.py', 'sample_params.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def preference_schedule():
    """Balanced action allocation, determined before reading any training outcome.

    Each instance gets 1536 actions. Its 512/1024/1536 horizon is unchanged;
    the final 512-action fragment of a 1024-horizon episode remains truncated.
    Greedy longest-fragment-first allocation balances the 21 global totals.
    """
    episodes = []
    for instance in range(24):
        horizon = 512 * (1 + instance % 3)
        for start in range(0, STEPS_PER_INSTANCE, horizon):
            episodes.append({'instance': instance, 'start_action': start,
                             'actions': min(horizon, STEPS_PER_INSTANCE-start),
                             'horizon': horizon})
    totals = [0] * 21
    for episode in sorted(episodes, key=lambda e: (-e['actions'], e['instance'], e['start_action'])):
        index = min(range(21), key=lambda i: (totals[i], i))
        episode['preference_index'] = index
        episode['preference'] = list(PREFERENCES[index])
        totals[index] += episode['actions']
    return sorted(episodes, key=lambda e: (e['instance'], e['start_action'])), totals


def protected_inputs(scale):
    v4 = read(ROOT / 'configs/ppo_models_parameter_revision_v4.json')
    v8 = read(ROOT / 'configs/ppo_objects_v8_exploratory.json')
    checked = {v4['dataset_manifest']: v4['dataset_manifest_sha256'],
               v8['checkpoint']: v8['sha256']}
    checked.update({v['checkpoint']: v['sha256'] for v in v4['models'].values()})
    for path, expected in checked.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Protected input hash mismatch: ' + path)
    manifest = read(ROOT / v4['dataset_manifest'])
    records = sorted((r for r in manifest['records'] if r['split']=='train' and r['scale']==scale),
                     key=lambda r: r['index'])
    if len(records) != 24 or [r['index'] for r in records] != list(range(24)):
        raise ValueError('Expected the original 24 ordered TRAIN instances')
    paths = [ROOT/'datasets/parameter_revision_v4'/r['path'] for r in records]
    actual_paths = sorted((ROOT/'datasets/parameter_revision_v4/train'/scale).glob('*.json'))
    if paths != actual_paths:
        raise ValueError('Training folder and manifest differ')
    for path, record in zip(paths, records):
        if sha(path) != record['sha256']:
            raise ValueError('Training data changed: ' + str(path))
        data = read(path)
        if data['b_C'] != record['b_C'] or data['b_R'] != record['b_R']:
            raise ValueError('Bound reference differs from manifest')
        checked[path.relative_to(ROOT).as_posix()] = record['sha256']
    return paths, checked


def tensors(states, device):
    return [torch.as_tensor(np.stack([s[i] for s in states]), device=device) for i in range(3)]


def encode_policy(policy, encoded, edges):
    return policy.encode(encoded[0], encoded[1], edges, encoded[2])


def reset_preference(env, preference):
    """Called only at episode boundaries, before recalculating reference metrics."""
    env.preference = tuple(preference)
    env.reset()


def train(scale_name, device_name='cuda', seed=20260909):
    scale = {'small':'Test-1', 'large':'Test-4'}[scale_name]
    paths, protected = protected_inputs(scale)
    dest = OUTPUT/'training'/scale_name
    checkpoint = OUTPUT/'models'/(scale_name+'_model.pt')
    if dest.exists() or checkpoint.exists():
        raise FileExistsError('Refusing to overwrite previous training or checkpoint')
    episodes, expected_totals = preference_schedule()
    schedule = {(e['instance'], e['start_action']):e for e in episodes}
    config = {
        'version':'ppo-frontier-v9-multipreference', 'scale':scale,
        'args':{'variant':'b','masks':'processing','dimension':96,'layers':1,'seed':seed},
        'sources':{p:sha(ROOT/p) for p in SOURCES}, 'protected_inputs':protected,
        'training_inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        'preference_grid':PREFERENCES, 'episode_schedule':episodes,
        'expected_preference_action_counts':expected_totals,
        'preference_allocation':'predeclared longest-fragment-first, least assigned actions, grid index tie break',
        'preference_coverage':'all 21 globally; individual instances see only 1 to 3 preferences',
        'sampling':'PPO operator plus three separately parameterized conditional 512 object heads',
        'state_class':'src.ppo_objects_v8.ObjectState', 'value_head':'Linear(96,1)',
        'shared_context':'Linear(101,96) -> GELU',
        'objective_normalization':'instance_reference',
        'repair':False, 'heuristic_insertion':False, 'NSGA_warm_start':False,
        'training':{'seed':seed,'instances':24,'rollout_steps':32,'updates':48,
                    'episode_horizons':[512*(1+i%3) for i in range(24)],
                    'epochs':3,'minibatch':96,'lr':.0003,'gamma':.95,'gae_lambda':.90,
                    'clip':.2,'entropy_coefficient':.02,'reward_scale':100.,
                    'invalid_reward':-.2,'no_change_reward':-.01,'gradient_clip':.5,
                    'advantage':'scalar then full-rollout standardization',
                    'initialization':'from scratch','checkpoint_selection':'final fixed update 48, no test selection',
                    'optimizer':'Adam','device':device_name,'torch_version':torch.__version__,
                    'actions':36864,'optimizer_steps':1152,
                    'training_sampling':'ordinary conditional samples at temperature 1; state memory is inference only',
                    'historical_seed_note':'same seed for both sizes; v8 seed, not the original formal v4 seeds'}}
    write(dest/'config.json', config)
    for file in SOURCES:
        target = dest/'source'/file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/file, target)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    device = torch.device(device_name)
    envs = []
    for i, path in enumerate(paths):
        data = read(path)
        state = ObjectState(params_from_json_data(data['params']), (data['b_C'],data['b_R']))
        envs.append(SearchEnvironment(state, deserialize_plan(data['reference_plan']),
                                     schedule[i,0]['preference'], 512*(1+i%3)))
    model = ObjectPolicy(96,1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=.0003)
    edges = torch.tensor(np.stack([e.state.edges for e in envs]), device=device)
    start = time.perf_counter()
    history, counts = [], Counter()
    coverage = np.zeros((24,21), dtype=np.int64)
    optimizer_steps = 0
    for update in range(48):
        buffer, rewards, values, dones = [], [], [], []
        model.eval()
        for step in range(32):
            action_index = update*32 + step
            encoded = [env.encode() for env in envs]
            with torch.no_grad():
                context = encode_policy(model, tensors(encoded,device), edges)
                v = model.value(context[1]).squeeze(-1).cpu().numpy()
                sampled = sample_batch(model,context,[e.state for e in envs],[e.plan for e in envs],rng)
            row_rewards, row_done = [], []
            for i,env in enumerate(envs):
                episode_start = (action_index // env.horizon) * env.horizon
                pref_index = schedule[i,episode_start]['preference_index']
                if env.preference != PREFERENCES[pref_index]:
                    raise AssertionError('Unexpected within-episode preference change')
                coverage[i,pref_index] += 1
                action,masks,logprob = sampled[i]
                reward,status = env.step(action)
                done = env.age >= env.horizon
                counts[status] += 1
                counts[OPS[action.operator]+'_attempts'] += 1
                if status=='accepted':counts[OPS[action.operator]+'_accepted'] += 1
                buffer.append((encoded[i],i,list(asdict(action).values()),masks,logprob))
                row_rewards.append(reward)
                row_done.append(done)
                if done:
                    if action_index+1 < STEPS_PER_INSTANCE:
                        reset_preference(env, schedule[i,action_index+1]['preference'])
                    else:
                        env.reset()  # same last preference; no zero-action schedule entry
            rewards.append(row_rewards)
            values.append(v)
            dones.append(row_done)
        with torch.no_grad():
            context = encode_policy(model,tensors([e.encode() for e in envs],device),edges)
            last = model.value(context[1]).squeeze(-1)
        rewards,values,dones = np.array(rewards),np.array(values),np.array(dones)
        advantage = np.zeros_like(rewards)
        gae, next_value = np.zeros(24), last.cpu().numpy()
        for t in reversed(range(32)):
            continuation = 1.-dones[t]
            delta = rewards[t]+.95*next_value*continuation-values[t]
            gae = delta+.95*.90*continuation*gae
            advantage[t],next_value = gae,values[t]
        returns = torch.tensor((advantage+values).flatten(),device=device,dtype=torch.float32)
        advantage = (advantage.flatten()-advantage.mean())/(advantage.std()+1e-8)
        adv = torch.tensor(advantage,device=device,dtype=torch.float32)
        features = tensors([item[0] for item in buffer],device)
        env_indices = torch.tensor([item[1] for item in buffer],device=device)
        actions = torch.tensor([item[2] for item in buffer],device=device)
        masks = [torch.tensor(np.stack([item[3][i] for item in buffer]),device=device) for i in range(4)]
        oldlog = torch.tensor([item[4] for item in buffer],device=device,dtype=torch.float32)
        model.train()
        losses,kls,gradients = [],[],[]
        for epoch in range(3):
            order = rng.permutation(len(buffer))
            for offset in range(0,len(buffer),96):
                idx = torch.tensor(order[offset:offset+96],device=device)
                logits,value = model(features[0][idx],features[1][idx],edges[env_indices[idx]],features[2][idx],actions[idx])
                dist = distributions(logits,[m[idx] for m in masks])
                logprob = sum(d.log_prob(actions[idx,i]) for i,d in enumerate(dist))
                entropy = sum(d.entropy() for d in dist).mean()
                ratio = (logprob-oldlog[idx]).exp()
                actor = -torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()
                loss = actor+.5*(value-returns[idx]).square().mean()-.02*entropy
                optim.zero_grad(set_to_none=True)
                loss.backward()
                gradients.append([float(h.weight.grad.norm()) for h in model.heads])
                torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
                optim.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
                kls.append(float((oldlog[idx]-logprob).mean().detach()))
        record = {'update':update+1,'attempts':(update+1)*32*24,'optimizer_steps':optimizer_steps,
                  'seconds':time.perf_counter()-start,
                  'mean_current_J':float(np.mean([e.metrics['weighted_objective'] for e in envs])),
                  'reward_mean':float(rewards.mean()),'loss':float(np.mean(losses)),
                  'approx_kl':float(np.mean(kls)),'head_gradient_norms':np.mean(gradients,axis=0).tolist(),
                  'counts':dict(counts),'preference_action_counts':coverage.sum(axis=0).tolist()}
        history.append(record)
        write(dest/'history.json',history)
        print(json.dumps(record),flush=True)
    expected = np.zeros((24,21),dtype=np.int64)
    for e in episodes:expected[e['instance'],e['preference_index']] += e['actions']
    if not np.array_equal(coverage,expected) or optimizer_steps != 1152:
        raise AssertionError('Training budget or preference schedule mismatch')
    coverage_record = {'grid':PREFERENCES,'instance_paths':[p.relative_to(ROOT).as_posix() for p in paths],
                       'instance_preference_action_counts':coverage.tolist(),
                       'preference_action_counts':coverage.sum(axis=0).tolist(),
                       'optimizer_sample_presentations_per_preference':(coverage.sum(axis=0)*3).tolist(),
                       'total_actions':int(coverage.sum()),'optimizer_steps':optimizer_steps}
    write(dest/'preference_coverage.json',coverage_record)
    for path,h in {**config['sources'],**protected}.items():
        if sha(ROOT/path) != h:raise AssertionError('Source/input changed while training: '+path)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    model.eval()
    torch.save({'format':'conditional_objects_v8c','training_version':'ppo-frontier-v9-multipreference',
                'model_state_dict':model.cpu().state_dict(),'dimension':96,'layers':1,
                'config':config,'history':history,'preference_coverage':coverage_record},checkpoint)
    write(dest/'completed.json',{'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
          'checkpoint_sha256':sha(checkpoint),'seconds':time.perf_counter()-start,
          'training_only':True,'training_actions':36864,'optimizer_steps':optimizer_steps,
          'source_config_sha256':sha(dest/'config.json'),
          'coverage_sha256':sha(dest/'preference_coverage.json'),'history_sha256':sha(dest/'history.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=['small','large','both'],required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=20260909)
    args = parser.parse_args()
    for scale in (['small','large'] if args.scale=='both' else [args.scale]):
        train(scale,args.device,args.seed)


if __name__=='__main__':main()

"""Reproducible full-object PPO training and single-instance scalar exploration.

Training reads only the 24 original large TRAIN instances. Evaluation reads one
fixed Test-4 and the scalar NSGA benchmark; genetic plans never enter the policy.
Every run is independently versioned and refuses to overwrite previous output.
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
from src.reproducibility import deserialize_plan,plan_to_canonical_data,plan_sha256
from src.ppo_objects_v7 import ObjectState,ObjectPolicy,SearchEnvironment,OPS,sample_action,distributions
from src.ppo_unique_sampling import sample_unique

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/ppo-objects-v8'
SEEDS=(3381415411,3078273277,958147205)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def protect():
    reg=read(ROOT/'configs/ppo_models_parameter_revision_v4.json')
    assert sha(ROOT/reg['dataset_manifest'])==reg['dataset_manifest_sha256']
    for item in reg['models'].values():assert sha(ROOT/item['checkpoint'])==item['sha256']


def initialize(args):
    protect();dest=OUT/args.mode/args.id
    if dest.exists():raise FileExistsError('Refusing to overwrite an earlier experiment')
    config={'args':vars(args),'exploratory':True,'operator_vocabulary':OPS,
            'object_heads':[512,512,512],'object_selection':'network logits + hard semantic/edge masks',
            'objective':'preference[0] * C / instance.b_C + preference[1] * R / instance.b_R',
            'repair':False,'heuristic_insertion':False,'NSGA_warm_start':False,
            'sources':{p:sha(ROOT/p) for p in ('run_ppo_objects_v8.py','src/ppo_objects_v7.py',
                        'src/ppo_objects_v7b.py','src/ppo_objects_v8.py','src/ppo_unique_sampling.py','src/solution_utils.py','hazardous_waste_model.py','sample_params.py')}}
    write(dest/'config.json',config)
    for file in config['sources']:
        target=dest/'source'/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,target)
    return dest,config


def load_case(path):
    data=read(path);params=params_from_json_data(data['params']);refs=(data['b_C'],data['b_R'])
    return ObjectState(params,refs),deserialize_plan(data['reference_plan']),data


def tensors(states,device):
    return [torch.as_tensor(np.stack([s[i] for s in states]),device=device) for i in range(3)]


def forward(policy,encoded,edges):
    return policy(encoded[0],encoded[1],edges,encoded[2])


def train(args):
    dest,config=initialize(args);rng=np.random.default_rng(args.seed)
    torch.manual_seed(args.seed);np.random.seed(args.seed % (2**32));torch.set_num_threads(2)
    device=torch.device(args.device)
    paths=sorted((ROOT/'datasets/parameter_revision_v4/train/Test-4').glob('*.json'))
    assert len(paths)==24
    envs=[]
    for index,path in enumerate(paths):
        state,plan,data=load_case(path)
        horizon=args.horizon*(1+index%3) if args.mixed_horizons else args.horizon
        envs.append(SearchEnvironment(state,plan,(.5,.5),horizon))
    config['training_inputs']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    config['training']={'seed':args.seed,'instances':24,'rollout_steps':args.rollout,'updates':args.updates,
        'episode_horizons':[e.horizon for e in envs],'epochs':args.epochs,'minibatch':args.minibatch,'lr':args.lr,
        'gamma':.95,'gae_lambda':.90,'clip':.2,'entropy_coefficient':args.entropy,'reward_scale':100.,
        'invalid_reward':-.2,'no_change_reward':-.01,'advantage':'scalar, then one batch normalization',
        'initialization':'from scratch','checkpoint_selection':'final fixed update, no test selection',
        'preference_training':'fixed 0.5/0.5 for this single-point exploratory specialist',
        'optimizer':'Adam','device':str(device),'torch_version':torch.__version__}
    write(dest/'config.json',config)
    model=ObjectPolicy(args.dimension,args.layers).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    edges=torch.tensor(np.stack([e.state.edges for e in envs]),device=device)
    start=time.perf_counter();history=[];counts=Counter()
    for update in range(args.updates):
        buffer=[];rewards=[];values=[];dones=[]
        model.eval()
        for step in range(args.rollout):
            encoded=[env.encode() for env in envs]
            with torch.no_grad():logits,v=forward(model,tensors(encoded,device),edges)
            scores=[x.cpu().numpy() for x in logits];v=v.cpu().numpy();row_rewards=[];row_done=[]
            for i,env in enumerate(envs):
                action,masks,logprob=sample_action(env.state,env.plan,[x[i] for x in scores],rng)
                reward,status=env.step(action);done=env.age>=env.horizon
                counts[status]+=1;counts[OPS[action.operator]+'_attempts']+=1
                if status=='accepted':counts[OPS[action.operator]+'_accepted']+=1
                buffer.append((encoded[i],i,list(asdict(action).values()),masks,logprob))
                row_rewards.append(reward);row_done.append(done)
                if done:env.reset()
            rewards.append(row_rewards);values.append(v);dones.append(row_done)
        with torch.no_grad():_,last=forward(model,tensors([e.encode() for e in envs],device),edges)
        rewards=np.array(rewards);values=np.array(values);dones=np.array(dones)
        advantage=np.zeros_like(rewards);gae=np.zeros(len(envs));next_value=last.cpu().numpy()
        for t in reversed(range(args.rollout)):
            continuation=1.-dones[t]
            delta=rewards[t]+.95*next_value*continuation-values[t]
            gae=delta+.95*.90*continuation*gae;advantage[t]=gae;next_value=values[t]
        returns=torch.tensor((advantage+values).flatten(),device=device,dtype=torch.float32)
        advantage=(advantage.flatten()-advantage.mean())/(advantage.std()+1e-8)
        adv=torch.tensor(advantage,device=device,dtype=torch.float32)
        features=tensors([item[0] for item in buffer],device)
        env_indices=torch.tensor([item[1] for item in buffer],device=device)
        actions=torch.tensor([item[2] for item in buffer],device=device)
        masks=[torch.tensor(np.stack([item[3][i] for item in buffer]),device=device) for i in range(4)]
        oldlog=torch.tensor([item[4] for item in buffer],device=device,dtype=torch.float32)
        model.train();losses=[];kls=[];gradients=[]
        for epoch in range(args.epochs):
            order=rng.permutation(len(buffer))
            for offset in range(0,len(buffer),args.minibatch):
                idx=torch.tensor(order[offset:offset+args.minibatch],device=device)
                logits,value=forward(model,[x[idx] for x in features],edges[env_indices[idx]])
                dist=distributions(logits,[m[idx] for m in masks])
                logprob=sum(d.log_prob(actions[idx,i]) for i,d in enumerate(dist))
                entropy=sum(d.entropy() for d in dist).mean()
                ratio=(logprob-oldlog[idx]).exp()
                actor=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()
                loss=actor+.5*(value-returns[idx]).square().mean()-args.entropy*entropy
                optim.zero_grad(set_to_none=True);loss.backward()
                gradients.append([float(h.weight.grad.norm()) for h in model.heads])
                torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optim.step()
                losses.append(float(loss.detach()));kls.append(float((oldlog[idx]-logprob).mean().detach()))
        record={'update':update+1,'attempts':(update+1)*args.rollout*len(envs),
            'seconds':time.perf_counter()-start,'mean_current_J':float(np.mean([e.metrics['weighted_objective'] for e in envs])),
            'min_current_J':min(e.metrics['weighted_objective'] for e in envs),
            'mean_visits':float(np.mean([sum(len(r)-2 for r in e.plan.values()) for e in envs])),
            'reward_mean':float(rewards.mean()),'loss':float(np.mean(losses)),'approx_kl':float(np.mean(kls)),
            'head_gradient_norms':np.mean(gradients,axis=0).tolist(),'counts':dict(counts)}
        history.append(record);write(dest/'history.json',history)
        print(json.dumps(record),flush=True)
    checkpoint=dest/'large_model.pt'
    model.eval();torch.save({'format':'full_objects_v7','model_state_dict':model.cpu().state_dict(),
        'dimension':args.dimension,'layers':args.layers,'config':config,'history':history},checkpoint)
    for path,h in config['sources'].items():assert sha(ROOT/path)==h,'Source changed while training'
    write(dest/'completed.json',{'checkpoint_sha256':sha(checkpoint),'seconds':time.perf_counter()-start,
        'optimizer_steps':args.updates*args.epochs*((len(envs)*args.rollout+args.minibatch-1)//args.minibatch),
        'training_only':True,'source_config_sha256':sha(dest/'config.json')})


def evaluate(args):
    dest,config=initialize(args);torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);device=torch.device(args.device)
    state,plan,data=load_case(ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json')
    env=SearchEnvironment(state,plan,(.5,.5),args.horizon)
    model=None
    if args.checkpoint:
        ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['format']=='full_objects_v7'
        if ckpt['config']['args'].get('variant','a')!=args.variant:
            raise ValueError('Checkpoint encoder variant does not match evaluation; no silent substitution')
        model=ObjectPolicy(ckpt['dimension'],ckpt['layers']).to(device);model.load_state_dict(ckpt['model_state_dict']);model.eval()
        config['checkpoint']={'path':args.checkpoint,'sha256':sha(args.checkpoint)}
    from run_ppo_scalar_exploration import benchmark
    _,target=benchmark();config['benchmark']=target
    config['input_sha256']=sha(ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json')
    config['feasibility_masks']=args.masks
    config['sampling']='all arguments independently scored by network, sequential legal masks; no cheapest insertion'
    if model is None:config['sampling']='UNIFORM CONTROL: no network, all operator and object scores zero'
    if args.object_ablation=='uniform':
        if model is None:raise ValueError('Object-only ablation requires a trained operator policy')
        config['sampling']='ABLATION: trained PPO operator, all three object logits replaced by uniform scores'
    config['object_heads_used']=model is not None and args.object_ablation=='none'
    config['candidate_pool']='network samples width tuples from SAME state, choose best feasible strict improvement'
    write(dest/'config.json',config)
    edge=torch.tensor(state.edges,device=device)[None];counts=Counter();trace=[];start=time.perf_counter()
    attempts=0;forbidden=[];memory_key=None
    while attempts<args.budget:
        encoded=env.encode()
        if model is None:scores=[np.zeros(6),*([np.zeros(512)]*3)]
        else:
            with torch.inference_mode():logits,_=forward(model,tensors([encoded],device),edge)
            scores=[x[0].cpu().numpy()/args.temperature for x in logits]
            if args.object_temperature is not None:
                scores[1:]=[x[0].cpu().numpy()/args.object_temperature for x in logits[1:]]
            if args.object_ablation=='uniform':scores[1:]=[np.zeros(512) for _ in range(3)]
        starting=(env.plan,env.solution,env.metrics);best=starting;best_action=None;block_trace=[]
        current_key=(id(env.plan),min(env.age,args.horizon))
        if not args.memoize or current_key!=memory_key:forbidden=[]
        memory_key=current_key
        for _ in range(min(args.width,args.budget-attempts)):
            env.plan,env.solution,env.metrics=starting
            if args.unique or args.memoize:
                action,masks,lp=sample_unique(state,env.plan,lambda chosen:scores[len(chosen)],rng,forbidden)
            else:action,masks,lp=sample_action(state,env.plan,scores,rng)
            _,status=env.step(action);attempts+=1;counts[status]+=1;counts[OPS[action.operator]+'_attempts']+=1
            if status=='accepted':counts[OPS[action.operator]+'_accepted_proposals']+=1
            if env.metrics['weighted_objective']<best[2]['weighted_objective']-1e-12:
                best=(env.plan,env.solution,env.metrics);best_action=asdict(action)
            block_trace.append({'attempt':attempts,'action':asdict(action),'status':status,'log_probability':lp})
        env.plan,env.solution,env.metrics=best
        if best_action:
            trace.append({'attempt':attempts,'action':best_action,'J':env.metrics['weighted_objective'],
                          'C':env.metrics['cost'],'R':env.metrics['risk']})
        if attempts % 400==0 or attempts==args.budget:
            progress={'attempts':attempts,'J':env.metrics['weighted_objective'],'seconds':time.perf_counter()-start,
                      'counts':dict(counts)}
            print(json.dumps(progress),flush=True);write(dest/'progress.json',progress)
        # Full sampled tuple trace, append-only scientific output, not source editing.
        with (dest/'actions.jsonl').open('a',encoding='utf8') as stream:
            for item in block_trace:stream.write(json.dumps(item)+'\n')
    result={'metrics':env.metrics,'plan':plan_to_canonical_data(env.plan),'solution_id':plan_sha256(env.plan),
            'counts':dict(counts),'trace':trace,'attempts':attempts,'seconds':time.perf_counter()-start,
            'beats_all_existing_nsga':env.metrics['weighted_objective']<target['target_J']-1e-8,
            'config_sha256':sha(dest/'config.json'),'checkpoint_used':model is not None,
            'object_heads_used':config['object_heads_used']}
    write(dest/'result.json',result)
    if args.audit:
        from audit_scalar_advantage_v5 import full_model_replay
        result['full_milp_audit']=full_model_replay(state.p,[(env.plan,env.metrics)])
        write(dest/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('plan','trace')}),flush=True)


def main():
    global ObjectState
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','evaluate']);parser.add_argument('--id',required=True)
    parser.add_argument('--seed',type=int,default=20260909);parser.add_argument('--device',default='cuda')
    parser.add_argument('--dimension',type=int,default=96);parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--updates',type=int,default=32);parser.add_argument('--rollout',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--minibatch',type=int,default=96)
    parser.add_argument('--horizon',type=int,default=1024);parser.add_argument('--lr',type=float,default=0.0003)
    parser.add_argument('--entropy',type=float,default=.003);parser.add_argument('--checkpoint')
    parser.add_argument('--budget',type=int,default=8000);parser.add_argument('--width',type=int,default=8)
    parser.add_argument('--temperature',type=float,default=1.);parser.add_argument('--audit',action='store_true')
    parser.add_argument('--object-temperature',type=float)
    parser.add_argument('--unique',action='store_true')
    parser.add_argument('--memoize',action='store_true',help='No repeat tuples while plan and capped age are unchanged')
    parser.add_argument('--variant',choices=['a','b'],default='b')
    parser.add_argument('--masks',choices=['base','processing'],default='processing')
    parser.add_argument('--mixed-horizons',action='store_true')
    parser.add_argument('--object-ablation',choices=['none','uniform'],default='none')
    args=parser.parse_args()
    if Path(args.id).name!=args.id:raise ValueError('ID must be a basename')
    if args.mode=='train' and args.object_ablation!='none':raise ValueError('Object ablation is evaluation-only')
    if args.variant=='b':
        from src.ppo_objects_v7b import ObjectState
    if args.layers != 1:raise ValueError('User forbids additional Transformer layers')
    if args.temperature<=0 or (args.object_temperature is not None and args.object_temperature<=0):raise ValueError('Temperatures must be positive')
    if args.mode=='evaluate' and args.budget>12000:raise ValueError('Fixed candidate budget maximum is 12000')
    if args.masks=='processing':
        if args.variant!='b':raise ValueError('Processing masks require the unchanged b encoder')
        from src.ppo_objects_v8 import ObjectState
    if args.mode=='train':train(args)
    else:evaluate(args)


if __name__=='__main__':main()

"""Exact regressions for invisible edges and ambiguous action identities."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import unittest

from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan
from src.ppo_objects_v7 import ObjectState,ObjectPolicy,Action,DUMMY,OPS,SearchEnvironment,sample_action,distributions

ROOT=Path(__file__).resolve().parents[1]


def case():
    data=json.loads((ROOT/'output/pareto-parameter-revision-v4/instances/Test-4.json').read_text(encoding='utf8'))
    return params_from_json_data(data['params']),deserialize_plan(data['reference_plan']),(data['b_C'],data['b_R'])


def test_unused_candidate_arc_is_visible():
    p,plan,refs=case();q=copy.deepcopy(p);q.distance['n::G10::S3','D2']=74.28453933333314
    a,b=ObjectState(p,refs),ObjectState(q,refs)
    assert not np.array_equal(a.edges,b.edges)
    assert a.edges.shape==(83,128,3)
    assert a.edges[a.node_index['n::G10::S3'],a.node_index['D2'],0]!=b.edges[b.node_index['n::G10::S3'],b.node_index['D2'],0]


def test_route_membership_and_stable_objects():
    p,plan,refs=case();s=ObjectState(p,refs);keys=[k for k in plan if k[1]==p.periods[0]]
    changed=copy.deepcopy(plan);changed[keys[0]],changed[keys[1]]=changed[keys[1]],changed[keys[0]]
    before=s.encode(plan,(.5,.5));after=s.encode(changed,(.5,.5))
    assert not np.array_equal(before[0][:s.r0,16],after[0][:s.r0,16])
    n=plan[keys[0]][1];slot=s.pickup_slot[n,keys[0][1]]
    assert s.pickups[slot]==(n,keys[0][1])
    assert s.membership(plan)[slot][0]==keys[0]
    assert s.membership(changed)[slot][0]==keys[1]


def test_exact_drop_and_no_modulo_alias():
    p,plan,refs=case();s=ObjectState(p,refs);key=next(k for k in plan if k[1]!=p.periods[-1]);node=plan[key][1]
    slot=s.pickup_slot[node,key[1]];out,status=s.apply(plan,Action(0,slot))
    assert status=='candidate';assert out[key]==[plan[key][0],*plan[key][2:]]
    assert all(out[k]==r for k,r in plan.items() if k!=key)
    assert s.apply(plan,Action(0,slot+512))[0] is None
    assert s.apply(plan,Action(0,s.pickup_slot[node,p.periods[-1]]))[0] is None


def test_third_head_controls_exact_insertion_not_cheapest():
    p,plan,refs=case();s=ObjectState(p,refs)
    for slot,(key,pos) in s.membership(plan).items():
        target=s.route_slot[key];mask=s.masks(plan,(1,slot,target))
        anchors=np.flatnonzero(mask)
        for anchor in anchors:
            if anchor>=s.f0:continue
            out,status=s.apply(plan,Action(1,slot,target,int(anchor)))
            if out is not None:
                n=s.pickups[slot][0]
                assert out[key].index(n)==(1 if anchor==target else out[key].index(s.pickups[anchor][0])+1)
                return
    raise AssertionError('Expected a directly executable relocation')


def test_three_independent_512_heads_and_gradients():
    torch.set_num_threads(1)
    p,plan,refs=case();s=ObjectState(p,refs);f,links,interaction=s.encode(plan,(.5,.5))
    model=ObjectPolicy(dimension=32)
    logits,value=model(torch.tensor(f)[None],torch.tensor(links)[None],torch.tensor(s.edges)[None],torch.tensor(interaction)[None])
    assert [x.shape for x in logits]==[(1,6),(1,512),(1,512),(1,512)]
    loss=sum(x.square().mean() for x in logits)+value.square().mean();loss.backward()
    assert all(h.weight.grad is not None and h.weight.grad.norm()>0 for h in model.heads)
    assert model.edge_encoder[0].weight.grad.norm()>0


def test_fixed_b_immutable_through_actions():
    p,plan,refs=case();s=ObjectState(p,refs);env=SearchEnvironment(s,plan)
    slot=int(np.flatnonzero(s.masks(plan,(0,)))[0]);env.step(Action(0,slot))
    assert s.refs==refs
    assert abs(env.metrics['weighted_objective']-(.5*env.metrics['cost']/refs[0]+.5*env.metrics['risk']/refs[1]))<1e-12


def test_masked_sampling_matches_ppo_log_probability():
    p,plan,refs=case();s=ObjectState(p,refs);rng=np.random.default_rng(4096)
    logits=[rng.normal(size=6),rng.normal(size=512),rng.normal(size=512),rng.normal(size=512)]
    for repeat in range(30):
        action,masks,old_log=sample_action(s,plan,logits,rng)
        dist=distributions([torch.tensor(x)[None] for x in logits],
                           [torch.tensor(m)[None] for m in masks])
        choices=[action.operator,action.first,action.second,action.third]
        replay=sum(d.log_prob(torch.tensor([choice])).item() for d,choice in zip(dist,choices))
        assert abs(replay-old_log)<1e-10
        assert all(mask[choice] for mask,choice in zip(masks,choices))


def test_unused_edge_changes_policy_outputs_not_only_raw_storage():
    torch.manual_seed(3)
    p,plan,refs=case();q=copy.deepcopy(p);q.distance['n::G10::S3','D2']=74.28453933333314
    a,b=ObjectState(p,refs),ObjectState(q,refs);model=ObjectPolicy(dimension=32).eval()
    def run(state):
        f,links,interaction=state.encode(plan,(.5,.5))
        with torch.no_grad():
            return model(torch.tensor(f)[None],torch.tensor(links)[None],
                         torch.tensor(state.edges)[None],torch.tensor(interaction)[None])[0]
    assert all(not torch.equal(x,y) for x,y in zip(run(a),run(b)))


def test_v7b_drop_mask_never_hides_a_feasible_drop():
    from src.ppo_objects_v7b import ObjectState as CapacityState
    from src.solution_utils import route_plan_to_solution,evaluate_solution
    p,plan,refs=case();base=ObjectState(p,refs);state=CapacityState(p,refs)
    env=SearchEnvironment(base,plan)
    for slot in np.flatnonzero(base.masks(plan,(0,)))[:120]:
        env.step(Action(0,int(slot)))
    plan=env.plan
    permitted=state.masks(plan,(0,));excluded=0
    for slot in np.flatnonzero(base.masks(plan,(0,))):
        if permitted[slot]:continue
        excluded+=1;candidate,status=base.apply(plan,Action(0,int(slot)))
        if candidate is not None:
            metric=evaluate_solution(p,route_plan_to_solution(p,candidate),(.5,.5),refs)
            assert not metric['feasible'],(slot,metric)
    assert excluded>0


def test_v7b_cross_period_links_and_absolute_risk_scale():
    from src.ppo_objects_v7b import ObjectState as CapacityState
    p,plan,refs=case();state=CapacityState(p,refs);f,links,interaction=state.encode(plan,(.5,.5))
    assert np.count_nonzero(f[:state.r0,49])>0
    changed=copy.deepcopy(p);changed.coload_risk.update({k:v*2 for k,v in p.coload_risk.items()})
    other=CapacityState(changed,refs).encode(plan,(.5,.5))[2]
    np.testing.assert_allclose(other[:16],interaction[:16]*2)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))

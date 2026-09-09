import unittest
import torch
import numpy as np
from src.ppo_objects_v8f import ObjectState,ObjectPolicy
from src.ppo_objects_v8c import ObjectPolicy as PreviousPolicy
from src.ppo_objects_v8 import ObjectState as PreviousState
from test_ppo_objects_v8 import case


def test_extra_input_matches_actual_route_loads_and_next_edges():
    p,plan,refs=case();state=ObjectState(p,refs);f,l,i=state.encode(plan,(.5,.5))
    assert f.shape==(state.size,80)
    old=PreviousState(p,refs).encode(plan,(.5,.5))[0]
    np.testing.assert_array_equal(f[:,:64],old)
    slot=50;n,t=state.pickups[slot];key,pos=state._members[slot]
    for index,s in enumerate(p.waste_types):
        assert abs(f[slot,64+index]-state._raw.get(('F',plan[key][pos-1],n,s,key[0],t),0.)/p.vehicle_capacity)<1e-7
    assert f[slot,79]==0
    assert np.isfinite(f).all()


def test_added_features_leave_initial_policy_unchanged_but_learn():
    p,plan,refs=case();state=ObjectState(p,refs)
    f,l,i=[torch.tensor(x)[None] for x in state.encode(plan,(.5,.5))];edges=torch.tensor(state.edges)[None]
    torch.manual_seed(88);old=PreviousPolicy(64,1)
    torch.manual_seed(88);new=ObjectPolicy(64,1)
    actions=torch.tensor([[1,50,320,511]])
    torch.set_num_threads(1)
    a,_=old(f[:,:,:64],l,edges,i,actions);b,v=new(f,l,edges,i,actions)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    loss=sum(x.square().mean() for x in b)+v.square().mean();loss.backward()
    assert new.features.weight.grad[:,64:].abs().sum()>0
    assert len(new.transformer.layers)==1


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))

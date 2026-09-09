"""Full-object conditional likelihood, gradient and routing-input regression."""
import unittest
import numpy as np
import torch
from src.ppo_objects_v8c import ObjectPolicy, sample_batch
from src.ppo_objects_v8 import ObjectState
from src.ppo_objects_v7 import distributions
from test_ppo_objects_v8 import case


def setup():
    torch.manual_seed(19);torch.set_num_threads(1)
    p,plan,refs=case();state=ObjectState(p,refs)
    model=ObjectPolicy(96,1)
    f,l,i=[torch.tensor(x)[None] for x in state.encode(plan,(.5,.5))]
    edges=torch.tensor(state.edges)[None]
    return model,state,plan,(f,l,edges,i)


def test_preceding_object_changes_later_logits():
    model,state,plan,inputs=setup();encoded=model.encode(*inputs)
    a=model.head_logits(encoded,torch.tensor([[1,0]]))
    b=model.head_logits(encoded,torch.tensor([[1,1]]))
    assert not torch.allclose(a,b,atol=1e-6)
    assert a.shape==(1,512)
    assert len(model.transformer.layers)==1
    assert len({id(head.weight) for head in model.heads})==3


def test_rollout_likelihood_matches_teacher_forcing():
    model,state,plan,inputs=setup();encoded=model.encode(*inputs)
    rng=np.random.default_rng(222)
    for _ in range(16):
        action,masks,lp=sample_batch(model,encoded,[state],[plan],rng)[0]
        actions=torch.tensor([[action.operator,action.first,action.second,action.third]])
        logits,value=model(*inputs,actions)
        dist=distributions(logits,[torch.tensor(mask)[None] for mask in masks])
        replay=sum(d.log_prob(actions[:,i]) for i,d in enumerate(dist))
        assert abs(float(replay)-lp)<2e-5
        model.zero_grad();(-replay+value.square()).sum().backward(retain_graph=True)
        assert any(h.weight.grad is not None and h.weight.grad.abs().sum()>0 for h in model.heads)


def test_candidate_edge_inputs_affect_conditional_scores():
    model,state,plan,inputs=setup()
    changed=list(inputs);changed[2]=inputs[2].clone()
    changed[2][:,:, :,0]*=.3
    actions=torch.tensor([[1,0,320,511]])
    before,_=model(*inputs,actions);after,_=model(*changed,actions)
    assert not torch.allclose(before[3],after[3],atol=1e-6)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))

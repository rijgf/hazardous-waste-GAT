import unittest
import torch
from torch import nn
from src.ppo_objects_v8v import ObjectPolicy
from src.ppo_objects_v8c import ObjectPolicy as PreviousPolicy


def test_only_critic_initial_parameters_change():
    torch.manual_seed(20260909);before=PreviousPolicy(96,1)
    torch.manual_seed(20260909);after=ObjectPolicy(96,1)
    for name,value in before.state_dict().items():
        if not name.startswith('value.'):
            torch.testing.assert_close(value,after.state_dict()[name],rtol=0,atol=0)
    assert isinstance(before.value,nn.Linear)
    assert isinstance(before.context[1],nn.GELU)
    assert isinstance(after.value[1],nn.GELU) and isinstance(after.value[-1],nn.Linear)
    assert sum(p.numel() for p in after.value.parameters())-sum(p.numel() for p in before.value.parameters())==9312
    assert len(after.transformer.layers)==1


def test_critic_has_curvature_and_both_signs_remain_representable():
    torch.manual_seed(33);m=ObjectPolicy(96,1);x=torch.randn(3,96,requires_grad=True)
    value=m.value(x);gradient=torch.autograd.grad(value.sum(),x,create_graph=True)[0]
    second=torch.autograd.grad(gradient.sum(),x)[0]
    assert second.abs().max()>1e-5
    with torch.no_grad():
        m.value[-1].weight.zero_();m.value[-1].bias.fill_(-4.)
    assert (m.value(x)<0).all()
    with torch.no_grad():m.value[-1].bias.fill_(4.)
    assert (m.value(x)>1).all()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))

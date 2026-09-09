import unittest
import numpy as np
from src.ppo_unique_sampling import sample_unique
from src.ppo_objects_v7 import ObjectState


class TinyState:
    mask=staticmethod(ObjectState.mask)
    def masks(self,plan,chosen):
        if not chosen:return np.array([True,True])
        return self.mask([0,1] if len(chosen)==1 else [511])


def test_exact_joint_conditional_probability_without_duplicates():
    state=TinyState();rng=np.random.default_rng(243)
    scores=[np.log([.3,.7]),np.zeros(512),np.zeros(512),np.zeros(512)]
    scores[1][:2]=np.log([.2,.8]);base={(0,0):.06,(0,1):.24,(1,0):.14,(1,1):.56}
    for _ in range(100):
        forbidden=[];remaining=1.;seen=set()
        for i in range(4):
            action,_,lp=sample_unique(state,{},lambda chosen:scores[len(chosen)],rng,forbidden)
            key=(action.operator,action.first)
            assert key not in seen;seen.add(key)
            assert abs(np.exp(lp)-base[key]/remaining)<1e-10
            remaining-=base[key]
        assert len(seen)==4


def test_first_draw_is_original_policy():
    state=TinyState();scores=[np.array([2.,1.]),*([np.zeros(512)]*3)]
    action,masks,lp=sample_unique(state,{},lambda chosen:scores[len(chosen)],np.random.default_rng(12),[])
    expected=np.exp(scores[0][action.operator])/np.exp(scores[0]).sum()*.5
    assert abs(np.exp(lp)-expected)<1e-12


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))

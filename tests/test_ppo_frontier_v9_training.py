import unittest
from collections import Counter
from train_ppo_frontier_v9 import (PREFERENCES, preference_schedule, protected_inputs,
                                   reset_preference, ObjectState, ROOT, read)
from src.ppo_objects_v8 import ObjectState as StrictState
from src.ppo_objects_v7 import SearchEnvironment
from src.reproducibility import deserialize_plan
from sample_params import params_from_json_data


def test_predeclared_schedule_covers_all_preferences_without_extra_actions():
    episodes, totals = preference_schedule()
    assert preference_schedule() == (episodes,totals)
    assert len(episodes) == 48 and sum(totals) == 36864 and min(totals) > 0
    assert max(totals)-min(totals) <= 512
    by_instance = Counter()
    for e in episodes:
        assert e['start_action'] % e['horizon'] == 0
        assert e['horizon'] == 512*(1+e['instance']%3)
        assert e['preference'] == list(PREFERENCES[e['preference_index']])
        by_instance[e['instance']] += e['actions']
    assert len(by_instance)==24 and set(by_instance.values())=={1536}
    assert PREFERENCES[0] == (0.,1.) and PREFERENCES[-1] == (1.,0.)


def test_training_uses_exact_manifest_and_strict_state():
    assert ObjectState is StrictState
    for scale in ['Test-1','Test-4']:
        paths, checked = protected_inputs(scale)
        assert len(paths)==24
        assert all('/train/' in str(p).replace('\\','/') for p in paths)
        assert all(p.relative_to(ROOT).as_posix() in checked for p in paths)


def test_preference_reset_recomputes_scalar_objective():
    paths,_ = protected_inputs('Test-1')
    data = read(paths[0])
    env = SearchEnvironment(ObjectState(params_from_json_data(data['params']),
                                       (data['b_C'],data['b_R'])),
                            deserialize_plan(data['reference_plan']),(.5,.5),512)
    env.metrics['weighted_objective'] = -99
    env.age = 512
    reset_preference(env,(.2,.8))
    expected = .2*env.metrics['cost']/data['b_C']+.8*env.metrics['risk']/data['b_R']
    assert abs(env.metrics['weighted_objective']-expected)<1e-12
    assert env.age == 0 and env.preference == (.2,.8)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items()
                              if k.startswith('test_') and callable(v))

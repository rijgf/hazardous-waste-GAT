"""Replay the exact state/age boundary at which search exclusions may persist."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def repeats(folder):
    config=json.loads((folder/'config.json').read_text(encoding='utf8'))['args']
    result=json.loads((folder/'result.json').read_text(encoding='utf8'))
    actions=[json.loads(line) for line in (folder/'actions.jsonl').read_text(encoding='utf8').splitlines()]
    boundaries={x['attempt'] for x in result['trace']};version=0;previous=None;seen=set();count=0
    for i in range(0,len(actions),config['width']):
        key=(version,min(i,config['horizon']))
        if key!=previous:seen=set()
        previous=key
        block=actions[i:i+config['width']]
        for item in block:
            value=tuple(item['action'].values())
            count+=value in seen;seen.add(value)
        if i+len(block) in boundaries:version+=1
    return count


def test_pool_only_uniqueness_does_not_prevent_cross_pool_repeats():
    assert repeats(ROOT/'output/ppo-objects-v8/evaluate/conditional96_unique_r0')>100


def test_memory_prevents_all_repeats_when_plan_and_input_are_unchanged():
    for index in range(3):
        assert repeats(ROOT/f'output/ppo-objects-v8/evaluate/conditional_memory_r{index}')==0


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))

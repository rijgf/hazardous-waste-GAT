import random
import unittest

from run_nsga_parameter_probe import CONFIGS, hv, parameterized
from src import pareto_experiment as core


class ParameterProbeTests(unittest.TestCase):
    def test_wrapper_preserves_exact_child_and_random_stream(self):
        original = core.MultiPeriodEncoding
        class Params:
            periods = [0, 1]
            pickup_nodes = ['G1', 'G2']
            vehicles = ['V1', 'V2']
            facilities = ['D1', 'D2']
        plain = original(Params())
        left = plain.random_genome(random.Random(1))
        right = plain.random_genome(random.Random(2))
        for _, cross, mutation in CONFIGS.values():
            a, b = random.Random(19), random.Random(19)
            with parameterized(cross, mutation) as counters:
                wrapped = core.MultiPeriodEncoding(Params())
                for _ in range(100):
                    self.assertEqual(plain.child(left,right,a,cross,mutation),wrapped.child(left,right,b))
                self.assertEqual(a.getstate(),b.getstate())
                self.assertEqual(counters['children'],100)
            self.assertIs(core.MultiPeriodEncoding,original)

    def test_hv_rectangle_union(self):
        points = [{'cost':.2,'risk':.8},{'cost':.8,'risk':.2}]
        self.assertAlmostEqual(hv(points,(1,1)),.45)
        self.assertAlmostEqual(hv(list(reversed(points)),(1,1)),.45)
        self.assertAlmostEqual(hv(points+[{'cost':2,'risk':.1}],(1,1)),.45)

    def test_exception_restores_original(self):
        original = core.MultiPeriodEncoding
        with self.assertRaises(RuntimeError):
            with parameterized(.9,.25):
                raise RuntimeError('probe')
        self.assertIs(core.MultiPeriodEncoding,original)

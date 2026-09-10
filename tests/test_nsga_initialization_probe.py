import random
import unittest
from run_nsga_initialization_probe import probability
from src import pareto_experiment as core


class InitializationTests(unittest.TestCase):
    class Params:
        periods=[0,1,2]
        pickup_nodes=['G1','G2']
        vehicles=['V1','V2']
        facilities=['D1','D2']

    def test_half_exactly_matches_original(self):
        original=core.MultiPeriodEncoding
        plain=original(self.Params())
        a,b=random.Random(19),random.Random(19)
        with probability(.5):
            changed=core.MultiPeriodEncoding(self.Params())
            for _ in range(100):
                self.assertEqual(plain.random_genome(a),changed.random_genome(b))
            self.assertEqual(a.getstate(),b.getstate())
        self.assertIs(core.MultiPeriodEncoding,original)

    def test_probability_does_not_change_other_genes(self):
        with probability(.8):
            x=core.MultiPeriodEncoding(self.Params()).random_genome(random.Random(7))
        with probability(.5):
            y=core.MultiPeriodEncoding(self.Params()).random_genome(random.Random(7))
        self.assertEqual([g[1:] for g in x],[g[1:] for g in y])
        self.assertTrue(all(a[0]>=b[0] for a,b in zip(x,y)))

import unittest
from unittest.mock import patch
import numpy as np
from src.ppo_frontier_v9 import next_unique,scalar_metrics
from tests.test_ppo_unique_sampling import TinyState


class FrontierAdapterTests(unittest.TestCase):
    def test_root_exhaustion_retains_partial_batch(self):
        scores=[np.zeros(2),*([np.zeros(512)]*3)]
        state=TinyState();rng=np.random.default_rng(7);forbidden=[];sampled=[]
        for _ in range(8):
            value=next_unique(state,{},lambda chosen:scores[len(chosen)],rng,forbidden)
            if value is None:break
            sampled.append(value)
        self.assertEqual(len(sampled),4)
        self.assertEqual(len({(x[0].operator,x[0].first) for x in sampled}),4)
        self.assertIsNone(next_unique(state,{},lambda chosen:scores[len(chosen)],rng,forbidden))

    def test_nonroot_exhaustion_is_not_silently_success(self):
        state=TinyState();scores=[np.zeros(2),*([np.zeros(512)]*3)]
        with patch('src.ppo_frontier_v9.sample_unique',side_effect=ValueError('No unseen positive-probability tuple remains')):
            with self.assertRaises(ValueError):
                next_unique(state,{},lambda chosen:scores[len(chosen)],np.random.default_rng(2),[])

    def test_scalar_metrics_recomputes_all_weighted_fields_without_mutation(self):
        point={'cost':30.,'risk':8.,'weighted_objective':99.,'weighted_objective_raw':99.,'feasible':True}
        metric=scalar_metrics(point,(.25,.75),(10.,4.))
        self.assertEqual(metric['weighted_objective'],2.25)
        self.assertEqual(metric['weighted_objective_normalized'],2.25)
        self.assertEqual(metric['weighted_objective_raw'],13.5)
        self.assertEqual(point['weighted_objective'],99.)


if __name__=='__main__':unittest.main()

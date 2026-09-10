import unittest
from run_nsga_spread_probe import metrics


class SpreadMetricsTests(unittest.TestCase):
    def make(self,points):
        return {'refs':[10,100],'result':{'points':[dict(cost=c,risk=r) for c,r in points]}}

    def test_requires_both_axes(self):
        a=metrics(self.make([(1,90),(9,89)]))
        b=metrics(self.make([(1,90),(3,60)]))
        self.assertGreater(b['balanced_span'],a['balanced_span'])

    def test_single_point_not_wide(self):
        m=metrics(self.make([(1,90)]))
        self.assertEqual(m['balanced_span'],0)
        self.assertEqual(m['cost_bin_occupancy'],1)

    def test_gap_and_bins(self):
        m=metrics(self.make([(1,90),(3,60)]))
        self.assertEqual(m['largest_cost_gap_fraction'],1)
        self.assertEqual(m['cost_bin_occupancy'],2)

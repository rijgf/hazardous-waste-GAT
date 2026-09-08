"""Curves are visual interpolations, never additional experimental solutions."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from plot_pareto_sensitivity_variants import curve_points

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pareto-parameter-revision-v4'


class SensitivityVariantsTests(unittest.TestCase):
    def test_interpolation_retains_knots_and_does_not_overshoot(self):
        knots, curve = curve_points([(5, 1), (1, 10), (2, 9), (2.001, 2)])
        for point in knots:
            np.testing.assert_allclose(curve[curve[:, 0] == point[0]][0], point)
        self.assertTrue(np.all(np.diff(curve[:, 1]) <= 1e-10))
        for left, right in zip(knots, knots[1:]):
            segment = curve[(curve[:, 0] >= left[0]) & (curve[:, 0] <= right[0]), 1]
            self.assertTrue(np.all((segment >= right[1] - 1e-10) & (segment <= left[1] + 1e-10)))

    def test_invalid_knots_fail_rather_than_silently_filter(self):
        for points in ([(1, 2), (1, 1)], [(1, 1), (2, 3)], [(1, 1)], [(1, 2), (2, float('nan'))]):
            with self.assertRaises(ValueError):
                curve_points(points)

    def test_delivered_curves_match_saved_data_and_document(self):
        evidence = json.loads((OUT / 'figures/sensitivity_variants.json').read_text(encoding='utf8'))
        source = OUT / evidence['source']
        self.assertEqual(evidence['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(evidence['plot_script_sha256'], hashlib.sha256(
            (ROOT / 'plot_pareto_sensitivity_variants.py').read_bytes()).hexdigest())
        self.assertTrue(evidence['visual_only'])
        self.assertEqual(evidence['new_optimization_runs'], 0)
        with source.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(evidence['scenarios']), 9)
        for name, entry in evidence['scenarios'].items():
            points = [(float(r['cost']), float(r['risk'])) for r in rows if r['scenario'] == name]
            knots, smooth = curve_points(points)
            np.testing.assert_array_equal(entry['original_points'], knots)
            np.testing.assert_array_equal(entry['smooth_visual_points'], smooth)
        for name, digest in evidence['artifacts'].items():
            self.assertEqual(hashlib.sha256((OUT / 'figures' / name).read_bytes()).hexdigest(), digest)
        paper = (ROOT / '供应链管理写作/数值实验与结果分析_论文稿.md').read_text(encoding='utf8')
        for name in ('sensitivity.png', 'sensitivity_line.png', 'sensitivity_smooth.png'):
            self.assertIn('/figures/' + name + ')', paper)
        self.assertIn('不表示两方案之间的成本—风险组合均可行', paper)
        self.assertIn('PCHIP', paper)


if __name__ == '__main__':
    unittest.main()

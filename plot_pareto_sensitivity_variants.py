"""Alternative displays of saved solutions; interpolated values are visual only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/pareto-parameter-revision-v4'
STYLES = {
    70: ('#009b9f', '-', '−30%'),
    80: ('#b49a00', '--', '−20%'),
    100: ('#bd3036', '-', '基准'),
    120: ('#df7f25', '-.', '+20%'),
    130: ('#589bbc', ':', '+30%'),
}


def curve_points(points):
    """Interpolate every original knot, with no filtering or extrapolation."""
    knots = np.asarray(sorted(points), dtype=float)
    if (knots.ndim != 2 or knots.shape[1] != 2 or len(knots) < 2
            or not np.isfinite(knots).all()):
        raise ValueError('At least two finite (cost, risk) points are required')
    if np.any(np.diff(knots[:, 0]) <= 0) or np.any(np.diff(knots[:, 1]) > 0):
        raise ValueError('Require unique increasing costs and nonincreasing risks; never silently discard points')
    interpolator = PchipInterpolator(knots[:, 0], knots[:, 1], extrapolate=False)
    # Sample each interval, including very short intervals, and retain every knot.
    costs = np.unique(np.concatenate([
        np.linspace(a, b, 25) for a, b in zip(knots[:-1, 0], knots[1:, 0])]))
    curve = np.column_stack((costs, interpolator(costs)))
    if not np.allclose(interpolator(knots[:, 0]), knots[:, 1], rtol=1e-12, atol=1e-9):
        raise ValueError('Interpolation failed to preserve original knots')
    if np.any(np.diff(curve[:, 1]) > 1e-8):
        raise ValueError('Interpolation introduced a risk reversal')
    return knots, curve


def plot(out=OUT):
    out = Path(out)
    source = out / 'tables/sensitivity_points.csv'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    scenarios = {}
    for row in rows:
        scenarios.setdefault(row['scenario'], []).append((float(row['cost']), float(row['risk'])))
    curves = {name: curve_points(points) for name, points in scenarios.items()}
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 10, 'svg.fonttype': 'none'})
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    for kind in ('line', 'smooth'):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharex=True, sharey=True)
        for axis, parameter, title in zip(axes, ('capacity', 'coload'),
                ('(a) 处理处置能力', '(b) 共载附加风险系数')):
            for level, (color, style, label) in STYLES.items():
                name = 'baseline' if level == 100 else f'{parameter}-{level}'
                knots, smooth = curves[name]
                points = knots if kind == 'line' else smooth
                axis.plot(points[:, 0], points[:, 1], color=color, linestyle=style,
                          linewidth=1.65 if level == 100 else 1.3, label=label,
                          zorder=4 if level == 100 else 3)
            axis.set_xlabel('经济成本 C（成本模型单位）')
            axis.set_title(title, pad=12)
            axis.grid(True, alpha=.18, linewidth=.6)
            axis.legend(frameon=True, fontsize=9, handlelength=3.2)
        axes[0].set_ylabel('系统风险 R（模型风险指标）')
        fig.tight_layout(w_pad=2)
        fig.savefig(figures / f'sensitivity_{kind}.svg', bbox_inches='tight')
        fig.savefig(figures / f'sensitivity_{kind}.png', dpi=350, bbox_inches='tight')
        plt.close(fig)
    evidence = {
        'source': 'tables/sensitivity_points.csv', 'source_sha256': before,
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method': 'PCHIP risk(cost); every original knot retained; no extrapolation',
        'visual_only': True, 'new_optimization_runs': 0,
        'scenarios': {name: {'original_points': knots.tolist(), 'smooth_visual_points': smooth.tolist()}
                      for name, (knots, smooth) in curves.items()},
        'artifacts': {f'sensitivity_{kind}.{ext}': hashlib.sha256(
            (figures / f'sensitivity_{kind}.{ext}').read_bytes()).hexdigest()
            for kind in ('line', 'smooth') for ext in ('png', 'svg')},
    }
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError('Source coordinates changed during plotting')
    (figures / 'sensitivity_variants.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    plot()

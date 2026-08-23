# Experiment Summary

## Method Summary

| scale | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | runtime_mean | feasible_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large | GA | 0.95846 | 16460.9 | 31357.2 | 1564.65 | 26.3196 | 1 |
| large | Heuristic | 1 | 16062.1 | 30353.9 | 1770.23 | 1.188e-05 | 1 |
| large | PPO | 0.640952 | 11493.8 | 23921.9 | 1051.96 | 4.2375 | 1 |
| small | GA | 1 | 765.216 | 1523.39 | 7.04355 | 2.94639 | 1 |
| small | Heuristic | 1 | 765.216 | 1523.39 | 7.04355 | 4.81999e-06 | 1 |
| small | MILP | 0.608635 | 722.763 | 1867.75 | 3.16226 | 116.497 | 1 |
| small | PPO | 0.808075 | 673.871 | 1373.3 | 5.36825 | 1.13765 | 1 |

## Gap Summary

| scale | gap_baseline | method | gap_percent_mean |
| --- | --- | --- | --- |
| large | PPO | GA | 49.8452 |
| large | PPO | Heuristic | 56.9692 |
| large | PPO | PPO | 0 |
| small | MILP | GA | 26.329 |
| small | MILP | Heuristic | 26.329 |
| small | MILP | MILP | 0 |
| small | MILP | PPO | 6.61864 |

## Output Files

- `results.csv`: raw method metrics.
- `results_with_gap.csv`: metrics plus unified gap column. Small-scale gaps use MILP as baseline; large-scale gaps use PPO as baseline.
- `training_history.csv`: PPO training curves.
- `models/`: saved PPO model checkpoints.
- `figures/`: generated result figures.
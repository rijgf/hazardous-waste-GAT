# Experiment Summary

## Method Summary

| scale | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | runtime_mean | feasible_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large | GA | 0.95846 | 16460.9 | 31357.2 | 1564.65 | 26.066 | 1 |
| large | Heuristic | 1 | 16062.1 | 30353.9 | 1770.23 | 9.36e-06 | 1 |
| large | PPO | 0.653606 | 11832 | 24478.1 | 1053.34 | 2.52154 | 1 |
| small | GA | 1 | 765.216 | 1523.39 | 7.04355 | 3.06235 | 1 |
| small | Heuristic | 1 | 765.216 | 1523.39 | 7.04355 | 6.35996e-06 | 1 |
| small | MILP | 0.608635 | 722.763 | 1867.75 | 3.16226 | 120.588 | 1 |
| small | PPO | 0.812384 | 668.204 | 1386.53 | 5.47194 | 0.392329 | 1 |

## Gap Summary

| scale | gap_baseline | method | gap_percent_mean |
| --- | --- | --- | --- |
| large | PPO | GA | 47.6233 |
| large | PPO | Heuristic | 54.8665 |
| large | PPO | PPO | 0 |
| small | MILP | GA | 21.3487 |
| small | MILP | Heuristic | 21.3487 |
| small | MILP | MILP | 0 |
| small | MILP | PPO | 3.97433 |

## Output Files

- `results.csv`: raw method metrics.
- `results_with_gap.csv`: metrics plus unified gap column. Small-scale gaps use MILP as baseline; large-scale gaps use PPO as baseline.
- `training_history.csv`: PPO training curves.
- `models/`: saved PPO model checkpoints.
- `figures/`: generated result figures.
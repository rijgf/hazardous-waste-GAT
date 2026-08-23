# Experiment Summary

## Method Summary

| scale | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | runtime_mean | feasible_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large | GA | 0.95846 | 16460.9 | 31357.2 | 1564.65 | 22.7984 | 1 |
| large | Heuristic | 1 | 16062.1 | 30353.9 | 1770.23 | 5.20002e-06 | 1 |
| large | PPO | 0.590689 | 11215.8 | 23407.6 | 926.964 | 2.03386 | 1 |
| small | GA | 0.92548 | 901.354 | 1820.39 | 9.95385 | 2.71156 | 1 |
| small | Heuristic | 1 | 902.219 | 1792.83 | 11.611 | 4.40003e-06 | 1 |
| small | MILP | 0.831858 | 900.267 | 2153.19 | 8.95238 | 27.5719 | 1 |
| small | PPO | 0.923747 | 901.334 | 1845.61 | 10.2471 | 0.350146 | 1 |

## Small-Scale Gap To MILP

| method | gap_to_milp_mean |
| --- | --- |
| GA | 0.200078 |
| Heuristic | 5.82081 |
| MILP | 0 |
| PPO | 3.36051e-15 |

## Large-Scale Gap To PPO

| method | gap_to_ppo_mean |
| --- | --- |
| GA | 64.5202 |
| Heuristic | 73.0226 |
| PPO | 0 |

## Output Files

- `results.csv`: raw method metrics.
- `results_with_gap.csv`: metrics plus small-scale gap to MILP.
- `training_history.csv`: PPO training curves.
- `models/`: saved PPO model checkpoints.
- `figures/`: generated result figures.
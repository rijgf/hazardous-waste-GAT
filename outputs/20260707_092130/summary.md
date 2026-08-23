# Experiment Summary

## Method Summary

| scale | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | runtime_mean | feasible_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large | GA | 0.95846 | 16460.9 | 31357.2 | 1564.65 | 23.3766 | 1 |
| large | Heuristic | 1 | 16062.1 | 30353.9 | 1770.23 | 4.50001e-06 | 1 |
| large | PPO | 0.590689 | 11215.8 | 23407.6 | 926.964 | 2.09341 | 1 |
| small | GA | 1 | 765.216 | 1523.39 | 7.04355 | 3.44995 | 1 |
| small | Heuristic | 1 | 765.216 | 1523.39 | 7.04355 | 5.97993e-06 | 1 |
| small | MILP | 0.681042 | 635.911 | 1552.95 | 4.5675 | 34.8341 | 1 |
| small | PPO | 0.808075 | 673.871 | 1373.3 | 5.38216 | 0.50803 | 1 |

## Small-Scale Gap To MILP

| method | gap_to_milp_mean |
| --- | --- |
| GA | 23.8831 |
| Heuristic | 23.8831 |
| MILP | 0 |
| PPO | 2.44078 |

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
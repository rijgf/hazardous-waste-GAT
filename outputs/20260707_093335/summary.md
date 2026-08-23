# Experiment Summary

## Method Summary

| scale | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | runtime_mean | feasible_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large | GA | 0.95846 | 16460.9 | 31357.2 | 1564.65 | 26.5202 | 1 |
| large | Heuristic | 1 | 16062.1 | 30353.9 | 1770.23 | 4.41999e-06 | 1 |
| large | PPO | 0.590689 | 11215.8 | 23407.6 | 926.964 | 2.57838 | 1 |
| small | GA | 1 | 765.216 | 1523.39 | 7.04355 | 2.87263 | 1 |
| small | Heuristic | 1 | 765.216 | 1523.39 | 7.04355 | 4.26001e-06 | 1 |
| small | MILP | 0.615175 | 722.809 | 1788.17 | 3.20833 | 47.815 | 1 |
| small | PPO | 0.808075 | 673.871 | 1373.3 | 5.38216 | 0.334414 | 1 |

## Small-Scale Gap To MILP

| method | gap_to_milp_mean |
| --- | --- |
| GA | 21.3487 |
| Heuristic | 21.3487 |
| MILP | 0 |
| PPO | 3.97433 |

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